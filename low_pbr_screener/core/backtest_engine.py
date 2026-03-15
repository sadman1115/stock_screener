"""AI 급등주 롤링 백테스트 엔진.

주별(매주 월요일) 롤링 윈도우 학습 → Top 20 예측 → 매매 시뮬레이션.
데이터를 1회 수집 후 사전 계산하여 반복 학습 시 재사용.

체크포인트 기능: Phase 5 매 5주마다 중간 결과 저장 → 중단 시 이어서 실행.
"""

import hashlib
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from .config import CACHE_DIR
from .ml_pattern import (
    SURGE_FEATURE_NAMES,
    compute_breakout_labels,
    compute_ml_features_surge,
)

warnings.filterwarnings("ignore", category=UserWarning)

BACKTEST_CACHE_DIR = CACHE_DIR / "backtest"
BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 파일 로거 설정
_LOG_FILE = BACKTEST_CACHE_DIR / "backtest.log"
_logger = logging.getLogger("backtest_engine")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    _logger.addHandler(_fh)

# 백테스트용 경량 LightGBM 파라미터
_BACKTEST_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "min_data_in_leaf": 30,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "verbose": -1,
    "seed": 42,
}

_BACKTEST_NUM_BOOST_ROUND = 400  # 기본값 400 (UI에서 300~800 설정 가능)
_CHECKPOINT_INTERVAL = 5  # 5주마다 체크포인트 저장
HOLD_PERIODS = [3, 4, 6, 8, 10, 12]  # 다기간 보유 주수

# 최소 요구사항
_MIN_OHLCV_DAYS = 200
_MIN_TRADE_VALUE = 3_000_000_000  # 30억
_MIN_MCAP = 100_000_000_000  # 1000억
_MIN_POSITIVE_SAMPLES = 10

# ETF 모드 최소 요구사항
_ETF_MIN_TRADE_VALUE = 100_000_000  # 1억
_ETF_MIN_OHLCV_DAYS = 200


# ──────────────────────────────────────────────
# 거래일 조회
# ──────────────────────────────────────────────


def _get_trading_days(start_date: str, end_date: str) -> list[str]:
    """pykrx로 거래일 목록 조회. 실패 또는 빈 결과 시 평일 fallback."""
    try:
        from pykrx import stock

        days = stock.get_previous_business_days(fromdate=start_date, todate=end_date)
        result = [d.strftime("%Y%m%d") for d in days]
        if result:
            return result
    except Exception:
        pass
    # fallback: 평일만
    s = datetime.strptime(start_date, "%Y%m%d")
    e = datetime.strptime(end_date, "%Y%m%d")
    days = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return days


def _next_trading_day(date_str: str, trading_days: list[str]) -> str | None:
    """date_str 이후 첫 번째 거래일 반환."""
    for d in trading_days:
        if d > date_str:
            return d
    return None


def _nearest_trading_day(date_str: str, trading_days: list[str]) -> str | None:
    """date_str 이후(포함) 가장 가까운 거래일."""
    for d in trading_days:
        if d >= date_str:
            return d
    return None


# ──────────────────────────────────────────────
# 데이터 수집 및 사전 계산
# ──────────────────────────────────────────────


def _find_nearest_bt_ohlcv(
    end_date: str, asset_type: str = "stock", data_start_str: str = None
) -> dict | None:
    """최근 7일 이내의 bt_ohlcv 캐시를 탐색하여 반환."""
    from datetime import datetime, timedelta

    _asset_tag = "_etf" if asset_type == "ETF" else ""
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    for days_back in range(1, 8):
        prev_date = (end_dt - timedelta(days=days_back)).strftime("%Y%m%d")
        if data_start_str:
            prev_tag = f"bt_ohlcv{_asset_tag}_{prev_date}_from{data_start_str}"
        else:
            prev_tag = f"bt_ohlcv{_asset_tag}_{prev_date}"
        prev_file = BACKTEST_CACHE_DIR / f"{prev_tag}.pkl"
        if prev_file.exists():
            try:
                with open(prev_file, "rb") as f:
                    data = pickle.load(f)
                if data and len(data) > 0:
                    _logger.info("이전 bt_ohlcv 캐시 발견: %s", prev_file.name)
                    return data
            except Exception:
                continue
    return None


def _prefetch_all_ohlcv(
    tickers, end_date, progress_cb=None, data_start_str=None, asset_type="stock"
):
    """전종목 OHLCV 일괄 수집 (캐시 활용).

    Args:
        data_start_str: 수집 시작일 (YYYYMMDD). None이면 3년(1110일) 전.
        asset_type: "stock" 또는 "ETF"

    Returns:
        dict[str, pd.DataFrame]: ticker → OHLCV DataFrame (DatetimeIndex)
    """
    from .data_fetcher import fetch_ohlcv_3year

    # 캐시 키: 수집 기간 및 자산 유형에 따라 다름
    _asset_tag = "_etf" if asset_type == "ETF" else ""
    if data_start_str:
        cache_tag = f"bt_ohlcv{_asset_tag}_{end_date}_from{data_start_str}"
    else:
        cache_tag = f"bt_ohlcv{_asset_tag}_{end_date}"
    cache_file = BACKTEST_CACHE_DIR / f"{cache_tag}.pkl"

    if cache_file.exists():
        if progress_cb:
            progress_cb("캐시된 OHLCV 데이터 로드 중...", 5)
        _logger.info("OHLCV 캐시 로드: %s", cache_file.name)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # ── 이전 날짜 bt_ohlcv 캐시 재사용 시도 ──
    prev_ohlcv = _find_nearest_bt_ohlcv(end_date, asset_type, data_start_str)
    if prev_ohlcv is not None:
        if progress_cb:
            progress_cb(f"이전 캐시 기반 증분 업데이트 ({len(prev_ohlcv)}종목)...", 5)
        _logger.info("OHLCV 증분 업데이트: 이전 캐시 %d종목 활용", len(prev_ohlcv))
        # Phase 1의 증분 OHLCV를 활용하여 각 ticker 갱신
        ohlcv_dict = {}
        total = len(tickers)
        for i, ticker in enumerate(tickers):
            if progress_cb and i % 50 == 0:
                pct = int(5 + (i / total) * 20)
                progress_cb(f"OHLCV 증분 업데이트: {i}/{total}", pct)
            try:
                df = fetch_ohlcv_3year(ticker, end_date)
                _min_days = (
                    _ETF_MIN_OHLCV_DAYS if asset_type == "ETF" else _MIN_OHLCV_DAYS
                )
                if df is not None and len(df) >= _min_days:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    ohlcv_dict[ticker] = df
            except Exception as e:
                # 증분 실패 시 이전 캐시 데이터 재사용
                if ticker in prev_ohlcv:
                    ohlcv_dict[ticker] = prev_ohlcv[ticker]
                else:
                    _logger.warning("OHLCV 수집 실패 %s: %s", ticker, e)
                continue
        with open(cache_file, "wb") as f:
            pickle.dump(ohlcv_dict, f)
        _logger.info("OHLCV 증분 업데이트 완료: %d종목", len(ohlcv_dict))
        if progress_cb:
            progress_cb(f"OHLCV 증분 업데이트 완료: {len(ohlcv_dict)}종목", 25)
        return ohlcv_dict

    # ── 전체 수집 (콜드 스타트) ──
    ohlcv_dict = {}
    total = len(tickers)
    _logger.info("OHLCV 수집 시작: %d종목 (시작=%s)", total, data_start_str or "3년전")
    for i, ticker in enumerate(tickers):
        if progress_cb and i % 20 == 0:
            pct = int(5 + (i / total) * 20)
            progress_cb(f"OHLCV 수집: {i}/{total} ({ticker})", pct)
        try:
            if data_start_str:
                # 커스텀 시작일 ~ end_date
                from pykrx import stock as pykrx_stock

                df = pykrx_stock.get_market_ohlcv(data_start_str, end_date, ticker)
            else:
                df = fetch_ohlcv_3year(ticker, end_date)
            _min_days = _ETF_MIN_OHLCV_DAYS if asset_type == "ETF" else _MIN_OHLCV_DAYS
            if df is not None and len(df) >= _min_days:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                ohlcv_dict[ticker] = df
        except Exception as e:
            _logger.warning("OHLCV 수집 실패 %s: %s", ticker, e)
            continue

    # 캐시 저장
    with open(cache_file, "wb") as f:
        pickle.dump(ohlcv_dict, f)

    _logger.info("OHLCV 수집 완료: %d종목", len(ohlcv_dict))
    if progress_cb:
        progress_cb(f"OHLCV 수집 완료: {len(ohlcv_dict)}종목", 25)
    return ohlcv_dict


def _precompute_features_labels(
    ohlcv_dict,
    kospi_close,
    end_date,
    progress_cb=None,
    label_horizon_days=63,
    label_target_pct=0.20,
    label_stop_pct=0.08,
    asset_type="stock",
):
    """전종목 피처/라벨 사전 계산 (캐시 활용).

    Args:
        label_horizon_days: 라벨링 horizon (거래일). 기본 63 (3개월).
        label_target_pct: 라벨링 목표수익률. 기본 0.20 (20%).
        label_stop_pct: 라벨링 손절비율. 기본 0.08 (8%).
        asset_type: "stock" 또는 "ETF" (캐시 키 분리용).

    Returns:
        features_dict: {ticker: DataFrame[dates × 55]}
        labels_dict: {ticker: (y_series, hard_series)}
    """
    # 캐시 확인 (라벨링 파라미터별 별도 캐시)
    _asset_tag = "_etf" if asset_type == "ETF" else ""
    cache_file = BACKTEST_CACHE_DIR / (
        f"bt_features{_asset_tag}_{end_date}"
        f"_h{label_horizon_days}_t{int(label_target_pct*100)}_s{int(label_stop_pct*100)}.pkl"
    )
    if cache_file.exists():
        if progress_cb:
            progress_cb("캐시된 피처/라벨 로드 중...", 28)
        _logger.info("피처/라벨 캐시 로드: %s", cache_file.name)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    features_dict = {}
    labels_dict = {}
    total = len(ohlcv_dict)
    errors = 0

    _logger.info("피처/라벨 계산 시작: %d종목", total)
    for i, (ticker, ohlcv) in enumerate(ohlcv_dict.items()):
        if progress_cb and i % 10 == 0:
            pct = int(25 + (i / total) * 10)
            progress_cb(f"피처 계산: {i}/{total} (완료 {len(features_dict)})", pct)
        try:
            feats = compute_ml_features_surge(ohlcv, kospi_close)
            if feats.empty:
                continue

            lbl, _fwd, hard = compute_breakout_labels(
                ohlcv,
                target_pct=label_target_pct,
                stop_pct=label_stop_pct,
                horizon_days=label_horizon_days,
            )
            valid = lbl.notna()
            if valid.sum() < 10:
                continue

            features_dict[ticker] = feats
            labels_dict[ticker] = (lbl, hard)
        except Exception as e:
            errors += 1
            if errors <= 5:
                _logger.warning("피처 계산 실패 %s: %s", ticker, e)
            continue

    _logger.info("피처/라벨 계산 완료: %d종목 (에러 %d)", len(features_dict), errors)

    # 캐시 저장
    with open(cache_file, "wb") as f:
        pickle.dump((features_dict, labels_dict), f)

    if progress_cb:
        progress_cb(f"피처 계산 완료: {len(features_dict)}종목", 35)
    return features_dict, labels_dict


# ──────────────────────────────────────────────
# 경량 LightGBM 학습 (백테스트용)
# ──────────────────────────────────────────────


def _train_backtest_model(
    X_arr, y_arr, hard_neg_arr, feat_names, num_boost_round=_BACKTEST_NUM_BOOST_ROUND
):
    """백테스트용 LightGBM 학습.

    Args:
        num_boost_round: 부스팅 라운드 수 (300~800, 기본 400)

    Returns:
        (model, use_lgb, auc)
    """
    # 가중치
    n_pos = int(y_arr.sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos < _MIN_POSITIVE_SAMPLES:
        return None, False, 0.0

    w_pos = max(n_neg / n_pos, 1.0)
    sample_weight = np.where(y_arr == 1, w_pos, 1.0)
    hard_mask = hard_neg_arr.astype(bool) & (y_arr == 0)
    sample_weight[hard_mask] = 1.5

    # NaN/Inf 처리
    X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # train/val split (85/15)
    n = len(X_clean)
    split = int(n * 0.85)
    X_tr, X_val = X_clean[:split], X_clean[split:]
    y_tr, y_val = y_arr[:split], y_arr[split:]
    sw_tr = sample_weight[:split]

    try:
        import lightgbm as lgb

        dtrain = lgb.Dataset(
            X_tr, y_tr, weight=sw_tr, feature_name=feat_names, free_raw_data=False
        )
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            _BACKTEST_LGB_PARAMS,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # AUC 계산
        from sklearn.metrics import roc_auc_score

        pred_val = model.predict(X_val)
        try:
            auc = roc_auc_score(y_val, pred_val)
        except Exception:
            auc = 0.5

        return model, True, auc
    except ImportError:
        # HGB fallback
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        hgb = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            min_samples_leaf=30,
            learning_rate=0.05,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        hgb.fit(X_clean, y_arr, sample_weight=sample_weight)

        pred = hgb.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, pred)
        except Exception:
            auc = 0.5

        return hgb, False, auc


# ──────────────────────────────────────────────
# 지수 조회 헬퍼
# ──────────────────────────────────────────────


def _lookup_index(index_close, date_str):
    """지수 시리즈에서 date_str(YYYYMMDD) 날짜의 종가를 조회.

    정확한 날짜가 없으면 직전 거래일 값을 반환한다.
    """
    if index_close is None or len(index_close) == 0:
        return None
    try:
        dt = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
        # 정확 매칭
        if dt in index_close.index:
            return float(index_close.at[dt])
        # 직전 거래일
        mask = index_close.index <= dt
        if mask.any():
            return float(index_close[mask].iloc[-1])
    except Exception:
        pass
    return None


def _lookup_index_after_n_days(index_close, date_str, trading_days_list, n_days=63):
    """date_str 이후 n_days 거래일 뒤의 지수 종가를 조회."""
    if index_close is None or not trading_days_list:
        return None
    try:
        future_days = [d for d in trading_days_list if d > date_str]
        if len(future_days) >= n_days:
            target_date = future_days[n_days - 1]
        elif future_days:
            target_date = future_days[-1]
        else:
            return None
        return _lookup_index(index_close, target_date)
    except Exception:
        return None


# ──────────────────────────────────────────────
# 매매 시뮬레이션
# ──────────────────────────────────────────────


def _simulate_trade(
    buy_price, ohlcv_forward, trailing_stop_pct, stop_loss_pct, min_profit_pct=0.0
):
    """단일 종목 매매 시뮬레이션.

    매도 조건 (우선순위):
      1) 손절: 저가가 매수가 × (1 - stop_loss_pct) 이하 → 손절가에 매도
      2) 트레일링스탑: 매수가 대비 A%(min_profit_pct) 이상 상승한 적이 있고,
         고점 대비 B%(trailing_stop_pct) 하락 시 매도.
         단, 트레일링가 < 매수가이면 보유 유지.
      3) 만기: 보유기간 도래 시 종가에 매도

    Args:
        buy_price: 매수가 (종가)
        ohlcv_forward: 매수 다음날부터 N개월간 OHLCV (DataFrame)
        trailing_stop_pct: 트레일링 스탑 비율 B% (0.10 = 10%)
        stop_loss_pct: 손절 비율 (0.10 = 10%)
        min_profit_pct: 트레일링 발동 최소 수익률 A% (0.10 = 10%)

    Returns:
        (sell_price, return_pct, exit_reason, hold_days)
    """
    if ohlcv_forward is None or len(ohlcv_forward) == 0:
        return buy_price, 0.0, "데이터없음", 0

    if buy_price <= 0:
        return 0, 0.0, "매수가없음", 0

    max_high = buy_price
    stop_price = buy_price * (1.0 - stop_loss_pct)
    min_profit_price = buy_price * (1.0 + min_profit_pct)  # A% 이상 도달 기준

    for day_idx in range(len(ohlcv_forward)):
        row = ohlcv_forward.iloc[day_idx]
        high = float(row.get("고가", row.get("high", buy_price)))
        low = float(row.get("저가", row.get("low", buy_price)))

        max_high = max(max_high, high)
        trailing_price = max_high * (1.0 - trailing_stop_pct)

        # 1) 손절: 매수가 대비 -stop_loss_pct%
        if low <= stop_price:
            sell = stop_price
            return sell, sell / buy_price - 1.0, "손절", day_idx + 1

        # 2) 트레일링 스탑: A% 이상 올랐던 적 있고 + 고점대비 B% 하락
        #    단, 매도가가 매수가 아래이면 보유 유지
        if (
            max_high >= min_profit_price
            and low <= trailing_price
            and trailing_price >= buy_price
        ):
            sell = trailing_price
            return sell, sell / buy_price - 1.0, "트레일링스탑", day_idx + 1

    # 3) 만기 매도
    last_close = float(
        ohlcv_forward.iloc[-1].get(
            "종가", ohlcv_forward.iloc[-1].get("close", buy_price)
        )
    )
    return last_close, last_close / buy_price - 1.0, "만기매도", len(ohlcv_forward)


def _simulate_trade_multi_period(
    buy_price,
    ohlcv_after_buy,
    trailing_stop_pct,
    stop_loss_pct,
    min_profit_pct,
    today_str,
    buy_date,
    periods=None,
):
    """지정 기간별 독립적으로 매매 시뮬레이션.

    각 기간별 독립 시뮬레이션 (기간별 OHLCV 슬라이스).
    buy_date + N주가 today를 초과하면 해당 기간은 None.

    Args:
        periods: 시뮬레이션할 기간 목록 (None이면 HOLD_PERIODS 전체)

    Returns:
        dict[int, dict | None]: {3: {sell_price, return_pct, exit_reason, hold_days}, ...}
    """
    results = {}
    buy_dt = datetime.strptime(buy_date, "%Y%m%d")
    today_dt = datetime.strptime(today_str, "%Y%m%d")

    for weeks in periods or HOLD_PERIODS:
        hold_days = weeks * 5  # 거래일 기준
        forward = ohlcv_after_buy.head(hold_days)
        if forward is None or len(forward) == 0:
            results[weeks] = None
            continue

        # 기간 종료일이 오늘을 초과 → 가용 데이터로 만기매도 처리
        period_end = buy_dt + timedelta(weeks=weeks)
        partial = period_end > today_dt

        sell_price, ret_pct, exit_reason, h_days = _simulate_trade(
            buy_price, forward, trailing_stop_pct, stop_loss_pct, min_profit_pct
        )
        if partial and exit_reason == "만기매도":
            exit_reason = "조기만기(데이터부족)"
        results[weeks] = {
            "sell_price": round(sell_price, 2),
            "return_pct": round(ret_pct, 4),
            "exit_reason": exit_reason,
            "hold_days": h_days,
        }

    return results


def _lookup_index_multi_period(
    index_close, buy_date, trading_days_list, today_str, periods=None
):
    """매수일 기준 지정 기간별 지수 조회.

    Args:
        periods: 조회할 기간 목록 (None이면 HOLD_PERIODS 전체)

    Returns:
        dict[int, float | None]
    """
    _periods = periods or HOLD_PERIODS
    results = {}
    if index_close is None:
        return {w: None for w in _periods}

    for weeks in _periods:
        n_days = weeks * 5
        results[weeks] = _lookup_index_after_n_days(
            index_close, buy_date, trading_days_list, n_days
        )

    return results


# ──────────────────────────────────────────────
# 체크포인트 관리
# ──────────────────────────────────────────────


def _checkpoint_path(date_str):
    return BACKTEST_CACHE_DIR / f"bt_checkpoint_{date_str}.pkl"


def _save_checkpoint(date_str, data):
    """체크포인트 저장."""
    path = _checkpoint_path(date_str)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    _logger.info("체크포인트 저장: %d주 완료", data["completed_weeks"])


def _load_checkpoint(date_str):
    """체크포인트 로드. 없으면 None."""
    path = _checkpoint_path(date_str)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _clear_checkpoint(date_str):
    """체크포인트 삭제."""
    path = _checkpoint_path(date_str)
    if path.exists():
        path.unlink()


def has_checkpoint(date_str=None):
    """체크포인트 존재 여부 확인."""
    if date_str:
        return _checkpoint_path(date_str).exists()
    # 아무 체크포인트나 있으면 True
    return any(BACKTEST_CACHE_DIR.glob("bt_checkpoint_*.pkl"))


def get_checkpoint_info(date_str):
    """체크포인트 정보 반환."""
    cp = _load_checkpoint(date_str)
    if cp is None:
        return None
    return {
        "completed_weeks": cp["completed_weeks"],
        "total_weeks": cp["total_weeks"],
        "cum_return": cp["cum_return"],
        "saved_at": cp.get("saved_at", ""),
    }


# ──────────────────────────────────────────────
# 메인 백테스트 함수
# ──────────────────────────────────────────────


def run_rolling_backtest(
    universe_df,
    date_str,
    progress_callback=None,
    train_window_years=2,
    top_k=20,
    trailing_stop_pct=0.05,
    stop_loss_pct=0.10,
    min_profit_pct=0.30,
    hold_months=3,
    resume=False,
    num_boost_round=_BACKTEST_NUM_BOOST_ROUND,
    label_horizon_months=None,
    buy_price_type="close",
    label_target_pct=0.20,
    label_stop_pct=0.08,
    label_horizon_weeks=4,
    target_hold_periods=None,
    asset_type="stock",
    min_avg_tv: float = 3_000_000_000,
):
    """주별 롤링 백테스트 실행.

    target_hold_periods: 시뮬레이션할 보유기간 목록 (None이면 HOLD_PERIODS 전체).

    Args:
        universe_df: 시총 포함 universe (MarketCap 컬럼)
        date_str: 기준일 (오늘, YYYYMMDD)
        progress_callback: fn(msg, pct) 진행 상태 콜백
        train_window_years: 학습 윈도우 (기본 2년, 1~4년)
        top_k: Top K 종목 (기본 20)
        trailing_stop_pct: 트레일링 스탑 비율 B% (고점대비 하락)
        stop_loss_pct: 손절 비율
        min_profit_pct: 트레일링 발동 최소 수익률 A% (매수가 대비)
        hold_months: (하위호환용, 무시됨) 기간별 자동 산출
        resume: True이면 체크포인트에서 이어서 실행
        num_boost_round: LightGBM 부스팅 라운드 수 (300~800, 기본 400)
        label_horizon_months: (deprecated) 하위호환용, label_horizon_weeks 우선
        buy_price_type: 매수가 기준 ("close"=종가, "open"=시가, 기본 종가)
        label_horizon_weeks: 수익목표기간 (주 단위, 기본 4주=20영업일)

    Returns:
        dict: 백테스트 결과 (다기간 수익률 포함)
    """
    _logger.info("=" * 60)
    _logger.info("백테스트 시작 (date=%s, resume=%s)", date_str, resume)

    def _cb(msg, pct=None):
        _logger.info(msg)
        if progress_callback:
            try:
                progress_callback(msg, pct)
            except Exception:
                # WebSocket 끊김 등으로 콜백 실패해도 계속 진행
                pass

    # ── Phase 0: 날짜 계산 ──
    today = datetime.strptime(date_str, "%Y%m%d")

    # 라벨링 horizon (주 → 거래일, 하위호환: months 지정 시 변환)
    # bt_end 계산에 필요하므로 먼저 확정
    if label_horizon_months is not None:
        # 하위호환: 기존 개월 단위 호출 → 주 단위로 변환
        label_horizon_weeks = label_horizon_months * 4
    label_horizon_days = label_horizon_weeks * 5

    bt_start_raw = today - relativedelta(years=2)  # 정확히 2년 전
    # 첫 일요일로 이동 (weekday: 0=월 … 6=일) → 매수일이 월요일
    days_to_sun = (6 - bt_start_raw.weekday()) % 7
    bt_start = (
        bt_start_raw + timedelta(days=days_to_sun) if days_to_sun else bt_start_raw
    )
    # 마지막 지정일: 직전 월요일까지 확장
    # 최근 주는 보유기간 미달이나 가용 데이터로 만기매도 처리
    bt_end_raw = today
    # 해당 날짜 또는 직전 월요일로 조정 (weekday 0=월)
    days_back = bt_end_raw.weekday()  # 월=0, 화=1, ..., 일=6
    bt_end = bt_end_raw - timedelta(days=days_back)  # 해당 주 월요일
    # 학습 데이터 시작점 (bt_start에서 추가 train_window_years 전)
    data_start = bt_start - relativedelta(years=train_window_years) - timedelta(days=30)

    bt_start_str = bt_start.strftime("%Y%m%d")
    bt_end_str = bt_end.strftime("%Y%m%d")
    data_start_str = data_start.strftime("%Y%m%d")

    _cb(
        f"백테스트 기간: {bt_start_str} ~ {bt_end_str} | "
        f"학습윈도우 {train_window_years}Y | "
        f"수익목표 {label_horizon_weeks}주(={label_horizon_days}일) | "
        f"Boost {num_boost_round}",
        1,
    )

    # ── Phase 1: 적격 종목 필터 ──
    if asset_type == "ETF":
        # ETF: 시총 필터 없음 — 전체 ETF 유니버스 사용
        tickers = universe_df.index.tolist()
        _cb(f"ETF 유니버스: {len(tickers)}개", 2)
    else:
        mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
        mask = mcap.fillna(0) >= _MIN_MCAP
        tickers = universe_df[mask].index.tolist()
        _cb(f"적격 종목: {len(tickers)}개 (시총≥{_MIN_MCAP / 1e8:.0f}억)", 2)

    if len(tickers) < 10:
        raise ValueError(f"적격 종목 부족: {len(tickers)}개 (최소 10개)")

    # ── Phase 2: 전종목 OHLCV 수집 ──
    # 3년 이상 학습윈도우 사용 시 data_start_str 전달
    _ohlcv_start = data_start_str if train_window_years > 2 else None
    ohlcv_dict = _prefetch_all_ohlcv(
        tickers,
        date_str,
        _cb,
        data_start_str=_ohlcv_start,
        asset_type=asset_type,
    )

    # 거래대금 필터 (현재 기준 20일 평균)
    _min_tv = _ETF_MIN_TRADE_VALUE if asset_type == "ETF" else min_avg_tv
    eligible_tickers = []
    for ticker in tickers:
        if ticker not in ohlcv_dict:
            continue
        ohlcv = ohlcv_dict[ticker]
        if "거래대금" in ohlcv.columns:
            avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
        else:
            avg_tv = (
                (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                .iloc[-20:]
                .mean()
            )
        if avg_tv >= _min_tv:
            eligible_tickers.append(ticker)

    _tv_label = f"{_min_tv / 1e8:.0f}억" if _min_tv >= 1e8 else f"{_min_tv / 1e4:.0f}만"
    _cb(f"거래대금 필터 후: {len(eligible_tickers)}종목 (≥{_tv_label})", 27)

    # ── Phase 3: KOSPI/KOSDAQ 지수 + 피처/라벨 사전 계산 (캐시) ──
    from .data_fetcher import fetch_kosdaq_index_3year, fetch_kospi_index_3year

    kospi_close = None
    kosdaq_close = None
    try:
        kospi_df = fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
    except Exception:
        pass
    try:
        kosdaq_df = fetch_kosdaq_index_3year(date_str)
        if kosdaq_df is not None and len(kosdaq_df) > 0:
            kosdaq_close = kosdaq_df["종가"].astype(float)
    except Exception:
        pass

    # 적격 종목만 피처 계산 (캐시 활용, 라벨링 파라미터 동적 전달)
    eligible_ohlcv = {t: ohlcv_dict[t] for t in eligible_tickers if t in ohlcv_dict}
    features_dict, labels_dict = _precompute_features_labels(
        eligible_ohlcv,
        kospi_close,
        date_str,
        _cb,
        label_horizon_days=label_horizon_days,
        label_target_pct=label_target_pct,
        label_stop_pct=label_stop_pct,
        asset_type=asset_type,
    )

    # ── Phase 4: 거래일 목록 + 주별 월요일 추출 ──
    all_trading_days = _get_trading_days(data_start_str, date_str)
    bt_trading_days = [d for d in all_trading_days if bt_start_str <= d <= bt_end_str]

    # 주별 첫 거래일 (매주 월요일 또는 해당 주 첫 거래일)
    weekly_dates = _extract_weekly_dates(bt_trading_days)
    _cb(f"주별 백테스트: {len(weekly_dates)}주", 36)

    if not weekly_dates:
        raise ValueError("백테스트 대상 주가 없습니다.")

    # 종목명 매핑
    name_map = {}
    if "Name" in universe_df.columns:
        name_map = universe_df["Name"].to_dict()

    # ── 설정별 결과 캐시 확인 ──
    params_hash = _compute_bt_params_hash(
        train_window_years,
        top_k,
        trailing_stop_pct,
        stop_loss_pct,
        min_profit_pct,
        num_boost_round,
        label_horizon_weeks,
        label_target_pct,
        label_stop_pct,
        buy_price_type,
        target_hold_periods,
        asset_type,
        min_avg_tv,
    )
    _bt_cache = _load_bt_results_cache(params_hash)
    cached_weeks = _bt_cache["weekly_results"] if _bt_cache else {}
    n_cached = sum(1 for d in weekly_dates if d in cached_weeks)
    if n_cached > 0:
        _cb(
            f"설정 캐시: {n_cached}/{len(weekly_dates)}주 재사용 가능 "
            f"(hash={params_hash})",
            37,
        )

    # ── Phase 5: 롤링 백테스트 (체크포인트 지원) ──
    daily_results = []
    start_wi = 0
    total_weeks = len(weekly_dates)
    cum_return = 1.0

    # 체크포인트 재개
    if resume:
        cp = _load_checkpoint(date_str)
        if cp and cp.get("completed_weeks", 0) > 0:
            daily_results = cp["daily_results"]
            start_wi = cp["completed_weeks"]
            cum_return = cp["cum_return"]
            _cb(
                f"체크포인트에서 재개: {start_wi}/{total_weeks}주 완료, "
                f"누적 {(cum_return - 1) * 100:+.1f}%",
                int(37 + (start_wi / total_weeks) * 50),
            )

    feat_names = list(SURGE_FEATURE_NAMES)
    periods = target_hold_periods or HOLD_PERIODS
    max_hold_trading_days = max(periods) * 5
    phase5_start = datetime.now()

    for wi in range(start_wi, total_weeks):
        test_date = weekly_dates[wi]
        pct = int(37 + (wi / total_weeks) * 50)

        # ── 설정별 결과 캐시 재사용 ──
        if test_date in cached_weeks:
            wr = cached_weeks[test_date]
            daily_results.append(wr)
            _cached_pr = wr.get("portfolio_return", 0.0)
            cum_return *= 1.0 + _cached_pr
            _cb(
                f"[{wi + 1}/{total_weeks}] {test_date} — "
                f"캐시 재사용 ({_cached_pr * 100:+.1f}%)",
                pct,
            )
            if (wi + 1) % _CHECKPOINT_INTERVAL == 0:
                _save_checkpoint(
                    date_str,
                    {
                        "daily_results": daily_results,
                        "completed_weeks": wi + 1,
                        "total_weeks": total_weeks,
                        "cum_return": cum_return,
                        "saved_at": datetime.now().isoformat(),
                    },
                )
            continue

        # 학습 윈도우: 지정일 전날까지 (지정일 자체는 예측일이므로 제외)
        test_dt = datetime.strptime(test_date, "%Y%m%d")
        train_end_dt = test_dt - timedelta(days=1)  # 지정일 전날
        train_start_dt = test_dt - relativedelta(years=train_window_years)
        train_start_str = train_start_dt.strftime("%Y%m%d")
        train_end_str = train_end_dt.strftime("%Y%m%d")

        _cb(
            f"[{wi + 1}/{total_weeks}] 지정일 {test_date} | "
            f"학습기간 {train_start_str}~{train_end_str}",
            pct,
        )

        # 종목별 피처/라벨 슬라이싱 → 학습 데이터 구성
        all_X, all_y, all_hard = [], [], []
        for ticker in eligible_tickers:
            if ticker not in features_dict or ticker not in labels_dict:
                continue

            feats = features_dict[ticker]
            y_series, hard_series = labels_dict[ticker]

            # 날짜 필터: [train_start, test_date 전날] — 지정일 데이터는 제외
            date_mask = (feats.index >= train_start_dt) & (feats.index <= train_end_dt)
            if date_mask.sum() < 10:
                continue

            # 유효 라벨만
            sliced_feats = feats[date_mask]
            sliced_y = y_series.reindex(sliced_feats.index)
            sliced_hard = hard_series.reindex(sliced_feats.index)

            valid = sliced_y.notna()
            if valid.sum() < 5:
                continue

            all_X.append(sliced_feats[valid])
            all_y.append(sliced_y[valid])
            all_hard.append(sliced_hard[valid])

        if not all_X:
            continue

        X = pd.concat(all_X)
        y = np.concatenate([s.values for s in all_y])
        hard = np.concatenate([s.values for s in all_hard])

        n_pos = int(y.sum())
        if n_pos < _MIN_POSITIVE_SAMPLES:
            continue

        # 피처 정렬 (SURGE_FEATURE_NAMES 순서)
        available_cols = [c for c in feat_names if c in X.columns]
        if len(available_cols) < 20:
            continue
        X_aligned = X[available_cols]

        # 학습
        model, use_lgb, auc = _train_backtest_model(
            X_aligned.values, y, hard, available_cols, num_boost_round=num_boost_round
        )
        if model is None:
            continue

        # ── 예측: 전 종목에서 Top K ──
        scored = []
        for ticker in eligible_tickers:
            if ticker not in features_dict:
                continue
            feats = features_dict[ticker]

            # test_date 이전 마지막 피처 행
            mask = feats.index <= test_dt
            if mask.sum() == 0:
                continue
            last_row = feats[mask].iloc[-1:]
            last_aligned = last_row.reindex(columns=available_cols, fill_value=0.0)
            x_pred = np.nan_to_num(
                last_aligned.values, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float64)

            if use_lgb:
                prob = float(model.predict(x_pred)[0])
            else:
                prob = float(model.predict_proba(x_pred)[:, 1][0])

            feat_snapshot = {
                c: float(last_aligned.iloc[0].get(c, 0.0)) for c in available_cols[:20]
            }
            scored.append((ticker, prob, feat_snapshot))

        # Top K 선정
        scored.sort(key=lambda x: -x[1])
        top_stocks = scored[:top_k]

        # ── 매매 시뮬레이션 (다기간) ──
        buy_date = _next_trading_day(test_date, all_trading_days)
        if not buy_date:
            continue

        stock_results = []
        for ticker, prob, feat_snap in top_stocks:
            if ticker not in ohlcv_dict:
                continue
            ohlcv = ohlcv_dict[ticker]
            buy_dt_ts = pd.Timestamp(datetime.strptime(buy_date, "%Y%m%d"))

            # 매수가: 매수일 종가 또는 시가
            buy_mask = ohlcv.index >= buy_dt_ts
            if buy_mask.sum() == 0:
                continue
            buy_idx = ohlcv.index[buy_mask][0]
            if buy_price_type == "open":
                buy_price = float(
                    ohlcv.at[buy_idx, "시가"]
                    if "시가" in ohlcv.columns
                    else ohlcv.at[buy_idx, ohlcv.columns[0]]
                )
            else:
                buy_price = float(ohlcv.at[buy_idx, "종가"])

            # 보유 기간 OHLCV (매수 다음날부터 최대 12주)
            after_buy = ohlcv.index > buy_idx
            forward_ohlcv = ohlcv[after_buy].head(max_hold_trading_days)

            # 다기간 시뮬레이션
            returns_by_period = _simulate_trade_multi_period(
                buy_price,
                forward_ohlcv,
                trailing_stop_pct,
                stop_loss_pct,
                min_profit_pct,
                date_str,
                buy_date,
                periods=periods,
            )

            # 기본값: 첫 번째 유효 기간 사용
            default_period = returns_by_period.get(periods[0])
            sr = {
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "predicted_prob": round(prob, 4),
                "features": feat_snap,
                "buy_price": buy_price,
                "returns_by_period": returns_by_period,
            }
            if default_period:
                sr["sell_price"] = default_period["sell_price"]
                sr["return_pct"] = default_period["return_pct"]
                sr["exit_reason"] = default_period["exit_reason"]
                sr["hold_days"] = default_period["hold_days"]
            else:
                # 첫 기간 결과 없으면 가장 짧은 유효 기간 사용
                first_valid = next(
                    (
                        returns_by_period[m]
                        for m in periods
                        if returns_by_period.get(m) is not None
                    ),
                    None,
                )
                if first_valid:
                    sr["sell_price"] = first_valid["sell_price"]
                    sr["return_pct"] = first_valid["return_pct"]
                    sr["exit_reason"] = first_valid["exit_reason"]
                    sr["hold_days"] = first_valid["hold_days"]
                else:
                    sr["sell_price"] = buy_price
                    sr["return_pct"] = 0.0
                    sr["exit_reason"] = "데이터없음"
                    sr["hold_days"] = 0

            stock_results.append(sr)

        if not stock_results:
            continue

        # 기간별 포트폴리오 수익률 계산
        portfolio_returns = {}
        win_counts = {}
        loss_counts = {}
        for months in periods:
            period_rets = [
                s["returns_by_period"][months]["return_pct"]
                for s in stock_results
                if s["returns_by_period"].get(months) is not None
            ]
            if period_rets:
                portfolio_returns[months] = round(float(np.mean(period_rets)), 4)
                win_counts[months] = sum(1 for r in period_rets if r > 0)
                loss_counts[months] = sum(1 for r in period_rets if r <= 0)
            else:
                portfolio_returns[months] = None
                win_counts[months] = 0
                loss_counts[months] = 0

        # 기본 수익률: 첫 번째 기간
        _first_p = periods[0]
        portfolio_return = portfolio_returns.get(_first_p) or 0.0
        win_count = win_counts.get(_first_p, 0)
        loss_count = loss_counts.get(_first_p, 0)

        # KOSPI/KOSDAQ 지수 조회: 매수일 + 다기간
        kospi_buy = _lookup_index(kospi_close, buy_date)
        kosdaq_buy = _lookup_index(kosdaq_close, buy_date)
        kospi_after = _lookup_index_multi_period(
            kospi_close, buy_date, all_trading_days, date_str, periods=periods
        )
        kosdaq_after = _lookup_index_multi_period(
            kosdaq_close, buy_date, all_trading_days, date_str, periods=periods
        )

        daily_results.append(
            {
                "date": test_date,
                "buy_date": buy_date,
                "stocks": stock_results,
                "portfolio_return": round(portfolio_return, 4),
                "portfolio_returns": portfolio_returns,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_counts": win_counts,
                "loss_counts": loss_counts,
                "model_auc": round(auc, 4),
                "n_samples": len(y),
                "n_positive": n_pos,
                "kospi_buy": kospi_buy,
                "kospi_after": kospi_after,
                "kosdaq_buy": kosdaq_buy,
                "kosdaq_after": kosdaq_after,
                # 하위호환
                "kospi_3m": kospi_after.get(_first_p),
                "kosdaq_3m": kosdaq_after.get(_first_p),
            }
        )

        # 주별 결과 콜백
        cum_return *= 1.0 + portfolio_return
        elapsed = (datetime.now() - phase5_start).total_seconds()
        elapsed_min = elapsed / 60

        # 다기간 수익률 로그
        ret_parts = []
        for m in periods:
            pr = portfolio_returns.get(m)
            if pr is not None:
                ret_parts.append(f"{m}주 {pr * 100:+.1f}%")
        ret_log = " | ".join(ret_parts) if ret_parts else "N/A"

        # 지수 변동 로그 (첫 기간 기준)
        _first_hp = periods[0]
        idx_log = ""
        kospi_hp_val = kospi_after.get(_first_hp)
        if kospi_buy and kospi_hp_val:
            kospi_chg = (kospi_hp_val / kospi_buy - 1) * 100
            idx_log += f" | KOSPI {kospi_buy:.0f}→{kospi_hp_val:.0f}({kospi_chg:+.1f}%)"
        kosdaq_hp_val = kosdaq_after.get(_first_hp)
        if kosdaq_buy and kosdaq_hp_val:
            kosdaq_chg = (kosdaq_hp_val / kosdaq_buy - 1) * 100
            idx_log += (
                f" | KOSDAQ {kosdaq_buy:.0f}→{kosdaq_hp_val:.0f}({kosdaq_chg:+.1f}%)"
            )

        _cb(
            f"  → {ret_log} | "
            f"승 {win_count}/패 {loss_count} | AUC {auc:.3f} | "
            f"누적(3M) {(cum_return - 1) * 100:+.1f}%{idx_log} | "
            f"경과 {elapsed_min:.0f}분",
            pct,
        )

        # 체크포인트 저장 (매 _CHECKPOINT_INTERVAL 주마다)
        if (wi + 1) % _CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(
                date_str,
                {
                    "daily_results": daily_results,
                    "completed_weeks": wi + 1,
                    "total_weeks": total_weeks,
                    "cum_return": cum_return,
                    "saved_at": datetime.now().isoformat(),
                },
            )

    total_elapsed = (datetime.now() - phase5_start).total_seconds()
    n_new_weeks = len(daily_results) - n_cached
    _cb(
        f"백테스트 완료 ({len(daily_results)}주, 캐시 {n_cached}주 + 신규 {n_new_weeks}주) | "
        f"소요시간 {total_elapsed / 60:.1f}분 | 통계 집계 중...",
        90,
    )

    # ── 설정별 결과 캐시 저장 ──
    updated_cache = dict(cached_weeks)
    for r in daily_results:
        updated_cache[r["date"]] = r
    _save_bt_results_cache(params_hash, updated_cache, date_str)

    # ── Phase 6: 결과 집계 ──
    summary = _compute_summary(daily_results, periods=periods, date_str=date_str)

    result = {
        "config": {
            "start_date": bt_start_str,
            "end_date": bt_end_str,
            "train_window": f"{train_window_years}Y",
            "hold_periods": periods,
            "trailing_stop": trailing_stop_pct,
            "stop_loss": stop_loss_pct,
            "min_profit": min_profit_pct,
            "top_k": top_k,
            "interval": "weekly",
            "num_boost_round": num_boost_round,
            "label_horizon_weeks": label_horizon_weeks,
            "label_horizon_days": label_horizon_days,
            "label_target_pct": label_target_pct,
            "label_stop_pct": label_stop_pct,
            "buy_price_type": buy_price_type,
            "asset_type": asset_type,
        },
        "daily_results": daily_results,
        "summary": summary,
    }

    # 체크포인트 정리 (완료했으므로)
    _clear_checkpoint(date_str)

    _cb("결과 저장 중...", 95)
    _sbp = summary.get("summary_by_period", {})
    _real_hw = _sbp.get(label_horizon_weeks, {}).get("realistic_cum_return", 1.0)
    _logger.info(
        "백테스트 완료: %d주, %d주 실투자수익률 %.1f%%, 소요 %.1f분",
        len(daily_results),
        label_horizon_weeks,
        (_real_hw - 1) * 100,
        total_elapsed / 60,
    )
    return result


def _extract_weekly_dates(trading_days):
    """거래일 목록에서 주별 첫 거래일 추출 (매주 1회)."""
    if not trading_days:
        return []

    weekly = []
    last_week = None
    for d in trading_days:
        dt = datetime.strptime(d, "%Y%m%d")
        iso_week = dt.isocalendar()[:2]  # (year, week_number)
        if iso_week != last_week:
            weekly.append(d)
            last_week = iso_week
    return weekly


def _compute_summary(daily_results, periods=None, date_str=None):
    """백테스트 결과 통계 요약."""
    if not daily_results:
        return {
            "total_weeks": 0,
            "avg_portfolio_return": 0.0,
            "cumulative_return": 1.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    # 하위호환: 3개월 기준 통계
    all_returns = [r["portfolio_return"] for r in daily_results]
    all_stock_returns = []
    for r in daily_results:
        all_stock_returns.extend(s["return_pct"] for s in r["stocks"])

    # 누적 수익률 (복리)
    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in all_returns:
        cum *= 1.0 + ret
        peak = max(peak, cum)
        dd = (cum - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    wins = [r for r in all_stock_returns if r > 0]
    losses = [r for r in all_stock_returns if r <= 0]

    # 샤프 비율 (겹치는 수익률 보정: √(52/horizon) 연환산)
    _primary_hw = periods[0] if periods else 4
    if len(all_returns) > 1:
        ret_arr = np.array(all_returns)
        mean_r = ret_arr.mean()
        std_r = ret_arr.std()
        sharpe = (mean_r / std_r * np.sqrt(52 / _primary_hw)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # ── 기간별 통계 ──
    _periods = periods or HOLD_PERIODS
    summary_by_period = {}
    for months in _periods:
        period_rets = []
        stock_rets = []
        for r in daily_results:
            pr = r.get("portfolio_returns", {}).get(months)
            if pr is not None:
                period_rets.append(pr)
            for s in r["stocks"]:
                rbp = s.get("returns_by_period", {}).get(months)
                if rbp is not None:
                    stock_rets.append(rbp["return_pct"])

        if period_rets:
            p_cum = 1.0
            p_peak = 1.0
            p_mdd = 0.0
            for pr in period_rets:
                p_cum *= 1.0 + pr
                p_peak = max(p_peak, p_cum)
                p_dd = (p_cum - p_peak) / p_peak if p_peak > 0 else 0
                p_mdd = min(p_mdd, p_dd)

            p_wins = [r for r in stock_rets if r > 0]
            p_losses = [r for r in stock_rets if r <= 0]

            # 샤프 비율 (겹치는 수익률 보정: √(52/horizon) 연환산)
            if len(period_rets) > 1:
                p_arr = np.array(period_rets)
                p_sharpe = (
                    (p_arr.mean() / p_arr.std() * np.sqrt(52 / months))
                    if p_arr.std() > 0
                    else 0.0
                )
            else:
                p_sharpe = 0.0

            # 기간별 exit_stats
            p_exit = {}
            for r in daily_results:
                for s in r["stocks"]:
                    rbp = s.get("returns_by_period", {}).get(months)
                    if rbp is not None:
                        reason = rbp["exit_reason"]
                        if reason not in p_exit:
                            p_exit[reason] = {"count": 0, "returns": []}
                        p_exit[reason]["count"] += 1
                        p_exit[reason]["returns"].append(rbp["return_pct"])
            p_exit_stats = {
                reason: {
                    "count": data["count"],
                    "avg_return": round(float(np.mean(data["returns"])), 4),
                }
                for reason, data in p_exit.items()
            }

            summary_by_period[months] = {
                "n_weeks": len(period_rets),
                "cumulative_return": round(p_cum, 4),
                "avg_return": round(float(np.mean(period_rets)), 4),
                "win_rate": round(len(p_wins) / len(stock_rets), 4)
                if stock_rets
                else 0.0,
                "avg_win": round(float(np.mean(p_wins)), 4) if p_wins else 0.0,
                "avg_loss": round(float(np.mean(p_losses)), 4) if p_losses else 0.0,
                "max_drawdown": round(p_mdd, 4),
                "sharpe_ratio": round(float(p_sharpe), 2),
                "exit_stats": p_exit_stats,
            }
        else:
            summary_by_period[months] = {
                "n_weeks": 0,
                "cumulative_return": 1.0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "exit_stats": {},
            }

    # ── 기간별 현실적 자금 시뮬레이션 (슬롯 모델 + 일괄매수) ──
    for hp in _periods:
        if summary_by_period[hp]["n_weeks"] > 0:
            realistic = _compute_realistic_portfolio(daily_results, hp)
            summary_by_period[hp]["realistic_cum_return"] = realistic[
                "realistic_cum_return"
            ]
            summary_by_period[hp]["n_slots"] = realistic["n_slots"]
            # 차트용: 주별 포트폴리오 가치 (date, total_value)
            summary_by_period[hp]["realistic_weekly_values"] = realistic[
                "weekly_values"
            ]
            # 일괄매수
            lumpsum = _compute_lumpsum_portfolio(daily_results, hp)
            summary_by_period[hp]["lumpsum_cum_return"] = lumpsum["lumpsum_cum_return"]
            summary_by_period[hp]["lumpsum_weekly_values"] = lumpsum[
                "lumpsum_weekly_values"
            ]
        else:
            summary_by_period[hp]["realistic_cum_return"] = 1.0
            summary_by_period[hp]["n_slots"] = 0
            summary_by_period[hp]["realistic_weekly_values"] = []
            summary_by_period[hp]["lumpsum_cum_return"] = 1.0
            summary_by_period[hp]["lumpsum_weekly_values"] = []

    # ── 서브기간 통계 (최근 1년, 6개월, 3개월) ──
    summary_by_subperiod = {}
    if date_str:
        today_dt = datetime.strptime(date_str, "%Y%m%d")
        sub_cutoffs = {
            "1Y": (today_dt - relativedelta(years=1)).strftime("%Y%m%d"),
            "6M": (today_dt - relativedelta(months=6)).strftime("%Y%m%d"),
            "3M": (today_dt - relativedelta(months=3)).strftime("%Y%m%d"),
            "1M": (today_dt - relativedelta(months=1)).strftime("%Y%m%d"),
        }
        for sub_label, cutoff_str in sub_cutoffs.items():
            filtered_daily = [r for r in daily_results if r["date"] >= cutoff_str]
            sub_by_period = {}
            for hp in _periods:
                sp_rets = []
                sp_stock_rets = []
                for r in filtered_daily:
                    pr = r.get("portfolio_returns", {}).get(hp)
                    if pr is not None:
                        sp_rets.append(pr)
                    for s in r["stocks"]:
                        rbp = s.get("returns_by_period", {}).get(hp)
                        if rbp is not None:
                            sp_stock_rets.append(rbp["return_pct"])

                if sp_rets:
                    sp_arr = np.array(sp_rets)
                    sp_cum = 1.0
                    sp_peak = 1.0
                    sp_mdd = 0.0
                    for pr in sp_rets:
                        sp_cum *= 1.0 + pr
                        sp_peak = max(sp_peak, sp_cum)
                        sp_dd = (sp_cum - sp_peak) / sp_peak if sp_peak > 0 else 0
                        sp_mdd = min(sp_mdd, sp_dd)

                    sp_wins = [r for r in sp_stock_rets if r > 0]
                    sp_losses = [r for r in sp_stock_rets if r <= 0]

                    sp_std = sp_arr.std() if len(sp_arr) > 1 else 0.0
                    sp_sharpe = (
                        (sp_arr.mean() / sp_std * np.sqrt(52 / hp))
                        if sp_std > 0
                        else 0.0
                    )

                    sp_realistic = _compute_realistic_portfolio(filtered_daily, hp)
                    sp_lumpsum = _compute_lumpsum_portfolio(filtered_daily, hp)

                    sub_by_period[hp] = {
                        "n_weeks": len(sp_rets),
                        "cumulative_return": round(sp_cum, 4),
                        "avg_return": round(float(sp_arr.mean()), 4),
                        "win_rate": round(len(sp_wins) / len(sp_stock_rets), 4)
                        if sp_stock_rets
                        else 0.0,
                        "avg_win": round(float(np.mean(sp_wins)), 4)
                        if sp_wins
                        else 0.0,
                        "avg_loss": round(float(np.mean(sp_losses)), 4)
                        if sp_losses
                        else 0.0,
                        "max_drawdown": round(sp_mdd, 4),
                        "sharpe_ratio": round(float(sp_sharpe), 2),
                        "realistic_cum_return": sp_realistic["realistic_cum_return"],
                        "n_slots": sp_realistic["n_slots"],
                        "lumpsum_cum_return": sp_lumpsum["lumpsum_cum_return"],
                    }
                else:
                    sub_by_period[hp] = {
                        "n_weeks": 0,
                        "cumulative_return": 1.0,
                        "avg_return": 0.0,
                        "win_rate": 0.0,
                        "avg_win": 0.0,
                        "avg_loss": 0.0,
                        "max_drawdown": 0.0,
                        "sharpe_ratio": 0.0,
                        "realistic_cum_return": 1.0,
                        "n_slots": 0,
                        "lumpsum_cum_return": 1.0,
                    }
            summary_by_subperiod[sub_label] = sub_by_period

    return {
        "total_weeks": len(daily_results),
        "avg_portfolio_return": round(float(np.mean(all_returns)), 4),
        "cumulative_return": round(cum, 4),
        "win_rate": round(len(wins) / len(all_stock_returns), 4)
        if all_stock_returns
        else 0.0,
        "avg_win": round(float(np.mean(wins)), 4) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 4) if losses else 0.0,
        "max_drawdown": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 2),
        "total_stocks_traded": len(all_stock_returns),
        "exit_stats": _exit_stats(daily_results),
        "summary_by_period": summary_by_period,
        "summary_by_subperiod": summary_by_subperiod,
    }


def _compute_realistic_portfolio(
    daily_results, period_weeks, initial_capital=100_000_000
):
    """현실적 자금 기반 포트폴리오 시뮬레이션 (슬롯 모델).

    보유기간에 따라 자금을 N개 슬롯으로 나눠서 순환 투자.
    예) 4주 보유 → N=4 슬롯, 초기 자금 1억 → 슬롯당 0.25억

      Week 1: slot[0] = 0.25억 투자 → 수익률 r₁
      Week 2: slot[1] = 0.25억 투자 → 수익률 r₂
      Week 3: slot[2] = 0.25억 투자 → 수익률 r₃
      Week 4: slot[3] = 0.25억 투자 → 수익률 r₄
      Week 5: slot[0] = 0.25억×(1+r₁) 재투자  ← Week1 매도 예수금
      ...

    각 슬롯은 독립적으로 복리 성장/축소.
    """
    if not daily_results:
        return {
            "realistic_cum_return": 1.0,
            "weekly_values": [],
            "n_slots": 0,
        }

    # 슬롯 수 = 보유기간(주) = 주별 순환 슬롯
    n_slots = max(4, period_weeks)
    per_slot = initial_capital / n_slots
    slot_capital = [per_slot] * n_slots

    weekly_values = []

    for wi, r in enumerate(daily_results):
        slot = wi % n_slots
        pr = r.get("portfolio_returns", {}).get(period_weeks)

        if pr is not None:
            # 이 슬롯의 배치 수익률 적용 (매도 예수금 반영)
            slot_capital[slot] *= 1.0 + pr

        # 전체 포트폴리오 가치 = 모든 슬롯의 합
        total_value = sum(slot_capital)
        weekly_values.append(
            {
                "date": r.get("buy_date", r.get("date", "")),
                "total_value": round(total_value),
                "slot": slot,
                "slot_capital": round(slot_capital[slot]),
            }
        )

    final = sum(slot_capital)
    return {
        "realistic_cum_return": round(final / initial_capital, 4),
        "weekly_values": weekly_values,
        "n_slots": n_slots,
    }


def _compute_lumpsum_portfolio(
    daily_results, period_weeks, initial_capital=100_000_000
):
    """원금 일괄매수 포트폴리오 시뮬레이션.

    전액을 한 번에 투자 → period_weeks 후 매도 → 재투자 반복.
    매 period_weeks 간격(0, period_weeks, 2*period_weeks, ...)의 수익률만 사용.
    """
    if not daily_results:
        return {
            "lumpsum_cum_return": 1.0,
            "lumpsum_weekly_values": [],
        }

    capital = initial_capital
    weekly_values = []

    for wi, r in enumerate(daily_results):
        if wi % period_weeks != 0:
            continue
        pr = r.get("portfolio_returns", {}).get(period_weeks)
        if pr is not None:
            capital *= 1.0 + pr
        weekly_values.append(
            {
                "date": r.get("buy_date", r.get("date", "")),
                "total_value": round(capital),
            }
        )

    return {
        "lumpsum_cum_return": round(capital / initial_capital, 4),
        "lumpsum_weekly_values": weekly_values,
    }


def _compute_lumpsum_from_rets(all_rets, hold_weeks):
    """주별 수익률 리스트에서 일괄매수 누적수익률 계산 (배수 반환).

    all_rets[0], all_rets[hold_weeks], all_rets[2*hold_weeks], ... 만 사용.
    """
    lump_rets = all_rets[::hold_weeks]
    capital = 1.0
    for pr in lump_rets:
        capital *= 1.0 + pr
    return capital


def _exit_stats(daily_results):
    """매도 사유별 통계."""
    stats = {}
    for r in daily_results:
        for s in r["stocks"]:
            reason = s["exit_reason"]
            if reason not in stats:
                stats[reason] = {"count": 0, "returns": []}
            stats[reason]["count"] += 1
            stats[reason]["returns"].append(s["return_pct"])

    result = {}
    for reason, data in stats.items():
        result[reason] = {
            "count": data["count"],
            "avg_return": round(float(np.mean(data["returns"])), 4),
        }
    return result


# ──────────────────────────────────────────────
# 저장/로드
# ──────────────────────────────────────────────


def cleanup_old_backtest_caches(max_age_days: int = 14) -> dict:
    """오래된 bt_ohlcv_*.pkl, bt_features_*.pkl 파일 삭제.

    Returns:
        {"deleted": [(파일명, 바이트)], "freed_bytes": int}
    """
    cutoff_dt = datetime.now() - timedelta(days=max_age_days)
    deleted = []
    freed = 0

    if not BACKTEST_CACHE_DIR.exists():
        return {"deleted": deleted, "freed_bytes": freed}

    for f in BACKTEST_CACHE_DIR.iterdir():
        if not f.is_file() or not f.name.endswith(".pkl"):
            continue
        # _latest 파일은 보존
        if "latest" in f.name:
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime < cutoff_dt:
            size = f.stat().st_size
            try:
                f.unlink()
                deleted.append((f.name, size))
                freed += size
            except Exception:
                pass

    return {"deleted": deleted, "freed_bytes": freed}


# ──────────────────────────────────────────────
# 설정별 백테스트 결과 캐시 (증분 재사용)
# ──────────────────────────────────────────────

_BT_RESULTS_CACHE_VERSION = 1


def _compute_bt_params_hash(
    train_window_years,
    top_k,
    trailing_stop_pct,
    stop_loss_pct,
    min_profit_pct,
    num_boost_round,
    label_horizon_weeks,
    label_target_pct,
    label_stop_pct,
    buy_price_type,
    target_hold_periods,
    asset_type,
    min_avg_tv,
) -> str:
    """백테스트 알고리즘 파라미터 해시 계산 (설정별 캐시 키)."""
    params = {
        "tw": train_window_years,
        "k": top_k,
        "ts": trailing_stop_pct,
        "sl": stop_loss_pct,
        "mp": min_profit_pct,
        "nb": num_boost_round,
        "hw": label_horizon_weeks,
        "tp": label_target_pct,
        "sp": label_stop_pct,
        "bp": buy_price_type,
        "hp": sorted(target_hold_periods or HOLD_PERIODS),
        "at": asset_type,
        "tv": min_avg_tv,
    }
    raw = json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _bt_results_cache_path(params_hash: str) -> Path:
    return BACKTEST_CACHE_DIR / f"bt_results_{params_hash}.pkl"


def _load_bt_results_cache(params_hash: str) -> dict | None:
    """설정별 백테스트 결과 캐시 로드. 없거나 버전 불일치 시 None."""
    path = _bt_results_cache_path(params_hash)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != _BT_RESULTS_CACHE_VERSION:
            return None
        if data.get("params_hash") != params_hash:
            return None
        return data
    except Exception:
        return None


def _save_bt_results_cache(params_hash: str, weekly_results: dict, last_date: str):
    """설정별 백테스트 결과 캐시 저장."""
    path = _bt_results_cache_path(params_hash)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "version": _BT_RESULTS_CACHE_VERSION,
                "params_hash": params_hash,
                "weekly_results": weekly_results,
                "last_run_date": last_date,
                "saved_at": datetime.now().isoformat(),
            },
            f,
        )
    _logger.info("백테스트 결과 캐시 저장: %s (%d주)", path.name, len(weekly_results))


def save_backtest_result(result, date_str=None):
    """백테스트 결과 pickle 저장."""
    BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_result_{date_str or ts}.pkl"
    filepath = BACKTEST_CACHE_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(result, f)
    # latest 링크
    latest = BACKTEST_CACHE_DIR / "backtest_result_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(result, f)
    return filepath


def load_backtest_result(path=None):
    """백테스트 결과 pickle 로드."""
    if path and Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    latest = BACKTEST_CACHE_DIR / "backtest_result_latest.pkl"
    if latest.exists():
        with open(latest, "rb") as f:
            return pickle.load(f)
    return None


def load_cached_ohlcv(date_str, asset_type="stock"):
    """캐시된 OHLCV 로드. 가장 적합한 캐시 파일 탐색."""
    _asset_tag = "_etf" if asset_type == "ETF" else ""
    # 정확히 일치하는 기본 캐시
    cache_file = BACKTEST_CACHE_DIR / f"bt_ohlcv{_asset_tag}_{date_str}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    # data_start 포함 캐시 (확장 학습윈도우)
    for p in sorted(
        BACKTEST_CACHE_DIR.glob(f"bt_ohlcv{_asset_tag}_{date_str}_from*.pkl")
    ):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ──────────────────────────────────────────────
# 최적 매도조건(A,B,C) 탐색
# ──────────────────────────────────────────────


_DEFAULT_A_RANGE = tuple(range(20, 41))  # A(최소수익률%): 20~40, 1단위 (21개)
_DEFAULT_B_RANGE = tuple(range(5, 11))  # B(트레일링스탑%): 5~10, 1단위 (6개)
_DEFAULT_C_RANGE = tuple(range(10, 21))  # C(손절기준%): 10~20, 1단위 (11개)


def optimize_trade_params(
    bt_result,
    ohlcv_dict,
    a_range=_DEFAULT_A_RANGE,
    b_range=_DEFAULT_B_RANGE,
    c_range=_DEFAULT_C_RANGE,
    target_period=3,
    progress_callback=None,
    date_str=None,
):
    """백테스트 결과의 종목 선정을 재사용하여 A,B,C 최적 조합 탐색.

    학습/예측을 다시 수행하지 않고, 이미 선정된 종목+매수가에 대해
    매도 조건(A,B,C)만 변경하여 시뮬레이션을 재수행한다 (수초~수십초).

    Args:
        bt_result: run_rolling_backtest 반환값
        ohlcv_dict: 전종목 OHLCV dict (load_cached_ohlcv로 로드)
        a_range: A% (최소수익률) 후보 목록 (정수 %)
        b_range: B% (고점대비하락) 후보 목록 (정수 %)
        c_range: C% (손절) 후보 목록 (정수 %)
        target_period: 최적화 대상 보유기간 (주)
        progress_callback: fn(msg, pct) 콜백
        date_str: 기준일 (YYYYMMDD) — 서브기간별 최적조합 산출용

    Returns:
        dict: grid_results, best, best_by_period, settings_text
    """
    daily_results = bt_result.get("daily_results", [])
    config = bt_result.get("config", {})

    if not daily_results or not ohlcv_dict:
        return {
            "grid_results": [],
            "best": None,
            "best_by_period": {},
            "settings_text": "데이터 없음",
        }

    # 매수 정보 사전 추출 (종목별 OHLCV forward 슬라이스)
    trade_info = []
    hold_days = target_period * 5
    for dr in daily_results:
        for stock in dr["stocks"]:
            ticker = stock["ticker"]
            buy_price = stock["buy_price"]
            buy_date = dr["buy_date"]
            if ticker not in ohlcv_dict:
                continue
            ohlcv = ohlcv_dict[ticker]
            buy_dt_ts = pd.Timestamp(datetime.strptime(buy_date, "%Y%m%d"))
            after_buy = ohlcv.index > buy_dt_ts
            forward = ohlcv[after_buy].head(hold_days)
            if len(forward) == 0:
                continue
            trade_info.append(
                {
                    "ticker": ticker,
                    "buy_price": buy_price,
                    "forward": forward,
                    "week_date": dr["date"],
                }
            )

    if not trade_info:
        return {
            "grid_results": [],
            "best": None,
            "best_by_period": {},
            "settings_text": "거래 데이터 없음",
        }

    # 주별 그룹핑 (포트폴리오 수익률 계산용)
    from collections import defaultdict

    week_trades = defaultdict(list)
    for t in trade_info:
        week_trades[t["week_date"]].append(t)

    # 서브기간 cutoff 계산 (date_str 있을 때)
    _sub_cutoffs = {}
    if date_str:
        _today_dt = datetime.strptime(date_str, "%Y%m%d")
        _sub_cutoffs = {
            "1Y": (_today_dt - relativedelta(years=1)).strftime("%Y%m%d"),
            "6M": (_today_dt - relativedelta(months=6)).strftime("%Y%m%d"),
            "3M": (_today_dt - relativedelta(months=3)).strftime("%Y%m%d"),
            "1M": (_today_dt - relativedelta(months=1)).strftime("%Y%m%d"),
        }

    total_combos = len(a_range) * len(b_range) * len(c_range)
    grid_results = []
    combo_idx = 0

    for a in a_range:
        for b in b_range:
            for c in c_range:
                combo_idx += 1
                if progress_callback and combo_idx % 50 == 0:
                    pct = int(combo_idx / total_combos * 100)
                    progress_callback(
                        f"그리드 서치: {combo_idx}/{total_combos} (A={a} B={b} C={c})",
                        pct,
                    )

                a_dec = a / 100.0
                b_dec = b / 100.0
                c_dec = c / 100.0

                # 주별 포트폴리오 수익률 계산
                all_rets = []  # (ret, week_date) 쌍
                wins = 0
                losses = 0

                sorted_weeks = sorted(week_trades.keys())
                for week_date in sorted_weeks:
                    trades = week_trades[week_date]
                    week_rets = []
                    for t in trades:
                        _, ret_pct, _, _ = _simulate_trade(
                            t["buy_price"],
                            t["forward"],
                            trailing_stop_pct=b_dec,
                            stop_loss_pct=c_dec,
                            min_profit_pct=a_dec,
                        )
                        week_rets.append(ret_pct)
                        if ret_pct > 0:
                            wins += 1
                        else:
                            losses += 1

                    if week_rets:
                        port_ret = float(np.mean(week_rets))
                        all_rets.append((port_ret, week_date))

                rets_only = [r for r, _ in all_rets]

                # 실투자 수익률 (슬롯 모델)
                n_slots = max(4, target_period)
                initial = 100_000_000
                slot_cap = [initial / n_slots] * n_slots
                for wi, pr in enumerate(rets_only):
                    slot_cap[wi % n_slots] *= 1.0 + pr
                realistic_cum = sum(slot_cap) / initial

                # 일괄매수 수익률
                lumpsum_cum = _compute_lumpsum_from_rets(rets_only, target_period)

                # 샤프 비율 (겹치는 수익률 보정: √(52/horizon))
                if len(rets_only) > 1:
                    ret_arr = np.array(rets_only)
                    std_r = ret_arr.std()
                    sharpe = (
                        (ret_arr.mean() / std_r * np.sqrt(52 / target_period))
                        if std_r > 0
                        else 0.0
                    )
                else:
                    sharpe = 0.0

                total = wins + losses
                _period_avg = (
                    round(float(np.mean(rets_only)) * 100, 2) if rets_only else 0
                )
                _weekly_avg = _period_avg  # N주 보유수익률 평균 (주간배치 단위)
                _row = {
                    "A%": a,
                    "B%": b,
                    "C%": c,
                    "cum_return": round(realistic_cum, 4),
                    "cum_pct": round((realistic_cum - 1) * 100, 2),
                    "lumpsum_pct": round((lumpsum_cum - 1) * 100, 2),
                    "avg_return": _weekly_avg,
                    "period_avg_return": _period_avg,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                    "sharpe": round(sharpe, 2),
                    "n_trades": total,
                    "n_slots": n_slots,
                }

                # 서브기간별 평균 주간수익률
                if _sub_cutoffs:
                    for _slbl, _scut in _sub_cutoffs.items():
                        _sub_r = [r for r, wd in all_rets if wd >= _scut]
                        if _sub_r:
                            _sp = round(float(np.mean(_sub_r)) * 100, 2)
                        else:
                            _sp = 0.0
                        _row[f"avg_return_{_slbl}"] = _sp

                grid_results.append(_row)

    # 최적 조합 찾기
    grid_results.sort(key=lambda x: -x["avg_return"])
    best = grid_results[0] if grid_results else None

    # 기간별 최적조합
    best_by_period = {}
    if best:
        best_by_period["전체"] = best
    if _sub_cutoffs and grid_results:
        _period_names = {
            "1Y": "최근 1년",
            "6M": "최근 6개월",
            "3M": "최근 3개월",
            "1M": "최근 1개월",
        }
        for _slbl in _sub_cutoffs:
            _key = f"avg_return_{_slbl}"
            _sorted = sorted(grid_results, key=lambda x: -x.get(_key, 0))
            best_by_period[_slbl] = _sorted[0]

    # txt 텍스트 생성
    lines = []
    lines.append("=" * 70)
    lines.append("최적 매도조건(A,B,C) 탐색 결과")
    lines.append(
        f"백테스트 기간: {config.get('start_date', '?')} ~ {config.get('end_date', '?')}"
    )
    lines.append(f"최적화 대상 기간: {target_period}주")
    lines.append(f"학습 윈도우: {config.get('train_window', '?')}")
    lines.append(f"Boost Rounds: {config.get('num_boost_round', '?')}")
    _lhw = config.get("label_horizon_weeks", config.get("label_horizon_months", "?"))
    lines.append(f"수익목표기간: {_lhw}주")
    n_slots_info = max(4, target_period)
    lines.append(f"총 조합: {total_combos}개")
    lines.append(f"수익률 산출: 실투자 (슬롯 모델, {n_slots_info}슬롯 순환)")
    lines.append("=" * 70)
    if best:
        lines.append("\n[최적 조합]")
        lines.append(f"  A(최소수익률) = {best['A%']}%")
        lines.append(f"  B(고점대비하락) = {best['B%']}%")
        lines.append(f"  C(손절) = {best['C%']}%")
        lines.append(
            f"  실투자 수익률(분할) = {best['cum_pct']:+.2f}% ({n_slots_info}슬롯)"
        )
        lines.append(f"  일괄투자 수익률 = {best['lumpsum_pct']:+.2f}%")
        lines.append(f"  평균 주간수익률 = {best['avg_return']:+.2f}%")
        lines.append(f"  기간평균수익률 = {best['period_avg_return']:+.2f}%")
        lines.append(f"  승률 = {best['win_rate']:.1f}%")
        lines.append(f"  샤프비율 = {best['sharpe']:.2f}")
        lines.append(f"  총거래수 = {best['n_trades']}")
    lines.append(f"\n{'─' * 70}")
    lines.append(
        f"{'순위':>4} {'A%':>4} {'B%':>4} {'C%':>4} {'누적수익률':>10} "
        f"{'주간수익':>8} {'승률':>6} {'샤프':>6} {'거래수':>6}"
    )
    lines.append(f"{'─' * 70}")
    for i, r in enumerate(grid_results[:50], 1):
        lines.append(
            f"{i:>4} {r['A%']:>4} {r['B%']:>4} {r['C%']:>4} "
            f"{r['cum_pct']:>+9.2f}% {r['avg_return']:>+7.2f}% "
            f"{r['win_rate']:>5.1f}% {r['sharpe']:>5.2f} {r['n_trades']:>6}"
        )
    lines.append(f"{'─' * 70}")

    settings_text = "\n".join(lines)

    return {
        "grid_results": grid_results,
        "best": best,
        "best_by_period": best_by_period,
        "settings_text": settings_text,
    }


# ──────────────────────────────────────────────
# 설정조합별 수익률검사 (Grid Search)
# ──────────────────────────────────────────────


def _aggregate_weekly_stats(weekly_results, hold_weeks):
    """주별 시뮬레이션 결과 목록에서 집계 통계를 계산.

    Args:
        weekly_results: list[dict] - date, port_ret, wins, losses, exits 키
        hold_weeks: 보유기간 (주)

    Returns:
        dict: 누적수익률, 실투자수익률, 기간평균수익률, 평균 주간수익률, ...
    """
    all_rets = [w["port_ret"] for w in weekly_results]
    wins = sum(w["wins"] for w in weekly_results)
    losses = sum(w["losses"] for w in weekly_results)
    exit_counts = {"손절": 0, "트레일링스탑": 0, "만기매도": 0}
    for w in weekly_results:
        for reason in exit_counts:
            exit_counts[reason] += w["exits"].get(reason, 0)

    total = wins + losses
    if not all_rets or total == 0:
        return {
            "실투자수익률": 0.0,
            "일괄투자수익률": 0.0,
            "기간평균수익률": 0.0,
            "평균 주간수익률": 0.0,
            "승률": 0.0,
            "샤프비율": 0.0,
            "총거래수": 0,
        }

    ret_arr = np.array(all_rets)

    # 실투자 수익률 (슬롯 모델)
    n_slots = max(4, hold_weeks)
    initial = 100_000_000
    slot_cap = [initial / n_slots] * n_slots
    for wi, pr in enumerate(all_rets):
        slot_cap[wi % n_slots] *= 1.0 + pr
    realistic_cum = sum(slot_cap) / initial

    # 일괄매수 수익률
    lumpsum_cum = _compute_lumpsum_from_rets(all_rets, hold_weeks)

    # 샤프 비율 (겹치는 수익률 보정: √(52/horizon))
    std_r = ret_arr.std() if len(ret_arr) > 1 else 0.0
    sharpe = (ret_arr.mean() / std_r * np.sqrt(52 / hold_weeks)) if std_r > 0 else 0.0

    _period_avg = round(float(ret_arr.mean()) * 100, 2)

    return {
        "실투자수익률": round((realistic_cum - 1) * 100, 2),
        "일괄투자수익률": round((lumpsum_cum - 1) * 100, 2),
        "기간평균수익률": _period_avg,
        "평균 주간수익률": _period_avg,  # N주 보유수익률 평균 (주간배치 단위)
        "승률": round(wins / total * 100, 1),
        "샤프비율": round(float(sharpe), 2),
        "총거래수": total,
    }


def _resim_grid_combo(
    week_trades,
    sorted_weeks,
    top_k,
    buy_type,
    a_dec,
    b_dec,
    c_dec,
    hold_days,
    hold_weeks,
    orig_buy_type,
    today_str,
    prob_threshold=None,
):
    """단일 trading 파라미터 조합에 대한 재시뮬레이션.

    week_trades: {week_date -> [trade_info, ...]}  (max_top_k 종목)
    각 trade_info: {ticker, buy_price_close, buy_price_open, forward, week_date}

    전체 기간 + 서브기간(1Y, 6M, 3M) 통계를 함께 반환.
    """
    # Phase 1: 전체 주별 시뮬레이션 실행, 결과를 날짜별로 저장
    weekly_results = []

    for week_date in sorted_weeks:
        trades = week_trades[week_date]
        top_trades = trades[:top_k]
        if prob_threshold is not None:
            top_trades = [
                t for t in top_trades if t.get("predicted_prob", 0) > prob_threshold
            ]
        week_rets = []
        w_wins = 0
        w_losses = 0
        w_exits = {"손절": 0, "트레일링스탑": 0, "만기매도": 0}

        for t in top_trades:
            bp = t["buy_price_open"] if buy_type == "open" else t["buy_price_close"]
            if bp is None or bp <= 0:
                continue
            forward = t["forward"]
            if forward is None or len(forward) == 0:
                continue
            _, ret_pct, reason, _ = _simulate_trade(bp, forward, b_dec, c_dec, a_dec)
            week_rets.append(ret_pct)
            if ret_pct > 0:
                w_wins += 1
            else:
                w_losses += 1
            if reason in w_exits:
                w_exits[reason] += 1

        if week_rets:
            weekly_results.append(
                {
                    "date": week_date,
                    "port_ret": float(np.mean(week_rets)),
                    "wins": w_wins,
                    "losses": w_losses,
                    "exits": dict(w_exits),
                }
            )

    # Phase 2: 전체 기간 기본 통계 (기존 호환 유지)
    all_rets = [w["port_ret"] for w in weekly_results]
    total_wins = sum(w["wins"] for w in weekly_results)
    total_losses = sum(w["losses"] for w in weekly_results)
    total = total_wins + total_losses
    exit_counts = {"손절": 0, "트레일링스탑": 0, "만기매도": 0}
    for w in weekly_results:
        for reason in exit_counts:
            exit_counts[reason] += w["exits"].get(reason, 0)

    if not all_rets or total == 0:
        empty = {
            "누적수익률": 0.0,
            "실투자수익률": 0.0,
            "일괄투자수익률": 0.0,
            "기간평균수익률": 0.0,
            "평균 주간수익률": 0.0,
            "승률": 0.0,
            "샤프비율": 0.0,
            "최대낙폭": 0.0,
            "총거래수": 0,
            "손절%": 0.0,
            "트레일링%": 0.0,
            "만기매도%": 0.0,
        }
        for sfx in ("_1Y", "_6M", "_3M", "_1M"):
            empty[f"실투자수익률{sfx}"] = 0.0
            empty[f"일괄투자수익률{sfx}"] = 0.0
            empty[f"평균 주간수익률{sfx}"] = 0.0
            empty[f"승률{sfx}"] = 0.0
        return empty

    # 누적 수익률 + 최대 낙폭
    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in all_rets:
        cum *= 1.0 + r
        peak = max(peak, cum)
        dd = (peak - cum) / peak
        max_dd = max(max_dd, dd)

    ret_arr = np.array(all_rets)
    std_r = ret_arr.std() if len(ret_arr) > 1 else 0.0
    sharpe = (ret_arr.mean() / std_r * np.sqrt(52 / hold_weeks)) if std_r > 0 else 0.0

    n_slots = max(4, hold_weeks)
    initial = 100_000_000
    slot_cap = [initial / n_slots] * n_slots
    for wi, pr in enumerate(all_rets):
        slot_cap[wi % n_slots] *= 1.0 + pr
    realistic_cum = sum(slot_cap) / initial

    # 일괄매수 수익률
    lumpsum_cum = _compute_lumpsum_from_rets(all_rets, hold_weeks)

    _period_avg = round(float(ret_arr.mean()) * 100, 2)

    result = {
        "누적수익률": round((cum - 1) * 100, 2),
        "실투자수익률": round((realistic_cum - 1) * 100, 2),
        "일괄투자수익률": round((lumpsum_cum - 1) * 100, 2),
        "기간평균수익률": _period_avg,
        "평균 주간수익률": _period_avg,  # N주 보유수익률 평균 (주간배치 단위)
        "승률": round(total_wins / total * 100, 1),
        "샤프비율": round(float(sharpe), 2),
        "최대낙폭": round(max_dd * 100, 2),
        "총거래수": total,
        "손절%": round(exit_counts["손절"] / total * 100, 1),
        "트레일링%": round(exit_counts["트레일링스탑"] / total * 100, 1),
        "만기매도%": round(exit_counts["만기매도"] / total * 100, 1),
    }

    # Phase 3: 서브기간 통계 (최근 1년, 6개월, 3개월, 1개월)
    today = datetime.strptime(today_str, "%Y%m%d")
    sub_periods = [
        ("_1Y", today - relativedelta(years=1)),
        ("_6M", today - relativedelta(months=6)),
        ("_3M", today - relativedelta(months=3)),
        ("_1M", today - relativedelta(months=1)),
    ]
    for suffix, cutoff in sub_periods:
        cutoff_str = cutoff.strftime("%Y%m%d")
        filtered = [w for w in weekly_results if w["date"] >= cutoff_str]
        sub_stats = _aggregate_weekly_stats(filtered, hold_weeks)
        result[f"실투자수익률{suffix}"] = sub_stats["실투자수익률"]
        result[f"일괄투자수익률{suffix}"] = sub_stats["일괄투자수익률"]
        result[f"평균 주간수익률{suffix}"] = sub_stats["평균 주간수익률"]
        result[f"승률{suffix}"] = sub_stats["승률"]

    return result


def _gs_checkpoint_path(date_str):
    """Grid search 체크포인트 경로."""
    return BACKTEST_CACHE_DIR / f"gs_checkpoint_{date_str}.pkl"


def _save_gs_checkpoint(date_str, data):
    path = _gs_checkpoint_path(date_str)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    _logger.info(
        "Grid search 체크포인트 저장: %d/%d 라벨링 조합 완료, 결과 %d행",
        data["completed_label_idx"],
        data["n_label"],
        len(data["results"]),
    )


def _load_gs_checkpoint(date_str):
    path = _gs_checkpoint_path(date_str)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _clear_gs_checkpoint(date_str):
    path = _gs_checkpoint_path(date_str)
    if path.exists():
        path.unlink()


def clear_gs_checkpoint(date_str):
    """Grid search 체크포인트 삭제 (외부 호출용)."""
    _clear_gs_checkpoint(date_str)


def has_gs_checkpoint(date_str):
    """Grid search 체크포인트 존재 여부."""
    return _gs_checkpoint_path(date_str).exists()


def get_gs_checkpoint_info(date_str):
    """Grid search 체크포인트 정보."""
    cp = _load_gs_checkpoint(date_str)
    if cp is None:
        return None
    return {
        "completed_label_idx": cp["completed_label_idx"],
        "n_label": cp["n_label"],
        "n_results": len(cp["results"]),
        "total_combos": cp["total_combos"],
        "labeling_combos": cp["labeling_combos"],
        "trading_combos_count": cp["n_trade"],
    }


def run_grid_search(
    universe_df,
    date_str,
    horizon_weeks_list,
    target_pct_list,
    stop_pct_list,
    top_k_list,
    buy_type_list,
    a_list,
    b_list,
    c_list,
    train_window_years=2,
    num_boost_round=_BACKTEST_NUM_BOOST_ROUND,
    progress_callback=None,
    log_callback=None,
    resume=False,
    labeling_combos_override=None,
    asset_type="stock",
):
    """설정조합별 수익률검사 — 모든 파라미터 조합 그리드 서치.

    Phase 1: 라벨링 조합(horizon, target, stop)별 run_rolling_backtest 1회
    Phase 2: 매매 조합(top_k, buy_type, A, B, C)별 빠른 re-simulation

    라벨링 조합 완료 시마다 체크포인트 저장 → 중단 후 이어서 실행 가능.

    Args:
        horizon_weeks_list: [3, 4, 6, ...] 주 단위
        target_pct_list: [20, 25, 30, ...] 정수 %
        stop_pct_list: [8, 10, 12, ...] 정수 %
        top_k_list: [5, 10, (5, 0.98), ...] — int 또는 (k, threshold) tuple
        buy_type_list: ["open", "close"]
        a_list: [20, 25, 30, ...] 정수 % (트레일링 목표가)
        b_list: [5, 7, 10, ...] 정수 % (고점 하락 매도)
        c_list: [8, 10, 12, ...] 정수 % (손절)
        progress_callback: fn(msg, pct_float_0to1)
        log_callback: fn(msg) — 로그 메시지 출력
        resume: True이면 체크포인트에서 이어서 실행
        labeling_combos_override: [(hw, tp_int, sp_int), ...] 명시적 라벨링 조합
            지정 시 horizon/target/stop 리스트 대신 이 목록 사용.

    Returns:
        list[dict]: 모든 조합의 결과 행 목록
    """
    import itertools
    from collections import defaultdict

    def _log(msg):
        _logger.info(msg)
        if log_callback:
            log_callback(msg)

    if labeling_combos_override is not None:
        labeling_combos = list(labeling_combos_override)
    else:
        labeling_combos = list(
            itertools.product(horizon_weeks_list, target_pct_list, stop_pct_list)
        )
    trading_combos = list(
        itertools.product(top_k_list, buy_type_list, a_list, b_list, c_list)
    )
    n_label = len(labeling_combos)
    n_trade = len(trading_combos)
    total_combos = n_label * n_trade
    max_top_k = max(tk[0] if isinstance(tk, tuple) else tk for tk in top_k_list)

    _log(
        f"=== Grid Search 시작: 학습 {n_label}개 × 매매 {n_trade}개 = "
        f"총 {total_combos:,}개 조합 ==="
    )

    # ── 체크포인트 재개 ──
    results = []
    start_li = 0
    if resume:
        cp = _load_gs_checkpoint(date_str)
        if cp is not None:
            # 파라미터가 일치하는지 확인
            if (
                cp.get("labeling_combos") == labeling_combos
                and cp.get("n_trade") == n_trade
            ):
                results = cp["results"]
                start_li = cp["completed_label_idx"]
                _log(
                    f"체크포인트에서 재개: {start_li}/{n_label} 라벨링 완료, "
                    f"결과 {len(results)}행 로드됨"
                )
            else:
                _log("체크포인트 파라미터 불일치 — 처음부터 실행")
                _clear_gs_checkpoint(date_str)

    done = start_li * n_trade

    for li in range(start_li, n_label):
        hw, tp_int, sp_int = labeling_combos[li]
        tp = tp_int / 100.0
        sp = sp_int / 100.0
        label_tag = f"horizon={hw}주 target={tp_int}% stop={sp_int}%"

        _log(f"[학습 {li + 1}/{n_label}] {label_tag} — 백테스트 시작")

        if progress_callback:
            progress_callback(
                f"[학습 {li + 1}/{n_label}] {label_tag} — 모델 학습 중...",
                done / total_combos,
            )

        # Phase 1: 라벨링 조합별 전체 백테스트 실행
        bt_result = run_rolling_backtest(
            universe_df=universe_df,
            date_str=date_str,
            top_k=max_top_k,
            label_horizon_weeks=hw,
            label_target_pct=tp,
            label_stop_pct=sp,
            trailing_stop_pct=0.10,
            stop_loss_pct=0.10,
            min_profit_pct=0.30,
            buy_price_type="close",
            train_window_years=train_window_years,
            num_boost_round=num_boost_round,
            asset_type=asset_type,
        )

        if "error" in bt_result:
            _log(f"  백테스트 실패: {bt_result['error']}")
            done += n_trade
            # 체크포인트 저장 (실패해도 다음으로 넘어감)
            _save_gs_checkpoint(
                date_str,
                {
                    "completed_label_idx": li + 1,
                    "n_label": n_label,
                    "n_trade": n_trade,
                    "total_combos": total_combos,
                    "labeling_combos": labeling_combos,
                    "results": results,
                },
            )
            continue

        daily = bt_result["daily_results"]
        config = bt_result.get("config", {})
        orig_buy_type = config.get("buy_price_type", "close")
        _log(f"  백테스트 완료: {len(daily)}주 데이터")

        # OHLCV 로드 (re-simulation용)
        ohlcv_dict = load_cached_ohlcv(date_str, asset_type=asset_type)
        if ohlcv_dict is None:
            _log("  OHLCV 캐시 없음 — 건너뜀")
            done += n_trade
            _save_gs_checkpoint(
                date_str,
                {
                    "completed_label_idx": li + 1,
                    "n_label": n_label,
                    "n_trade": n_trade,
                    "total_combos": total_combos,
                    "labeling_combos": labeling_combos,
                    "results": results,
                },
            )
            continue

        # 매수 정보 사전 추출 (주별 그룹, max_top_k 종목)
        hold_days = hw * 5
        week_trades = defaultdict(list)
        for dr in daily:
            buy_date = dr.get("buy_date", "")
            if not buy_date:
                continue
            for s in dr.get("stocks", []):
                ticker = s["ticker"]
                if ticker not in ohlcv_dict:
                    continue
                ohlcv = ohlcv_dict[ticker]
                buy_dt_ts = pd.Timestamp(datetime.strptime(buy_date, "%Y%m%d"))
                after_buy = ohlcv.index > buy_dt_ts
                forward = ohlcv[after_buy].head(hold_days)
                if len(forward) == 0:
                    continue
                # 시가/종가 모두 저장
                if buy_dt_ts in ohlcv.index:
                    buy_row = ohlcv.loc[buy_dt_ts]
                else:
                    idx_arr = ohlcv.index.get_indexer([buy_dt_ts], method="nearest")
                    if idx_arr[0] < 0:
                        continue
                    buy_row = ohlcv.iloc[idx_arr[0]]
                bp_close = float(
                    buy_row.get("종가", buy_row.get("Close", s["buy_price"]))
                )
                bp_open = float(buy_row.get("시가", buy_row.get("Open", bp_close)))
                week_trades[dr["date"]].append(
                    {
                        "ticker": ticker,
                        "predicted_prob": s.get("predicted_prob", 0.0),
                        "buy_price_close": bp_close,
                        "buy_price_open": bp_open,
                        "forward": forward,
                    }
                )
        sorted_weeks = sorted(week_trades.keys())

        _log(f"  매매 시뮬레이션 시작: {n_trade}개 조합")

        # Phase 2: 매매 조합별 재시뮬레이션
        for ti, (top_k_spec, buy_type, a_pct, b_pct, c_pct) in enumerate(
            trading_combos
        ):
            if isinstance(top_k_spec, tuple):
                top_k, prob_threshold = top_k_spec
            else:
                top_k, prob_threshold = top_k_spec, None

            done += 1
            _tk_label = (
                f"{top_k}(>{int(prob_threshold * 100)})"
                if prob_threshold is not None
                else str(top_k)
            )
            if progress_callback and (ti + 1) % max(1, n_trade // 20) == 0:
                progress_callback(
                    f"[{done}/{total_combos}] {label_tag} | "
                    f"K={_tk_label} {'시가' if buy_type == 'open' else '종가'} "
                    f"A={a_pct}% B={b_pct}% C={c_pct}%",
                    done / total_combos,
                )

            metrics = _resim_grid_combo(
                week_trades,
                sorted_weeks,
                top_k=top_k,
                buy_type=buy_type,
                a_dec=a_pct / 100.0,
                b_dec=b_pct / 100.0,
                c_dec=c_pct / 100.0,
                hold_days=hold_days,
                hold_weeks=hw,
                orig_buy_type=orig_buy_type,
                today_str=date_str,
                prob_threshold=prob_threshold,
            )

            results.append(
                {
                    "수익목표기간": f"{hw}주",
                    "목표수익률": f"{tp_int}%",
                    "학습손절기준": f"{sp_int}%",
                    "TopK": _tk_label,
                    "매수가기준": "시가" if buy_type == "open" else "종가",
                    "A_트레일링목표": f"{a_pct}%",
                    "B_고점하락매도": f"{b_pct}%",
                    "C_손절": f"{c_pct}%",
                    **metrics,
                }
            )

        _log(f"  학습 {li + 1}/{n_label} 완료 — " f"누적 결과: {len(results)}행")

        # ── 라벨링 조합 완료 시마다 체크포인트 저장 ──
        _save_gs_checkpoint(
            date_str,
            {
                "completed_label_idx": li + 1,
                "n_label": n_label,
                "n_trade": n_trade,
                "total_combos": total_combos,
                "labeling_combos": labeling_combos,
                "results": results,
            },
        )

    # 완료 — 체크포인트 삭제
    _clear_gs_checkpoint(date_str)

    if progress_callback:
        progress_callback(f"완료: {total_combos}개 조합", 1.0)

    _log(f"=== Grid Search 완료: {len(results)}개 결과 ===")

    # 실투자수익률 기준 내림차순 정렬
    results.sort(key=lambda x: -x.get("실투자수익률", 0))
    return results
