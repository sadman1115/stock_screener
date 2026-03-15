"""AI추천주 강화학습 분석/규칙 생성 모듈.

AI추천주 백테스트 결과에서 고수익/저수익 패턴을 분석하여
학습규칙을 자동 생성하고, 개선된 모델로 예측한다.

핵심 원칙: AI강화학습(reinforcement.py) 코드/캐시에 영향 0.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .recommend_model import SURGE_FEATURE_NAMES, SURGE_FEATURE_NAMES_KR

RECOMMEND_RF_DIR = CACHE_DIR / "recommend_reinforced"

# ── AI급등주(_train_sklearn_surge)와 동일한 LGB 파라미터 ──
# backtest_engine._BACKTEST_LGB_PARAMS 와는 다름 (백테스트용은 31 leaves/0.05 lr)
_SURGE_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "verbose": -1,
    "seed": 42,
}
_SURGE_NUM_BOOST_ROUND = 1000
_SURGE_EARLY_STOPPING = 50

# 고수익/저수익 임계치
_HIGH_RETURN_THR = 0.15  # +15% 이상
_LOSS_RETURN_THR = -0.05  # -5% 이하

# 자동 생성 패턴 최대 개수
_MAX_PATTERN_BOOSTS = 5


# ──────────────────────────────────────────────
# 1. 패턴 분석
# ──────────────────────────────────────────────


def analyze_recommend_patterns(backtest_result: dict) -> dict:
    """백테스트 결과에서 고수익/저수익 패턴 분석.

    Args:
        backtest_result: run_rolling_backtest() 반환값

    Returns:
        dict with keys:
            feature_return_corr: {feat: spearman_corr}
            high_return_profile: {feat: mean_val}
            loss_profile: {feat: mean_val}
            profile_diff: {feat: high_mean - loss_mean}
            optimal_ranges: {feat: (q25, q75)}
            high_return_stocks: list[dict]
            loss_stocks: list[dict]
    """
    daily_results = backtest_result.get("daily_results", [])
    if not daily_results:
        raise ValueError("백테스트 결과가 비어있습니다.")

    # 모든 종목의 (features, return) 수집
    all_preds = []
    for day in daily_results:
        for stock in day["stocks"]:
            feats = stock.get("features", {})
            if not feats:
                continue
            all_preds.append(
                {
                    "ticker": stock["ticker"],
                    "name": stock.get("name", stock["ticker"]),
                    "date": day["date"],
                    "features": feats,
                    "return_pct": stock["return_pct"],
                    "exit_reason": stock["exit_reason"],
                    "hold_days": stock["hold_days"],
                    "predicted_prob": stock.get("predicted_prob", 0),
                }
            )

    if len(all_preds) < 50:
        raise ValueError(f"분석에 필요한 데이터 부족: {len(all_preds)}건 (최소 50건)")

    # 사용 가능한 피처 이름 (교집합)
    available_feats = sorted(
        set.intersection(*[set(p["features"].keys()) for p in all_preds])
    )
    if not available_feats:
        available_feats = sorted(all_preds[0]["features"].keys())

    # 피처-수익률 Spearman 상관관계
    returns = np.array([p["return_pct"] for p in all_preds])
    feature_return_corr = {}
    for feat_name in available_feats:
        vals = np.array([p["features"].get(feat_name, 0.0) for p in all_preds])
        if np.std(vals) < 1e-10:
            feature_return_corr[feat_name] = 0.0
            continue
        feature_return_corr[feat_name] = float(_spearman_corr(vals, returns))

    # 고수익/저수익 그룹 분리
    high_return = [p for p in all_preds if p["return_pct"] >= _HIGH_RETURN_THR]
    loss_group = [p for p in all_preds if p["return_pct"] < _LOSS_RETURN_THR]

    # 그룹별 피처 프로파일
    high_profile = _compute_group_profile(high_return, available_feats)
    loss_profile = _compute_group_profile(loss_group, available_feats)

    profile_diff = {}
    for feat in available_feats:
        h_val = high_profile.get(feat, 0.0)
        l_val = loss_profile.get(feat, 0.0)
        profile_diff[feat] = h_val - l_val

    # 최적 피처 범위 (고수익 그룹의 25~75% 분위)
    optimal_ranges = {}
    if high_return:
        for feat in available_feats:
            vals = sorted(p["features"].get(feat, 0.0) for p in high_return)
            n = len(vals)
            q25 = vals[max(0, int(n * 0.25))]
            q75 = vals[min(n - 1, int(n * 0.75))]
            optimal_ranges[feat] = (round(q25, 4), round(q75, 4))

    return {
        "feature_return_corr": feature_return_corr,
        "high_return_profile": high_profile,
        "loss_profile": loss_profile,
        "profile_diff": profile_diff,
        "optimal_ranges": optimal_ranges,
        "high_return_stocks": high_return,
        "loss_stocks": loss_group,
        "total_predictions": len(all_preds),
        "available_features": available_feats,
        "stats": {
            "total": len(all_preds),
            "high_return_count": len(high_return),
            "loss_count": len(loss_group),
            "high_return_avg": float(np.mean([p["return_pct"] for p in high_return]))
            if high_return
            else 0.0,
            "loss_avg": float(np.mean([p["return_pct"] for p in loss_group]))
            if loss_group
            else 0.0,
        },
    }


def _spearman_corr(x, y):
    """Spearman 순위 상관계수."""
    from scipy.stats import spearmanr

    try:
        corr, _ = spearmanr(x, y, nan_policy="omit")
        return corr if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def _compute_group_profile(group, feature_names):
    """그룹의 피처 평균값 계산."""
    if not group:
        return dict.fromkeys(feature_names, 0.0)
    profile = {}
    for feat in feature_names:
        vals = [p["features"].get(feat, 0.0) for p in group]
        profile[feat] = round(float(np.mean(vals)), 6)
    return profile


# ──────────────────────────────────────────────
# 2. 학습규칙 자동 생성
# ──────────────────────────────────────────────


def generate_recommend_rf_rules(
    pattern_analysis: dict,
    backtest_result: dict,
    version: int | None = None,
) -> dict:
    """분석 결과로 강화학습 규칙 자동 생성.

    모든 버전에서 패턴 분석 결과를 반영하여 규칙 생성.

    Args:
        pattern_analysis: analyze_backtest_patterns() 반환값
        backtest_result: 원본 백테스트 결과
        version: 명시적 버전 번호 (None이면 latest에서 자동 판단)

    Returns:
        recommend_rf_rules dict
    """
    corr = pattern_analysis["feature_return_corr"]
    available_feats = pattern_analysis["available_features"]
    summary = backtest_result.get("summary", {})

    # 버전 결정
    if version is not None:
        next_version = version
    else:
        prev_rules = load_recommend_rf_rules()
        next_version = (prev_rules.get("version", 0) + 1) if prev_rules else 1

    # A. 피처 중요도 가중치 (참고용으로만 저장)
    feature_weights = _compute_feature_weights(corr, available_feats)

    # B. 고수익 패턴 부스트 (패턴 분석 기반)
    pattern_boosts = _detect_pattern_boosts(pattern_analysis)

    # C. LightGBM 하이퍼파라미터 (버전별 seed + 성능 기반 조정)
    lgb_params = _suggest_lgb_params(summary, version=next_version)

    # D. Triple-Barrier 조정
    target_pct, stop_pct = _suggest_barrier_params(backtest_result)

    # E. 피처 선택 (상관관계 기반 제외, 버전별 threshold 차등)
    excluded_features = _select_excluded_features(
        corr, available_feats, version=next_version
    )

    # F. 적격종목 필터
    min_mcap = 100_000_000_000
    min_trade_value = 3_000_000_000

    return {
        "version": next_version,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backtest_period": f"{backtest_result['config']['start_date']} ~ "
        f"{backtest_result['config']['end_date']}",
        "backtest_stats": {
            "avg_return": summary.get("avg_portfolio_return", 0),
            "win_rate": summary.get("win_rate", 0),
            "cumulative_return": summary.get("cumulative_return", 1.0),
            "sharpe_ratio": summary.get("sharpe_ratio", 0),
        },
        "feature_weights": feature_weights,
        "pattern_boosts": pattern_boosts,
        "lgb_params": lgb_params,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "excluded_features": excluded_features,
        "min_mcap": min_mcap,
        "min_trade_value": min_trade_value,
    }


def _compute_feature_weights(corr: dict, available_feats: list) -> dict:
    """피처-수익률 상관관계 기반 가중치 계산.

    상관관계 높은 피처 → 가중치 증가, 낮은 피처 → 가중치 감소.
    가중치 범위: 0.5 ~ 2.0
    """
    if not corr:
        return {}

    abs_corrs = {f: abs(corr.get(f, 0)) for f in available_feats}
    max_corr = max(abs_corrs.values()) if abs_corrs else 1.0
    if max_corr < 1e-10:
        return dict.fromkeys(available_feats, 1.0)

    weights = {}
    for feat in available_feats:
        # 정규화: 0 → 0.5, max → 2.0
        normalized = abs_corrs[feat] / max_corr
        weight = 0.5 + normalized * 1.5
        weights[feat] = round(weight, 2)

    return weights


def _detect_pattern_boosts(pattern_analysis: dict) -> list[dict]:
    """고수익 그룹의 피처 분포에서 유의미한 패턴 자동 감지.

    과적합 방지를 위해 엄격한 기준 적용:
    - 상관계수 >= 0.08 이상인 피처만 사용
    - 매칭 종목 >= max(10, 고수익 종목의 20%) 이상
    - 가중치 상한 1.15x (LightGBM이 자체적으로 hard example을 처리하므로
      수동 가중치는 미세 조정에 그쳐야 함)
    """
    corr = pattern_analysis["feature_return_corr"]
    optimal = pattern_analysis["optimal_ranges"]
    high_stocks = pattern_analysis["high_return_stocks"]

    if not high_stocks or not optimal or len(high_stocks) < 20:
        return []

    # 상관관계 상위 피처: 0.08 이상만 (통계적 유의미성 확보)
    sorted_feats = sorted(corr.items(), key=lambda x: -abs(x[1]))
    top_feats = [f for f, c in sorted_feats[:15] if abs(c) >= 0.08]

    if len(top_feats) < 2:
        return []

    # 피처 쌍 조합으로 패턴 후보 생성
    min_match = max(10, int(len(high_stocks) * 0.20))
    pattern_candidates = []
    for i in range(min(len(top_feats), 6)):
        for j in range(i + 1, min(len(top_feats), 6)):
            f1, f2 = top_feats[i], top_feats[j]
            if f1 not in optimal or f2 not in optimal:
                continue

            r1 = optimal[f1]
            r2 = optimal[f2]
            match_count = 0
            match_returns = []
            for stock in high_stocks:
                v1 = stock["features"].get(f1, 0)
                v2 = stock["features"].get(f2, 0)
                if r1[0] <= v1 <= r1[1] and r2[0] <= v2 <= r2[1]:
                    match_count += 1
                    match_returns.append(stock["return_pct"])

            if match_count >= min_match:
                avg_ret = float(np.mean(match_returns))
                pattern_candidates.append(
                    {
                        "features": (f1, f2),
                        "ranges": {f1: list(r1), f2: list(r2)},
                        "match_count": match_count,
                        "avg_return": avg_ret,
                    }
                )

    # 평균 수익률 순으로 상위 N개 선택
    pattern_candidates.sort(key=lambda x: -x["avg_return"])
    boosts = []
    used_feats = set()
    for cand in pattern_candidates:
        f1, f2 = cand["features"]
        if f1 in used_feats and f2 in used_feats:
            continue
        used_feats.update([f1, f2])

        kr1 = SURGE_FEATURE_NAMES_KR.get(f1, f1)
        kr2 = SURGE_FEATURE_NAMES_KR.get(f2, f2)
        # 가중치: 1.05 ~ 1.15 범위 (미세 조정만)
        w = round(1.05 + min(cand["avg_return"], 0.5) * 0.2, 2)
        w = min(w, 1.15)
        boosts.append(
            {
                "name": f"{kr1}+{kr2}",
                "conditions": cand["ranges"],
                "weight_multiplier": w,
                "match_count": cand["match_count"],
                "avg_return": round(cand["avg_return"], 4),
            }
        )
        if len(boosts) >= _MAX_PATTERN_BOOSTS:
            break

    return boosts


def _suggest_lgb_params(summary: dict, version: int = 1) -> dict:
    """백테스트 성능 분석 기반 LightGBM 하이퍼파라미터 제안.

    버전별로 다른 seed를 부여하여 모델 탐색 다양성을 확보하고,
    이전 백테스트 성능 지표(샤프, 승률, 평균수익률)에 따라
    정규화/학습률/용량을 조정한다.
    """
    sharpe = summary.get("sharpe_ratio", 0)
    win_rate = summary.get("win_rate", 0)
    avg_ret = summary.get("avg_portfolio_return", 0)

    num_leaves = 31
    min_data = 30
    lr = 0.05
    l1 = 0.1
    l2 = 1.0
    ff = 0.8

    if sharpe < 0.8:
        l2 = 2.0
        min_data = 40
        ff = 0.7
    elif sharpe > 1.5:
        num_leaves = 40
        min_data = 25
        lr = 0.06

    if win_rate > 0.52 and avg_ret < 0.01:
        lr = 0.03
        num_leaves = 25

    if win_rate < 0.45:
        l1 = 0.3
        l2 = 1.5
        ff = 0.7

    return {
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data,
        "learning_rate": lr,
        "lambda_l1": l1,
        "lambda_l2": l2,
        "feature_fraction": ff,
        "seed": 42 + version,
    }


def _suggest_barrier_params(backtest_result: dict) -> tuple[float, float]:
    """Triple-Barrier 라벨링 파라미터.

    기존 AI급등주 백테스트에서 사용한 라벨 파라미터를 그대로 상속하여
    동일한 양성/음성 정의를 유지한다.
    (예: 페이지4에서 target=30%, stop=10%로 실행했다면 그 값을 사용)
    """
    config = backtest_result.get("config", {})
    target_pct = config.get("label_target_pct", 0.30)
    stop_pct = config.get("label_stop_pct", 0.10)
    return target_pct, stop_pct


def _select_excluded_features(
    corr: dict, available_feats: list, version: int = 1
) -> list[str]:
    """상관관계 기반 피처 제외 (버전별 threshold 차등).

    버전에 따라 제외 threshold를 미세 조정하여 각 버전이
    다른 피처 조합으로 학습하게 한다.
    """
    if not corr or len(available_feats) < 15:
        return []

    threshold = min(0.01 + (version - 1) * 0.002, 0.03)
    candidates = sorted(
        [f for f in available_feats if abs(corr.get(f, 0)) < threshold],
        key=lambda f: abs(corr.get(f, 0)),
    )
    exclude_pct = min(0.05 + (version - 1) * 0.005, 0.10)
    max_exclude = max(1, int(len(available_feats) * exclude_pct))
    return candidates[:max_exclude]


# ──────────────────────────────────────────────
# 3. 강화학습 모델 학습 + 예측
# ──────────────────────────────────────────────


def train_recommend_rf_model(
    universe_df: pd.DataFrame,
    date_str: str,
    rules: dict,
    progress_callback=None,
) -> dict:
    """AI추천주 강화된 규칙으로 모델 학습 + Top 20 예측.

    Args:
        universe_df: 전체 종목 DataFrame
        date_str: 기준일 (YYYYMMDD)
        rules: generate_recommend_rf_rules() 결과
        progress_callback: fn(msg, pct)

    Returns:
        dict with top20, model_info, comparison
    """
    from .data_fetcher import fetch_kospi_index_3year, fetch_ohlcv_3year
    from .recommend_model import (
        compute_breakout_labels,
        compute_ml_features_surge,
    )

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    # 적격종목 필터 (AI급등주와 동일: 시총 ≥ 1000억)
    min_mcap = rules.get("min_mcap", 100_000_000_000)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    mask = mcap.fillna(0) >= min_mcap
    tickers = universe_df[mask].index.tolist()
    _cb(f"적격 종목: {len(tickers)}개 (시총≥{min_mcap / 1e8:.0f}억)", 5)

    # KOSPI
    kospi_close = None
    try:
        kospi_df = fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
    except Exception:
        pass

    # 종목별 OHLCV 수집 + (v2+일 경우) 학습 데이터 수집
    feat_names = list(SURGE_FEATURE_NAMES)
    target_pct = rules.get("target_pct", 0.20)
    stop_pct = rules.get("stop_pct", 0.08)
    min_tv = rules.get("min_trade_value", 3_000_000_000)  # 거래대금 ≥ 30억
    is_v1 = rules.get("version", 1) == 1

    all_X, all_y, all_hard, all_weights = [], [], [], []
    ohlcv_dict_all = {}  # 전종목 OHLCV (predict_recommend용)
    ticker_meta = {}  # 종목별 시총/거래대금
    name_map = universe_df["Name"].to_dict() if "Name" in universe_df.columns else {}
    mcap_map = (
        universe_df["MarketCap"].to_dict() if "MarketCap" in universe_df.columns else {}
    )
    n_skipped_tv = 0

    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            _cb(f"데이터 수집: {i}/{total}", int(5 + (i / total) * 40))
        try:
            ohlcv = fetch_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 60:
                continue

            if not isinstance(ohlcv.index, pd.DatetimeIndex):
                ohlcv.index = pd.to_datetime(ohlcv.index)

            # 거래대금 필터 (AI급등주와 동일: 최근 20일 평균 ≥ 30억)
            if "거래대금" in ohlcv.columns:
                _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
            else:
                _avg_tv = (
                    (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                    .iloc[-20:]
                    .mean()
                )
            if _avg_tv < min_tv:
                n_skipped_tv += 1
                continue

            # 예측용 OHLCV 저장 (predict_recommend는 ≥60일이면 처리 가능)
            ohlcv_dict_all[ticker] = ohlcv
            ticker_meta[ticker] = {
                "avg_tv": _avg_tv,
                "mcap": mcap_map.get(ticker, 0),
            }

            # v1은 학습 불필요 → OHLCV 수집만
            if is_v1:
                continue

            # v2+: 학습 데이터 수집 (≥200일인 종목만)
            if len(ohlcv) < 200:
                continue

            feats = compute_ml_features_surge(ohlcv, kospi_close)
            if feats.empty:
                continue

            # 피처 컬럼 정렬 (train_model_surge와 동일)
            for col in feat_names:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[feat_names]

            lbl, _fwd, hard = compute_breakout_labels(
                ohlcv, target_pct=target_pct, stop_pct=stop_pct
            )
            valid = lbl.notna()
            if valid.sum() < 10:
                continue

            all_X.append(feats[valid])
            all_y.append(lbl[valid])
            all_hard.append(hard[valid])

            # 패턴 부스트 가중치 계산
            boost_w = _compute_boost_weights(
                feats[valid], rules.get("pattern_boosts", [])
            )
            all_weights.append(boost_w)

        except Exception:
            continue

    if n_skipped_tv:
        _cb(f"거래대금 부족 제외: {n_skipped_tv}종목", 45)

    # ── v1: 기존 AI추천주 캐시 모델을 그대로 사용 (baseline과 100% 동일) ──
    if is_v1:
        from .recommend_model import load_recommend_model, predict_recommend

        # target_pct=0.25 (25%급등주) 우선, 없으면 0.30, 없으면 기본 모델
        cached = None
        for tp in [0.25, 0.30, None]:
            cached = load_recommend_model(mode="surge", target_pct=tp)
            if cached is not None:
                _cb(
                    f"v1: 기존 AI추천주 캐시 모델 사용 "
                    f"(target={tp}, AUC={cached.get('validation_auc', 0):.3f})",
                    55,
                )
                break

        if cached is not None:
            _cb("전종목 예측 중 (AI추천주 모델)...", 60)
            pred_df = predict_recommend(
                cached,
                ohlcv_dict_all,
                list(ohlcv_dict_all.keys()),
                kospi_close,
            )

            passed = pred_df[pred_df["AI_Pass"]].sort_values(
                "AI_Score", ascending=False
            )
            top20 = []
            for ticker in passed.head(20).index:
                row = passed.loc[ticker]
                meta = ticker_meta.get(ticker, {})
                top20.append(
                    {
                        "ticker": ticker,
                        "name": name_map.get(ticker, ticker),
                        "predicted_prob": round(row["AI_Score"] / 100, 4),
                        "score": row["AI_Score"],
                        "avg_tv_억": round(meta.get("avg_tv", 0) / 1e8, 1),
                        "시총_억": round(meta.get("mcap", 0) / 1e8, 0),
                    }
                )

            _cb("강화학습 완료 (v1: AI추천주 모델 동일)", 100)
            return {
                "top20": top20,
                "model_info": {
                    "train_auc": cached.get("train_auc", 0),
                    "wf_avg_auc": cached.get("wf_avg_auc", 0),
                    "n_samples": cached.get("n_samples", 0),
                    "n_positive": cached.get("n_positive", 0),
                    "excluded_features": [],
                    "n_features_used": len(feat_names),
                },
                "rules_version": 1,
                "rules_created_at": rules.get("created_at", ""),
                "date": date_str,
            }

        # v1이지만 캐시 모델이 없으면 아래 학습 경로로 fallback
        _cb("⚠️ AI추천주 캐시 모델 없음 → 새로 학습합니다", 48)
        # 이미 수집한 ohlcv_dict_all에서 학습 데이터 구축
        for _ticker, ohlcv in ohlcv_dict_all.items():
            if len(ohlcv) < 200:
                continue
            try:
                feats = compute_ml_features_surge(ohlcv, kospi_close)
                if feats.empty:
                    continue
                for col in feat_names:
                    if col not in feats.columns:
                        feats[col] = 0.0
                feats = feats[feat_names]
                lbl, _fwd, hard = compute_breakout_labels(
                    ohlcv, target_pct=target_pct, stop_pct=stop_pct
                )
                valid = lbl.notna()
                if valid.sum() < 10:
                    continue
                all_X.append(feats[valid])
                all_y.append(lbl[valid])
                all_hard.append(hard[valid])
                all_weights.append(np.ones(valid.sum(), dtype=np.float64))
            except Exception:
                continue

    # ── v2+ (또는 v1 fallback): 강화학습 모델 학습 ──
    if not all_X:
        raise ValueError("학습 데이터 없음")

    _cb("모델 학습 중...", 50)

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True).values
    hard = pd.concat(all_hard, ignore_index=True).values
    boost_weights = np.concatenate(all_weights)

    # 학습 실행
    model_info = _train_reinforced_lgb(
        X, y, hard, boost_weights, rules, feat_names, _cb
    )

    if model_info is None:
        raise ValueError("모델 학습 실패")

    # ── 예측: predict_recommend() 호출 (AI추천주와 동일한 코드 경로) ──
    _cb("전종목 예측 중...", 75)
    from .recommend_model import predict_recommend

    model_data_for_pred = {
        "model": model_info["model"],
        "use_lgbm": model_info["use_lgb"],
        "feature_importances": model_info.get("feature_importances", {}),
        "feature_names": model_info.get("feature_names", feat_names),
    }

    pred_df = predict_recommend(
        model_data_for_pred,
        ohlcv_dict_all,
        list(ohlcv_dict_all.keys()),
        kospi_close,
    )

    # Top 20 추출 + 시총/거래대금 부착
    passed = pred_df[pred_df["AI_Pass"]].sort_values("AI_Score", ascending=False)
    top20 = []
    for ticker in passed.head(20).index:
        row = passed.loc[ticker]
        meta = ticker_meta.get(ticker, {})
        top20.append(
            {
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "predicted_prob": round(row["AI_Score"] / 100, 4),
                "score": row["AI_Score"],
                "avg_tv_억": round(meta.get("avg_tv", 0) / 1e8, 1),
                "시총_억": round(meta.get("mcap", 0) / 1e8, 0),
            }
        )

    _cb("강화학습 완료", 100)

    return {
        "top20": top20,
        "model_info": {
            "train_auc": model_info.get("train_auc", 0),
            "wf_avg_auc": model_info.get("wf_avg_auc", 0),
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "excluded_features": [],
            "n_features_used": len(feat_names),
        },
        "rules_version": rules.get("version", 0),
        "rules_created_at": rules.get("created_at", ""),
        "date": date_str,
    }


def _compute_boost_weights(feats_df, pattern_boosts):
    """패턴 부스트 조건에 맞는 행에 가중치 할당.

    Returns:
        np.ndarray: 행별 부스트 가중치 (기본 1.0)
    """
    n = len(feats_df)
    weights = np.ones(n, dtype=np.float64)

    for boost in pattern_boosts:
        conditions = boost.get("conditions", {})
        multiplier = boost.get("weight_multiplier", 1.5)

        mask = np.ones(n, dtype=bool)
        for feat, (lo, hi) in conditions.items():
            if feat in feats_df.columns:
                vals = feats_df[feat].values
                mask &= (vals >= lo) & (vals <= hi)

        weights[mask] = np.maximum(weights[mask], multiplier)

    return weights


def _train_reinforced_lgb(X, y, hard, boost_weights, rules, feat_names, _cb):
    """강화학습 규칙이 적용된 LightGBM 학습.

    AI급등주(_train_sklearn_surge)와 **동일한 모델 아키텍처** 기반.
    v1(rules 비어있음) → baseline과 동일한 결과.
    v2+ → rules의 lgb_params/pattern_boosts로 점진 개선.
    """
    valid_mask = ~np.isnan(y)
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    hard_clean = hard[valid_mask].copy() if hard is not None else np.zeros(len(y_clean))
    boost_w = boost_weights[valid_mask]

    n_pos = int(y_clean.sum())
    n_neg = int((y_clean == 0).sum())
    if n_pos < 20:
        _cb(f"양성 샘플 부족: {n_pos}건")
        return None

    # 가중치: AI급등주와 동일 (불균형 보정 + Hard Negative)
    w_pos = max(n_neg / n_pos, 1.0)
    sample_weight = np.where(y_clean == 1, w_pos, 1.0)
    hard_mask = hard_clean.astype(bool) & (y_clean == 0)
    sample_weight[hard_mask] = 1.5
    # 패턴 부스트 가중치 (v1은 boost_w=1.0이므로 무영향)
    sample_weight *= boost_w

    X_arr = np.nan_to_num(
        X_clean.values if hasattr(X_clean, "values") else X_clean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float64)
    y_arr = y_clean.values if hasattr(y_clean, "values") else y_clean
    used_feat_names = list(X_clean.columns) if hasattr(X_clean, "columns") else None

    available_feats = used_feat_names or feat_names

    # LGB 파라미터: AI급등주(_train_sklearn_surge)와 동일한 기반
    # is_unbalance=True: AI급등주도 True + 수동 w_pos를 동시 사용 → 동일하게 맞춤
    lgb_params = dict(_SURGE_LGB_PARAMS)
    # v2+ 규칙 오버라이드 (v1은 빈 dict이므로 무영향)
    rule_lgb = rules.get("lgb_params", {})
    if rule_lgb:
        lgb_params.update(rule_lgb)

    from sklearn.metrics import roc_auc_score

    n_total = len(X_arr)
    split_idx = int(n_total * 0.85)

    try:
        import lightgbm as lgb

        dtrain = lgb.Dataset(
            X_arr[:split_idx],
            y_arr[:split_idx],
            weight=sample_weight[:split_idx],
            feature_name=available_feats,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_arr[split_idx:],
            y_arr[split_idx:],
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=_SURGE_NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(_SURGE_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # AUC는 validation set에서만 계산
        val_prob = model.predict(X_arr[split_idx:])
        try:
            auc = roc_auc_score(y_arr[split_idx:], val_prob)
        except Exception:
            auc = 0.5

        # Feature Importance
        imp_gain = model.feature_importance(importance_type="gain")
        imp_sum = imp_gain.sum() if imp_gain.sum() > 0 else 1
        feature_importances = {
            fn: float(ig / imp_sum)
            for fn, ig in zip(available_feats, imp_gain, strict=False)
        }

        _cb(f"  Train AUC: {auc:.3f}")
        top_fi = sorted(feature_importances.items(), key=lambda x: -x[1])[:10]
        for i, (fn, v) in enumerate(top_fi, 1):
            kr = SURGE_FEATURE_NAMES_KR.get(fn, fn)
            _cb(f"  Top{i}: {kr} ({v:.1%})")

        return {
            "model": model,
            "use_lgb": True,
            "feature_names": available_feats,
            "feature_importances": feature_importances,
            "train_auc": auc,
            "wf_avg_auc": 0,
        }
    except ImportError:
        # HGB fallback — AI급등주와 동일 파라미터
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=6,
            min_samples_leaf=50,
            learning_rate=0.03,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
        )
        hgb.fit(X_arr, y_arr, sample_weight=sample_weight)
        pred = hgb.predict_proba(X_arr[split_idx:])[:, 1]
        try:
            auc = roc_auc_score(y_arr[split_idx:], pred)
        except Exception:
            auc = 0.5

        _cb(f"  Train AUC (HGB): {auc:.3f}")

        return {
            "model": hgb,
            "use_lgb": False,
            "feature_names": available_feats,
            "feature_importances": {},
            "train_auc": auc,
            "wf_avg_auc": 0,
        }


# ──────────────────────────────────────────────
# 4. 강화학습 롤링 백테스트
# ──────────────────────────────────────────────


def _train_reinforced_backtest_model(
    X_arr, y_arr, hard_neg_arr, feat_names, rules, num_boost_round=400
):
    """강화학습 규칙이 적용된 백테스트용 LightGBM 학습.

    _train_backtest_model()과 **동일한 구조** 기반.
    v1(rules 비어있음) → baseline과 동일한 결과.
    v2+ → rules의 lgb_params/pattern_boosts로 점진 개선.

    Returns:
        (model, use_lgb, auc) — 기존 백테스트 학습 함수와 동일
    """
    n_pos = int(y_arr.sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos < 10:
        return None, False, 0.0

    # 가중치: _train_backtest_model()과 동일
    w_pos = max(n_neg / n_pos, 1.0)
    sample_weight = np.where(y_arr == 1, w_pos, 1.0)
    hard_mask = hard_neg_arr.astype(bool) & (y_arr == 0)
    sample_weight[hard_mask] = 1.5

    # 패턴 부스트 가중치 (v1은 빈 리스트이므로 무영향)
    pattern_boosts = rules.get("pattern_boosts", [])
    if pattern_boosts and hasattr(X_arr, "shape"):
        boost_w = _compute_boost_weights(
            pd.DataFrame(X_arr, columns=feat_names), pattern_boosts
        )
        sample_weight *= boost_w

    # NaN/Inf 처리
    X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # LGB 파라미터: _train_backtest_model()과 동일 (is_unbalance=True 포함)
    from .backtest_engine import _BACKTEST_LGB_PARAMS

    lgb_params = dict(_BACKTEST_LGB_PARAMS)  # is_unbalance=True 그대로 유지
    # v2+ 규칙 오버라이드 (v1은 빈 dict이므로 무영향)
    rule_lgb = rules.get("lgb_params", {})
    if rule_lgb:
        lgb_params.update(rule_lgb)

    # Train/Val split (85/15) — baseline과 동일
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
            lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(20, verbose=False),  # baseline과 동일: 고정 20
                lgb.log_evaluation(0),
            ],
        )

        from sklearn.metrics import roc_auc_score

        pred_val = model.predict(X_val)
        try:
            auc = roc_auc_score(y_val, pred_val)
        except Exception:
            auc = 0.5

        return model, True, auc
    except ImportError:
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


def run_recommend_rf_backtest(
    universe_df: pd.DataFrame,
    date_str: str,
    rules: dict,
    progress_callback=None,
    train_window_years: int = 2,
    top_k: int = 20,
    trailing_stop_pct: float = 0.05,
    stop_loss_pct: float = 0.10,
    min_profit_pct: float = 0.30,
    label_horizon_weeks: int = 4,
    target_hold_periods: list = None,
    buy_price_type: str = "close",
    num_boost_round: int = 400,
    label_stop_pct: float = None,
) -> dict:
    """AI추천주 강화학습 규칙을 적용한 롤링 백테스트.

    backtest_engine.run_rolling_backtest()와 동일한 구조를 사용하되,
    학습 단계에서 rules의 excluded_features, pattern_boosts, lgb_params를 적용.

    Returns:
        기존 run_rolling_backtest()와 100% 동일한 출력 포맷
    """
    from datetime import timedelta

    from dateutil.relativedelta import relativedelta

    from .backtest_engine import (
        _MIN_MCAP,
        _MIN_POSITIVE_SAMPLES,
        _MIN_TRADE_VALUE,
        HOLD_PERIODS,
        _compute_summary,
        _extract_weekly_dates,
        _get_trading_days,
        _lookup_index,
        _lookup_index_multi_period,
        _next_trading_day,
        _precompute_features_labels,
        _prefetch_all_ohlcv,
        _simulate_trade_multi_period,
    )
    from .data_fetcher import fetch_kosdaq_index_3year, fetch_kospi_index_3year

    def _cb(msg, pct=None):
        if progress_callback:
            try:
                progress_callback(msg, pct)
            except Exception:
                pass

    # ── Phase 0: 날짜 계산 ──
    today = datetime.strptime(date_str, "%Y%m%d")
    label_horizon_days = label_horizon_weeks * 5

    # 라벨 파라미터: rules에서 가져오기 (기존 백테스트와의 핵심 차이점)
    label_target_pct = rules.get("target_pct", 0.20)
    if label_stop_pct is None:
        label_stop_pct = rules.get("stop_pct", 0.08)

    bt_start_raw = today - relativedelta(years=2)
    days_to_sun = (6 - bt_start_raw.weekday()) % 7
    bt_start = (
        bt_start_raw + timedelta(days=days_to_sun) if days_to_sun else bt_start_raw
    )
    bt_end_raw = today
    days_back = bt_end_raw.weekday()
    bt_end = bt_end_raw - timedelta(days=days_back)
    data_start = bt_start - relativedelta(years=train_window_years) - timedelta(days=30)

    bt_start_str = bt_start.strftime("%Y%m%d")
    bt_end_str = bt_end.strftime("%Y%m%d")
    data_start_str = data_start.strftime("%Y%m%d")

    _cb(
        f"AI추천주 강화학습 백테스트: {bt_start_str} ~ {bt_end_str} | "
        f"학습윈도우 {train_window_years}Y | "
        f"수익목표 {label_horizon_weeks}주 | "
        f"규칙 v{rules.get('version', '?')}",
        1,
    )

    # ── Phase 1: 적격 종목 필터 ──
    min_mcap = rules.get("min_mcap", _MIN_MCAP)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    mask = mcap.fillna(0) >= min_mcap
    tickers = universe_df[mask].index.tolist()
    _cb(f"적격 종목: {len(tickers)}개 (시총≥{min_mcap / 1e8:.0f}억)", 2)

    if len(tickers) < 10:
        raise ValueError(f"적격 종목 부족: {len(tickers)}개 (최소 10개)")

    # ── Phase 2: 전종목 OHLCV 수집 ──
    _ohlcv_start = data_start_str if train_window_years > 2 else None
    ohlcv_dict = _prefetch_all_ohlcv(
        tickers, date_str, _cb, data_start_str=_ohlcv_start
    )

    # 거래대금 필터
    min_tv = rules.get("min_trade_value", _MIN_TRADE_VALUE)
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
        if avg_tv >= min_tv:
            eligible_tickers.append(ticker)

    _cb(f"거래대금 필터 후: {len(eligible_tickers)}종목", 27)

    # ── Phase 3: 피처/라벨 사전 계산 ──
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

    eligible_ohlcv = {t: ohlcv_dict[t] for t in eligible_tickers if t in ohlcv_dict}
    features_dict, labels_dict = _precompute_features_labels(
        eligible_ohlcv,
        kospi_close,
        date_str,
        _cb,
        label_horizon_days=label_horizon_days,
        label_target_pct=label_target_pct,
        label_stop_pct=label_stop_pct,
    )

    # ── Phase 4: 주별 날짜 추출 ──
    all_trading_days = _get_trading_days(data_start_str, date_str)
    bt_trading_days = [d for d in all_trading_days if bt_start_str <= d <= bt_end_str]
    weekly_dates = _extract_weekly_dates(bt_trading_days)
    _cb(f"주별 백테스트: {len(weekly_dates)}주", 36)

    if not weekly_dates:
        raise ValueError("백테스트 대상 주가 없습니다.")

    name_map = {}
    if "Name" in universe_df.columns:
        name_map = universe_df["Name"].to_dict()

    # ── Phase 5: 롤링 백테스트 (강화학습 규칙 적용) ──
    daily_results = []
    total_weeks = len(weekly_dates)
    cum_return = 1.0

    # 피처 필터: excluded_features 제외
    excluded = set(rules.get("excluded_features", []))
    feat_names = [f for f in SURGE_FEATURE_NAMES if f not in excluded]

    periods = target_hold_periods or HOLD_PERIODS
    max_hold_trading_days = max(periods) * 5
    phase5_start = datetime.now()

    for wi in range(total_weeks):
        test_date = weekly_dates[wi]
        pct = int(37 + (wi / total_weeks) * 50)

        test_dt = datetime.strptime(test_date, "%Y%m%d")
        train_end_dt = test_dt - timedelta(days=1)
        train_start_dt = test_dt - relativedelta(years=train_window_years)
        train_start_str = train_start_dt.strftime("%Y%m%d")
        train_end_str = train_end_dt.strftime("%Y%m%d")

        _cb(
            f"[{wi + 1}/{total_weeks}] 지정일 {test_date} | "
            f"학습 {train_start_str}~{train_end_str}",
            pct,
        )

        # 종목별 피처/라벨 슬라이싱
        all_X, all_y, all_hard = [], [], []
        for ticker in eligible_tickers:
            if ticker not in features_dict or ticker not in labels_dict:
                continue
            feats = features_dict[ticker]
            y_series, hard_series = labels_dict[ticker]

            date_mask = (feats.index >= train_start_dt) & (feats.index <= train_end_dt)
            if date_mask.sum() < 10:
                continue

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

        # 피처 정렬 (excluded 제외된 feat_names 순서)
        available_cols = [c for c in feat_names if c in X.columns]
        if len(available_cols) < 20:
            continue
        X_aligned = X[available_cols]

        # 강화학습 규칙 적용 학습
        model, use_lgb, auc = _train_reinforced_backtest_model(
            X_aligned.values,
            y,
            hard,
            available_cols,
            rules,
            num_boost_round=num_boost_round,
        )
        if model is None:
            continue

        # ── 예측: 전 종목에서 Top K ──
        scored = []
        for ticker in eligible_tickers:
            if ticker not in features_dict:
                continue
            feats = features_dict[ticker]
            mask_t = feats.index <= test_dt
            if mask_t.sum() == 0:
                continue
            last_row = feats[mask_t].iloc[-1:]
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

        scored.sort(key=lambda x: -x[1])
        top_stocks = scored[:top_k]

        # ── 매매 시뮬레이션 ──
        buy_date = _next_trading_day(test_date, all_trading_days)
        if not buy_date:
            continue

        stock_results = []
        for ticker, prob, feat_snap in top_stocks:
            if ticker not in ohlcv_dict:
                continue
            ohlcv = ohlcv_dict[ticker]
            buy_dt_ts = pd.Timestamp(datetime.strptime(buy_date, "%Y%m%d"))

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

            after_buy = ohlcv.index > buy_idx
            forward_ohlcv = ohlcv[after_buy].head(max_hold_trading_days)

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

        # 기간별 포트폴리오 수익률
        portfolio_returns = {}
        win_counts = {}
        loss_counts = {}
        for m in periods:
            period_rets = [
                s["returns_by_period"][m]["return_pct"]
                for s in stock_results
                if s["returns_by_period"].get(m) is not None
            ]
            if period_rets:
                portfolio_returns[m] = round(float(np.mean(period_rets)), 4)
                win_counts[m] = sum(1 for r in period_rets if r > 0)
                loss_counts[m] = sum(1 for r in period_rets if r <= 0)
            else:
                portfolio_returns[m] = None
                win_counts[m] = 0
                loss_counts[m] = 0

        _first_p = periods[0]
        portfolio_return = portfolio_returns.get(_first_p) or 0.0
        win_count = win_counts.get(_first_p, 0)
        loss_count = loss_counts.get(_first_p, 0)

        # 지수 조회
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
                "kospi_3m": kospi_after.get(_first_p),
                "kosdaq_3m": kosdaq_after.get(_first_p),
            }
        )

        cum_return *= 1.0 + portfolio_return
        elapsed = (datetime.now() - phase5_start).total_seconds()
        _cb(
            f"  → {_first_p}주 {portfolio_return * 100:+.1f}% | "
            f"승 {win_count}/패 {loss_count} | AUC {auc:.3f} | "
            f"누적 {(cum_return - 1) * 100:+.1f}% | {elapsed / 60:.0f}분",
            pct,
        )

    total_elapsed = (datetime.now() - phase5_start).total_seconds()
    _cb(
        f"AI추천주 강화학습 백테스트 완료 ({len(daily_results)}주, {total_elapsed / 60:.1f}분)",
        90,
    )

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
            "asset_type": "stock",
            "model_type": "recommend_rf",
        },
        "daily_results": daily_results,
        "summary": summary,
        "rules_version": rules.get("version"),
    }

    _cb("완료!", 100)
    return result


# ──────────────────────────────────────────────
# 5. 저장/로드
# ──────────────────────────────────────────────


def save_recommend_rf_rules(rules: dict) -> Path:
    """AI추천주 강화학습 규칙을 pickle로 저장."""
    RECOMMEND_RF_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RECOMMEND_RF_DIR / f"recommend_rf_rules_{ts}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(rules, f)
    # latest 파일도 동시 저장
    latest = RECOMMEND_RF_DIR / "recommend_rf_rules_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(rules, f)
    return filepath


def load_recommend_rf_rules() -> dict | None:
    """저장된 규칙 로드 (latest)."""
    latest = RECOMMEND_RF_DIR / "recommend_rf_rules_latest.pkl"
    if latest.exists():
        with open(latest, "rb") as f:
            return pickle.load(f)
    return None


def save_recommend_rf_result(result: dict) -> Path:
    """AI추천주 강화학습 예측 결과 저장."""
    RECOMMEND_RF_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RECOMMEND_RF_DIR / f"recommend_rf_result_{ts}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(result, f)
    latest = RECOMMEND_RF_DIR / "recommend_rf_result_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(result, f)
    return filepath


def load_recommend_rf_result() -> dict | None:
    """AI추천주 강화학습 예측 결과 로드."""
    latest = RECOMMEND_RF_DIR / "recommend_rf_result_latest.pkl"
    if latest.exists():
        with open(latest, "rb") as f:
            return pickle.load(f)
    return None


def save_recommend_rf_backtest(result: dict) -> Path:
    """AI추천주 강화학습 백테스트 결과 저장."""
    RECOMMEND_RF_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RECOMMEND_RF_DIR / f"recommend_rf_backtest_{ts}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(result, f)
    latest = RECOMMEND_RF_DIR / "recommend_rf_backtest_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(result, f)
    return filepath


def load_recommend_rf_backtest() -> dict | None:
    """최신 AI추천주 강화학습 백테스트 결과 로드."""
    latest = RECOMMEND_RF_DIR / "recommend_rf_backtest_latest.pkl"
    if latest.exists():
        with open(latest, "rb") as f:
            return pickle.load(f)
    return None


def reset_recommend_rf_cache() -> int:
    """AI추천주 강화학습 캐시 전체 초기화 (v1부터 재시작).

    Returns:
        삭제된 파일 수
    """
    if not RECOMMEND_RF_DIR.exists():
        return 0
    deleted = 0
    for f in RECOMMEND_RF_DIR.glob("*.pkl"):
        f.unlink()
        deleted += 1
    return deleted


def rollback_recommend_rf_rules(filepath: str) -> dict:
    """특정 버전의 규칙 파일을 latest로 복원.

    Args:
        filepath: 복원할 규칙 pkl 파일명 (예: recommend_rf_rules_20260223_002415.pkl)

    Returns:
        복원된 rules dict
    """
    src = RECOMMEND_RF_DIR / filepath
    if not src.exists():
        raise FileNotFoundError(f"규칙 파일을 찾을 수 없습니다: {filepath}")

    with open(src, "rb") as f:
        rules = pickle.load(f)

    # latest로 덮어쓰기
    latest = RECOMMEND_RF_DIR / "recommend_rf_rules_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(rules, f)

    return rules


def list_recommend_rf_rule_versions() -> list[dict]:
    """저장된 규칙 버전 목록 반환 (최신 먼저).

    Returns:
        [{"version": 2, "filename": "...", "created_at": "...", "win_rate": 0.4}, ...]
    """
    if not RECOMMEND_RF_DIR.exists():
        return []

    versions = []
    for f in sorted(RECOMMEND_RF_DIR.glob("recommend_rf_rules_2*.pkl"), reverse=True):
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            versions.append(
                {
                    "version": r.get("version", "?"),
                    "filename": f.name,
                    "created_at": r.get("created_at", "?"),
                    "win_rate": r.get("backtest_stats", {}).get("win_rate", 0),
                }
            )
        except Exception:
            pass
    return versions
