"""AI추천주 모델 래퍼 — 독립 캐시 + ml_pattern 위임.

Phase A.5: 24주/50% 안정형 모델 파라미터 + 보수적 1차필터 (시총≥1000억, 거래대금≥50억).
Phase B: 자체 2모델 구조(P(up) - λ·P(loss))로 교체 예정.

핵심 원칙: AI급등주(ml_pattern) 코드/캐시에 영향 0.
"""

import pickle
from pathlib import Path

import pandas as pd

from . import backtest_engine as _bt_engine
from . import ml_pattern
from .config import CACHE_DIR
from .ml_pattern import (
    SURGE_FEATURE_NAMES,
    SURGE_FEATURE_NAMES_KR,
    compute_breakout_labels,
    compute_ml_features_surge,
)

# ──────────────────────────────────────────────
# 독립 캐시 경로
# ──────────────────────────────────────────────

RECOMMEND_MODEL_DIR = CACHE_DIR / "recommend_models"
RECOMMEND_BT_DIR = CACHE_DIR / "recommend_backtest"

# 디렉토리 생성
RECOMMEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RECOMMEND_BT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 1. 학습 (ml_pattern 위임)
# ──────────────────────────────────────────────


def train_recommend_model(
    universe_df: pd.DataFrame,
    date_str: str,
    progress_callback=None,
    use_gpu: bool = False,
    horizon_days: int = 120,
    target_pct: float = 0.50,
    stop_pct: float = 0.25,
    min_avg_tv: float = 5_000_000_000,
) -> dict:
    """AI추천주 모델 학습 (24주/50% 안정형).

    ml_pattern.train_multi_period_with_cache() 위임.
    결과를 RECOMMEND_MODEL_DIR에 저장.

    Returns:
        {"surge": model_data, "surge_1y": ..., "surge_6m": ..., "surge_3m": ...}
    """
    result = ml_pattern.train_multi_period_with_cache(
        universe_df=universe_df,
        date_str=date_str,
        progress_callback=progress_callback,
        use_gpu=use_gpu,
        horizon_days=horizon_days,
        target_pct=target_pct,
        stop_pct=stop_pct,
        min_avg_tv=min_avg_tv,
    )

    # 결과를 AI추천주 독립 캐시에 저장
    if result:
        for mode, model_data in result.items():
            if model_data:
                save_recommend_model(
                    model_data, date_str, mode=mode, target_pct=target_pct
                )

    return result


# ──────────────────────────────────────────────
# 2. 예측 (ml_pattern 위임)
# ──────────────────────────────────────────────


def predict_recommend(
    model_data: dict,
    ohlcv_dict: dict,
    tickers: list,
    kospi_close: pd.Series = None,
    min_avg_tv: float = 5_000_000_000,
    **kwargs,
) -> pd.DataFrame:
    """AI추천주 예측 (거래대금 ≥50억 기본).

    ml_pattern.predict_current_surge() 위임.
    """
    return ml_pattern.predict_current_surge(
        model_data=model_data,
        ohlcv_dict=ohlcv_dict,
        tickers=tickers,
        kospi_close=kospi_close,
        min_avg_tv=min_avg_tv,
        **kwargs,
    )


# ──────────────────────────────────────────────
# 3. 모델 저장/로드 (독립 캐시)
# ──────────────────────────────────────────────


def save_recommend_model(
    model_data: dict,
    date_str: str,
    mode: str = "surge",
    target_pct: float | None = None,
) -> Path:
    """AI추천주 모델을 독립 캐시에 pkl 저장.

    AI급등주의 cache/ml_models/ 와는 완전 분리.
    """
    RECOMMEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prefix = f"recommend_{mode}"
    if target_pct is not None:
        prefix += f"_t{int(target_pct * 100)}"

    # 날짜별
    path = RECOMMEND_MODEL_DIR / f"{prefix}_{date_str}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    # latest
    latest = RECOMMEND_MODEL_DIR / f"{prefix}_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(model_data, f)

    return path


def load_recommend_model(
    date_str: str = None,
    mode: str = "surge",
    target_pct: float | None = None,
) -> dict | None:
    """AI추천주 모델 로드. 없으면 None."""
    prefix = f"recommend_{mode}"
    if target_pct is not None:
        prefix += f"_t{int(target_pct * 100)}"

    if date_str:
        path = RECOMMEND_MODEL_DIR / f"{prefix}_{date_str}.pkl"
    else:
        path = RECOMMEND_MODEL_DIR / f"{prefix}_latest.pkl"

    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    return None


# ──────────────────────────────────────────────
# 4. 백테스트 (backtest_engine 위임 + 독립 저장)
# ──────────────────────────────────────────────


def run_recommend_backtest(
    universe_df,
    date_str,
    progress_callback=None,
    train_window_years=2,
    top_k=20,
    trailing_stop_pct=0.05,
    stop_loss_pct=0.10,
    min_profit_pct=0.30,
    resume=False,
    num_boost_round=400,
    buy_price_type="close",
    label_target_pct=0.50,
    label_stop_pct=0.25,
    label_horizon_weeks=24,
    target_hold_periods=None,
    asset_type="stock",
    min_avg_tv: float = 5_000_000_000,
) -> dict:
    """AI추천주 백테스트 실행 (24주/50% 기본).

    backtest_engine.run_rolling_backtest() 위임.
    결과를 RECOMMEND_BT_DIR에 저장.
    """
    result = _bt_engine.run_rolling_backtest(
        universe_df=universe_df,
        date_str=date_str,
        progress_callback=progress_callback,
        train_window_years=train_window_years,
        top_k=top_k,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_pct=stop_loss_pct,
        min_profit_pct=min_profit_pct,
        resume=resume,
        num_boost_round=num_boost_round,
        buy_price_type=buy_price_type,
        label_target_pct=label_target_pct,
        label_stop_pct=label_stop_pct,
        label_horizon_weeks=label_horizon_weeks,
        target_hold_periods=target_hold_periods,
        asset_type=asset_type,
        min_avg_tv=min_avg_tv,
    )

    # 결과를 AI추천주 독립 캐시에 저장
    if result:
        save_recommend_backtest(result, date_str)

    return result


def save_recommend_backtest(result, date_str=None) -> Path:
    """AI추천주 백테스트 결과 pickle 저장."""
    from datetime import datetime

    RECOMMEND_BT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recommend_bt_{date_str or ts}.pkl"
    filepath = RECOMMEND_BT_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(result, f)
    # latest
    latest = RECOMMEND_BT_DIR / "recommend_bt_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(result, f)
    return filepath


def load_recommend_backtest(path=None) -> dict | None:
    """AI추천주 백테스트 결과 로드."""
    if path and Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    latest = RECOMMEND_BT_DIR / "recommend_bt_latest.pkl"
    if latest.exists():
        with open(latest, "rb") as f:
            return pickle.load(f)
    return None


# ──────────────────────────────────────────────
# 5. 공유 상수/함수 re-export
# ──────────────────────────────────────────────
# Phase B에서 독자 피처/라벨로 교체할 때
# import 경로 변경 없이 여기서만 교체하면 됨.

# SURGE_FEATURE_NAMES — 이미 상단에서 import
# SURGE_FEATURE_NAMES_KR — 이미 상단에서 import
# compute_ml_features_surge — 이미 상단에서 import
# compute_breakout_labels — 이미 상단에서 import

__all__ = [
    # 학습/예측
    "train_recommend_model",
    "predict_recommend",
    # 모델 저장/로드
    "save_recommend_model",
    "load_recommend_model",
    # 백테스트
    "run_recommend_backtest",
    "save_recommend_backtest",
    "load_recommend_backtest",
    # 상수/함수 re-export
    "SURGE_FEATURE_NAMES",
    "SURGE_FEATURE_NAMES_KR",
    "compute_ml_features_surge",
    "compute_breakout_labels",
    # 캐시 경로
    "RECOMMEND_MODEL_DIR",
    "RECOMMEND_BT_DIR",
]
