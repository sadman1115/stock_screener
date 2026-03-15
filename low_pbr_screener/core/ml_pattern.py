"""AI 매수패턴 학습 모듈 (v6.0).

과거 2년 OHLCV에서 '급등 직전 시작점'을 추출하고,
해당 시점의 기술적 피처 패턴을 학습하여 현재 종목에 적용한다.

v6.0 핵심 변경 (근본적 재설계):
- Initiation-Point-Only 라벨링: 급등 이벤트당 첫 날만 양성 + 15일 쿨다운
  · 기존: 조건 충족 연속 5~15일 양성 → 모멘텀 학습 (잘못된 패턴)
  · v6.0: 이벤트당 1일만 양성 → 시작점 패턴 학습 (올바른 패턴)
- 임계값 30% (종가 기준, 진정한 급등만 포착)
- 엄격한 pre-surge 필터: 10d<10%, 20d<15%
- 알고리즘: RandomForest → HistGradientBoosting (AUC 향상)
- 30개 피처 (Drawdown_60d, Prior_Surge_60d 포함)
- 예측: 학습과 동일한 엄격 하드필터 + 모델 점수 판단
- 포착: 바닥→반전, 횡보→돌파, 조정→재상승 (급등 중 배제)
- sklearn HistGradientBoosting (없으면 numpy fallback)
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import indicators
from .config import CACHE_DIR

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

FEATURE_NAMES = [
    "RSI14",
    "MACD_norm",
    "MACD_Hist_norm",
    "StochK",
    "StochD",
    "Close_MA20_ratio",
    "Close_MA60_ratio",
    "Ret_1d",
    "Ret_5d",
    "Ret_10d",
    "Ret_20d",
    "Volatility_20d",
    "VolumeRatio_20d",
    "Drawdown_rolling",
    "NearLow_rolling",
    "ATR_ratio",
    "MACD_Cross_3d",
    "RSI_Oversold_Exit_3d",
    "Stoch_Golden_3d",
    "MA20_Cross_3d",
    # v3.2 신규: 에너지 축적 / 돌파 직전 피처
    "BB_Width_20",
    "BB_Width_Ratio",
    "MA_Convergence",
    "Vol_Contraction",
    "Price_Position",
    "Range_Compression",
    "MACD_Hist_Slope",
    "Accum_Dist_Ratio",
    # v5 신규: 급등 이력 감지 피처
    "Drawdown_60d",
    "Prior_Surge_60d",
]

FEATURE_NAMES_KR = {
    "RSI14": "RSI(14)",
    "MACD_norm": "MACD",
    "MACD_Hist_norm": "MACD히스토그램",
    "StochK": "스토캐스틱K",
    "StochD": "스토캐스틱D",
    "Close_MA20_ratio": "종가/MA20",
    "Close_MA60_ratio": "종가/MA60",
    "Ret_1d": "1일수익률",
    "Ret_5d": "5일수익률",
    "Ret_10d": "10일수익률",
    "Ret_20d": "20일수익률",
    "Volatility_20d": "20일변동성",
    "VolumeRatio_20d": "거래량비율",
    "Drawdown_rolling": "고점대비하락",
    "NearLow_rolling": "저점대비상승",
    "ATR_ratio": "ATR/종가",
    "MACD_Cross_3d": "MACD골든크로스",
    "RSI_Oversold_Exit_3d": "RSI과매도탈출",
    "Stoch_Golden_3d": "Stoch저점골든",
    "MA20_Cross_3d": "MA20돌파",
    # v3.2 신규
    "BB_Width_20": "BB밴드폭",
    "BB_Width_Ratio": "BB폭변화율",
    "MA_Convergence": "이평밀집도",
    "Vol_Contraction": "거래량위축",
    "Price_Position": "가격위치",
    "Range_Compression": "레인지압축",
    "MACD_Hist_Slope": "MACD기울기",
    "Accum_Dist_Ratio": "매집비율",
    # v5 신규
    "Drawdown_60d": "60일고점하락",
    "Prior_Surge_60d": "과거급등이력",
}

# 정밀학습 피처 (30개 기본 + 30개 시계열 = 60개)
PRECISE_FEATURE_NAMES = FEATURE_NAMES + [
    # ── Cat 1: 지표 기울기/방향 (6개) ──
    "RSI_slope_5d",
    "RSI_slope_10d",
    "StochK_slope_5d",
    "MACD_Hist_accel",
    "MA20_slope_10d",
    "MA60_slope_20d",
    # ── Cat 2: 연속 패턴 (4개) ──
    "Volume_up_streak",
    "Close_up_streak",
    "Close_above_MA20_days",
    "MACD_Hist_pos_days",
    # ── Cat 3: 다중 시간대 수익률 (3개) ──
    "Ret_3d",
    "Ret_40d",
    "Ret_60d",
    # ── Cat 4: 변동성 체제 (3개) ──
    "Volatility_5d",
    "Volatility_10d",
    "Vol_regime_ratio",
    # ── Cat 5: 시차 피처 (5개) ──
    "RSI14_lag5",
    "RSI14_lag10",
    "StochK_lag5",
    "VolumeRatio_lag5",
    "Close_MA20_ratio_lag5",
    # ── Cat 6: 가격-거래량 패턴 (3개) ──
    "OBV_slope_10d",
    "PV_divergence",
    "Accum_phase_score",
    # ── Cat 7: 고급 패턴 (6개) ──
    "Higher_lows_count",
    "Range_breakout_prox",
    "MA_alignment",
    "Consolidation_days",
    "Volume_dry_up",
    "Price_acceleration",
]

PRECISE_FEATURE_NAMES_KR = {
    **FEATURE_NAMES_KR,
    "RSI_slope_5d": "RSI기울기(5d)",
    "RSI_slope_10d": "RSI기울기(10d)",
    "StochK_slope_5d": "StochK기울기",
    "MACD_Hist_accel": "MACD가속도",
    "MA20_slope_10d": "MA20기울기",
    "MA60_slope_20d": "MA60기울기",
    "Volume_up_streak": "거래량연속↑",
    "Close_up_streak": "종가연속↑",
    "Close_above_MA20_days": "MA20위일수",
    "MACD_Hist_pos_days": "MACDHist+일수",
    "Ret_3d": "3일수익률",
    "Ret_40d": "40일수익률",
    "Ret_60d": "60일수익률",
    "Volatility_5d": "5일변동성",
    "Volatility_10d": "10일변동성",
    "Vol_regime_ratio": "변동성비율(5/20)",
    "RSI14_lag5": "RSI(5일전)",
    "RSI14_lag10": "RSI(10일전)",
    "StochK_lag5": "StochK(5일전)",
    "VolumeRatio_lag5": "거래량비(5일전)",
    "Close_MA20_ratio_lag5": "종가/MA20(5일전)",
    "OBV_slope_10d": "OBV기울기",
    "PV_divergence": "가격거래량괴리",
    "Accum_phase_score": "매집점수",
    "Higher_lows_count": "저점상승횟수",
    "Range_breakout_prox": "레인지돌파근접",
    "MA_alignment": "이평정배열",
    "Consolidation_days": "횡보일수",
    "Volume_dry_up": "거래량건조",
    "Price_acceleration": "가격가속도",
}

MODEL_DIR = CACHE_DIR / "ml_models"
FEATURE_CACHE_DIR = CACHE_DIR / "feature_cache"

_ROLLING_CACHE_VERSION = 3


def _rolling_cache_path(horizon_days: int, target_pct: float, stop_pct: float) -> Path:
    _t = int(target_pct * 100)
    _s = int(stop_pct * 100)
    return FEATURE_CACHE_DIR / f"surge_rolling_h{horizon_days}_t{_t}_s{_s}.pkl"


def _load_rolling_cache(
    horizon_days: int, target_pct: float, stop_pct: float
) -> dict | None:
    """Rolling 피처 캐시 로드. 없거나 버전 불일치 시 None."""
    path = _rolling_cache_path(horizon_days, target_pct, stop_pct)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != _ROLLING_CACHE_VERSION:
            return None
        return data
    except Exception:
        return None


def _save_rolling_cache(
    tickers_data: dict,
    date_str: str,
    horizon_days: int,
    target_pct: float,
    stop_pct: float,
):
    """Rolling 피처 캐시 저장."""
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _rolling_cache_path(horizon_days, target_pct, stop_pct)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "tickers_data": tickers_data,
                "end_date": date_str,
                "version": _ROLLING_CACHE_VERSION,
            },
            f,
        )


# 기간별 급등주 모델 설정: mode → (라벨, lookback 일수)
SURGE_PERIODS = {
    # mode: (label, lookback_days, min_mcap, lgb_overrides)
    "surge": ("2Y — 전체", 365 * 2, 80_000_000_000, {}),
    "surge_1y": ("1Y — 최근 1년", 365, 80_000_000_000, {}),
    "surge_6m": (
        "6M — 최근 6개월",
        183,
        70_000_000_000,
        {"num_leaves": 47, "min_data_in_leaf": 40},
    ),
    "surge_3m": (
        "3M — 최근 3개월",
        91,
        50_000_000_000,
        {"num_leaves": 31, "min_data_in_leaf": 30},
    ),
}

# ETF 기간별 모델 설정: mode → (라벨, lookback 일수, min_avg_tv, lgb_overrides)
# 장기 모델일수록 안정적 유동성 요구 → 거래대금 차등
ETF_SURGE_PERIODS = {
    # mode: (label, lookback_days, min_avg_trading_value, lgb_overrides)
    "etf_surge": ("2Y — 전체", 365 * 2, 300_000_000, {}),  # 3억
    "etf_surge_1y": ("1Y — 최근 1년", 365, 200_000_000, {}),  # 2억
    "etf_surge_6m": (
        "6M — 최근 6개월",
        183,
        100_000_000,  # 1억
        {"num_leaves": 47, "min_data_in_leaf": 40},
    ),
    "etf_surge_3m": (
        "3M — 최근 3개월",
        91,
        50_000_000,  # 5천만
        {"num_leaves": 31, "min_data_in_leaf": 30},
    ),
}
_ETF_MIN_TV_ALL = min(v[2] for v in ETF_SURGE_PERIODS.values())  # 5천만

# v7.0 "고요한 폭풍 전야" 피처 (50개) — 모멘텀 누수 제거
V7_FEATURE_NAMES = [
    # ══ A. 기저 상태 (Base State) — 11개 ══
    "RSI14",
    "RSI_distance_from_30",
    "Price_to_20d_low",
    "Drawdown_from_60d",
    "Drawdown_from_250d",
    "BB_Width_20",
    "BB_Width_Ratio",
    "Range_Compression",
    "Consolidation_days",
    "ATR_ratio",
    "Vol_regime_ratio",
    # ══ B. 매집/수급 (Accumulation) — 9개 ══
    "OBV_slope_10d",
    "OBV_price_divergence",
    "Vol_Contraction",
    "Volume_dry_up",
    "Accum_Dist_Ratio",
    "VolumeRatio_20d",
    "PV_divergence",
    "Accum_phase_score",
    "Up_volume_ratio_10d",
    # ══ C. 구조 패턴 (Structure) — 8개 ══
    "Higher_lows_count",
    "Lower_highs_count",
    "Support_strength",
    "MA_Convergence",
    "Close_MA20_ratio",
    "Close_MA60_ratio",
    "MA20_slope_10d",
    "Price_Position",
    # ══ D. 기술 반전 신호 (Reversal) — 10개 ══
    "StochK",
    "StochD",
    "MACD_norm",
    "MACD_Hist_norm",
    "MACD_Hist_Slope",
    "MACD_Hist_accel",
    "RSI_slope_5d",
    "RSI_slope_10d",
    "StochK_slope_5d",
    "MACD_Cross_3d",
    # ══ E. 수익률/낙폭 (Returns) — 5개 ══
    "Ret_10d",
    "Ret_20d",
    "Ret_60d",
    "NearLow_rolling",
    "Volatility_20d",
    # ══ F. 시차 변화 (Temporal) — 7개 ══
    "RSI14_lag5",
    "RSI14_lag10",
    "StochK_lag5",
    "VolumeRatio_lag5",
    "BB_Width_change_10d",
    "Volatility_change_10d",
    "Close_MA20_ratio_lag5",
]

V7_FEATURE_NAMES_KR = {
    "RSI14": "RSI(14)",
    "RSI_distance_from_30": "RSI과매도거리",
    "Price_to_20d_low": "20일저가비",
    "Drawdown_from_60d": "60일고점하락",
    "Drawdown_from_250d": "52주고점하락",
    "BB_Width_20": "BB밴드폭",
    "BB_Width_Ratio": "BB폭변화율",
    "Range_Compression": "레인지압축",
    "Consolidation_days": "횡보일수",
    "ATR_ratio": "ATR/종가",
    "Vol_regime_ratio": "변동성비율(5/20)",
    "OBV_slope_10d": "OBV기울기",
    "OBV_price_divergence": "OBV매집괴리",
    "Vol_Contraction": "거래량위축",
    "Volume_dry_up": "거래량건조",
    "Accum_Dist_Ratio": "매집비율",
    "VolumeRatio_20d": "거래량비율",
    "PV_divergence": "가격거래량괴리",
    "Accum_phase_score": "매집점수",
    "Up_volume_ratio_10d": "상승일거래량비",
    "Higher_lows_count": "저점상승횟수",
    "Lower_highs_count": "고점하락횟수",
    "Support_strength": "지지선강도",
    "MA_Convergence": "이평밀집도",
    "Close_MA20_ratio": "종가/MA20",
    "Close_MA60_ratio": "종가/MA60",
    "MA20_slope_10d": "MA20기울기",
    "Price_Position": "가격위치",
    "StochK": "스토캐스틱K",
    "StochD": "스토캐스틱D",
    "MACD_norm": "MACD",
    "MACD_Hist_norm": "MACD히스토그램",
    "MACD_Hist_Slope": "MACD기울기",
    "MACD_Hist_accel": "MACD가속도",
    "RSI_slope_5d": "RSI기울기(5d)",
    "RSI_slope_10d": "RSI기울기(10d)",
    "StochK_slope_5d": "StochK기울기",
    "MACD_Cross_3d": "MACD골든크로스",
    "Ret_10d": "10일수익률",
    "Ret_20d": "20일수익률",
    "Ret_60d": "60일수익률",
    "NearLow_rolling": "저점대비상승",
    "Volatility_20d": "20일변동성",
    "RSI14_lag5": "RSI(5일전)",
    "RSI14_lag10": "RSI(10일전)",
    "StochK_lag5": "StochK(5일전)",
    "VolumeRatio_lag5": "거래량비(5일전)",
    "BB_Width_change_10d": "BB폭변화(10d)",
    "Volatility_change_10d": "변동성변화(10d)",
    "Close_MA20_ratio_lag5": "종가/MA20(5일전)",
}

# v8.0 "Base Breakout" 피처 (58개) — v7 50개 + 바닥형성/상대강도/거래량 8개
V8_FEATURE_NAMES = V7_FEATURE_NAMES + [
    # ══ G. 바닥 형성 패턴 (Base Formation) — 4개 ══
    "Base_duration",  # 60일 저가 ±10% 이내 횡보 연속일수
    "Base_tightness",  # 40일 종가 표준편차/평균 (타이트한 바닥)
    "Volume_trend_in_base",  # 바닥 구간 내 거래량 기울기 (매집 감지)
    "Breakout_proximity",  # 40일 범위 내 위치 (1 = 돌파 임박)
    # ══ H. 시장 대비 상대강도 — 2개 ══
    "RS_vs_market_20d",  # 20일 수익률 - KOSPI 20일 수익률
    "RS_vs_market_60d",  # 60일 수익률 - KOSPI 60일 수익률
    # ══ I. 거래량 인텔리전스 — 2개 ══
    "Large_volume_days_ratio",  # 20일 중 거래량>2배평균 일수 비율
    "Volume_pattern_score",  # OBV↑ + 거래량위축 + 상승일거래량↑ 복합 점수
]

V8_FEATURE_NAMES_KR = {
    **V7_FEATURE_NAMES_KR,
    "Base_duration": "바닥횡보기간",
    "Base_tightness": "바닥타이트니스",
    "Volume_trend_in_base": "바닥내거래량추세",
    "Breakout_proximity": "돌파임박도",
    "RS_vs_market_20d": "시장대비강도(20d)",
    "RS_vs_market_60d": "시장대비강도(60d)",
    "Large_volume_days_ratio": "대량거래일비율",
    "Volume_pattern_score": "거래량패턴점수",
}

# ── AI_급등주 v2: Triple-Barrier 55개 피처 ──
SURGE_FEATURE_NAMES = [
    # ══ A. 가격/추세 (Price & Trend) — 15개 ══
    "Ret_1d",  # 1일 수익률
    "Ret_5d",  # 5일 수익률
    "Ret_10d",  # 10일 수익률
    "Ret_20d",  # 20일 수익률
    "Ret_60d",  # 60일 수익률
    "MA_align_score",  # MA정배열 점수 (0~5: close>5>20>60>120>200)
    "Close_MA20_ratio",  # 종가/MA20 이격도
    "Close_MA60_ratio",  # 종가/MA60 이격도
    "Close_MA120_ratio",  # 종가/MA120 이격도
    "Pct_from_52w_high",  # 52주고점 대비 (-1~0)
    "Pct_from_52w_low",  # 52주저점 대비 (0~∞)
    "Trend_template_score",  # Minervini 추세 점수 (0~5)
    "MA20_slope_10d",  # MA20 10일 기울기
    "MA60_slope_20d",  # MA60 20일 기울기
    "Higher_lows_count",  # 30일간 저점 상승 횟수
    # ══ B. 거래량/유동성 (Volume & Liquidity) — 10개 ══
    "Vol_ratio_20d",  # 당일거래량/20일평균
    "Vol_contraction_20d",  # 최근10일/이전10일 거래량비 (수축=<1)
    "Up_down_vol_ratio",  # 상승일/하락일 거래량비 (20일)
    "Vol_trend_slope",  # 20일 거래량 선형회귀 기울기
    "OBV_slope_20d",  # OBV 20일 기울기 (매집)
    "OBV_price_diverge",  # OBV-가격 괴리 (매집 신호)
    "Vol_dry_up_5d",  # 최근5일거래량/60일평균 (건조=매집완료)
    "Large_vol_days",  # 20일 중 거래량≥2x평균 일수 비율
    "Avg_trade_val_log",  # log(20일 평균 거래대금)
    "Vol_surge_ratio",  # 당일거래량/5일평균 (단기 급증)
    # ══ C. 변동성/압축 (Volatility & Compression) — 8개 ══
    "BB_width_20",  # 볼린저밴드 폭 (절대)
    "BB_width_ratio_60d",  # 현재BB폭/60일평균BB폭 (수축도)
    "ATR_ratio",  # ATR(14)/종가
    "ATR_contraction",  # 최근ATR/60일평균ATR
    "Price_tight_10d",  # 10일 종가 range/mean
    "Range_compress_20d",  # 20일 고저범위/60일 고저범위
    "Volatility_20d",  # 20일 수익률 std
    "Vol_regime_ratio",  # 5일변동성/20일변동성 (확장 시작)
    # ══ D. 패턴/베이스 품질 (Pattern & Base) — 10개 ══
    "Base_length_days",  # ±15% 범위 내 횡보 연속일
    "Base_depth_pct",  # 베이스 깊이 (고점대비 최대하락)
    "VCP_contract_count",  # 변동성 수축 횟수 (4주간)
    "Pivot_proximity",  # 피벗(20일 고점) 대비 현재가 (0~1+)
    "Breakout_proximity",  # 40일 고점 대비 현재가 (0~1+)
    "Days_since_breakout",  # 마지막 20일고점 돌파 후 경과일
    "Consol_tightness",  # 30일 종가 std/mean
    "Support_touches",  # 20일 내 지지선 터치 횟수
    "Accum_bar_count",  # 매집봉 횟수 (거대거래량+가격복귀)
    "Price_pos_in_range",  # 40일 범위 내 위치 (0~1)
    # ══ E. 시장 대비 (Market Relative) — 5개 ══
    "RS_vs_kospi_20d",  # 20일 수익률 - KOSPI 20일 수익률
    "RS_vs_kospi_60d",  # 60일 상대강도
    "RS_rank_pct",  # 종목 상대강도 백분위
    "Market_trend_20d",  # KOSPI 20일 수익률 (시장 환경)
    "Market_vol_20d",  # KOSPI 20일 변동성
    # ══ F. 기술신호/모멘텀 (Technical Signals) — 7개 ══
    "RSI14",  # RSI(14)
    "MACD_hist_norm",  # MACD히스토그램/종가
    "MACD_cross_3d",  # MACD 골든크로스 (최근3일)
    "Stoch_cross_3d",  # 스토캐스틱 골든크로스 (최근3일)
    "RSI_slope_5d",  # RSI 5일 기울기
    "MACD_hist_slope",  # MACD히스토그램 5일 기울기
    "BB_position",  # BB내 가격위치 (0=하단, 1=상단)
]

SURGE_FEATURE_NAMES_KR = {
    "Ret_1d": "1일수익률",
    "Ret_5d": "5일수익률",
    "Ret_10d": "10일수익률",
    "Ret_20d": "20일수익률",
    "Ret_60d": "60일수익률",
    "MA_align_score": "MA정배열",
    "Close_MA20_ratio": "종가/MA20",
    "Close_MA60_ratio": "종가/MA60",
    "Close_MA120_ratio": "종가/MA120",
    "Pct_from_52w_high": "52주고점대비",
    "Pct_from_52w_low": "52주저점대비",
    "Trend_template_score": "추세점수",
    "MA20_slope_10d": "MA20기울기",
    "MA60_slope_20d": "MA60기울기",
    "Higher_lows_count": "저점상승횟수",
    "Vol_ratio_20d": "거래량비율",
    "Vol_contraction_20d": "거래량수축",
    "Up_down_vol_ratio": "상승/하락거래량비",
    "Vol_trend_slope": "거래량추세",
    "OBV_slope_20d": "OBV기울기",
    "OBV_price_diverge": "OBV매집괴리",
    "Vol_dry_up_5d": "거래량건조",
    "Large_vol_days": "대량거래일비율",
    "Avg_trade_val_log": "거래대금(log)",
    "Vol_surge_ratio": "거래량급증",
    "BB_width_20": "BB밴드폭",
    "BB_width_ratio_60d": "BB수축도",
    "ATR_ratio": "ATR/종가",
    "ATR_contraction": "ATR수축",
    "Price_tight_10d": "가격타이트",
    "Range_compress_20d": "레인지압축",
    "Volatility_20d": "20일변동성",
    "Vol_regime_ratio": "변동성비율",
    "Base_length_days": "베이스기간",
    "Base_depth_pct": "베이스깊이",
    "VCP_contract_count": "VCP수축횟수",
    "Pivot_proximity": "피벗근접",
    "Breakout_proximity": "돌파근접",
    "Days_since_breakout": "돌파후경과일",
    "Consol_tightness": "횡보타이트",
    "Support_touches": "지지선터치",
    "Accum_bar_count": "매집봉횟수",
    "Price_pos_in_range": "가격위치",
    "RS_vs_kospi_20d": "시장대비(20d)",
    "RS_vs_kospi_60d": "시장대비(60d)",
    "RS_rank_pct": "상대강도백분위",
    "Market_trend_20d": "시장추세",
    "Market_vol_20d": "시장변동성",
    "RSI14": "RSI(14)",
    "MACD_hist_norm": "MACD히스토그램",
    "MACD_cross_3d": "MACD골든크로스",
    "Stoch_cross_3d": "Stoch골든크로스",
    "RSI_slope_5d": "RSI기울기",
    "MACD_hist_slope": "MACD기울기",
    "BB_position": "BB가격위치",
}


# ── 매집주 피처 (43개 = 기본 30 + 매집 전용 13) ──
ACCUM_FEATURE_NAMES = FEATURE_NAMES + [
    # ══ A. 매집 강도 (Accumulation Intensity) — 5개 ══
    "OBV_slope_20d",  # OBV 20일 기울기
    "OBV_price_divergence",  # 주가 vs OBV 괴리
    "Up_down_vol_ratio_20d",  # 상승일/하락일 거래량 비율
    "Positive_candle_ratio",  # 최근 20일 양봉 비율
    "Accum_bar_count",  # 매집봉 출현 횟수
    # ══ B. 거래량 수축/건조 — 4개 ══
    "Vol_contraction_40d",  # 최근 20일 거래량 / 이전 20일 거래량
    "Vol_dry_up_5d",  # 최근 5일 평균 거래량 / 60일 평균
    "Vol_contraction_streak",  # 연속 거래량 감소 일수
    "BB_width_ratio",  # 현재 BB폭 / 60일 평균 BB폭
    # ══ C. 베이스 성숙도 — 2개 ══
    "Base_maturity_score",  # 횡보일수 × 거래량수축도 × BB수축도
    "MA_convergence_score",  # MA5/20/60 수렴도
    # ══ D. 시장 대비 — 2개 ══
    "RS_vs_market_20d",  # 20일 수익률 - KOSPI
    "RS_vs_market_60d",  # 60일 상대강도
]

ACCUM_FEATURE_NAMES_KR = {
    **FEATURE_NAMES_KR,
    "OBV_slope_20d": "OBV기울기(20d)",
    "OBV_price_divergence": "OBV-가격괴리",
    "Up_down_vol_ratio_20d": "상승/하락거래량비",
    "Positive_candle_ratio": "양봉밀집도",
    "Accum_bar_count": "매집봉횟수",
    "Vol_contraction_40d": "거래량수축(40d)",
    "Vol_dry_up_5d": "거래량건조(5d)",
    "Vol_contraction_streak": "거래량연속감소일",
    "BB_width_ratio": "BB수축비율",
    "Base_maturity_score": "베이스성숙도",
    "MA_convergence_score": "이평수렴도",
    "RS_vs_market_20d": "시장대비강도(20d)",
    "RS_vs_market_60d": "시장대비강도(60d)",
}


# ──────────────────────────────────────────────
# 1. 피처 엔지니어링
# ──────────────────────────────────────────────


def compute_ml_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame → 28개 피처 DataFrame 반환.

    기존 indicators 모듈의 calc_rsi/macd/stochastic/ma 재사용.
    52주 고저는 rolling(250) 사용 (look-ahead bias 방지).
    v3.2: 에너지 축적/돌파 직전 피처 8개 추가.
    """
    if ohlcv_df is None or len(ohlcv_df) < 60:
        return pd.DataFrame()

    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    low = ohlcv_df["저가"].astype(float)
    volume = ohlcv_df["거래량"].astype(float)
    idx = ohlcv_df.index

    # 기술지표 계산 (기존 모듈 재사용)
    rsi = indicators.calc_rsi(close, 14)
    macd_df = indicators.calc_macd(close, 12, 26, 9)
    stoch_df = indicators.calc_stochastic(high, low, close, 14, 3)
    ma20 = indicators.calc_ma(close, 20)
    ma60 = indicators.calc_ma(close, 60)

    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # rolling 52주 고저 (250거래일, 데이터 부족 시 전체)
    roll_win = min(250, len(close))
    rolling_high = close.rolling(roll_win, min_periods=60).max()
    rolling_low = close.rolling(roll_win, min_periods=60).min()

    # Boolean 신호 (3일 내 크로스 여부 → int)
    macd_line = macd_df["MACD"]
    macd_sig = macd_df["MACD_Signal"]
    stoch_k = stoch_df["StochK"]
    stoch_d = stoch_df["StochD"]

    def _cross_within(a, b, days=3):
        """a가 b를 상향 돌파한 적이 최근 days일 내 있으면 1."""
        crossed = (a > b) & (a.shift(1) <= b.shift(1))
        return crossed.rolling(days, min_periods=1).max().fillna(0).astype(int)

    def _rsi_exit_oversold(rsi_s, threshold=30, days=3):
        crossed = (rsi_s > threshold) & (rsi_s.shift(1) <= threshold)
        return crossed.rolling(days, min_periods=1).max().fillna(0).astype(int)

    def _ma_cross(close_s, ma_s, days=3):
        crossed = (close_s > ma_s) & (close_s.shift(1) <= ma_s.shift(1))
        return crossed.rolling(days, min_periods=1).max().fillna(0).astype(int)

    def _stoch_golden_low(k, d, k_max=30, days=3):
        crossed = (k > d) & (k.shift(1) <= d.shift(1)) & (k <= k_max)
        return crossed.rolling(days, min_periods=1).max().fillna(0).astype(int)

    vol_avg20 = volume.rolling(20).mean().replace(0, np.nan)

    features = pd.DataFrame(
        {
            "RSI14": rsi,
            "MACD_norm": macd_line / close,
            "MACD_Hist_norm": macd_df["MACD_Hist"] / close,
            "StochK": stoch_k,
            "StochD": stoch_d,
            "Close_MA20_ratio": close / ma20,
            "Close_MA60_ratio": close / ma60,
            "Ret_1d": close.pct_change(1),
            "Ret_5d": close.pct_change(5),
            "Ret_10d": close.pct_change(10),
            "Ret_20d": close.pct_change(20),
            "Volatility_20d": close.pct_change().rolling(20).std(),
            "VolumeRatio_20d": volume / vol_avg20,
            "Drawdown_rolling": (rolling_high - close) / rolling_high,
            "NearLow_rolling": (close - rolling_low) / rolling_low.replace(0, np.nan),
            "ATR_ratio": atr14 / close,
            "MACD_Cross_3d": _cross_within(macd_line, macd_sig, 3),
            "RSI_Oversold_Exit_3d": _rsi_exit_oversold(rsi, 30, 3),
            "Stoch_Golden_3d": _stoch_golden_low(stoch_k, stoch_d, 30, 3),
            "MA20_Cross_3d": _ma_cross(close, ma20, 3),
        },
        index=idx,
    )

    # ── v3.2 신규 피처: 에너지 축적 / 돌파 직전 ──

    # 21. 볼린저밴드 폭 (좁을수록 수렴 → 돌파 에너지 축적)
    bb_std_20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std_20
    bb_lower = ma20 - 2 * bb_std_20
    features["BB_Width_20"] = (bb_upper - bb_lower) / ma20

    # 22. 밴드 폭 변화율 (<1 = 최근 수렴 가속)
    ma5 = indicators.calc_ma(close, 5)
    bb_width_5 = (close.rolling(5).std() * 4) / ma5.replace(0, np.nan)
    bb_width_20 = features["BB_Width_20"]
    features["BB_Width_Ratio"] = bb_width_5 / bb_width_20.replace(0, np.nan)

    # 23. 이평선 밀집도 (0에 가까울수록 밀집 → 돌파 임박)
    features["MA_Convergence"] = (ma5 - ma20).abs() / ma20

    # 24. 거래량 위축 비율 (<0.7 = 거래 건조 → 돌파 전 침묵)
    vol_5 = volume.rolling(5).mean()
    vol_20 = volume.rolling(20).mean().replace(0, np.nan)
    features["Vol_Contraction"] = vol_5 / vol_20

    # 25. 20일 범위 내 가격 위치 (0=바닥, 1=천장, 0.3~0.5=반등 초입)
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    price_range_20 = (high_20 - low_20).replace(0, np.nan)
    features["Price_Position"] = (close - low_20) / price_range_20

    # 26. 가격 레인지 압축 (좁을수록 횡보 수렴)
    features["Range_Compression"] = price_range_20 / close

    # 27. MACD 히스토그램 3일 기울기 (음→양 전환 = 반전 시작)
    macd_hist = macd_df["MACD_Hist"]
    features["MACD_Hist_Slope"] = (macd_hist - macd_hist.shift(3)) / close

    # 28. 매집 비율 (10일) (>1 = 가격은 안 올랐는데 매수세 유입)
    daily_ret_raw = close.pct_change()
    up_vol = volume.where(daily_ret_raw > 0, 0).rolling(10).sum()
    down_vol = volume.where(daily_ret_raw < 0, 0).rolling(10).sum().replace(0, np.nan)
    features["Accum_Dist_Ratio"] = up_vol / down_vol

    # ── v5 신규 피처: 급등 이력 감지 ──

    # 29. 60일 고점 대비 하락률 (높을수록 바닥 → 매수 기회, 낮으면 고점 근처)
    high_60d = close.rolling(60, min_periods=20).max()
    features["Drawdown_60d"] = (high_60d - close) / high_60d

    # 30. 60일 내 최대 20일 상승률 (높으면 최근 급등 이력 → post-surge 위험)
    ret_20d_hist = close.pct_change(20)
    features["Prior_Surge_60d"] = ret_20d_hist.rolling(60, min_periods=1).max()

    return features


# ──────────────────────────────────────────────
# 1-b. 정밀 피처 엔지니어링 (30개 시계열 추가)
# ──────────────────────────────────────────────


def _rolling_slope(s, window):
    """rolling 선형회귀 기울기 (정규화)."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    if x_var == 0:
        return pd.Series(np.nan, index=s.index)

    def _slope(vals):
        if len(vals) < window:
            return np.nan
        return ((x - x_mean) * (vals - vals.mean())).sum() / x_var

    return s.rolling(window, min_periods=window).apply(_slope, raw=True)


def _streak(condition_series):
    """True 연속 일수 계산."""
    groups = (condition_series != condition_series.shift()).cumsum()
    streaks = condition_series.groupby(groups).cumsum()
    return streaks


def compute_ml_features_precise(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """v6 호환 래퍼 — v7.0으로 리다이렉트."""
    return compute_ml_features_v7(ohlcv_df)


def compute_ml_features_v7(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """v7.0: 50개 '고요한 폭풍 전야' 피처 (모멘텀 누수 완전 제거).

    모멘텀 인코딩 피처(Prior_Surge_60d, Close_up_streak 등) 제거.
    기저 상태·매집·구조·반전 신호·낙폭·시차 변화 피처로 재구성.
    """
    if ohlcv_df is None or len(ohlcv_df) < 60:
        return pd.DataFrame()

    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    low = ohlcv_df["저가"].astype(float)
    volume = ohlcv_df["거래량"].astype(float)
    idx = ohlcv_df.index

    # ── 기술지표 계산 ──
    rsi = indicators.calc_rsi(close, 14)
    macd_df = indicators.calc_macd(close, 12, 26, 9)
    stoch_df = indicators.calc_stochastic(high, low, close, 14, 3)
    ma5 = indicators.calc_ma(close, 5)
    ma20 = indicators.calc_ma(close, 20)
    ma60 = indicators.calc_ma(close, 60)

    macd_line = macd_df["MACD"]
    macd_sig = macd_df["MACD_Signal"]
    macd_hist = macd_df["MACD_Hist"]
    stoch_k = stoch_df["StochK"]
    stoch_d = stoch_df["StochD"]

    daily_ret = close.pct_change(1)
    vol_avg20 = volume.rolling(20).mean().replace(0, np.nan)
    vol_avg60 = volume.rolling(60, min_periods=20).mean().replace(0, np.nan)

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # Bollinger Band
    bb_std_20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std_20
    bb_lower = ma20 - 2 * bb_std_20
    bb_width_20 = (bb_upper - bb_lower) / ma20.replace(0, np.nan)

    bb_std_5 = close.rolling(5).std()
    bb_width_5 = (bb_std_5 * 4) / ma5.replace(0, np.nan)

    # Rolling highs/lows
    roll_win = min(250, len(close))
    rolling_high = close.rolling(roll_win, min_periods=60).max()
    rolling_low = close.rolling(roll_win, min_periods=60).min()
    high_60d = close.rolling(60, min_periods=20).max()
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    range_20 = (high_20 - low_20).replace(0, np.nan)

    # OBV
    sign = np.sign(close.pct_change())
    obv = (volume * sign).cumsum()

    # MACD cross helper
    def _cross_within(a, b, days=3):
        crossed = (a > b) & (a.shift(1) <= b.shift(1))
        return crossed.rolling(days, min_periods=1).max().fillna(0).astype(int)

    features = pd.DataFrame(index=idx)

    # ══ A. 기저 상태 (Base State) — 11개 ══
    features["RSI14"] = rsi
    features["RSI_distance_from_30"] = rsi - 30
    features["Price_to_20d_low"] = close / low_20.replace(0, np.nan)
    features["Drawdown_from_60d"] = (high_60d - close) / high_60d.replace(0, np.nan)
    features["Drawdown_from_250d"] = (rolling_high - close) / rolling_high.replace(
        0, np.nan
    )
    features["BB_Width_20"] = bb_width_20
    features["BB_Width_Ratio"] = bb_width_5 / bb_width_20.replace(0, np.nan)
    features["Range_Compression"] = range_20 / close
    is_consolidating = (close.rolling(5).std() / close) < 0.03
    features["Consolidation_days"] = _streak(is_consolidating)
    features["ATR_ratio"] = atr14 / close
    vol_20d_std = daily_ret.rolling(20).std().replace(0, np.nan)
    features["Vol_regime_ratio"] = daily_ret.rolling(5).std() / vol_20d_std

    # ══ B. 매집/수급 (Accumulation) — 9개 ══
    features["OBV_slope_10d"] = _rolling_slope(obv, 10)
    # OBV-가격 괴리: OBV 상승 but 가격 횡보 → 매집
    obv_norm = obv / obv.rolling(20).std().replace(0, np.nan)
    obv_change_10d = obv_norm - obv_norm.shift(10)
    price_change_10d = close.pct_change(10)
    features["OBV_price_divergence"] = obv_change_10d - price_change_10d * 10

    vol_5 = volume.rolling(5).mean()
    vol_20 = volume.rolling(20).mean().replace(0, np.nan)
    features["Vol_Contraction"] = vol_5 / vol_20
    features["Volume_dry_up"] = 1 - (vol_5 / vol_avg60).clip(0, 2) / 2
    up_vol = volume.where(daily_ret > 0, 0).rolling(10).sum()
    down_vol = volume.where(daily_ret < 0, 0).rolling(10).sum().replace(0, np.nan)
    features["Accum_Dist_Ratio"] = up_vol / down_vol
    features["VolumeRatio_20d"] = volume / vol_avg20
    price_down = (close.pct_change(10) < -0.02).astype(float)
    vol_up = (vol_5 > vol_20).astype(float)
    features["PV_divergence"] = price_down * vol_up
    price_range_10 = (high.rolling(10).max() - low.rolling(10).min()) / close
    vol_increase = vol_5 / vol_avg20
    features["Accum_phase_score"] = (
        1 - price_range_10.clip(0, 0.10) / 0.10
    ) * vol_increase.fillna(0)
    # 상승일 거래량 비율
    up_day = (close > close.shift(1)).astype(float)
    up_day_vol = (volume * up_day).rolling(10).sum()
    total_vol_10 = volume.rolling(10).sum().replace(0, np.nan)
    features["Up_volume_ratio_10d"] = up_day_vol / total_vol_10

    # ══ C. 구조 패턴 (Structure) — 8개 ══
    rolling_low_5 = low.rolling(5).min()
    higher_low_signals = (rolling_low_5 > rolling_low_5.shift(5)).astype(float)
    features["Higher_lows_count"] = higher_low_signals.rolling(20, min_periods=5).sum()
    rolling_high_5 = high.rolling(5).max()
    lower_high_signals = (rolling_high_5 < rolling_high_5.shift(5)).astype(float)
    features["Lower_highs_count"] = lower_high_signals.rolling(20, min_periods=5).sum()
    # 지지선 강도: 60일 저점 근처(±2%)에서 반등 횟수
    low_zone = close.rolling(60, min_periods=20).min() * 1.02
    bounce_signals = ((low <= low_zone) & (close > low_zone)).astype(float)
    features["Support_strength"] = bounce_signals.rolling(60, min_periods=20).sum()
    features["MA_Convergence"] = (ma5 - ma20).abs() / ma20.replace(0, np.nan)
    features["Close_MA20_ratio"] = close / ma20
    features["Close_MA60_ratio"] = close / ma60
    features["MA20_slope_10d"] = _rolling_slope(ma20, 10)
    features["Price_Position"] = (close - low_20) / range_20

    # ══ D. 기술 반전 신호 (Reversal) — 10개 ══
    features["StochK"] = stoch_k
    features["StochD"] = stoch_d
    features["MACD_norm"] = macd_line / close
    features["MACD_Hist_norm"] = macd_hist / close
    features["MACD_Hist_Slope"] = (macd_hist - macd_hist.shift(3)) / close
    macd_hist_slope = macd_hist - macd_hist.shift(1)
    features["MACD_Hist_accel"] = macd_hist_slope - macd_hist_slope.shift(1)
    features["RSI_slope_5d"] = _rolling_slope(rsi, 5)
    features["RSI_slope_10d"] = _rolling_slope(rsi, 10)
    features["StochK_slope_5d"] = _rolling_slope(stoch_k, 5)
    features["MACD_Cross_3d"] = _cross_within(macd_line, macd_sig, 3)

    # ══ E. 수익률/낙폭 (Returns) — 5개 ══
    features["Ret_10d"] = close.pct_change(10)
    features["Ret_20d"] = close.pct_change(20)
    features["Ret_60d"] = close.pct_change(60)
    features["NearLow_rolling"] = (close - rolling_low) / rolling_low.replace(0, np.nan)
    features["Volatility_20d"] = daily_ret.rolling(20).std()

    # ══ F. 시차 변화 (Temporal) — 7개 ══
    features["RSI14_lag5"] = rsi.shift(5)
    features["RSI14_lag10"] = rsi.shift(10)
    features["StochK_lag5"] = stoch_k.shift(5)
    features["VolumeRatio_lag5"] = (volume / vol_avg20).shift(5)
    features["BB_Width_change_10d"] = bb_width_20 - bb_width_20.shift(10)
    features["Volatility_change_10d"] = daily_ret.rolling(20).std() - daily_ret.rolling(
        20
    ).std().shift(10)
    features["Close_MA20_ratio_lag5"] = (close / ma20).shift(5)

    return features


def compute_ml_features_v8(
    ohlcv_df: pd.DataFrame, kospi_close: pd.Series = None
) -> pd.DataFrame:
    """v8.0: 58개 'Base Breakout' 피처 (v7 50개 + 바닥형성/상대강도/거래량 8개).

    Args:
        ohlcv_df: 종목 OHLCV DataFrame
        kospi_close: KOSPI 지수 종가 Series (없으면 RS 피처 0으로 채움)
    """
    feats = compute_ml_features_v7(ohlcv_df)
    if feats.empty:
        return feats

    close = ohlcv_df["종가"].astype(float)
    volume = ohlcv_df["거래량"].astype(float)
    vol_avg20 = volume.rolling(20).mean().replace(0, np.nan)

    # OBV (재계산)
    sign = np.sign(close.pct_change())
    obv = (volume * sign).cumsum()

    # ══ G. 바닥 형성 패턴 (Base Formation) — 4개 ══
    low_60d = close.rolling(60, min_periods=20).min()
    is_in_base = (close / low_60d.replace(0, np.nan)) < 1.10
    feats["Base_duration"] = _streak(is_in_base)

    close_std_40 = close.rolling(40, min_periods=10).std()
    close_mean_40 = close.rolling(40, min_periods=10).mean().replace(0, np.nan)
    feats["Base_tightness"] = close_std_40 / close_mean_40

    vol_slope_20 = _rolling_slope(volume, 20)
    in_base = feats["Base_duration"] > 10
    feats["Volume_trend_in_base"] = np.where(in_base, vol_slope_20, 0)

    high_40 = close.rolling(40, min_periods=10).max()
    low_40 = close.rolling(40, min_periods=10).min()
    range_40 = (high_40 - low_40).replace(0, np.nan)
    feats["Breakout_proximity"] = (close - low_40) / range_40

    # ══ H. 시장 대비 상대강도 — 2개 ══
    stock_ret_20d = close.pct_change(20)
    stock_ret_60d = close.pct_change(60)
    if kospi_close is not None and len(kospi_close) > 0:
        # 인덱스 정렬
        kospi_aligned = kospi_close.reindex(close.index, method="ffill")
        kospi_ret_20d = kospi_aligned.pct_change(20)
        kospi_ret_60d = kospi_aligned.pct_change(60)
        feats["RS_vs_market_20d"] = stock_ret_20d - kospi_ret_20d
        feats["RS_vs_market_60d"] = stock_ret_60d - kospi_ret_60d
    else:
        feats["RS_vs_market_20d"] = 0.0
        feats["RS_vs_market_60d"] = 0.0

    # ══ I. 거래량 인텔리전스 — 2개 ══
    large_vol = (volume > vol_avg20 * 2).astype(float)
    feats["Large_volume_days_ratio"] = large_vol.rolling(20, min_periods=5).mean()

    # Volume_pattern_score: OBV 기울기 + 거래량 위축도 + 상승일 거래량 비율 복합
    obv_slope_norm = _rolling_slope(obv, 10)
    obv_slope_rank = obv_slope_norm.rank(pct=True)
    vol_contraction = feats.get("Vol_Contraction", pd.Series(1.0, index=close.index))
    up_vol_ratio = feats.get("Up_volume_ratio_10d", pd.Series(0.5, index=close.index))
    feats["Volume_pattern_score"] = (
        obv_slope_rank * 0.4
        + (1 - vol_contraction.clip(0, 2) / 2) * 0.3
        + up_vol_ratio * 0.3
    )

    return feats


def compute_ml_features_surge(
    ohlcv_df: pd.DataFrame, kospi_close: pd.Series = None
) -> pd.DataFrame:
    """AI_급등주 v2: 55개 피처 계산 (6개 카테고리).

    모든 피처는 해당 날짜 이전 데이터만 사용 (미래 누수 없음).
    """
    if ohlcv_df is None or len(ohlcv_df) < 60:
        return pd.DataFrame()

    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    low = ohlcv_df["저가"].astype(float)
    volume = ohlcv_df["거래량"].astype(float)
    idx = close.index

    daily_ret = close.pct_change(1)
    vol_avg20 = volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    vol_avg60 = volume.rolling(60, min_periods=20).mean().replace(0, np.nan)

    # 이동평균
    ma5 = close.rolling(5, min_periods=3).mean()
    ma20 = close.rolling(20, min_periods=10).mean()
    ma60 = close.rolling(60, min_periods=20).mean()
    ma120 = close.rolling(120, min_periods=40).mean()
    ma200 = close.rolling(200, min_periods=80).mean()

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=7).mean()

    # RSI
    rsi = indicators.calc_rsi(close, 14)

    # MACD
    macd_df = indicators.calc_macd(close, 12, 26, 9)
    macd_hist = macd_df["MACD_Hist"]
    macd_line = macd_df["MACD"]
    macd_sig = macd_df["MACD_Signal"]

    # Stochastic
    stoch_df = indicators.calc_stochastic(high, low, close, 14, 3)
    stoch_k = stoch_df["StochK"]
    stoch_d = stoch_df["StochD"]

    # BB
    bb_std = close.rolling(20, min_periods=10).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / ma20.replace(0, np.nan)

    # 52주 고저
    high_52w = close.rolling(250, min_periods=60).max()
    low_52w = close.rolling(250, min_periods=60).min()

    # OBV
    sign = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (sign * volume).cumsum()

    feats = pd.DataFrame(index=idx)

    # ══ A. 가격/추세 (15개) ══
    feats["Ret_1d"] = daily_ret
    feats["Ret_5d"] = close.pct_change(5)
    feats["Ret_10d"] = close.pct_change(10)
    feats["Ret_20d"] = close.pct_change(20)
    feats["Ret_60d"] = close.pct_change(60)

    # MA정배열 점수 (0~5)
    ma_score = pd.Series(0.0, index=idx)
    ma_score += (close > ma5).astype(float)
    ma_score += (ma5 > ma20).astype(float)
    ma_score += (ma20 > ma60).astype(float)
    ma_score += (ma60 > ma120).astype(float)
    ma_score += (ma120 > ma200).astype(float)
    feats["MA_align_score"] = ma_score

    feats["Close_MA20_ratio"] = close / ma20.replace(0, np.nan)
    feats["Close_MA60_ratio"] = close / ma60.replace(0, np.nan)
    feats["Close_MA120_ratio"] = close / ma120.replace(0, np.nan)
    feats["Pct_from_52w_high"] = close / high_52w.replace(0, np.nan) - 1
    feats["Pct_from_52w_low"] = close / low_52w.replace(0, np.nan) - 1

    # Minervini Trend Template 점수 (0~5)
    tt_score = pd.Series(0.0, index=idx)
    ma150 = close.rolling(150, min_periods=50).mean()
    tt_score += (close > ma150).astype(float)
    tt_score += (close > ma200).astype(float)
    tt_score += (ma150 > ma200).astype(float)
    tt_score += (close > high_52w * 0.75).astype(float)
    tt_score += (close > low_52w * 1.25).astype(float)
    feats["Trend_template_score"] = tt_score

    feats["MA20_slope_10d"] = _rolling_slope(ma20, 10)
    feats["MA60_slope_20d"] = _rolling_slope(ma60, 20)

    # Higher_lows_count (30일간)
    low_5d = low.rolling(5, min_periods=3).min()
    hl_count = pd.Series(0.0, index=idx)
    for shift_val in [5, 10, 15, 20, 25]:
        prev_low = low_5d.shift(shift_val)
        curr_low = low_5d.shift(shift_val - 5) if shift_val > 5 else low_5d
        hl_count += (curr_low > prev_low).astype(float)
    feats["Higher_lows_count"] = hl_count

    # ══ B. 거래량/유동성 (10개) ══
    feats["Vol_ratio_20d"] = volume / vol_avg20
    vol_recent_10 = volume.rolling(10, min_periods=5).mean()
    vol_prev_10 = volume.shift(10).rolling(10, min_periods=5).mean().replace(0, np.nan)
    feats["Vol_contraction_20d"] = vol_recent_10 / vol_prev_10

    up_mask = (daily_ret > 0).astype(float)
    dn_mask = (daily_ret < 0).astype(float)
    up_vol = (volume * up_mask).rolling(20, min_periods=5).sum()
    dn_vol = (volume * dn_mask).rolling(20, min_periods=5).sum().replace(0, np.nan)
    feats["Up_down_vol_ratio"] = up_vol / dn_vol

    feats["Vol_trend_slope"] = _rolling_slope(volume, 20)

    # OBV 기울기 & 가격 괴리
    feats["OBV_slope_20d"] = _rolling_slope(obv, 20)
    price_ret_20 = close.pct_change(20)
    obv_norm = obv / obv.rolling(60, min_periods=20).std().replace(0, 1)
    obv_ret_20 = obv_norm.diff(20)
    feats["OBV_price_diverge"] = obv_ret_20 - price_ret_20 * 10

    vol_5 = volume.rolling(5, min_periods=3).mean()
    feats["Vol_dry_up_5d"] = vol_5 / vol_avg60

    large_vol = (volume >= vol_avg20 * 2).astype(float)
    feats["Large_vol_days"] = large_vol.rolling(20, min_periods=5).mean()

    # 거래대금 (log)
    if "거래대금" in ohlcv_df.columns:
        trade_val = ohlcv_df["거래대금"].astype(float)
    else:
        trade_val = close * volume
    avg_tv_20 = trade_val.rolling(20, min_periods=5).mean()
    feats["Avg_trade_val_log"] = np.log1p(avg_tv_20)

    vol_avg5 = volume.rolling(5, min_periods=3).mean().replace(0, np.nan)
    feats["Vol_surge_ratio"] = volume / vol_avg5

    # ══ C. 변동성/압축 (8개) ══
    feats["BB_width_20"] = bb_width
    bb_w_avg60 = bb_width.rolling(60, min_periods=20).mean().replace(0, np.nan)
    feats["BB_width_ratio_60d"] = bb_width / bb_w_avg60

    feats["ATR_ratio"] = atr14 / close.replace(0, np.nan)
    atr_avg60 = atr14.rolling(60, min_periods=20).mean().replace(0, np.nan)
    feats["ATR_contraction"] = atr14 / atr_avg60

    close_range_10 = (
        close.rolling(10, min_periods=5).max() - close.rolling(10, min_periods=5).min()
    )
    close_mean_10 = close.rolling(10, min_periods=5).mean().replace(0, np.nan)
    feats["Price_tight_10d"] = close_range_10 / close_mean_10

    range_20 = (
        high.rolling(20, min_periods=10).max() - low.rolling(20, min_periods=10).min()
    )
    range_60 = (
        high.rolling(60, min_periods=20).max() - low.rolling(60, min_periods=20).min()
    )
    feats["Range_compress_20d"] = range_20 / range_60.replace(0, np.nan)

    feats["Volatility_20d"] = daily_ret.rolling(20, min_periods=10).std()
    vol_5d = daily_ret.rolling(5, min_periods=3).std()
    vol_20d = daily_ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    feats["Vol_regime_ratio"] = vol_5d / vol_20d

    # ══ D. 패턴/베이스 품질 (10개) ══
    recent_high = close.rolling(60, min_periods=20).max()
    in_base = (close / recent_high.replace(0, np.nan)) > 0.85
    feats["Base_length_days"] = _streak(in_base)
    feats["Base_depth_pct"] = close / recent_high.replace(0, np.nan) - 1

    # VCP 수축 횟수
    atr5 = (high - low).rolling(5, min_periods=3).mean()
    vcp_cnt = pd.Series(0.0, index=idx)
    for sv in [5, 10, 15]:
        prev_atr = atr5.shift(sv)
        curr_atr = atr5.shift(sv - 5) if sv > 5 else atr5
        vcp_cnt += (curr_atr < prev_atr).astype(float)
    feats["VCP_contract_count"] = vcp_cnt

    # 피벗/돌파 근접도
    pivot_20 = close.shift(1).rolling(20, min_periods=10).max()
    feats["Pivot_proximity"] = close / pivot_20.replace(0, np.nan)

    pivot_40 = close.shift(1).rolling(40, min_periods=20).max()
    feats["Breakout_proximity"] = close / pivot_40.replace(0, np.nan)

    # Days_since_breakout
    broke_out = close > pivot_20
    dsb = pd.Series(np.nan, index=idx)
    last_bo = -9999
    for i in range(len(close)):
        if broke_out.iloc[i]:
            last_bo = i
        dsb.iloc[i] = i - last_bo if last_bo >= 0 else 999
    feats["Days_since_breakout"] = dsb.clip(upper=250)

    # 횡보 타이트니스
    feats["Consol_tightness"] = close.rolling(30, min_periods=10).std() / close.rolling(
        30, min_periods=10
    ).mean().replace(0, np.nan)

    # 지지선 터치 (20일 저가 근처 ±2% 내 도달 횟수)
    low_20 = close.rolling(20, min_periods=10).min()
    near_support = ((close / low_20.replace(0, np.nan) - 1) < 0.02).astype(float)
    feats["Support_touches"] = near_support.rolling(20, min_periods=5).sum()

    # 매집봉 횟수 (60일 내: 거래량 2배+ AND 이후 5일 내 ±5% 복귀)
    n = len(close)
    accum_cnt = pd.Series(0.0, index=idx)
    for i in range(65, n):
        cnt = 0
        for j in range(max(0, i - 60), i):
            if vol_avg20.iloc[j] > 0 and volume.iloc[j] >= vol_avg20.iloc[j] * 2.0:
                for k in range(j + 1, min(j + 6, i + 1)):
                    if abs(close.iloc[k] / close.iloc[j] - 1) < 0.05:
                        cnt += 1
                        break
        accum_cnt.iloc[i] = cnt
    feats["Accum_bar_count"] = accum_cnt

    # 40일 범위 내 위치
    high_40 = close.rolling(40, min_periods=20).max()
    low_40 = close.rolling(40, min_periods=20).min()
    range_40 = (high_40 - low_40).replace(0, np.nan)
    feats["Price_pos_in_range"] = (close - low_40) / range_40

    # ══ E. 시장 대비 (5개) ══
    stock_ret_20 = close.pct_change(20)
    stock_ret_60 = close.pct_change(60)
    if kospi_close is not None and len(kospi_close) > 20:
        k_aligned = kospi_close.reindex(idx, method="ffill")
        k_ret_20 = k_aligned.pct_change(20)
        k_ret_60 = k_aligned.pct_change(60)
        feats["RS_vs_kospi_20d"] = stock_ret_20 - k_ret_20
        feats["RS_vs_kospi_60d"] = stock_ret_60 - k_ret_60
        feats["Market_trend_20d"] = k_ret_20
        feats["Market_vol_20d"] = (
            k_aligned.pct_change().rolling(20, min_periods=10).std()
        )
    else:
        feats["RS_vs_kospi_20d"] = 0.0
        feats["RS_vs_kospi_60d"] = 0.0
        feats["Market_trend_20d"] = 0.0
        feats["Market_vol_20d"] = 0.0
    feats["RS_rank_pct"] = 0.0  # 학습 시 전체 종목 대비 계산, 예측 시 0

    # ══ F. 기술신호/모멘텀 (7개) ══
    feats["RSI14"] = rsi
    feats["MACD_hist_norm"] = macd_hist / close.replace(0, np.nan)

    def _cross_within(a, b, days=3):
        crossed = (a > b) & (a.shift(1) <= b.shift(1))
        return crossed.rolling(days, min_periods=1).max().fillna(0).astype(int)

    feats["MACD_cross_3d"] = _cross_within(macd_line, macd_sig, 3)

    def _stoch_golden(k, d, k_max=30, days=3):
        crossed = (k > d) & (k.shift(1) <= d.shift(1)) & (k <= k_max)
        return crossed.rolling(days, min_periods=1).max().fillna(0).astype(int)

    feats["Stoch_cross_3d"] = _stoch_golden(stoch_k, stoch_d, 30, 3)
    feats["RSI_slope_5d"] = rsi - rsi.shift(5)
    feats["MACD_hist_slope"] = (macd_hist - macd_hist.shift(5)) / close.replace(
        0, np.nan
    )

    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    feats["BB_position"] = (close - bb_lower) / bb_range

    return feats


def compute_ml_features_accum(
    ohlcv_df: pd.DataFrame, kospi_close: pd.Series = None
) -> pd.DataFrame:
    """AI_매집주: 기본 30 + 매집 전용 13 = 43개 피처 계산."""
    base = compute_ml_features(ohlcv_df)
    if base.empty:
        return base

    close = ohlcv_df["종가"].astype(float)
    volume = ohlcv_df["거래량"].astype(float)
    op = ohlcv_df["시가"].astype(float) if "시가" in ohlcv_df.columns else close
    n = len(close)
    feats = base.copy()

    # ── OBV 계산 ──
    sign = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (sign * volume).cumsum()

    # A. 매집 강도
    # OBV 기울기 (20일)
    def _slope(s, w):
        x = np.arange(w, dtype=float)
        x -= x.mean()
        denom = (x * x).sum()
        if denom == 0:
            return s * 0
        return s.rolling(w, min_periods=w).apply(
            lambda y: np.dot(y - y.mean(), x) / denom, raw=True
        )

    feats["OBV_slope_20d"] = _slope(obv, 20)

    # OBV vs 가격 괴리
    price_ret_20 = close.pct_change(20)
    obv_norm = obv / obv.rolling(60, min_periods=20).std().replace(0, 1)
    obv_ret_20 = obv_norm.diff(20)
    feats["OBV_price_divergence"] = obv_ret_20 - price_ret_20 * 10  # 스케일 보정

    # 상승/하락일 거래량 비율
    up_vol = volume.where(close > close.shift(1), 0)
    dn_vol = volume.where(close < close.shift(1), 0)
    up_sum = up_vol.rolling(20, min_periods=5).sum()
    dn_sum = dn_vol.rolling(20, min_periods=5).sum().replace(0, 1)
    feats["Up_down_vol_ratio_20d"] = up_sum / dn_sum

    # 양봉 밀집도
    pos_candle = (close >= op).astype(float)
    feats["Positive_candle_ratio"] = pos_candle.rolling(20, min_periods=10).mean()

    # 매집봉 횟수 (60일 내: 거래량 2배+ AND 이후 5일 내 가격 복귀 ±5%)
    vol_ma20 = volume.rolling(20, min_periods=10).mean()
    accum_count = pd.Series(0.0, index=close.index)
    for i in range(65, n):
        cnt = 0
        for j in range(max(0, i - 60), i):
            if vol_ma20.iloc[j] > 0 and volume.iloc[j] >= vol_ma20.iloc[j] * 2.0:
                # 이후 5일 내 가격 복귀 확인
                for k in range(j + 1, min(j + 6, i + 1)):
                    if abs(close.iloc[k] / close.iloc[j] - 1) < 0.05:
                        cnt += 1
                        break
        accum_count.iloc[i] = cnt
    feats["Accum_bar_count"] = accum_count

    # B. 거래량 수축/건조
    vol_20_recent = volume.rolling(20, min_periods=10).mean()
    vol_20_prev = volume.shift(20).rolling(20, min_periods=10).mean().replace(0, 1)
    feats["Vol_contraction_40d"] = vol_20_recent / vol_20_prev

    vol_5 = volume.rolling(5, min_periods=3).mean()
    vol_60 = volume.rolling(60, min_periods=20).mean().replace(0, 1)
    feats["Vol_dry_up_5d"] = vol_5 / vol_60

    # 연속 거래량 감소 일수
    vol_dec = (volume < volume.shift(1)).astype(float)
    streak = pd.Series(0.0, index=close.index)
    for i in range(1, n):
        if vol_dec.iloc[i] == 1:
            streak.iloc[i] = streak.iloc[i - 1] + 1
        else:
            streak.iloc[i] = 0
    feats["Vol_contraction_streak"] = streak

    # BB 수축 비율
    bb_ma = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_w = (2 * bb_std / bb_ma.replace(0, 1)).fillna(0)
    bb_w_avg60 = bb_w.rolling(60, min_periods=20).mean().replace(0, 1)
    feats["BB_width_ratio"] = bb_w / bb_w_avg60

    # C. 베이스 성숙도
    consol = feats.get("Consolidation_days", pd.Series(0, index=close.index))
    if "Consolidation_days" not in feats.columns:
        # 간이 횡보일수: 고점 대비 ±10% 내 연속일
        hi_60 = close.rolling(60, min_periods=20).max()
        in_range = (close / hi_60 > 0.90).astype(float)
        consol = pd.Series(0.0, index=close.index)
        for i in range(1, n):
            consol.iloc[i] = (consol.iloc[i - 1] + 1) * in_range.iloc[i]

    base_norm = consol.clip(0, 120) / 120
    vc_norm = (1 - feats["Vol_contraction_40d"].clip(0, 2) / 2).clip(0, 1)
    bb_norm = (1 - feats["BB_width_ratio"].clip(0, 2) / 2).clip(0, 1)
    feats["Base_maturity_score"] = base_norm * vc_norm * bb_norm

    # 이평 수렴도
    ma5 = close.rolling(5, min_periods=3).mean()
    ma20 = close.rolling(20, min_periods=10).mean()
    ma60 = close.rolling(60, min_periods=20).mean()
    ma_stack = pd.concat([ma5, ma20, ma60], axis=1)
    ma_std = ma_stack.std(axis=1)
    feats["MA_convergence_score"] = 1 - (ma_std / close.replace(0, 1) * 100).clip(0, 1)

    # D. 시장 대비
    if kospi_close is not None and len(kospi_close) > 20:
        aligned = kospi_close.reindex(close.index, method="ffill")
        k_ret_20 = aligned.pct_change(20).fillna(0)
        k_ret_60 = aligned.pct_change(60).fillna(0)
        feats["RS_vs_market_20d"] = close.pct_change(20).fillna(0) - k_ret_20
        feats["RS_vs_market_60d"] = close.pct_change(60).fillna(0) - k_ret_60
    else:
        feats["RS_vs_market_20d"] = 0.0
        feats["RS_vs_market_60d"] = 0.0

    return feats


# ──────────────────────────────────────────────
# 2. 라벨 생성 (이중: 이진 + 연속)
# ──────────────────────────────────────────────


def compute_labels(ohlcv_df: pd.DataFrame, horizon: int = 20, threshold: float = 0.30):
    """매 거래일: 향후 horizon일 내 고가 최대값 기준 수익률 산출.

    ⚠️ v3.2 이후 학습에는 compute_initiation_labels()을 사용.
    이 함수는 패턴 분석 등 호환용으로 유지.

    Returns:
        (binary_label, continuous_return) tuple of pd.Series
        binary_label: 1 if forward_return >= threshold, else 0
        continuous_return: 실제 최대 수익률 (상관분석용)
    """
    high = ohlcv_df["고가"].astype(float)
    close = ohlcv_df["종가"].astype(float)

    # 향후 horizon일 내 고가의 최대값 (역순 rolling으로 효율적 계산)
    forward_max = (
        high.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )
    forward_return = forward_max / close - 1

    binary = (forward_return >= threshold).astype(float)
    binary.iloc[-horizon:] = np.nan
    forward_return_out = forward_return.copy()
    forward_return_out.iloc[-horizon:] = np.nan

    return binary, forward_return_out


def compute_initiation_labels(
    ohlcv_df: pd.DataFrame,
    horizon: int = 20,
    threshold: float = 0.30,
    max_spike_3d: float = 0.06,
    max_spike_1d: float = 0.05,
    max_ret_10d: float = 0.10,
    max_ret_20d: float = 0.15,
    cooldown: int = 15,
):
    """급등 직전 시작점 라벨링 (v6.0 — Initiation-Point-Only).

    v6.0 핵심 변경: 급등 이벤트당 **첫 날만** 양성 + 쿨다운.
    기존(v5.2)은 조건 충족 연속 5~15일을 양성으로 표시하여
    모델이 "모멘텀"을 학습하는 라벨 오염 문제가 있었음.

    양성(label=1) 조건 (5가지 모두 충족 + initiation-only):
      ① 향후 horizon일 내 종가 최대 수익률 ≥ threshold (30%)
      ② 직전 3일 수익률 < max_spike_3d (6%) — 단기 급등 아님
      ③ 직전 5일 중 하루 최대 상승 < max_spike_1d (5%) — 갭 없음
      ④ 직전 10일 수익률 < max_ret_10d (10%) — 이미 상승 중 아님
      ⑤ 직전 20일 수익률 < max_ret_20d (15%) — 이미 상승 완료 아님
      ⑥ 이전 양성 라벨로부터 cooldown(15)일 이상 경과

    효과:
      - 급등 1회당 양성 1건 (기존 5~15건 → 1건)
      - 모델이 "시작점"의 피처 패턴만 학습
      - 양성 비율 대폭 감소 → 더 엄격하고 정확한 신호

    Returns:
        (binary_label, continuous_return) tuple of pd.Series
    """
    close = ohlcv_df["종가"].astype(float)

    # ① 향후 수익률 — 종가 기반
    forward_close_max = (
        close.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )
    forward_return = forward_close_max / close - 1

    # ②③ 단기 spike/갭 필터
    prior_3d = close.pct_change(3)
    daily_ret = close.pct_change(1)
    max_daily_5d = daily_ret.rolling(5, min_periods=1).max()

    # ④⑤ 중기 추세 필터 (엄격화: 10d<10%, 20d<15%)
    ret_10d = close.pct_change(10)
    ret_20d = close.pct_change(20)

    is_pre_surge = (
        (prior_3d < max_spike_3d)  # 3일 급등 아님
        & (max_daily_5d < max_spike_1d)  # 갭 상승 없음
        & (ret_10d < max_ret_10d)  # 10일간 이미 10%↑ 아님
        & (ret_20d < max_ret_20d)  # 20일간 이미 15%↑ 아님
    )

    raw_positive = (forward_return >= threshold) & is_pre_surge

    # ★ Initiation-Point-Only: 급등 이벤트당 첫 날만 양성 + 쿨다운
    label = pd.Series(0.0, index=close.index)
    last_label_pos = -cooldown - 1
    for i in range(len(close)):
        if raw_positive.iloc[i] and (i - last_label_pos) >= cooldown:
            label.iloc[i] = 1.0
            last_label_pos = i

    label.iloc[-horizon:] = np.nan  # 미래 데이터 부족 구간
    label.iloc[:20] = np.nan  # 사전 지표 계산 부족 구간

    forward_return_out = forward_return.copy()
    forward_return_out.iloc[-horizon:] = np.nan

    return label, forward_return_out


def compute_quiet_storm_labels(
    ohlcv_df: pd.DataFrame, horizon: int = 20, target: float = 0.30, cooldown: int = 20
):
    """v7.0: 고요한 폭풍 전야 라벨링.

    양성 = 향후 horizon일 내 종가 최대 +target AND "완벽한 고요 상태"
    7가지 조건 모두 충족 시에만 양성 라벨 부여.

    핵심 차이 (vs v6.0 compute_initiation_labels):
      v6.0: Ret_3d<6%, Ret_10d<10%, Ret_20d<15% (느슨 → 모멘텀 오염)
      v7.0: 5일변동<3%, Ret_5d·10d<±5%, RSI<50, 20일저가5%이내 (초엄격)

    Returns:
        (label, forward_return, hard_neg) — hard_neg는 "이미 급등중" 마커
    """
    close = ohlcv_df["종가"].astype(float)
    volume = ohlcv_df["거래량"].astype(float)

    # ── 향후 수익률 ──
    forward_close_max = (
        close.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )
    forward_return = forward_close_max / close - 1

    # ── 기본 지표 ──
    daily_ret = close.pct_change(1)
    ret_5d = close.pct_change(5)
    ret_10d = close.pct_change(10)
    ret_20d = close.pct_change(20)
    low_20d = close.rolling(20).min()
    vol_avg20 = volume.rolling(20).mean().replace(0, np.nan)
    rsi = indicators.calc_rsi(close, 14)

    # ── 7가지 "고요" 조건 (ALL 충족) ──
    quiet = (
        (daily_ret.rolling(5, min_periods=1).max() < 0.03)  # ① 5일내 일일상승 3%미만
        & (
            daily_ret.rolling(5, min_periods=1).min() > -0.03
        )  # ① 5일내 일일하락도 3%미만
        & (ret_5d.abs() < 0.05)  # ② 5일수익률 ±5%이내
        & (ret_10d.abs() < 0.05)  # ③ 10일수익률 ±5%이내
        & (ret_20d < 0.08)
        & (ret_20d > -0.15)  # ④ 20일수익률 -15%~+8%
        & (close / low_20d.replace(0, np.nan) < 1.05)  # ⑤ 20일저가의 5%이내
        & (volume / vol_avg20 < 1.3)  # ⑥ 거래량 평균130%미만
        & (rsi < 50)  # ⑦ RSI<50
    )

    raw_positive = (forward_return >= target) & quiet

    # Initiation-only + 20일 쿨다운
    label = pd.Series(0.0, index=close.index)
    last_label_pos = -cooldown - 1
    for i in range(len(close)):
        if raw_positive.iloc[i] and (i - last_label_pos) >= cooldown:
            label.iloc[i] = 1.0
            last_label_pos = i

    label.iloc[-horizon:] = np.nan  # 미래 데이터 부족 구간
    label.iloc[:30] = np.nan  # 사전 지표 계산 부족 구간

    # ★ Hard Negative 마킹 (이미 급등 중인 날) ★
    hard_neg = pd.Series(0.0, index=close.index)
    hard_neg_mask = (
        (ret_10d > 0.10)
        | (ret_20d > 0.15)
        | (daily_ret.rolling(5, min_periods=1).max() > 0.05)
    )
    hard_neg[hard_neg_mask] = 1.0
    # 양성 라벨은 hard_neg에서 제외
    hard_neg[label == 1.0] = 0.0
    hard_neg[label.isna()] = np.nan

    forward_return_out = forward_return.copy()
    forward_return_out.iloc[-horizon:] = np.nan

    return label, forward_return_out, hard_neg


def compute_base_breakout_labels(
    ohlcv_df: pd.DataFrame,
    min_surge: float = 0.25,
    max_horizon: int = 60,
    cooldown: int = 20,
):
    """v8.0: 급등 시작점 역추적 라벨링 (Base Breakout).

    v7.0 "고요한 폭풍 전야"와 반대 방향:
      v7.0: 조용한 상태 정의 → 급등 필터 → 0.10% 양성
      v8.0: 급등 이벤트 탐지 → 시작점(로컬 최저점) 역추적 → 0.3-0.7% 양성

    알고리즘:
      1. 로컬 최저점 탐지 (20일 윈도우 최저 + MA60 미만 또는 20일 고점 대비 -10%)
      2. 각 최저점에서 60거래일 내 +25% 이상 상승 확인
      3. 최저점 당일을 양성 라벨 (1 label per event, 20일 쿨다운)
      4. 급등 피크 직후 20일을 Hard Negative (꼭대기 매수 방지)

    Returns:
        (label, forward_return, hard_neg)
    """
    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    ma60 = close.rolling(60, min_periods=20).mean()

    label = pd.Series(0.0, index=close.index)
    hard_neg = pd.Series(0.0, index=close.index)

    # Step 1: 로컬 최저점 탐지
    local_mins = []
    for i in range(20, len(close) - max_horizon):
        window = close.iloc[max(0, i - 10) : i + 11]
        if close.iloc[i] <= window.min() * 1.01:  # 윈도우 최저의 1% 이내
            # MA60 이하 또는 20일 고점 대비 -10% 이상 하락
            if (
                close.iloc[i] < ma60.iloc[i] * 1.02
                or close.iloc[i] < high.iloc[max(0, i - 20) : i + 1].max() * 0.90
            ):
                local_mins.append(i)

    # Step 2: 각 최저점에서 급등 확인 + 라벨링
    last_label_pos = -cooldown - 1
    for min_idx in local_mins:
        if (min_idx - last_label_pos) < cooldown:
            continue
        min_price = close.iloc[min_idx]
        future = close.iloc[min_idx : min(min_idx + max_horizon + 1, len(close))]
        if len(future) < 10:
            continue
        max_future_price = future.max()
        surge_return = max_future_price / min_price - 1

        if surge_return >= min_surge:
            label.iloc[min_idx] = 1.0
            last_label_pos = min_idx

            # Hard Negative: 피크 이후 20일 (꼭대기 매수 방지)
            peak_offset = int(future.values.argmax())
            peak_idx = min_idx + peak_offset
            for j in range(peak_idx, min(peak_idx + 20, len(close))):
                hard_neg.iloc[j] = 1.0

    # 양성 라벨은 hard_neg에서 제외
    hard_neg[label == 1.0] = 0.0

    # 미래 데이터 부족 구간 / 사전 지표 부족 구간
    label.iloc[-max_horizon:] = np.nan
    label.iloc[:60] = np.nan
    hard_neg[label.isna()] = np.nan

    # forward return (참고용)
    forward_close_max = (
        close.iloc[::-1].rolling(max_horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )
    forward_return = forward_close_max / close - 1
    forward_return.iloc[-max_horizon:] = np.nan

    return label, forward_return, hard_neg


# ── (horizon, target) → stop 2D 매핑 ──────────────────────
def compute_recommended_stop(
    horizon_weeks: int, target_pct: int, mode: str = "stock"
) -> int:
    """(horizon, target) → stop 추천값 (2D 매핑).

    기존 1D 맵(base_target 기준 경험적 최적값)에 target 비율을 적용.
    ``stop = base_stop[horizon] × (target / base_target)``, clamped.

    Args:
        horizon_weeks: 수익목표기간 (주).
        target_pct: 목표수익률 (정수 %). 예: 30, 50.
        mode: ``"stock"`` | ``"etf"``.

    Returns:
        추천 stop_pct (정수 %).
    """
    if mode == "etf":
        _BASE_MAP = {3: 4, 4: 5, 6: 6, 8: 8, 10: 10, 12: 12}
        _BASE_TARGET = 10
        _MIN, _MAX, _FALLBACK = 3, 20, 5
    else:
        _BASE_MAP = {
            3: 8,
            4: 10,
            6: 12,
            8: 15,
            10: 18,
            12: 20,
            16: 22,
            20: 25,
            24: 25,
        }
        _MIN, _MAX, _FALLBACK = 5, 30, 10

    base_stop = _BASE_MAP.get(horizon_weeks, _FALLBACK)

    if mode == "etf":
        base_target = _BASE_TARGET
    else:
        # horizons ≤12주: 기존 맵이 target 30% 기준
        # horizons ≥16주: 기존 맵이 target 50% 기준
        base_target = 30 if horizon_weeks <= 12 else 50

    adjusted = round(base_stop * (target_pct / base_target))
    return max(_MIN, min(_MAX, adjusted))


def compute_breakout_labels(
    ohlcv_df: pd.DataFrame,
    target_pct: float = 0.20,
    stop_pct: float = 0.08,
    horizon_days: int = 15,
    pos_cooldown: int = 10,
):
    """AI_급등주 v2: Daily Triple-Barrier 라벨링.

    모든 거래일에 대해 Triple-Barrier 적용:
      - 목표: +20% (고가 기준)
      - 손절: -8% (저가 기준)
      - 기간: 15거래일
      - Outcome-First: 먼저 도달한 쪽이 승 (labeled.xlsx 규칙)
      - 둘 다 미도달 → NaN (학습 제외)
      - 양성 쿨다운: label=1 후 10일간 NaN (연속 양성 방지)

    Returns:
        (label, forward_return, hard_neg)
        - label: 0/1/NaN
        - forward_return: horizon 내 최대 상승률 (MFE, 참고용)
        - hard_neg: 손절 도달 음성 = 1.0 (Hard Negative 가중치용)
    """
    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    low = ohlcv_df["저가"].astype(float)
    n = len(close)

    close_arr = close.values
    high_arr = high.values
    low_arr = low.values

    label_arr = np.full(n, np.nan)
    hard_neg_arr = np.zeros(n)
    fwd_ret_arr = np.full(n, np.nan)

    # 벡터화된 forward return (MFE)
    for i in range(n - 1):
        end = min(i + horizon_days + 1, n)
        if end <= i + 1:
            continue
        fwd_highs = high_arr[i + 1 : end]
        fwd_lows = low_arr[i + 1 : end]
        entry = close_arr[i]
        if entry <= 0:
            continue

        fwd_ret_arr[i] = fwd_highs.max() / entry - 1

        target_price = entry * (1 + target_pct)
        stop_price = entry * (1 - stop_pct)

        target_day = -1
        stop_day = -1
        for j in range(len(fwd_highs)):
            if target_day < 0 and fwd_highs[j] >= target_price:
                target_day = j
            if stop_day < 0 and fwd_lows[j] <= stop_price:
                stop_day = j
            if target_day >= 0 and stop_day >= 0:
                break

        if target_day >= 0 and (stop_day < 0 or target_day <= stop_day):
            label_arr[i] = 1.0  # 목표가 먼저 도달
        elif stop_day >= 0 and (target_day < 0 or stop_day < target_day):
            label_arr[i] = 0.0  # 손절 먼저 도달
            hard_neg_arr[i] = 1.0
        # else: 둘 다 미도달 → NaN 유지

    # 양성 쿨다운: label=1 후 pos_cooldown일간 NaN 처리 (연속 양성 방지)
    last_pos = -pos_cooldown - 1
    for i in range(n):
        if label_arr[i] == 1.0:
            if (i - last_pos) < pos_cooldown:
                label_arr[i] = np.nan  # 쿨다운 내 양성 → 제외
                hard_neg_arr[i] = 0.0
            else:
                last_pos = i

    # 사전 지표 부족 구간 + 미래 데이터 부족 구간
    label_arr[:60] = np.nan
    label_arr[-(horizon_days + 1) :] = np.nan
    hard_neg_arr[np.isnan(label_arr)] = 0.0
    fwd_ret_arr[-(horizon_days + 1) :] = np.nan

    label = pd.Series(label_arr, index=close.index)
    forward_return = pd.Series(fwd_ret_arr, index=close.index)
    hard_neg = pd.Series(hard_neg_arr, index=close.index)

    return label, forward_return, hard_neg


def compute_accumulation_labels(
    ohlcv_df: pd.DataFrame,
    surge_threshold: float = 0.25,
    surge_horizon: int = 20,
    base_range_pct: float = 0.10,
    min_base_days: int = 30,
    max_base_days: int = 120,
    min_quality_checks: int = 3,
    cooldown: int = 40,
    fail_drop_pct: float = 0.15,
):
    """AI_매집주: 급등→역추적→매집기간 감지 라벨링.

    Algorithm:
      1. 급등 이벤트 탐지 (+25% in 20d)
      2. 매집 기간 역추적 (±10%, ≥30일 횡보)
      3. 매집 품질 확인 (OBV/거래량수축/BB수축/양봉밀집 중 3개+)
      4. 양성 라벨 = 매집 기간 마지막 5일
      5. Hard Negative = 횡보 후 하락 또는 품질 미달

    Returns:
        (label, forward_return, hard_neg)
    """
    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    low = ohlcv_df["저가"].astype(float)
    volume = ohlcv_df["거래량"].astype(float)
    op = ohlcv_df["시가"].astype(float) if "시가" in ohlcv_df.columns else close

    n = len(close)
    label = pd.Series(np.nan, index=close.index)
    hard_neg = pd.Series(0.0, index=close.index)

    # OBV 계산
    obv = pd.Series(0.0, index=close.index)
    for i in range(1, n):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    # 볼린저밴드 폭
    bb_ma = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_width = (2 * bb_std / bb_ma).fillna(0)

    last_label_pos = -cooldown - 1

    # Step 1: 급등 이벤트 탐지 (forward scan)
    for i in range(min_base_days + 60, n - surge_horizon):
        # 쿨다운
        if (i - last_label_pos) < cooldown:
            continue

        entry_price = close.iloc[i]
        # 향후 surge_horizon일 내 +25% 도달?
        future_high = high.iloc[i + 1 : i + surge_horizon + 1]
        if len(future_high) == 0:
            continue
        if future_high.max() < entry_price * (1 + surge_threshold):
            continue
        # 이 시점에서 급등 이벤트 확인됨

        # Step 2: 매집 기간 역추적
        # i 직전 구간에서 ±base_range_pct 범위 내 횡보 탐색
        base_end = i
        base_start = None
        ref_price = close.iloc[i]
        upper = ref_price * (1 + base_range_pct)
        lower = ref_price * (1 - base_range_pct)

        # 역방향으로 횡보 구간 탐색
        for j in range(i - 1, max(i - max_base_days - 1, 59), -1):
            if close.iloc[j] > upper or close.iloc[j] < lower:
                break
            base_start = j

        if base_start is None:
            continue
        base_len = base_end - base_start
        if base_len < min_base_days:
            continue

        # Step 3: 매집 품질 확인
        quality = 0
        half = base_start + base_len // 2

        # ① OBV 상승
        obv_start_val = obv.iloc[base_start]
        obv_end_val = obv.iloc[base_end]
        if obv_end_val > obv_start_val:
            quality += 1

        # ② 거래량 수축: 후반 평균 < 전반 × 0.8
        vol_first = volume.iloc[base_start:half].mean()
        vol_second = volume.iloc[half:base_end].mean()
        if vol_first > 0 and vol_second < vol_first * 0.8:
            quality += 1

        # ③ BB 수축: 후반 BB폭 < 전반 × 0.8
        bb_first = bb_width.iloc[base_start:half].mean()
        bb_second = bb_width.iloc[half:base_end].mean()
        if bb_first > 0 and bb_second < bb_first * 0.8:
            quality += 1

        # ④ 양봉 밀집: 양봉 비율 ≥ 50%
        pos_candles = (
            close.iloc[base_start:base_end] >= op.iloc[base_start:base_end]
        ).sum()
        total_candles = base_end - base_start
        if total_candles > 0 and pos_candles / total_candles >= 0.50:
            quality += 1

        if quality >= min_quality_checks:
            # 양성: 매집 기간 마지막 5일에 라벨
            lbl_start = max(base_end - 5, base_start)
            for k in range(lbl_start, base_end):
                label.iloc[k] = 1.0
            last_label_pos = i  # 급등 시작 시점 기준 쿨다운
        else:
            # Hard Negative: 품질 미달 횡보
            lbl_start = max(base_end - 5, base_start)
            for k in range(lbl_start, base_end):
                label.iloc[k] = 0.0
                hard_neg.iloc[k] = 1.0
            last_label_pos = i

    # Hard Negative: 횡보 후 하락 (급등 없이)
    # 별도 스캔: ±10% 횡보 ≥30일 + 이후 -15% 하락
    for i in range(min_base_days + 60, n - 20):
        if label.iloc[i] == label.iloc[i]:  # not NaN = 이미 라벨됨
            continue
        ref_p = close.iloc[i]
        up = ref_p * (1 + base_range_pct)
        lo = ref_p * (1 - base_range_pct)
        bs = None
        for j in range(i - 1, max(i - max_base_days - 1, 59), -1):
            if close.iloc[j] > up or close.iloc[j] < lo:
                break
            bs = j
        if bs is None or (i - bs) < min_base_days:
            continue
        # 이후 20일 내 -15% 하락?
        future_low = low.iloc[i + 1 : min(i + 21, n)]
        if len(future_low) > 0 and future_low.min() <= ref_p * (1 - fail_drop_pct):
            lbl_s = max(i - 5, bs)
            for k in range(lbl_s, i):
                if np.isnan(label.iloc[k]):
                    label.iloc[k] = 0.0
                    hard_neg.iloc[k] = 1.0

    # 초기 구간 제외
    label.iloc[:60] = np.nan
    hard_neg[label.isna()] = 0.0

    # forward return (참고용)
    fwd_max = (
        close.iloc[::-1]
        .rolling(surge_horizon, min_periods=1)
        .max()
        .iloc[::-1]
        .shift(-1)
    )
    forward_return = fwd_max / close - 1
    forward_return.iloc[-surge_horizon:] = np.nan

    return label, forward_return, hard_neg


# ──────────────────────────────────────────────
# 3. 패턴 분석
# ──────────────────────────────────────────────


def analyze_patterns(X: pd.DataFrame, y_binary: pd.Series, y_return: pd.Series) -> dict:
    """상승종목 공통 패턴 + 상승폭 상관관계 분석.

    Returns:
        dict with correlations, positive_profile, overall_profile,
        feature_coverage, pattern_summary
    """
    pos_mask = y_binary == 1
    n_pos = int(pos_mask.sum())
    if n_pos < 10:
        return {
            "correlations": {},
            "positive_profile": {},
            "overall_profile": {},
            "feature_coverage": {},
            "pattern_summary": [],
        }

    # 1. Spearman 상관: 각 피처 ↔ continuous_return
    correlations = {}
    for col in X.columns:
        valid = X[col].notna() & y_return.notna()
        if valid.sum() > 100:
            correlations[col] = float(
                X.loc[valid, col].corr(y_return[valid], method="spearman")
            )

    # 2. 양성 프로필 vs 전체 프로필
    positive_profile = {}
    overall_profile = {}
    for col in X.columns:
        pos_vals = X.loc[pos_mask, col].dropna()
        all_vals = X[col].dropna()
        if len(pos_vals) > 10 and len(all_vals) > 100:
            positive_profile[col] = {
                "mean": float(pos_vals.mean()),
                "std": float(pos_vals.std()),
                "median": float(pos_vals.median()),
            }
            overall_profile[col] = {
                "mean": float(all_vals.mean()),
                "std": float(all_vals.std()),
                "median": float(all_vals.median()),
            }

    # 3. 패턴 커버리지: 양성 케이스 중 피처가 "급등 방향"인 비율
    feature_coverage = {}
    for col in X.columns:
        if col not in correlations or col not in positive_profile:
            continue
        corr = correlations[col]
        all_mean = overall_profile[col]["mean"]

        # 급등 방향 판정: 상관계수 부호로 결정
        pos_vals = X.loc[pos_mask, col].dropna()
        if corr < 0:
            # 음의 상관 → 낮을수록 좋음 → 양성 평균 이하인 비율
            threshold_val = all_mean  # 전체 평균 이하
            coverage = float((pos_vals <= threshold_val).mean())
        else:
            # 양의 상관 → 높을수록 좋음 → 전체 평균 이상인 비율
            threshold_val = all_mean
            coverage = float((pos_vals >= threshold_val).mean())
        feature_coverage[col] = coverage

    # 4. 종합 순위: |상관계수| 기준 정렬
    pattern_summary = []
    for col in X.columns:
        if col not in correlations:
            continue
        corr = correlations[col]
        pos_p = positive_profile.get(col, {})
        all_p = overall_profile.get(col, {})
        cov = feature_coverage.get(col, 0)
        kr_name = FEATURE_NAMES_KR.get(col, col)

        # 방향 설명
        if corr < -0.05:
            direction = "낮을수록 상승폭↑"
        elif corr > 0.05:
            direction = "높을수록 상승폭↑"
        else:
            direction = "약한 관계"

        pattern_summary.append(
            {
                "name": col,
                "name_kr": kr_name,
                "corr": corr,
                "pos_mean": pos_p.get("mean", 0),
                "all_mean": all_p.get("mean", 0),
                "coverage": cov,
                "direction": direction,
            }
        )

    pattern_summary.sort(key=lambda x: abs(x["corr"]), reverse=True)

    return {
        "correlations": correlations,
        "positive_profile": positive_profile,
        "overall_profile": overall_profile,
        "feature_coverage": feature_coverage,
        "pattern_summary": pattern_summary,
    }


# ──────────────────────────────────────────────
# 4. sklearn / numpy fallback
# ──────────────────────────────────────────────


def _try_import_sklearn():
    """sklearn 사용 가능 여부 확인 (v6.0: HistGradientBoosting)."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit

        return True, {
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
            "TimeSeriesSplit": TimeSeriesSplit,
            "roc_auc_score": roc_auc_score,
        }
    except ImportError:
        return False, None


def _train_sklearn(X, y, sk):
    """sklearn HistGradientBoosting 학습 + TimeSeriesSplit 검증 (v6.0)."""
    # sample_weight로 클래스 불균형 보정 (HistGradientBoosting은 class_weight 미지원)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    w_pos = len(y) / (2 * max(n_pos, 1))
    w_neg = len(y) / (2 * max(n_neg, 1))
    sample_weight = np.where(y == 1, w_pos, w_neg)

    clf = sk["HistGradientBoostingClassifier"](
        max_iter=300,
        max_depth=6,
        min_samples_leaf=50,
        learning_rate=0.05,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )

    # TimeSeriesSplit 교차검증 (시간 순서 보존)
    tscv = sk["TimeSeriesSplit"](n_splits=3)
    val_aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sw_tr = sample_weight[train_idx]
        clf.fit(X_tr, y_tr, sample_weight=sw_tr)
        if len(y_val.unique()) < 2:
            continue
        prob = clf.predict_proba(X_val)[:, 1]
        auc = sk["roc_auc_score"](y_val, prob)
        val_aucs.append(auc)

    # 전체 데이터로 최종 학습
    clf.fit(X, y, sample_weight=sample_weight)

    # ── 학습 데이터 확률 분포 저장 ──
    train_probs = clf.predict_proba(X)[:, 1]
    pos_rate = float(y.mean())

    # HGB: raw probability 백분위 임계값 (sigmoid 보정 불필요)
    # sample_weight 균형으로 raw_prob이 과대추정되지만,
    # 백분위 기반 임계값으로 자기참조 → 보정 불필요
    prob_pct95 = float(np.percentile(train_probs, 95))
    prob_pct98 = float(np.percentile(train_probs, 98))
    prob_pct99 = float(np.percentile(train_probs, 99))

    # 피처별 상관 기반 중요도 (feature_importances_ 대체)
    importances = {}
    for col in X.columns:
        valid = X[col].notna()
        if valid.sum() > 100:
            importances[col] = abs(float(X.loc[valid, col].corr(y[valid])))
        else:
            importances[col] = 0.0
    # 정규화
    total_imp = sum(importances.values())
    if total_imp > 0:
        importances = {k: v / total_imp for k, v in importances.items()}

    return {
        "type": "sklearn_hgb",
        "model": clf,
        "feature_importances": importances,
        "validation_aucs": val_aucs,
        "validation_auc": float(np.mean(val_aucs)) if val_aucs else 0.0,
        "pos_rate": pos_rate,
        "prob_pct95": prob_pct95,
        "prob_pct98": prob_pct98,
        "prob_pct99": prob_pct99,
    }


def _train_sklearn_v7(X, y, hard_neg, sk, _cb=None):
    """v7.0: HGB 학습 + Hard Negative 가중치 + Walk-Forward 검증.

    Args:
        X: 피처 DataFrame
        y: 이진 라벨 (0/1)
        hard_neg: Hard Negative 마스크 (이미 급등 중인 날)
        sk: sklearn 모듈 딕셔너리
        _cb: 진행 콜백
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    w_pos = len(y) / (2 * max(n_pos, 1))
    w_neg = len(y) / (2 * max(n_neg, 1))

    # ★ Hard Negative: 이미 급등 중인 날은 2배 가중치 ★
    sample_weight = np.where(y == 1, w_pos, w_neg)
    hard_neg_mask = hard_neg.values > 0 if hasattr(hard_neg, "values") else hard_neg > 0
    sample_weight[hard_neg_mask & (y == 0)] = w_neg * 2.0

    # v7 강화 HGB 파라미터
    def _make_hgb():
        return sk["HistGradientBoostingClassifier"](
            max_iter=500,
            max_depth=5,
            min_samples_leaf=80,
            learning_rate=0.03,
            l2_regularization=1.0,
            max_bins=128,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            scoring="average_precision",
        )

    # ── Walk-Forward 검증 ──
    # 3년 데이터를 4등분 → 3개 Window
    n = len(X)
    quarter = n // 4
    windows = [
        (0, quarter * 2, quarter * 2, quarter * 3),  # Train[Q1-Q2] → Test[Q3]
        (0, quarter * 3, quarter * 3, n - quarter // 2),  # Train[Q1-Q3] → Test[Q3.5-Q4]
        (0, n - quarter, n - quarter, n),  # Train[Q1-Q3.5] → Test[Q4]
    ]

    val_aucs = []
    val_precisions = []
    for w_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        if te_end <= te_start or tr_end <= tr_start:
            continue
        X_tr = X.iloc[tr_start:tr_end]
        y_tr = y.iloc[tr_start:tr_end]
        sw_tr = sample_weight[tr_start:tr_end]
        X_te = X.iloc[te_start:te_end]
        y_te = y.iloc[te_start:te_end]

        if y_tr.sum() < 5 or len(y_te.unique()) < 2:
            continue

        clf_w = _make_hgb()
        clf_w.fit(X_tr, y_tr, sample_weight=sw_tr)
        prob_te = clf_w.predict_proba(X_te)[:, 1]

        try:
            auc = sk["roc_auc_score"](y_te, prob_te)
            val_aucs.append(auc)
        except Exception:
            pass

        # Precision@Top50: 상위 50개 예측 중 실제 양성 비율 + Lift
        base_rate = float(y_te.mean()) if len(y_te) > 0 else 0.001
        top_k = min(50, max(1, int(len(prob_te) * 0.01)))  # 상위 1% 또는 50개
        top_idx = np.argsort(prob_te)[-top_k:]
        prec_at_k = float(y_te.iloc[top_idx].mean())
        lift = prec_at_k / max(base_rate, 1e-6)
        val_precisions.append(prec_at_k)

        if _cb:
            _cb(
                f"  Walk-Forward {w_idx+1}/3: AUC={auc:.3f}, "
                f"Top{top_k} Prec={prec_at_k:.1%} "
                f"(Lift {lift:.1f}x vs 기저 {base_rate:.2%})",
                None,
            )

    # ── 전체 데이터로 최종 학습 ──
    clf = _make_hgb()
    clf.fit(X, y, sample_weight=sample_weight)

    # ── 학습 데이터 확률 분포 ──
    train_probs = clf.predict_proba(X)[:, 1]
    pos_rate = float(y.mean())

    prob_pct95 = float(np.percentile(train_probs, 95))
    prob_pct98 = float(np.percentile(train_probs, 98))
    prob_pct99 = float(np.percentile(train_probs, 99))

    # 피처 중요도 (HGB 내장 + 상관 기반 보정)
    importances = {}
    if hasattr(clf, "feature_importances_"):
        for i, col in enumerate(X.columns):
            importances[col] = float(clf.feature_importances_[i])
    else:
        for col in X.columns:
            valid = X[col].notna()
            if valid.sum() > 100:
                importances[col] = abs(float(X.loc[valid, col].corr(y[valid])))
            else:
                importances[col] = 0.0
    total_imp = sum(importances.values())
    if total_imp > 0:
        importances = {k: v / total_imp for k, v in importances.items()}

    mean_auc = float(np.mean(val_aucs)) if val_aucs else 0.0
    mean_prec = float(np.mean(val_precisions)) if val_precisions else 0.0

    if _cb:
        base_rate_all = float(y.mean())
        mean_lift = mean_prec / max(base_rate_all, 1e-6)
        _cb(
            f"  Walk-Forward 평균: AUC={mean_auc:.3f}, "
            f"TopK Prec={mean_prec:.1%} "
            f"(Lift {mean_lift:.1f}x vs 기저 {base_rate_all:.2%})",
            None,
        )

    return {
        "type": "sklearn_hgb",
        "model": clf,
        "model_type": "sklearn_hgb",
        "feature_importances": importances,
        "validation_aucs": val_aucs,
        "validation_auc": mean_auc,
        "validation_precisions": val_precisions,
        "validation_precision_mean": mean_prec,
        "pos_rate": pos_rate,
        "prob_pct95": prob_pct95,
        "prob_pct98": prob_pct98,
        "prob_pct99": prob_pct99,
    }


def _train_sklearn_v8(X, y, hard_neg, sk, _cb=None):
    """v8.0: HGB 학습 + Hard Negative 1.5x + Walk-Forward 검증.

    v7.0 대비 변경: max_depth=6, min_samples_leaf=50, hard_neg 1.5x.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    w_pos = len(y) / (2 * max(n_pos, 1))
    w_neg = len(y) / (2 * max(n_neg, 1))

    # ★ Hard Negative: 급등 피크 직후 → 1.5x 가중치 (v7: 2.0x) ★
    sample_weight = np.where(y == 1, w_pos, w_neg)
    hard_neg_mask = hard_neg.values > 0 if hasattr(hard_neg, "values") else hard_neg > 0
    sample_weight[hard_neg_mask & (y == 0)] = w_neg * 1.5

    # v8 HGB 파라미터 (양성 더 많으므로 깊은 트리, 작은 리프)
    def _make_hgb():
        return sk["HistGradientBoostingClassifier"](
            max_iter=500,
            max_depth=6,
            min_samples_leaf=50,
            learning_rate=0.03,
            l2_regularization=1.0,
            max_bins=128,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            scoring="average_precision",
        )

    # ── Walk-Forward 검증 ──
    n = len(X)
    quarter = n // 4
    windows = [
        (0, quarter * 2, quarter * 2, quarter * 3),
        (0, quarter * 3, quarter * 3, n - quarter // 2),
        (0, n - quarter, n - quarter, n),
    ]

    val_aucs = []
    val_precisions = []
    for w_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        if te_end <= te_start or tr_end <= tr_start:
            continue
        X_tr = X.iloc[tr_start:tr_end]
        y_tr = y.iloc[tr_start:tr_end]
        sw_tr = sample_weight[tr_start:tr_end]
        X_te = X.iloc[te_start:te_end]
        y_te = y.iloc[te_start:te_end]

        if y_tr.sum() < 5 or len(y_te.unique()) < 2:
            continue

        clf_w = _make_hgb()
        clf_w.fit(X_tr, y_tr, sample_weight=sw_tr)
        prob_te = clf_w.predict_proba(X_te)[:, 1]

        try:
            auc = sk["roc_auc_score"](y_te, prob_te)
            val_aucs.append(auc)
        except Exception:
            pass

        base_rate = float(y_te.mean()) if len(y_te) > 0 else 0.001
        top_k = min(50, max(1, int(len(prob_te) * 0.01)))
        top_idx = np.argsort(prob_te)[-top_k:]
        prec_at_k = float(y_te.iloc[top_idx].mean())
        lift = prec_at_k / max(base_rate, 1e-6)
        val_precisions.append(prec_at_k)

        if _cb:
            _cb(
                f"  Walk-Forward {w_idx+1}/3: AUC={auc:.3f}, "
                f"Top{top_k} Prec={prec_at_k:.1%} "
                f"(Lift {lift:.1f}x vs 기저 {base_rate:.2%})",
                None,
            )

    # ── 전체 데이터로 최종 학습 ──
    clf = _make_hgb()
    clf.fit(X, y, sample_weight=sample_weight)

    train_probs = clf.predict_proba(X)[:, 1]
    pos_rate = float(y.mean())

    prob_pct95 = float(np.percentile(train_probs, 95))
    prob_pct98 = float(np.percentile(train_probs, 98))
    prob_pct99 = float(np.percentile(train_probs, 99))

    importances = {}
    if hasattr(clf, "feature_importances_"):
        for i, col in enumerate(X.columns):
            importances[col] = float(clf.feature_importances_[i])
    else:
        for col in X.columns:
            valid = X[col].notna()
            if valid.sum() > 100:
                importances[col] = abs(float(X.loc[valid, col].corr(y[valid])))
            else:
                importances[col] = 0.0
    total_imp = sum(importances.values())
    if total_imp > 0:
        importances = {k: v / total_imp for k, v in importances.items()}

    mean_auc = float(np.mean(val_aucs)) if val_aucs else 0.0
    mean_prec = float(np.mean(val_precisions)) if val_precisions else 0.0

    if _cb:
        base_rate_all = float(y.mean())
        mean_lift = mean_prec / max(base_rate_all, 1e-6)
        _cb(
            f"  Walk-Forward 평균: AUC={mean_auc:.3f}, "
            f"TopK Prec={mean_prec:.1%} "
            f"(Lift {mean_lift:.1f}x vs 기저 {base_rate_all:.2%})",
            None,
        )

    return {
        "type": "sklearn_hgb",
        "model": clf,
        "model_type": "sklearn_hgb",
        "feature_importances": importances,
        "validation_aucs": val_aucs,
        "validation_auc": mean_auc,
        "validation_precisions": val_precisions,
        "validation_precision_mean": mean_prec,
        "pos_rate": pos_rate,
        "prob_pct95": prob_pct95,
        "prob_pct98": prob_pct98,
        "prob_pct99": prob_pct99,
    }


def _try_import_lightgbm():
    """LightGBM 사용 가능 여부 확인."""
    try:
        import lightgbm as lgb

        return True, lgb
    except ImportError:
        return False, None


def _train_sklearn_surge(X, y, hard_neg, sk, _cb=None, lgb_overrides=None):
    """AI_급등주 v2: LightGBM 학습 (primary) + HGB fallback.

    Walk-Forward 4-window 검증 + Feature Importance.

    Args:
        lgb_overrides: 기간별 LightGBM 파라미터 오버라이드 dict.
            예: {"num_leaves": 31, "min_data_in_leaf": 30}
    """
    if _cb is None:

        def _cb(msg, pct=None):
            return None

    valid_mask = ~np.isnan(y)
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    hard_neg_clean = (
        hard_neg[valid_mask].copy() if hard_neg is not None else np.zeros(len(y_clean))
    )

    n_pos = int(y_clean.sum())
    n_neg = int((y_clean == 0).sum())
    _cb(f"  학습 데이터: {len(y_clean)}건 (양성={n_pos}, 음성={n_neg})")
    if n_pos < 20:
        raise ValueError(f"양성 샘플 부족: {n_pos}건 (최소 20건)")

    # 가중치: 불균형 보정 + Hard Negative 1.5x
    w_pos = max(n_neg / n_pos, 1.0)
    sample_weight = np.where(y_clean == 1, w_pos, 1.0)
    hard_neg_mask = hard_neg_clean.astype(bool) & (y_clean == 0)
    sample_weight[hard_neg_mask] = 1.5
    _cb(f"  가중치: pos={w_pos:.1f}, hard_neg=1.5x ({int(hard_neg_mask.sum())}건)")

    # NaN/Inf 처리
    X_arr = np.nan_to_num(
        X_clean.values if hasattr(X_clean, "values") else X_clean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float64)
    y_arr = y_clean.values if hasattr(y_clean, "values") else y_clean
    feat_names = list(X_clean.columns) if hasattr(X_clean, "columns") else None

    from sklearn.metrics import roc_auc_score

    # ── Walk-Forward 4-window 검증 (5일 embargo) ──
    n_total = len(X_arr)
    embargo = 5
    quarters = [
        (0, int(n_total * 0.58), int(n_total * 0.58) + embargo, int(n_total * 0.70)),
        (0, int(n_total * 0.68), int(n_total * 0.68) + embargo, int(n_total * 0.80)),
        (0, int(n_total * 0.78), int(n_total * 0.78) + embargo, int(n_total * 0.88)),
        (0, int(n_total * 0.86), int(n_total * 0.86) + embargo, n_total),
    ]

    lgb_ok, lgb = _try_import_lightgbm()
    use_lgb = lgb_ok

    wf_results = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(quarters):
        if te_e <= te_s or tr_e <= tr_s + 100:
            continue
        X_tr, y_tr, sw_tr = X_arr[tr_s:tr_e], y_arr[tr_s:tr_e], sample_weight[tr_s:tr_e]
        X_te, y_te = X_arr[te_s:te_e], y_arr[te_s:te_e]
        if y_te.sum() < 3 or len(np.unique(y_te)) < 2:
            continue

        if use_lgb:
            dtrain = lgb.Dataset(
                X_tr, y_tr, weight=sw_tr, feature_name=feat_names, free_raw_data=False
            )
            dval = lgb.Dataset(X_te, y_te, reference=dtrain, free_raw_data=False)
            params = {
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
            if lgb_overrides:
                params.update(lgb_overrides)
            m_wf = lgb.train(
                params,
                dtrain,
                num_boost_round=800,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            prob_te = m_wf.predict(X_te)
        else:
            HGB = sk["HistGradientBoostingClassifier"]
            m_wf = HGB(
                max_iter=500,
                max_depth=6,
                min_samples_leaf=50,
                learning_rate=0.03,
                l2_regularization=0.1,
                max_bins=255,
                random_state=42,
                verbose=0,
            )
            m_wf.fit(X_tr, y_tr, sample_weight=sw_tr)
            prob_te = m_wf.predict_proba(X_te)[:, 1]

        try:
            wf_auc = roc_auc_score(y_te, prob_te)
        except ValueError:
            wf_auc = 0.5
        top_k = max(10, int(len(y_te) * 0.02))
        top_idx = prob_te.argsort()[-top_k:]
        top_rate = float(y_te[top_idx].mean())
        base_rate = float(y_te.mean())
        lift = top_rate / max(base_rate, 1e-6)
        wf_results.append({"auc": wf_auc, "lift": lift, "top_rate": top_rate})
        _cb(
            f"  WF-{wi+1}/4: AUC={wf_auc:.3f}, Lift={lift:.1f}x, "
            f"Top{top_k}적중={top_rate:.1%}"
        )

    wf_avg_auc = float(np.mean([r["auc"] for r in wf_results])) if wf_results else 0.5

    # ── 전체 데이터로 최종 모델 학습 ──
    _cb(f"  전체 데이터 최종 학습 ({'LightGBM' if use_lgb else 'HGB'})...")
    feature_importances = {}

    if use_lgb:
        params_final = {
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
        if lgb_overrides:
            params_final.update(lgb_overrides)
        # early stopping 위해 마지막 15%를 validation으로 사용
        split_idx = int(n_total * 0.85)
        dtrain_sub = lgb.Dataset(
            X_arr[:split_idx],
            y_arr[:split_idx],
            weight=sample_weight[:split_idx],
            feature_name=feat_names,
            free_raw_data=False,
        )
        dval_sub = lgb.Dataset(
            X_arr[split_idx:],
            y_arr[split_idx:],
            reference=dtrain_sub,
            free_raw_data=False,
        )
        model = lgb.train(
            params_final,
            dtrain_sub,
            num_boost_round=1000,
            valid_sets=[dval_sub],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        train_prob = model.predict(X_arr)
        if feat_names:
            imp_gain = model.feature_importance(importance_type="gain")
            imp_sum = imp_gain.sum() if imp_gain.sum() > 0 else 1
            feature_importances = {
                fn: float(ig / imp_sum)
                for fn, ig in zip(feat_names, imp_gain, strict=False)
            }
        model_type = "surge_lgbm"
    else:
        HGB = sk["HistGradientBoostingClassifier"]
        model = HGB(
            max_iter=500,
            max_depth=6,
            min_samples_leaf=50,
            learning_rate=0.03,
            l2_regularization=0.1,
            max_bins=255,
            random_state=42,
            verbose=0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
        )
        model.fit(X_arr, y_arr, sample_weight=sample_weight)
        train_prob = model.predict_proba(X_arr)[:, 1]
        if feat_names:
            imp = model.feature_importances_
            imp_sum = imp.sum() if imp.sum() > 0 else 1
            feature_importances = {
                fn: float(ig / imp_sum) for fn, ig in zip(feat_names, imp, strict=False)
            }
        model_type = "surge_hgb"

    auc = roc_auc_score(y_arr, train_prob)
    _cb(f"  Train AUC: {auc:.3f} | WF Avg AUC: {wf_avg_auc:.3f}")

    # Feature Importance Top 10 보고
    if feature_importances:
        top_fi = sorted(feature_importances.items(), key=lambda x: -x[1])[:10]
        _cb("  Feature Importance Top 10:")
        for i, (fn, imp_val) in enumerate(top_fi, 1):
            kr = SURGE_FEATURE_NAMES_KR.get(fn, fn)
            _cb(f"    {i}. {kr} ({imp_val:.1%})")

    return {
        "model": model,
        "model_type": model_type,
        "version": "surge_v2",
        "feature_names": feat_names,
        "feature_importances": feature_importances,
        "train_auc": auc,
        "wf_avg_auc": wf_avg_auc,
        "wf_results": wf_results,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "use_lgbm": use_lgb,
    }


def _train_sklearn_accum(X, y, hard_neg, sk, _cb=None):
    """AI_매집주: HGB 학습 + Hard Negative 1.5x + Walk-Forward 검증.

    v8 _train_sklearn_v8 기반, 매집주 전용 파라미터.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    w_pos = len(y) / (2 * max(n_pos, 1))
    w_neg = len(y) / (2 * max(n_neg, 1))

    # Hard Negative: 품질 미달 횡보 / 횡보 후 하락 → 1.5x 가중치
    sample_weight = np.where(y == 1, w_pos, w_neg)
    hard_neg_mask = hard_neg.values > 0 if hasattr(hard_neg, "values") else hard_neg > 0
    sample_weight[hard_neg_mask & (y == 0)] = w_neg * 1.5

    def _make_hgb():
        return sk["HistGradientBoostingClassifier"](
            max_iter=500,
            max_depth=6,
            min_samples_leaf=50,
            learning_rate=0.03,
            l2_regularization=1.0,
            max_bins=128,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            scoring="average_precision",
        )

    # Walk-Forward 3-window 검증
    n = len(X)
    quarter = n // 4
    windows = [
        (0, quarter * 2, quarter * 2, quarter * 3),
        (0, quarter * 3, quarter * 3, n - quarter // 2),
        (0, n - quarter, n - quarter, n),
    ]

    val_aucs = []
    val_precisions = []
    for w_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        if te_end <= te_start or tr_end <= tr_start:
            continue
        X_tr = X.iloc[tr_start:tr_end]
        y_tr = y.iloc[tr_start:tr_end]
        sw_tr = sample_weight[tr_start:tr_end]
        X_te = X.iloc[te_start:te_end]
        y_te = y.iloc[te_start:te_end]

        if y_tr.sum() < 5 or len(y_te.unique()) < 2:
            continue

        clf_w = _make_hgb()
        clf_w.fit(X_tr, y_tr, sample_weight=sw_tr)
        prob_te = clf_w.predict_proba(X_te)[:, 1]

        try:
            auc = sk["roc_auc_score"](y_te, prob_te)
            val_aucs.append(auc)
        except Exception:
            pass

        base_rate = float(y_te.mean()) if len(y_te) > 0 else 0.001
        top_k = min(50, max(1, int(len(prob_te) * 0.01)))
        top_idx = np.argsort(prob_te)[-top_k:]
        prec_at_k = float(y_te.iloc[top_idx].mean())
        lift = prec_at_k / max(base_rate, 1e-6)
        val_precisions.append(prec_at_k)

        if _cb:
            _cb(
                f"  Walk-Forward {w_idx+1}/3: AUC={auc:.3f}, "
                f"Top{top_k} Prec={prec_at_k:.1%} "
                f"(Lift {lift:.1f}x vs 기저 {base_rate:.2%})",
                None,
            )

    # 전체 데이터로 최종 학습
    clf = _make_hgb()
    clf.fit(X, y, sample_weight=sample_weight)

    train_probs = clf.predict_proba(X)[:, 1]
    pos_rate = float(y.mean())

    prob_pct95 = float(np.percentile(train_probs, 95))
    prob_pct98 = float(np.percentile(train_probs, 98))
    prob_pct99 = float(np.percentile(train_probs, 99))

    importances = {}
    if hasattr(clf, "feature_importances_"):
        for i, col in enumerate(X.columns):
            importances[col] = float(clf.feature_importances_[i])
    else:
        for col in X.columns:
            valid = X[col].notna()
            if valid.sum() > 100:
                importances[col] = abs(float(X.loc[valid, col].corr(y[valid])))
            else:
                importances[col] = 0.0
    total_imp = sum(importances.values())
    if total_imp > 0:
        importances = {k: v / total_imp for k, v in importances.items()}

    mean_auc = float(np.mean(val_aucs)) if val_aucs else 0.0

    if _cb:
        _cb(f"  Walk-Forward 평균: AUC={mean_auc:.3f}", None)

    return {
        "type": "sklearn_hgb",
        "model": clf,
        "model_type": "accum_hgb",
        "feature_importances": importances,
        "validation_aucs": val_aucs,
        "validation_auc": mean_auc,
        "pos_rate": pos_rate,
        "prob_pct95": prob_pct95,
        "prob_pct98": prob_pct98,
        "prob_pct99": prob_pct99,
    }


def _train_numpy_fallback(X, y):
    """numpy 전용 fallback: 양성 프로필 기반 가중 스코어링."""
    pos_mask = y == 1
    pos_mean = X[pos_mask].mean()
    pos_std = X[pos_mask].std().replace(0, 1)
    all_mean = X.mean()
    all_std = X.std().replace(0, 1)

    # 피처 가중치: 양성 평균과 전체 평균의 표준화 차이
    weights = ((pos_mean - all_mean) / all_std).abs()
    weights = weights / weights.sum()  # 정규화

    # 양성 방향: 양성 평균이 전체보다 높으면 +, 낮으면 -
    direction = np.sign(pos_mean - all_mean)

    # ── 학습 데이터 score 분포 산출 (예측 시 보정용) ──
    cols = list(X.columns)
    X_norm = (X[cols] - all_mean[cols]) / all_std[cols].replace(0, 1)
    weighted_z = X_norm * direction[cols] * weights[cols]
    train_scores = weighted_z.sum(axis=1)

    score_mean = float(train_scores.mean())
    score_std = float(train_scores.std())
    if score_std < 1e-6:
        score_std = 1.0

    pos_rate = float(pos_mask.mean())

    return {
        "type": "numpy_fallback",
        "pos_mean": pos_mean.to_dict(),
        "pos_std": pos_std.to_dict(),
        "all_mean": all_mean.to_dict(),
        "all_std": all_std.to_dict(),
        "weights": weights.to_dict(),
        "direction": direction.to_dict(),
        "feature_importances": weights.to_dict(),
        "validation_auc": 0.0,  # fallback은 AUC 미산출
        "validation_aucs": [],
        "score_mean": score_mean,
        "score_std": score_std,
        "pos_rate": pos_rate,
    }


def _predict_sklearn(model_data, X):
    """sklearn 모델로 확률 예측 (v6.0).

    HGB: raw probability 직접 사용 (백분위 기반 임계값과 연동).
    RF (legacy): z-score sigmoid 보정.
    """
    clf = model_data["model"]
    raw_prob = clf.predict_proba(X)[:, 1]

    # HGB는 raw probability 사용 (sigmoid 보정 불필요)
    # → predict_current에서 백분위 기반 임계값으로 판정
    if model_data.get("model_type") == "sklearn_hgb":
        return raw_prob

    # Legacy RF: 학습 분포 기반 sigmoid 보정
    prob_mean = model_data.get("prob_mean")
    prob_std = model_data.get("prob_std")
    pos_rate = model_data.get("pos_rate")

    if prob_mean is not None and prob_std is not None and pos_rate is not None:
        z = (raw_prob - prob_mean) / max(prob_std, 1e-6)
        bias = np.log(max(pos_rate, 1e-6) / max(1 - pos_rate, 1e-6))
        calibrated = 1 / (1 + np.exp(-(z + bias)))
        return calibrated

    return raw_prob


def _predict_numpy_fallback(model_data, X):
    """numpy fallback: 양성 프로필 유사도 점수 (보정된 확률)."""
    weights = pd.Series(model_data["weights"])
    direction = pd.Series(model_data["direction"])
    all_mean = pd.Series(model_data["all_mean"])
    all_std = pd.Series(model_data["all_std"])

    # 공통 피처만
    cols = [c for c in X.columns if c in weights.index]
    if not cols:
        return np.zeros(len(X))

    # 각 피처의 z-score (전체 기준)
    X_norm = (X[cols] - all_mean[cols]) / all_std[cols].replace(0, 1)

    # 방향 가중 합산: 양성 방향이면 z-score 그대로, 반대면 부호 반전
    dir_vals = direction[cols]
    weighted_z = X_norm * dir_vals * weights[cols]
    raw_score = weighted_z.sum(axis=1)

    # ── 학습 분포 기반 보정된 sigmoid ──
    # raw_score를 학습 데이터 분포 기준 z-score로 변환
    score_mean = model_data.get("score_mean", 0)
    score_std = model_data.get("score_std", 1)
    pos_rate = model_data.get("pos_rate", 0.05)

    z = (raw_score - score_mean) / max(score_std, 1e-6)

    # 보정 bias: 평균 종목(z=0) → 양성비율과 동일한 확률 출력
    # logit(pos_rate) = log(pos_rate / (1 - pos_rate))
    # 예: pos_rate=5% → bias=-2.94 → sigmoid(0 + (-2.94)) = 0.05
    bias = np.log(max(pos_rate, 1e-6) / max(1 - pos_rate, 1e-6))
    prob = 1 / (1 + np.exp(-(z + bias)))

    return prob.values


# ──────────────────────────────────────────────
# 4-b. LSTM 시계열 모델 (PyTorch 선택적)
# ──────────────────────────────────────────────


def _try_import_torch():
    """PyTorch 사용 가능 여부 확인."""
    try:
        import torch
        import torch.nn as nn

        return True, {"torch": torch, "nn": nn}
    except ImportError:
        return False, None


def _build_lstm_model_v7(torch_modules, input_dim=50, hidden_dim=64, num_layers=2):
    """v7.0: 양방향 LSTM + Attention 모델."""
    nn = torch_modules["nn"]
    torch = torch_modules["torch"]

    class PreSurgeSequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.3 if num_layers > 1 else 0,
                bidirectional=True,
            )
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
            attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
            context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)
            return self.fc(context).squeeze(-1)

    return PreSurgeSequenceModel()


def _prepare_sequences(features_df, labels, seq_len=30):
    """시계열 데이터를 LSTM 입력용 시퀀스로 변환.

    Returns:
        X_seq: (N-seq_len+1, seq_len, F) numpy array
        y_seq: (N-seq_len+1,) numpy array
    """
    X = features_df.values.astype(np.float32)
    y = labels.values.astype(np.float32)

    sequences = []
    targets = []
    for i in range(seq_len - 1, len(X)):
        sequences.append(X[i - seq_len + 1 : i + 1])
        targets.append(y[i])

    return np.array(sequences), np.array(targets)


def _get_device(torch_modules):
    """CUDA 사용 가능 시 GPU, 아니면 CPU 반환."""
    torch = torch_modules["torch"]
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_lstm_v7(X_seq, y_seq, torch_modules, epochs=50, lr=0.001):
    """v7/v8: 양방향 LSTM + Attention 학습 (GPU 자동 감지)."""
    torch = torch_modules["torch"]
    nn = torch_modules["nn"]
    device = _get_device(torch_modules)

    mean = X_seq.mean(axis=(0, 1))
    std = X_seq.std(axis=(0, 1))
    std[std < 1e-8] = 1.0
    X_norm = (X_seq - mean) / std

    split = int(len(X_norm) * 0.8)
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_w = n_neg / max(n_pos, 1)

    input_dim = X_seq.shape[2]
    model = _build_lstm_model_v7(torch_modules, input_dim=input_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    batch_size = min(512, len(X_train))
    best_val_loss = float("inf")
    patience = 7
    no_improve = 0
    best_state = None

    for _epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_t))
        for start in range(0, len(X_train_t), batch_size):
            idx = indices[start : start + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            pred = model(xb)
            weight = torch.where(yb == 1, pos_w, 1.0)
            loss = nn.functional.binary_cross_entropy(pred, yb, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 배치 단위 validation (대량 데이터 OOM 방지)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vs in range(0, len(X_val_t), batch_size):
                xv = X_val_t[vs : vs + batch_size].to(device)
                yv = y_val_t[vs : vs + batch_size].to(device)
                vp = model(xv)
                vl = nn.functional.binary_cross_entropy(vp, yv).item()
                val_losses.append(vl * len(xv))
            val_loss = sum(val_losses) / len(X_val_t)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    val_auc = 0.0
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    # 배치 단위 val 예측
    all_val_probs = []
    with torch.no_grad():
        for vs in range(0, len(X_val_t), batch_size):
            xv = X_val_t[vs : vs + batch_size].to(device)
            vp = model(xv).cpu().numpy()
            all_val_probs.append(vp)
    val_probs = np.concatenate(all_val_probs)

    if len(np.unique(y_val)) >= 2:
        try:
            from sklearn.metrics import roc_auc_score

            val_auc = roc_auc_score(y_val, val_probs)
        except Exception:
            pass

    return {
        "state_dict": best_state or {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler_mean": mean,
        "scaler_std": std,
        "input_dim": input_dim,
        "val_auc": val_auc,
        "version": "v8_bilstm_attention",
    }


def _predict_lstm_v7(lstm_data, feature_sequence, torch_modules):
    """v7/v8 LSTM 예측. feature_sequence: (seq_len, n_features)."""
    torch = torch_modules["torch"]
    device = _get_device(torch_modules)

    mean = lstm_data["scaler_mean"]
    std = lstm_data["scaler_std"]
    X_norm = (feature_sequence - mean) / std

    X_t = torch.tensor(X_norm[np.newaxis], dtype=torch.float32).to(device)

    model = _build_lstm_model_v7(torch_modules, input_dim=lstm_data["input_dim"])
    model.load_state_dict(lstm_data["state_dict"])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        prob = model(X_t).cpu().item()

    return prob


def _ensemble_predict_v7(hgb_prob, lstm_prob):
    """v7.0: HGB 60% + LSTM 40%."""
    return 0.6 * hgb_prob + 0.4 * lstm_prob


# ──────────────────────────────────────────────
# 5. 학습 파이프라인
# ──────────────────────────────────────────────


def train_model(
    universe_df: pd.DataFrame, date_str: str, progress_callback=None
) -> dict:
    """AI 매수패턴 학습 (v6.0: Initiation-Point-Only + HistGradientBoosting).

    1. 시가총액≥700억, 0<PER≤12, 0<PBR≤2 필터링
    2. 각 종목 2년 OHLCV → 30개 피처 + 시작점 라벨 (5중 필터 + cooldown)
    3. 패턴 분석 (상관계수, 프로필)
    4. 모델 학습 (HistGradientBoosting 또는 fallback)
    """
    from . import data_fetcher

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    # ── 종목 필터링 ──
    _cb("종목 필터링...", 1)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    per = universe_df.get("PER", pd.Series(np.nan, index=universe_df.index))
    pbr = universe_df.get("PBR", pd.Series(np.nan, index=universe_df.index))

    mcap = mcap.fillna(0)
    per = per.fillna(0)
    pbr = pbr.fillna(0)

    # ROE = PBR / PER (PER>0 일 때만 유효)
    roe = pbr / per.replace(0, np.nan)

    mask = (
        (mcap >= 70_000_000_000)  # 700억
        & (per > 0)
        & (per <= 12)
        & (pbr > 0)
        & (pbr <= 2.0)
        & (roe.fillna(-999) > 0)  # ROE > 0 (적자기업 제외)
    )
    tickers = universe_df[mask].index.tolist()
    _cb(f"학습 대상: {len(tickers)}종목 (시총≥700억, PER≤12, PBR≤2, ROE>0)", 3)

    if len(tickers) < 10:
        raise ValueError(f"학습 대상 종목 부족: {len(tickers)}개 (최소 10개)")

    # ── 데이터 수집 + 피처/라벨 계산 ──
    all_X = []
    all_y_bin = []
    all_y_ret = []
    n_fetched = 0
    n_failed = 0

    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            pct = 3 + int(67 * i / len(tickers))
            _cb(
                f"2년 OHLCV + 피처 계산: {i+1}/{len(tickers)} "
                f"(수집완료 {n_fetched}, 실패 {n_failed})",
                pct,
            )

        try:
            ohlcv_2y = data_fetcher.fetch_ohlcv_2year(ticker, date_str)
        except Exception:
            n_failed += 1
            continue

        if ohlcv_2y is None or len(ohlcv_2y) < 100:
            n_failed += 1
            continue

        feats = compute_ml_features(ohlcv_2y)
        if feats.empty:
            n_failed += 1
            continue

        y_bin, y_ret = compute_initiation_labels(ohlcv_2y)

        # 유효한 행만 (피처 전부 존재 + 라벨 존재)
        valid = feats.notna().all(axis=1) & y_bin.notna()
        if valid.sum() < 30:
            n_failed += 1
            continue

        all_X.append(feats[valid])
        all_y_bin.append(y_bin[valid])
        all_y_ret.append(y_ret[valid])
        n_fetched += 1

    _cb(f"데이터 수집 완료: {n_fetched}종목, {n_failed}종목 실패", 70)

    if n_fetched < 10:
        raise ValueError(f"유효 데이터 부족: {n_fetched}종목 (최소 10개)")

    X = pd.concat(all_X, ignore_index=True)
    y_binary = pd.concat(all_y_bin, ignore_index=True).astype(int)
    y_return = pd.concat(all_y_ret, ignore_index=True)

    n_total = len(X)
    n_positive = int(y_binary.sum())
    _cb(
        f"학습 데이터: {n_total:,}건, 양성(종가30%↑ 시작점) {n_positive:,}건 "
        f"({n_positive/n_total*100:.1f}%)",
        72,
    )

    if n_positive < 30:
        raise ValueError(f"양성 사례 부족: {n_positive}건 (최소 30건)")

    # ── 패턴 분석 ──
    _cb("패턴 분석 (상관계수, 프로필)...", 75)
    pattern_analysis = analyze_patterns(X, y_binary, y_return)

    # ── 모델 학습 ──
    _cb("모델 학습...", 80)
    sklearn_available, sk = _try_import_sklearn()

    if sklearn_available:
        _cb("sklearn HistGradientBoosting 학습...", 82)
        model_result = _train_sklearn(X, y_binary, sk)
    else:
        _cb("sklearn 미설치 → numpy fallback 학습...", 82)
        warnings.warn("sklearn 미설치. pip install scikit-learn 권장.", stacklevel=2)
        model_result = _train_numpy_fallback(X, y_binary)

    auc = model_result.get("validation_auc", 0)
    _cb(f"학습 완료 (검증 AUC: {auc:.3f})", 95)

    # ── 결과 조합 ──
    result = {
        "model_type": model_result["type"],
        "model": model_result.get("model"),  # sklearn만
        "feature_names": list(X.columns),
        "feature_importances": model_result.get("feature_importances", {}),
        "pattern_analysis": pattern_analysis,
        "validation_auc": auc,
        "validation_aucs": model_result.get("validation_aucs", []),
        "n_samples": n_total,
        "n_positive": n_positive,
        "n_stocks": n_fetched,
        "train_date": date_str,
    }

    # 확률 보정 데이터 (sklearn/numpy 공통)
    for key in (
        "prob_mean",
        "prob_std",
        "pos_rate",
        "score_mean",
        "score_std",
        "prob_pct95",
        "prob_pct98",
        "prob_pct99",
    ):
        if key in model_result:
            result[key] = model_result[key]

    # numpy fallback 전용 데이터
    if model_result["type"] == "numpy_fallback":
        for key in (
            "pos_mean",
            "pos_std",
            "all_mean",
            "all_std",
            "weights",
            "direction",
        ):
            result[key] = model_result[key]

    return result


def train_model_precise(
    universe_df: pd.DataFrame, date_str: str, progress_callback=None
) -> dict:
    """AI 매수패턴 정밀학습 v8.0: Base Breakout.

    v8.0 핵심 변경 (v7.0 대비):
    - 라벨링: 급등 이벤트 → 시작점(로컬 최저점) 역추적 (0.3-0.7% 양성)
    - 피처: 58개 (v7 50개 + 바닥형성/상대강도/거래량 8개)
    - 데이터: 3년 × 확대 유니버스 + KOSPI 지수
    - 모델: HGB (depth=6, leaf=50) + BiLSTM+Attention 앙상블 (55:45)
    - Hard Negative: 급등 피크 직후 20일 (1.5x 가중치)
    - 검증: Walk-Forward
    """
    import importlib

    from . import data_fetcher

    importlib.reload(data_fetcher)

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    # ── 종목 필터링 (확대된 유니버스) ──
    _cb("v8.0 종목 필터링 (확대 유니버스)...", 1)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    per = universe_df.get("PER", pd.Series(np.nan, index=universe_df.index))
    pbr = universe_df.get("PBR", pd.Series(np.nan, index=universe_df.index))
    mcap = mcap.fillna(0)
    per = per.fillna(0)
    pbr = pbr.fillna(0)
    roe = pbr / per.replace(0, np.nan)

    mask = (
        (mcap >= 50_000_000_000)  # 500억
        & (per > 0)
        & (per <= 20)  # PER≤20
        & (pbr > 0)
        & (pbr <= 3.0)  # PBR≤3
        & (roe.fillna(-999) > -0.05)  # ROE>-5% (턴어라운드 포함)
    )
    tickers = universe_df[mask].index.tolist()
    _cb(f"학습 대상: {len(tickers)}종목 (시총≥500억, PER≤20, PBR≤3)", 3)

    if len(tickers) < 10:
        raise ValueError(f"학습 대상 종목 부족: {len(tickers)}개 (최소 10개)")

    # PyTorch 사용 가능 여부
    torch_available, torch_modules = _try_import_torch()
    seq_len = 40  # v8: 40일 시퀀스 (바닥 형성 기간 포착)

    # ── KOSPI 지수 수집 (시장 대비 상대강도 피처용) ──
    _cb("KOSPI 지수 데이터 수집...", 4)
    kospi_close = None
    try:
        kospi_df = data_fetcher.fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
            _cb(f"KOSPI 지수: {len(kospi_close)}일 수집 완료", 5)
    except Exception as e:
        _cb(f"KOSPI 지수 수집 실패 ({e}), RS 피처 0으로 대체", 5)

    # ── 3년 데이터 수집 + v8 피처/라벨 계산 ──
    all_X = []
    all_y_bin = []
    all_y_ret = []
    all_hard_neg = []
    all_sequences = []  # LSTM용
    n_fetched = 0
    n_failed = 0

    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            pct = 5 + int(50 * i / len(tickers))
            _cb(
                f"v8 3년 OHLCV + 피처: {i+1}/{len(tickers)} "
                f"(완료 {n_fetched}, 실패 {n_failed})",
                pct,
            )

        try:
            ohlcv = data_fetcher.fetch_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 200:
                ohlcv = data_fetcher.fetch_ohlcv_2year(ticker, date_str)
        except Exception:
            n_failed += 1
            continue

        if ohlcv is None or len(ohlcv) < 100:
            n_failed += 1
            continue

        # v8: 58개 피처 (KOSPI 상대강도 포함)
        feats = compute_ml_features_v8(ohlcv, kospi_close)
        if feats.empty:
            n_failed += 1
            continue

        # v8: Base Breakout 라벨링 (Hard Negative 포함)
        y_bin, y_ret, hard_neg = compute_base_breakout_labels(ohlcv)

        valid = feats.notna().all(axis=1) & y_bin.notna()
        if valid.sum() < 30:
            n_failed += 1
            continue

        all_X.append(feats[valid])
        all_y_bin.append(y_bin[valid])
        all_y_ret.append(y_ret[valid])
        all_hard_neg.append(hard_neg[valid])

        # LSTM 시퀀스 수집
        if torch_available:
            valid_feats = feats[valid]
            valid_labels = y_bin[valid]
            if len(valid_feats) >= seq_len:
                seqs, seq_labels = _prepare_sequences(
                    valid_feats, valid_labels, seq_len
                )
                all_sequences.append((seqs, seq_labels))

        n_fetched += 1

    _cb(f"데이터 수집 완료: {n_fetched}종목, {n_failed}종목 실패", 55)

    if n_fetched < 10:
        raise ValueError(f"유효 데이터 부족: {n_fetched}종목 (최소 10개)")

    X = pd.concat(all_X, ignore_index=True)
    y_binary = pd.concat(all_y_bin, ignore_index=True).astype(int)
    y_return = pd.concat(all_y_ret, ignore_index=True)
    hard_neg_all = pd.concat(all_hard_neg, ignore_index=True)

    n_total = len(X)
    n_positive = int(y_binary.sum())
    n_hard_neg = int((hard_neg_all > 0).sum())
    _cb(
        f"학습 데이터: {n_total:,}건, 양성(바닥→25%↑) {n_positive:,}건 "
        f"({n_positive/n_total*100:.2f}%), "
        f"Hard부정(피크후) {n_hard_neg:,}건",
        58,
    )

    if n_positive < 20:
        raise ValueError(f"양성 사례 부족: {n_positive}건 (최소 20건)")

    # ── 패턴 분석 ──
    _cb("패턴 분석...", 60)
    pattern_analysis = analyze_patterns(X, y_binary, y_return)

    # ── HGB 학습 (v8: 58개 피처 + Hard Negative 가중치 1.5x) ──
    _cb("v8 HGB 모델 학습 (Hard Negative 강조)...", 65)
    sklearn_available, sk = _try_import_sklearn()

    if sklearn_available:
        model_result = _train_sklearn_v8(X, y_binary, hard_neg_all, sk, _cb)
    else:
        warnings.warn("sklearn 미설치. pip install scikit-learn 권장.", stacklevel=2)
        model_result = _train_numpy_fallback(X, y_binary)

    auc_hgb = model_result.get("validation_auc", 0)
    _cb(f"HGB 학습 완료 (AUC: {auc_hgb:.3f})", 80)

    # ── LSTM 학습 (선택적) ──
    lstm_data = None
    if torch_available and all_sequences:
        _cb("v8 BiLSTM+Attention 학습...", 82)
        try:
            X_all_seq = np.concatenate([s[0] for s in all_sequences], axis=0)
            y_all_seq = np.concatenate([s[1] for s in all_sequences], axis=0)

            n_seq_total = len(y_all_seq)
            n_seq_pos = int(y_all_seq.sum())

            # ★ 메모리/속도 보호: 층화 서브샘플링 ★
            MAX_SEQ = 50_000
            if n_seq_total > MAX_SEQ and n_seq_pos >= 10:
                _cb(
                    f"  LSTM 시퀀스 {n_seq_total:,}개 → {MAX_SEQ:,}개 서브샘플링...",
                    None,
                )
                rng = np.random.RandomState(42)
                pos_idx = np.where(y_all_seq == 1)[0]
                neg_idx = np.where(y_all_seq == 0)[0]
                n_neg_sample = min(MAX_SEQ - len(pos_idx), len(neg_idx))
                neg_sampled = rng.choice(neg_idx, size=n_neg_sample, replace=False)
                keep_idx = np.sort(np.concatenate([pos_idx, neg_sampled]))
                X_all_seq = X_all_seq[keep_idx]
                y_all_seq = y_all_seq[keep_idx]
                _cb(
                    f"  서브샘플링 완료: {len(y_all_seq):,}개 "
                    f"(양성 {int(y_all_seq.sum()):,}개 전수 포함)",
                    None,
                )

            if n_seq_pos >= 10:
                lstm_data = _train_lstm_v7(X_all_seq, y_all_seq, torch_modules)
                _cb(f"BiLSTM 학습 완료 (val AUC: {lstm_data['val_auc']:.3f})", 88)

                # ★ 앙상블 임계값 재보정 (v8: 55:45) ★
                _cb("앙상블 임계값 보정 중...", 89)
                try:
                    torch = torch_modules["torch"]
                    device = _get_device(torch_modules)
                    lstm_model = _build_lstm_model_v7(
                        torch_modules, input_dim=lstm_data["input_dim"]
                    )
                    lstm_model.load_state_dict(lstm_data["state_dict"])
                    lstm_model = lstm_model.to(device)
                    lstm_model.eval()

                    n_cal = min(5000, len(X_all_seq))
                    cal_rng = np.random.RandomState(42)
                    cal_idx = cal_rng.choice(len(X_all_seq), size=n_cal, replace=False)
                    X_cal = X_all_seq[cal_idx]

                    sc_mean = lstm_data["scaler_mean"]
                    sc_std = lstm_data["scaler_std"]
                    X_norm = (X_cal - sc_mean) / sc_std
                    # 배치 단위 LSTM 예측
                    cal_probs = []
                    cal_bs = 512
                    with torch.no_grad():
                        for cs in range(0, len(X_norm), cal_bs):
                            xc = torch.tensor(
                                X_norm[cs : cs + cal_bs], dtype=torch.float32
                            ).to(device)
                            cp = lstm_model(xc).cpu().numpy()
                            cal_probs.append(cp)
                    lstm_probs_cal = np.concatenate(cal_probs)

                    feat_names = list(X.columns)
                    last_rows = pd.DataFrame(X_cal[:, -1, :], columns=feat_names)
                    hgb_clf = model_result["model"]
                    hgb_probs_cal = hgb_clf.predict_proba(last_rows)[:, 1]

                    # v8: 앙상블 55:45
                    ens_probs = 0.55 * hgb_probs_cal + 0.45 * lstm_probs_cal

                    model_result["prob_pct95"] = float(np.percentile(ens_probs, 95))
                    model_result["prob_pct98"] = float(np.percentile(ens_probs, 98))
                    model_result["prob_pct99"] = float(np.percentile(ens_probs, 99))

                    _cb(
                        f"  앙상블 임계값: "
                        f"Watch≥{model_result['prob_pct95']:.3f}, "
                        f"Buy≥{model_result['prob_pct98']:.3f}, "
                        f"Strong≥{model_result['prob_pct99']:.3f}",
                        None,
                    )
                    del lstm_model, X_cal, X_norm
                except Exception as e:
                    _cb(f"  앙상블 보정 실패 ({e}), HGB 임계값 사용", None)

                _cb("LSTM+앙상블 보정 완료", 90)
            else:
                _cb("LSTM 양성 시퀀스 부족 → HGB only", 90)
        except Exception as e:
            _cb(f"LSTM 학습 실패: {e} → HGB only", 90)
    elif not torch_available:
        _cb("PyTorch 미설치 → HGB only", 90)

    _cb("결과 집계...", 92)

    # ── 결과 조합 ──
    model_type = "v8_ensemble" if lstm_data else "v8_hgb"

    result = {
        "model_type": model_type,
        "model": model_result.get("model"),
        "feature_names": list(X.columns),
        "feature_importances": model_result.get("feature_importances", {}),
        "pattern_analysis": pattern_analysis,
        "validation_auc": auc_hgb,
        "validation_aucs": model_result.get("validation_aucs", []),
        "n_samples": n_total,
        "n_positive": n_positive,
        "n_hard_neg": n_hard_neg,
        "n_stocks": n_fetched,
        "train_date": date_str,
        "lstm_data": lstm_data,
        "version": "v8.0",
    }

    for key in (
        "prob_mean",
        "prob_std",
        "pos_rate",
        "score_mean",
        "score_std",
        "prob_pct95",
        "prob_pct98",
        "prob_pct99",
    ):
        if key in model_result:
            result[key] = model_result[key]

    if model_result["type"] == "numpy_fallback":
        for key in (
            "pos_mean",
            "pos_std",
            "all_mean",
            "all_std",
            "weights",
            "direction",
        ):
            result[key] = model_result[key]

    _cb(f"정밀학습 v8.0 완료! (모델: {model_type}, AUC: {auc_hgb:.3f})", 100)
    return result


def train_model_surge(
    universe_df: pd.DataFrame, date_str: str, progress_callback=None
) -> dict:
    """AI_급등주 v2 학습: Daily Triple-Barrier + LightGBM.

    v2 핵심 변경:
    - 라벨링: 모든 거래일 Triple-Barrier (+20%/-8%/15일)
    - 피처: 55개 (6개 카테고리)
    - 모델: LightGBM (primary) + HGB (fallback)
    - 검증: Walk-Forward 4-window + 5일 embargo
    - 유니버스: 시총 ≥ 1000억 + 거래대금 ≥ 30억
    """
    import importlib

    from . import data_fetcher

    importlib.reload(data_fetcher)

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    # ── 종목 필터링 ──
    _cb("급등주 v2 종목 필터링 (시총≥1000억)...", 1)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    mcap = mcap.fillna(0)
    mask = mcap >= 100_000_000_000  # 1000억
    tickers = universe_df[mask].index.tolist()
    _cb(f"학습 대상: {len(tickers)}종목 (시총≥1000억)", 3)

    if len(tickers) < 10:
        raise ValueError(f"학습 대상 종목 부족: {len(tickers)}개 (최소 10개)")

    # ── KOSPI 지수 수집 ──
    _cb("KOSPI 지수 데이터 수집...", 4)
    kospi_close = None
    try:
        kospi_df = data_fetcher.fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
            _cb(f"  KOSPI 지수: {len(kospi_df)}일")
    except Exception as e:
        _cb(f"  KOSPI 수집 실패 (무시): {e}")

    # ── 2년 OHLCV + 피처/라벨 생성 ──
    _cb("2년 OHLCV 수집 + Triple-Barrier 라벨링...", 5)

    all_X, all_y, all_hard, all_ret = [], [], [], []
    done = 0
    total = len(tickers)
    n_skipped_tv = 0  # 거래대금 부족 스킵

    for ticker in tickers:
        done += 1
        if done % 50 == 0 or done == total:
            pct = 5 + int(done / total * 55)
            _cb(f"  [{done}/{total}] 종목 처리 중...", pct)
        try:
            ohlcv = data_fetcher.fetch_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 200:
                continue

            # 거래대금 필터 (30억 이상)
            if "거래대금" in ohlcv.columns:
                _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
            else:
                _avg_tv = (
                    (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                    .iloc[-20:]
                    .mean()
                )
            if _avg_tv < 3_000_000_000:
                n_skipped_tv += 1
                continue

            lbl, fwd_ret, hard = compute_breakout_labels(ohlcv)
            feats = compute_ml_features_surge(ohlcv, kospi_close)
            if feats.empty:
                continue

            # 피처 컬럼 정렬
            for col in SURGE_FEATURE_NAMES:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[SURGE_FEATURE_NAMES]

            # 유효 행 (라벨 ≠ NaN)
            valid = lbl.notna()
            if valid.sum() < 10:
                continue

            all_X.append(feats[valid])
            all_y.append(lbl[valid])
            all_hard.append(hard[valid])
            all_ret.append(fwd_ret[valid])
        except Exception:
            continue

    if not all_X:
        raise ValueError("피처/라벨 생성 실패: 유효 종목 없음")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True).values
    hard_neg_arr = pd.concat(all_hard, ignore_index=True).values

    n_pos = int((y == 1).sum())
    n_hard = int(hard_neg_arr.sum())
    n_total = len(y)
    pos_rate = n_pos / n_total if n_total > 0 else 0
    _cb(
        f"전체: {n_total:,}건 | 양성: {n_pos:,}건 ({pos_rate:.1%}) | "
        f"Hard Neg: {n_hard:,}건 | 거래대금스킵: {n_skipped_tv}",
        62,
    )

    if n_pos < 20:
        raise ValueError(f"양성 샘플 부족: {n_pos}건 (최소 20건)")

    # ── 모델 학습 (LightGBM primary, HGB fallback) ──
    _cb("LightGBM/HGB 학습 + Walk-Forward 4-window 검증...", 65)
    _sk_ok, sk = _try_import_sklearn()
    if not _sk_ok:
        raise ImportError("sklearn 필요: pip install scikit-learn")

    model_data = _train_sklearn_surge(X, y, hard_neg_arr, sk, _cb)

    # ── 결과 업데이트 ──
    model_data.update(
        {
            "train_date": date_str,
            "n_tickers": done - n_skipped_tv,
            "n_samples": n_total,
            "n_positive": n_pos,
            "n_hard_neg": n_hard,
            "pos_rate": pos_rate,
            "feature_names": SURGE_FEATURE_NAMES,
            "validation_auc": model_data.get(
                "wf_avg_auc", model_data.get("train_auc", 0)
            ),
        }
    )

    engine = "LightGBM" if model_data.get("use_lgbm") else "HGB"
    _cb(
        f"급등주 v2 학습 완료: {engine} | AUC={model_data['validation_auc']:.3f} | "
        f"양성={n_pos:,}건 ({pos_rate:.1%})",
        100,
    )

    return model_data


def train_model_accum(
    universe_df: pd.DataFrame, date_str: str, progress_callback=None
) -> dict:
    """AI_매집주 학습: 매집 완료 직전 종목 탐지 (HGB + LSTM 앙상블).

    설계계획 기반:
    - 라벨링: 급등→역추적→매집기간 감지 (Outcome-First)
    - 피처: 43개 (기본 30 + 매집 전용 13)
    - 모델: HGB + BiLSTM+Attention 앙상블 (55:45)
    - 유니버스: 시총 ≥ 1000억 + 거래대금 ≥ 70억
    - 기간: 2년
    """
    import importlib

    from . import data_fetcher

    importlib.reload(data_fetcher)

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    # ── 종목 필터링 (시총≥1000억) ──
    _cb("매집주 종목 필터링 (시총≥1000억)...", 1)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    mcap = mcap.fillna(0)
    mask = mcap >= 100_000_000_000  # 1000억
    tickers_all = universe_df[mask].index.tolist()
    _cb(f"학습 대상: {len(tickers_all)}종목 (시총≥1000억)", 3)

    if len(tickers_all) < 10:
        raise ValueError(f"학습 대상 종목 부족: {len(tickers_all)}개 (최소 10개)")

    # PyTorch 사용 가능 여부
    torch_available, torch_modules = _try_import_torch()
    seq_len = 40  # 매집 기간 포착용 40일 시퀀스

    # ── KOSPI 지수 수집 ──
    _cb("KOSPI 지수 수집...", 4)
    kospi_close = None
    try:
        kospi_df = data_fetcher.fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
            _cb(f"  KOSPI 지수: {len(kospi_df)}일")
    except Exception as e:
        _cb(f"  KOSPI 수집 실패 (무시): {e}")

    # ── 2년 OHLCV 수집 + 피처/라벨 생성 ──
    _cb("2년 OHLCV 수집 + 매집 피처/라벨 생성...", 5)

    all_X, all_y_bin, all_y_ret, all_hard_neg = [], [], [], []
    all_sequences = []  # LSTM용
    n_fetched = 0
    n_failed = 0
    total = len(tickers_all)

    for i, ticker in enumerate(tickers_all):
        if i % 20 == 0:
            pct = 5 + int(50 * i / total)
            _cb(
                f"  [{i+1}/{total}] 종목 처리 중 "
                f"(완료 {n_fetched}, 실패 {n_failed})",
                pct,
            )

        try:
            ohlcv = data_fetcher.fetch_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 200:
                ohlcv = data_fetcher.fetch_ohlcv_2year(ticker, date_str)
        except Exception:
            n_failed += 1
            continue

        if ohlcv is None or len(ohlcv) < 100:
            n_failed += 1
            continue

        # 거래대금 필터 (70억 이상)
        if "거래대금" in ohlcv.columns:
            _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
        else:
            _avg_tv = (
                (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                .iloc[-20:]
                .mean()
            )
        if _avg_tv < 7_000_000_000:
            n_failed += 1
            continue

        # 매집 라벨링
        lbl, fwd_ret, hard = compute_accumulation_labels(ohlcv)
        # 매집 피처 (43개)
        feats = compute_ml_features_accum(ohlcv, kospi_close)
        if feats.empty:
            n_failed += 1
            continue

        # 피처 컬럼 정렬
        for col in ACCUM_FEATURE_NAMES:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[ACCUM_FEATURE_NAMES]

        # 유효 행 = 라벨이 NaN이 아닌 행
        valid = feats.notna().all(axis=1) & lbl.notna()
        if valid.sum() < 10:
            n_failed += 1
            continue

        all_X.append(feats[valid])
        all_y_bin.append(lbl[valid])
        all_y_ret.append(fwd_ret[valid])
        all_hard_neg.append(hard[valid])

        # LSTM 시퀀스 수집
        if torch_available:
            valid_feats = feats[valid]
            valid_labels = lbl[valid]
            if len(valid_feats) >= seq_len:
                seqs, seq_labels = _prepare_sequences(
                    valid_feats, valid_labels, seq_len
                )
                all_sequences.append((seqs, seq_labels))

        n_fetched += 1

    _cb(f"데이터 수집 완료: {n_fetched}종목, {n_failed}종목 실패", 55)

    if n_fetched < 10:
        raise ValueError(f"유효 데이터 부족: {n_fetched}종목 (최소 10개)")

    X = pd.concat(all_X, ignore_index=True)
    y_binary = pd.concat(all_y_bin, ignore_index=True).astype(int)
    y_return = pd.concat(all_y_ret, ignore_index=True)
    hard_neg_all = pd.concat(all_hard_neg, ignore_index=True)

    n_total = len(X)
    n_positive = int(y_binary.sum())
    n_hard_neg = int((hard_neg_all > 0).sum())
    pos_rate = n_positive / n_total if n_total > 0 else 0
    _cb(
        f"학습 데이터: {n_total:,}건, 양성(매집완료) {n_positive:,}건 "
        f"({pos_rate:.2%}), "
        f"Hard부정(품질미달/하락) {n_hard_neg:,}건",
        58,
    )

    if n_positive < 20:
        raise ValueError(f"양성 사례 부족: {n_positive}건 (최소 20건)")

    # ── 패턴 분석 ──
    _cb("패턴 분석...", 60)
    pattern_analysis = analyze_patterns(X, y_binary, y_return)

    # ── HGB 학습 ──
    _cb("매집주 HGB 모델 학습...", 65)
    sklearn_available, sk = _try_import_sklearn()

    if sklearn_available:
        model_result = _train_sklearn_accum(X, y_binary, hard_neg_all, sk, _cb)
    else:
        warnings.warn("sklearn 미설치. pip install scikit-learn 권장.", stacklevel=2)
        model_result = _train_numpy_fallback(X, y_binary)

    auc_hgb = model_result.get("validation_auc", 0)
    _cb(f"HGB 학습 완료 (AUC: {auc_hgb:.3f})", 80)

    # ── LSTM 학습 (선택적) ──
    lstm_data = None
    if torch_available and all_sequences:
        _cb("매집주 BiLSTM+Attention 학습...", 82)
        try:
            X_all_seq = np.concatenate([s[0] for s in all_sequences], axis=0)
            y_all_seq = np.concatenate([s[1] for s in all_sequences], axis=0)

            n_seq_total = len(y_all_seq)
            n_seq_pos = int(y_all_seq.sum())

            # 메모리/속도 보호: 층화 서브샘플링
            MAX_SEQ = 50_000
            if n_seq_total > MAX_SEQ and n_seq_pos >= 10:
                _cb(
                    f"  LSTM 시퀀스 {n_seq_total:,}개 → {MAX_SEQ:,}개 서브샘플링...",
                    None,
                )
                rng = np.random.RandomState(42)
                pos_idx = np.where(y_all_seq == 1)[0]
                neg_idx = np.where(y_all_seq == 0)[0]
                n_neg_sample = min(MAX_SEQ - len(pos_idx), len(neg_idx))
                neg_sampled = rng.choice(neg_idx, size=n_neg_sample, replace=False)
                keep_idx = np.sort(np.concatenate([pos_idx, neg_sampled]))
                X_all_seq = X_all_seq[keep_idx]
                y_all_seq = y_all_seq[keep_idx]
                _cb(
                    f"  서브샘플링 완료: {len(y_all_seq):,}개 "
                    f"(양성 {int(y_all_seq.sum()):,}개 전수 포함)",
                    None,
                )

            if n_seq_pos >= 10:
                lstm_data = _train_lstm_v7(X_all_seq, y_all_seq, torch_modules)
                _cb(f"BiLSTM 학습 완료 (val AUC: {lstm_data['val_auc']:.3f})", 88)

                # 앙상블 임계값 재보정 (55:45)
                _cb("앙상블 임계값 보정 중...", 89)
                try:
                    torch = torch_modules["torch"]
                    device = _get_device(torch_modules)
                    lstm_model = _build_lstm_model_v7(
                        torch_modules, input_dim=lstm_data["input_dim"]
                    )
                    lstm_model.load_state_dict(lstm_data["state_dict"])
                    lstm_model = lstm_model.to(device)
                    lstm_model.eval()

                    n_cal = min(5000, len(X_all_seq))
                    cal_rng = np.random.RandomState(42)
                    cal_idx = cal_rng.choice(len(X_all_seq), size=n_cal, replace=False)
                    X_cal = X_all_seq[cal_idx]

                    sc_mean = lstm_data["scaler_mean"]
                    sc_std = lstm_data["scaler_std"]
                    X_norm = (X_cal - sc_mean) / sc_std
                    cal_probs = []
                    cal_bs = 512
                    with torch.no_grad():
                        for cs in range(0, len(X_norm), cal_bs):
                            xc = torch.tensor(
                                X_norm[cs : cs + cal_bs], dtype=torch.float32
                            ).to(device)
                            cp = lstm_model(xc).cpu().numpy()
                            cal_probs.append(cp)
                    lstm_probs_cal = np.concatenate(cal_probs)

                    feat_names = list(X.columns)
                    last_rows = pd.DataFrame(X_cal[:, -1, :], columns=feat_names)
                    hgb_clf = model_result["model"]
                    hgb_probs_cal = hgb_clf.predict_proba(last_rows)[:, 1]

                    # 앙상블 55:45
                    ens_probs = 0.55 * hgb_probs_cal + 0.45 * lstm_probs_cal

                    model_result["prob_pct95"] = float(np.percentile(ens_probs, 95))
                    model_result["prob_pct98"] = float(np.percentile(ens_probs, 98))
                    model_result["prob_pct99"] = float(np.percentile(ens_probs, 99))

                    _cb(
                        f"  앙상블 임계값: "
                        f"Watch≥{model_result['prob_pct95']:.3f}, "
                        f"Buy≥{model_result['prob_pct98']:.3f}, "
                        f"Strong≥{model_result['prob_pct99']:.3f}",
                        None,
                    )
                    del lstm_model, X_cal, X_norm
                except Exception as e:
                    _cb(f"  앙상블 보정 실패 ({e}), HGB 임계값 사용", None)

                _cb("LSTM+앙상블 보정 완료", 90)
            else:
                _cb("LSTM 양성 시퀀스 부족 → HGB only", 90)
        except Exception as e:
            _cb(f"LSTM 학습 실패: {e} → HGB only", 90)
    elif not torch_available:
        _cb("PyTorch 미설치 → HGB only", 90)

    _cb("결과 집계...", 92)

    # ── 결과 조합 ──
    model_type = "accum_ensemble" if lstm_data else "accum_hgb"

    result = {
        "model_type": model_type,
        "model": model_result.get("model"),
        "feature_names": list(X.columns),
        "feature_importances": model_result.get("feature_importances", {}),
        "pattern_analysis": pattern_analysis,
        "validation_auc": auc_hgb,
        "validation_aucs": model_result.get("validation_aucs", []),
        "n_samples": n_total,
        "n_positive": n_positive,
        "n_hard_neg": n_hard_neg,
        "n_stocks": n_fetched,
        "train_date": date_str,
        "lstm_data": lstm_data,
        "version": "accum_v1",
    }

    for key in ("pos_rate", "prob_pct95", "prob_pct98", "prob_pct99"):
        if key in model_result:
            result[key] = model_result[key]

    if model_result.get("type") == "numpy_fallback":
        for key in (
            "pos_mean",
            "pos_std",
            "all_mean",
            "all_std",
            "weights",
            "direction",
            "score_mean",
            "score_std",
        ):
            if key in model_result:
                result[key] = model_result[key]

    _cb(f"매집주 학습 완료! (모델: {model_type}, AUC: {auc_hgb:.3f})", 100)
    return result


# ──────────────────────────────────────────────
# 6. 예측
# ──────────────────────────────────────────────


def predict_current(
    model_data: dict,
    ohlcv_dict: dict,
    tickers: list,
    max_spike_3d: float = 0.06,
    max_spike_1d: float = 0.05,
    max_ret_10d: float = 0.10,
    max_ret_20d: float = 0.15,
) -> pd.DataFrame:
    """현재 종목에 학습된 패턴 적용 (v6.0: 엄격한 필터 + 모델 판단).

    학습 라벨과 동일한 엄격 하드필터로 일관성 확보:
    - 직전 3일 수익률 ≥ 6% → 단기 급등 spike (배제)
    - 직전 5일 중 하루 최대 ≥ 5% → 갭 상승 (배제)
    - 10일 수익률 ≥ 10% → 이미 상승 중 (배제)
    - 20일 수익률 ≥ 15% → 이미 상승 완료 (배제)

    Args:
        model_data: train_model() 반환값
        ohlcv_dict: {ticker: ohlcv_100d_df}
        tickers: 예측할 티커 목록
        max_spike_3d: 3일 급등 상한 (기본 6%)
        max_spike_1d: 일간 급등 상한 (기본 5%)
        max_ret_10d: 10일 수익률 상한 (기본 10%)
        max_ret_20d: 20일 수익률 상한 (기본 15%)

    Returns:
        DataFrame indexed by ticker: AI_Pass, AI_Score, AI_Timing, AI_Reason
    """
    results = []

    def _fail(t, reason=""):
        return {
            "ticker": t,
            "AI_Pass": False,
            "AI_Score": 0.0,
            "AI_Timing": "",
            "AI_Reason": reason,
        }

    for ticker in tickers:
        ohlcv = ohlcv_dict.get(ticker) if ohlcv_dict else None
        if ohlcv is None or len(ohlcv) < 60:
            results.append(_fail(ticker))
            continue

        # ── 학습 라벨과 동일한 하드필터 ──
        close = ohlcv["종가"].astype(float)
        daily_ret = close.pct_change()

        ret_3d = close.iloc[-1] / close.iloc[-4] - 1 if len(close) >= 4 else 0
        max_daily_5d = daily_ret.iloc[-5:].max() if len(close) >= 5 else 0
        ret_10d = close.iloc[-1] / close.iloc[-11] - 1 if len(close) >= 11 else 0
        ret_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0

        if (
            ret_3d >= max_spike_3d
            or max_daily_5d >= max_spike_1d
            or ret_10d >= max_ret_10d
            or ret_20d >= max_ret_20d
        ):
            results.append(_fail(ticker, "이미상승"))
            continue

        feats = compute_ml_features(ohlcv)
        if feats.empty:
            results.append(_fail(ticker))
            continue

        # 마지막 행 (현재 시점) 피처
        last_row = feats.iloc[[-1]].copy()

        # NaN 피처는 0으로 채움 (안전)
        last_row = last_row.fillna(0)

        # 학습 피처와 정렬
        feat_names = model_data.get("feature_names", FEATURE_NAMES)
        for col in feat_names:
            if col not in last_row.columns:
                last_row[col] = 0
        last_row = last_row[feat_names]

        # 예측
        if model_data.get("model_type") in ("sklearn_rf", "sklearn_hgb"):
            prob = float(_predict_sklearn(model_data, last_row)[0])
        else:
            prob = float(_predict_numpy_fallback(model_data, last_row)[0])

        prob = max(0.0, min(1.0, prob))

        # 판정: 모델 타입별 임계값 (v6.0)
        if model_data.get("model_type") == "sklearn_hgb":
            # HGB: 학습 데이터 확률 백분위 기반 임계값
            # 95th → WATCH, 98th → BUY, 99th → STRONG_BUY
            thr_pass = model_data.get("prob_pct95", 0.30)
            thr_buy = model_data.get("prob_pct98", 0.45)
            thr_strong = model_data.get("prob_pct99", 0.60)
        else:
            # Legacy RF / numpy fallback: pos_rate 기반
            base_rate = model_data.get("pos_rate", 0.05)
            thr_pass = max(base_rate * 5.0, 0.12)
            thr_buy = max(base_rate * 8.0, 0.20)
            thr_strong = max(base_rate * 12.0, 0.35)

        ai_pass = prob >= thr_pass
        ai_score = prob * 100

        if prob >= thr_strong:
            ai_timing = "STRONG_BUY"
        elif prob >= thr_buy:
            ai_timing = "BUY_NOW"
        elif prob >= thr_pass:
            ai_timing = "WATCH"
        else:
            ai_timing = ""

        # 사유 태그: 상위 기여 피처
        ai_reason = _generate_ai_reason(last_row.iloc[0], model_data)

        results.append(
            {
                "ticker": ticker,
                "AI_Pass": ai_pass,
                "AI_Score": ai_score,
                "AI_Timing": ai_timing,
                "AI_Reason": ai_reason,
            }
        )

    result_df = pd.DataFrame(results).set_index("ticker")
    return result_df


def predict_current_precise(
    model_data: dict,
    ohlcv_dict: dict,
    tickers: list,
    kospi_close: pd.Series = None,
    **kwargs,
) -> pd.DataFrame:
    """v8.0: Base Breakout 예측 — 완화 하드필터 + 모델 위임 + 후처리 가드.

    v8.0 변경:
    - 58개 v8 피처 (KOSPI 상대강도 포함)
    - 하드필터: 5일 +8%, 10일 +15%, 20일 +25%
    - 후처리 가드: 60일 +30%
    - 앙상블: HGB 55% + LSTM 45%
    - seq_len: 40
    """
    results = []

    def _fail(t, reason=""):
        return {
            "ticker": t,
            "AI_Pass": False,
            "AI_Score": 0.0,
            "AI_Timing": "",
            "AI_Reason": reason,
        }

    lstm_data = model_data.get("lstm_data")
    torch_available, torch_modules = (False, None)
    if lstm_data:
        torch_available, torch_modules = _try_import_torch()

    seq_len = 40  # v8: 40일 시퀀스 (바닥 형성 포착)

    for ticker in tickers:
        ohlcv = ohlcv_dict.get(ticker) if ohlcv_dict else None
        if ohlcv is None or len(ohlcv) < 60:
            results.append(_fail(ticker))
            continue

        close = ohlcv["종가"].astype(float)
        daily_ret = close.pct_change()

        # ── v8.0 예측 하드필터 (극단적 급등만 배제) ──
        ret_10d = close.iloc[-1] / close.iloc[-11] - 1 if len(close) >= 11 else 0
        ret_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
        max_daily_5d = daily_ret.iloc[-5:].max() if len(close) >= 5 else 0

        if (
            max_daily_5d >= 0.08  # 5일내 +8% 급등일
            or ret_10d >= 0.15  # 10일 +15% 초과
            or ret_20d >= 0.25
        ):  # 20일 +25% 초과
            results.append(_fail(ticker, "이미상승"))
            continue

        # v8: 58개 피처 (KOSPI 상대강도 포함)
        feats = compute_ml_features_v8(ohlcv, kospi_close)
        if feats.empty:
            results.append(_fail(ticker))
            continue

        last_row = feats.iloc[[-1]].copy().fillna(0)

        feat_names = model_data.get("feature_names", V8_FEATURE_NAMES)
        for col in feat_names:
            if col not in last_row.columns:
                last_row[col] = 0
        last_row = last_row[feat_names]

        # HGB 예측
        model_type = model_data.get("model_type", "")
        if model_type in (
            "v8_hgb",
            "v8_ensemble",
            "v7_hgb",
            "v7_ensemble",
            "sklearn_hgb",
        ):
            hgb_prob = float(_predict_sklearn(model_data, last_row)[0])
        else:
            hgb_prob = float(_predict_numpy_fallback(model_data, last_row)[0])

        # LSTM 예측 (v8: 40일 시퀀스)
        lstm_prob = None
        if lstm_data and torch_available and len(feats) >= seq_len:
            try:
                seq = feats.iloc[-seq_len:].fillna(0)
                for col in feat_names:
                    if col not in seq.columns:
                        seq[col] = 0
                seq_arr = seq[feat_names].values.astype(np.float32)
                lstm_prob = _predict_lstm_v7(lstm_data, seq_arr, torch_modules)
            except Exception:
                lstm_prob = None

        # v8 앙상블 (55:45)
        if lstm_prob is not None:
            prob = 0.55 * hgb_prob + 0.45 * lstm_prob
        else:
            prob = hgb_prob

        prob = max(0.0, min(1.0, prob))

        # 판정 (백분위 기반)
        if model_type in (
            "v8_hgb",
            "v8_ensemble",
            "v7_hgb",
            "v7_ensemble",
            "sklearn_hgb",
        ):
            thr_pass = model_data.get("prob_pct95", 0.30)
            thr_buy = model_data.get("prob_pct98", 0.45)
            thr_strong = model_data.get("prob_pct99", 0.60)
        else:
            base_rate = model_data.get("pos_rate", 0.05)
            thr_pass = max(base_rate * 5.0, 0.12)
            thr_buy = max(base_rate * 8.0, 0.20)
            thr_strong = max(base_rate * 12.0, 0.35)

        ai_pass = prob >= thr_pass
        ai_score = prob * 100

        # ★ 후처리 가드: 60일 수익률 +30% 초과 → 최종 제거 (v7: +20%) ★
        if ai_pass:
            ret_60d = close.iloc[-1] / close.iloc[-61] - 1 if len(close) >= 61 else 0
            if ret_60d > 0.30:
                ai_pass = False
                ai_score = ai_score * 0.5

        if prob >= thr_strong:
            ai_timing = "STRONG_BUY"
        elif prob >= thr_buy:
            ai_timing = "BUY_NOW"
        elif prob >= thr_pass:
            ai_timing = "WATCH"
        else:
            ai_timing = ""

        if not ai_pass:
            ai_timing = ""

        ai_reason = _generate_ai_reason_v8(
            last_row.iloc[0], model_data, lstm_prob is not None
        )

        results.append(
            {
                "ticker": ticker,
                "AI_Pass": ai_pass,
                "AI_Score": ai_score,
                "AI_Timing": ai_timing,
                "AI_Reason": ai_reason,
            }
        )

    return pd.DataFrame(results).set_index("ticker")


def predict_current_surge(
    model_data: dict,
    ohlcv_dict: dict,
    tickers: list,
    kospi_close: pd.Series = None,
    min_avg_tv: float = 3_000_000_000,
    **kwargs,
) -> pd.DataFrame:
    """AI_급등주 v2: 전종목 스코어링 → Top 20 리포트.

    v2 핵심 변경:
    1. 하드필터 최소화 (극단 과열만 배제)
    2. 추세필터 제거 (모델이 직접 판단)
    3. Top 20 랭킹 기반 (임계값 pct95/98/99 폐기)
    4. 종목별 근거 피처 표시
    """
    results = []
    scored = []  # (ticker, prob, feats, reason)

    model = model_data.get("model")
    if model is None:
        for t in tickers:
            results.append(
                {
                    "ticker": t,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
        return pd.DataFrame(results).set_index("ticker")

    use_lgb = model_data.get("use_lgbm", False)

    for ticker in tickers:
        ohlcv = ohlcv_dict.get(ticker) if ohlcv_dict else None
        if ohlcv is None or len(ohlcv) < 60:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
            continue

        close = ohlcv["종가"].astype(float)
        volume = ohlcv["거래량"].astype(float)
        daily_ret = close.pct_change()

        # ── 거래대금 필터 (30억 이상) ──
        if "거래대금" in ohlcv.columns:
            _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
        else:
            _avg_tv = (close * volume).iloc[-20:].mean()
        if _avg_tv < min_avg_tv:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "거래대금부족",
                }
            )
            continue

        # ── 과열 필터 (극단만 배제) ──
        ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else 0
        max_daily_5d = daily_ret.iloc[-5:].max() if len(close) >= 5 else 0
        if max_daily_5d >= 0.20 or ret_5d >= 0.30:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "과열",
                }
            )
            continue

        # 피처 계산
        feats = compute_ml_features_surge(ohlcv, kospi_close)
        if feats.empty:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
            continue

        for col in SURGE_FEATURE_NAMES:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[SURGE_FEATURE_NAMES]

        last_feats = feats.iloc[-1:]
        if last_feats.isna().all(axis=1).iloc[0]:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
            continue

        # 모델 예측
        X_pred = np.nan_to_num(
            last_feats.values, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64)

        if use_lgb:
            prob = float(model.predict(X_pred)[0])
        else:
            prob = float(model.predict_proba(X_pred)[:, 1][0])

        prob = max(0.0, min(1.0, prob))

        # 후처리: 60일 +30% 이상이면 감점
        ret_60d = close.iloc[-1] / close.iloc[-61] - 1 if len(close) >= 61 else 0
        if ret_60d >= 0.30:
            prob *= 0.5  # 이미 급등 종목 확률 감점

        ai_reason = _generate_ai_reason_surge(last_feats.iloc[0], model_data)
        scored.append((ticker, prob, ai_reason))

    # ── Top 20 랭킹 ──
    scored.sort(key=lambda x: -x[1])
    for rank, (ticker, prob, ai_reason) in enumerate(scored, 1):
        if rank <= 5:
            timing = "즉시매수"
            ai_pass = True
        elif rank <= 10:
            timing = "매수"
            ai_pass = True
        elif rank <= 20:
            timing = "관심"
            ai_pass = True
        else:
            timing = ""
            ai_pass = False

        results.append(
            {
                "ticker": ticker,
                "AI_Pass": ai_pass,
                "AI_Score": round(prob * 100, 1),
                "AI_Timing": timing,
                "AI_Reason": ai_reason,
            }
        )

    return pd.DataFrame(results).set_index("ticker")


def predict_current_accum(
    model_data: dict,
    ohlcv_dict: dict,
    tickers: list,
    kospi_close: pd.Series = None,
    **kwargs,
) -> pd.DataFrame:
    """AI_매집주: 매집 완료 직전 종목 예측.

    매집주는 아직 급등 전이므로 하드필터를 완화.
    거래대금 70억 + 과열 필터만 적용 + 모델 점수 판정.
    HGB + LSTM 앙상블 (55:45).
    """
    results = []

    def _fail(t, reason=""):
        return {
            "ticker": t,
            "AI_Pass": False,
            "AI_Score": 0.0,
            "AI_Timing": "",
            "AI_Reason": reason,
        }

    lstm_data = model_data.get("lstm_data")
    torch_available, torch_modules = (False, None)
    if lstm_data:
        torch_available, torch_modules = _try_import_torch()

    seq_len = 40

    for ticker in tickers:
        ohlcv = ohlcv_dict.get(ticker) if ohlcv_dict else None
        if ohlcv is None or len(ohlcv) < 60:
            results.append(_fail(ticker))
            continue

        close = ohlcv["종가"].astype(float)
        volume = ohlcv["거래량"].astype(float)
        daily_ret = close.pct_change()

        # ── 거래대금 필터 (70억 이상) ──
        if "거래대금" in ohlcv.columns:
            _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
        else:
            _avg_tv = (close * volume).iloc[-20:].mean()
        if _avg_tv < 7_000_000_000:
            results.append(_fail(ticker, "거래대금부족"))
            continue

        # ── 하드필터: 이미 급등한 종목 배제 (매집주는 급등 전) ──
        ret_10d = close.iloc[-1] / close.iloc[-11] - 1 if len(close) >= 11 else 0
        ret_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
        max_daily_5d = daily_ret.iloc[-5:].max() if len(close) >= 5 else 0

        if (
            max_daily_5d >= 0.10  # 5일내 +10% 급등일
            or ret_10d >= 0.20  # 10일 +20% 초과
            or ret_20d >= 0.30
        ):  # 20일 +30% 초과
            results.append(_fail(ticker, "이미상승"))
            continue

        # 피처 계산 (43개)
        feats = compute_ml_features_accum(ohlcv, kospi_close)
        if feats.empty:
            results.append(_fail(ticker))
            continue

        for col in ACCUM_FEATURE_NAMES:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[ACCUM_FEATURE_NAMES]

        last_row = feats.iloc[[-1]].copy().fillna(0)

        feat_names = model_data.get("feature_names", ACCUM_FEATURE_NAMES)
        for col in feat_names:
            if col not in last_row.columns:
                last_row[col] = 0
        last_row = last_row[feat_names]

        # HGB 예측
        model_type = model_data.get("model_type", "")
        if model_type in ("accum_hgb", "accum_ensemble", "sklearn_hgb"):
            hgb_prob = float(_predict_sklearn(model_data, last_row)[0])
        else:
            hgb_prob = float(_predict_numpy_fallback(model_data, last_row)[0])

        # LSTM 예측 (40일 시퀀스)
        lstm_prob = None
        if lstm_data and torch_available and len(feats) >= seq_len:
            try:
                seq = feats.iloc[-seq_len:].fillna(0)
                for col in feat_names:
                    if col not in seq.columns:
                        seq[col] = 0
                seq_arr = seq[feat_names].values.astype(np.float32)
                lstm_prob = _predict_lstm_v7(lstm_data, seq_arr, torch_modules)
            except Exception:
                lstm_prob = None

        # 앙상블 (55:45)
        if lstm_prob is not None:
            prob = 0.55 * hgb_prob + 0.45 * lstm_prob
        else:
            prob = hgb_prob

        prob = max(0.0, min(1.0, prob))

        # 판정 (백분위 기반)
        thr_pass = model_data.get("prob_pct95", 0.30)
        thr_buy = model_data.get("prob_pct98", 0.45)
        thr_strong = model_data.get("prob_pct99", 0.60)

        ai_pass = prob >= thr_pass
        ai_score = prob * 100

        # 후처리 가드: 60일 +25% 이상이면 격하
        if ai_pass:
            ret_60d = close.iloc[-1] / close.iloc[-61] - 1 if len(close) >= 61 else 0
            if ret_60d > 0.25:
                ai_pass = False
                ai_score = ai_score * 0.5

        if prob >= thr_strong:
            ai_timing = "즉시매수"
        elif prob >= thr_buy:
            ai_timing = "매수"
        elif prob >= thr_pass:
            ai_timing = "관심"
        else:
            ai_timing = ""

        if not ai_pass:
            ai_timing = ""

        ai_reason = _generate_ai_reason_accum(
            last_row.iloc[0], model_data, lstm_prob is not None
        )

        results.append(
            {
                "ticker": ticker,
                "AI_Pass": ai_pass,
                "AI_Score": round(prob * 100, 1),
                "AI_Timing": ai_timing,
                "AI_Reason": ai_reason,
            }
        )

    return pd.DataFrame(results).set_index("ticker")


def _generate_ai_reason_accum(feature_values, model_data, has_lstm=False):
    """AI_매집주 사유 태그 생성."""
    importances = model_data.get("feature_importances", {})
    if not importances and model_data.get("model") is not None:
        try:
            m = model_data["model"]
            fnames = model_data.get("feature_names", ACCUM_FEATURE_NAMES)
            imp = dict(zip(fnames, m.feature_importances_, strict=False))
            importances = imp
        except Exception:
            pass

    if not importances:
        prefix = "AI매집(+LSTM)" if has_lstm else "AI매집"
        return prefix

    # 상위 5 중요 피처
    top_feats = sorted(importances.items(), key=lambda x: -abs(x[1]))[:5]
    tags = []
    for fname, _ in top_feats:
        kr = ACCUM_FEATURE_NAMES_KR.get(fname, fname)
        val = feature_values.get(fname, None)
        if val is not None and not np.isnan(val):
            if abs(val) < 1:
                tags.append(f"{kr}({val:.2f})")
            else:
                tags.append(f"{kr}({val:.1f})")
        else:
            tags.append(kr)

    prefix = "AI매집(+LSTM)" if has_lstm else "AI매집"
    return f"{prefix}: " + "+".join(tags[:3])


def _generate_ai_reason_surge(feature_values, model_data):
    """AI_급등주 v2 사유 태그 생성 — Top 5 기여 피처."""
    importances = model_data.get("feature_importances", {})
    if not importances and model_data.get("model") is not None:
        try:
            m = model_data["model"]
            fnames = model_data.get("feature_names", SURGE_FEATURE_NAMES)
            if hasattr(m, "feature_importance"):
                imp_arr = m.feature_importance(importance_type="gain")
                importances = dict(zip(fnames, imp_arr, strict=False))
            elif hasattr(m, "feature_importances_"):
                importances = dict(zip(fnames, m.feature_importances_, strict=False))
        except Exception:
            pass

    if not importances:
        return "AI급등주v2"

    top_feats = sorted(importances.items(), key=lambda x: -abs(x[1]))[:5]
    tags = []
    for fname, _ in top_feats:
        kr = SURGE_FEATURE_NAMES_KR.get(fname, fname)
        val = feature_values.get(fname, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            if abs(val) < 1:
                tags.append(f"{kr}({val:.2f})")
            elif abs(val) < 100:
                tags.append(f"{kr}({val:.1f})")
            else:
                tags.append(f"{kr}({val:.0f})")
        else:
            tags.append(kr)

    return "AI급등v2: " + "+".join(tags[:3])


def _generate_ai_reason_v7(feature_values, model_data, has_lstm=False):
    """v7.0 정밀학습 사유 태그."""
    importances = model_data.get("feature_importances", {})
    if not importances:
        return "AI정밀v7"

    sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[
        :3
    ]

    tags = []
    for feat, _imp in sorted_feats:
        kr = V7_FEATURE_NAMES_KR.get(feat, feat)
        val = feature_values.get(feat, 0)
        if isinstance(val, int | float | np.integer | np.floating):
            if abs(val) < 1:
                tags.append(f"{kr}({val:.2f})")
            else:
                tags.append(f"{kr}({val:.1f})")
        else:
            tags.append(kr)

    prefix = "AI정밀v7(+LSTM)" if has_lstm else "AI정밀v7"
    return f"{prefix}: " + "+".join(tags)


def _generate_ai_reason_v8(feature_values, model_data, has_lstm=False):
    """v8.0 Base Breakout 사유 태그."""
    importances = model_data.get("feature_importances", {})
    if not importances:
        return "AI정밀v8"

    sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[
        :3
    ]

    tags = []
    for feat, _imp in sorted_feats:
        kr = V8_FEATURE_NAMES_KR.get(feat, feat)
        val = feature_values.get(feat, 0)
        if isinstance(val, int | float | np.integer | np.floating):
            if abs(val) < 1:
                tags.append(f"{kr}({val:.2f})")
            else:
                tags.append(f"{kr}({val:.1f})")
        else:
            tags.append(kr)

    prefix = "AI정밀v8(+LSTM)" if has_lstm else "AI정밀v8"
    return f"{prefix}: " + "+".join(tags)


def _generate_ai_reason_precise(feature_values, model_data, has_lstm=False):
    """v6 호환 래퍼 — v8.0으로 리다이렉트."""
    return _generate_ai_reason_v8(feature_values, model_data, has_lstm)


def _generate_ai_reason(feature_values: pd.Series, model_data: dict) -> str:
    """상위 기여 피처 기반 한국어 사유 태그."""
    importances = model_data.get("feature_importances", {})
    if not importances:
        return "AI패턴"

    # 중요도 상위 3개 피처
    sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[
        :3
    ]

    tags = []
    for feat, _imp in sorted_feats:
        kr = FEATURE_NAMES_KR.get(feat, feat)
        val = feature_values.get(feat, 0)
        if isinstance(val, int | float | np.integer | np.floating):
            if abs(val) < 1:
                tags.append(f"{kr}({val:.2f})")
            else:
                tags.append(f"{kr}({val:.1f})")
        else:
            tags.append(kr)

    return "AI: " + "+".join(tags)


# ──────────────────────────────────────────────
# 7. 모델 저장/로드
# ──────────────────────────────────────────────


def save_model(
    model_data: dict,
    date_str: str,
    mode: str = "fast",
    target_pct: float | None = None,
) -> Path:
    """모델을 pkl로 저장. mode: 'fast', 'precise', 'surge', 'accum'.

    target_pct가 주어지면 파일명에 ``_t{pct}`` 접미사를 붙여
    같은 mode라도 목표수익률별 별도 파일로 저장한다.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prefix = f"buy_pattern_{mode}"
    if target_pct is not None:
        prefix += f"_t{int(target_pct * 100)}"

    # 날짜별
    path = MODEL_DIR / f"{prefix}_{date_str}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    # latest
    latest = MODEL_DIR / f"{prefix}_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(model_data, f)

    return path


def load_model(
    date_str: str = None,
    mode: str = "fast",
    target_pct: float | None = None,
) -> dict | None:
    """저장된 모델 로드. mode: 'fast', 'precise', 'surge', 'accum'. 없으면 None.

    target_pct가 주어지면 ``_t{pct}`` 접미사 파일을 먼저 시도한다.
    """
    prefix = f"buy_pattern_{mode}"
    if target_pct is not None:
        prefix += f"_t{int(target_pct * 100)}"

    if date_str:
        path = MODEL_DIR / f"{prefix}_{date_str}.pkl"
    else:
        path = MODEL_DIR / f"{prefix}_latest.pkl"

    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    # fast 모드: 이전 형식 호환 (buy_pattern_{date}.pkl / buy_pattern_latest.pkl)
    if mode == "fast" and target_pct is None:
        if date_str:
            old_path = MODEL_DIR / f"buy_pattern_{date_str}.pkl"
        else:
            old_path = MODEL_DIR / "buy_pattern_latest.pkl"
        if old_path.exists() and old_path != path:
            try:
                with open(old_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    return None


# ──────────────────────────────────────────────
# 8. 학습 결과 요약
# ──────────────────────────────────────────────


def get_learned_patterns(model_data: dict) -> dict:
    """학습 결과를 한국어 요약으로 변환."""
    pa = model_data.get("pattern_analysis", {})
    summary = pa.get("pattern_summary", [])
    n_samples = model_data.get("n_samples", 0)
    n_positive = model_data.get("n_positive", 0)
    n_stocks = model_data.get("n_stocks", 0)
    auc = model_data.get("validation_auc", 0)
    pos_rate = n_positive / n_samples * 100 if n_samples > 0 else 0
    version = model_data.get("version", "")

    lines = []
    if version.startswith("accum"):
        # AI_매집주 리포트
        n_hard_neg = model_data.get("n_hard_neg", 0)
        lstm_data = model_data.get("lstm_data")
        lines.append("=== AI 매집주 — 매집 완료 직전 탐지 ===")
        lines.append(
            f"{n_stocks}종목 2년치 분석 | "
            f"총 {n_samples:,}건 | "
            f"양성(매집완료) {n_positive:,}건 ({pos_rate:.2f}%)"
        )
        lines.append(f"Hard Negative(품질미달/하락) {n_hard_neg:,}건 (가중치 1.5x)")
        if auc > 0:
            lines.append(f"Walk-Forward AUC: {auc:.3f}")
        if lstm_data:
            lstm_auc = lstm_data.get("val_auc", 0)
            lines.append(f"BiLSTM+Attention AUC: {lstm_auc:.3f}")
            lines.append("앙상블: HGB 55% + LSTM 45%")
        lines.append("")
    elif version.startswith("surge"):
        # AI_급등주 v2 Triple-Barrier 리포트
        n_pos = model_data.get("n_pos", model_data.get("n_positive", n_positive))
        n_hard = model_data.get("n_hard_neg", 0)
        n_tickers = model_data.get("n_tickers", n_stocks)
        engine = "LightGBM" if model_data.get("use_lgbm") else "HGB"
        lines.append(f"=== AI 급등주 v2 — Triple-Barrier ({engine}) ===")
        lines.append(
            f"{n_tickers}종목 2년치 분석 | "
            f"총 {n_samples:,}건 | "
            f"양성(+20%도달) {n_pos:,}건 ({pos_rate:.2f}%)"
        )
        lines.append(f"Hard Negative(손절도달) {n_hard:,}건 (가중치 1.5x)")
        if auc > 0:
            lines.append(f"Walk-Forward Avg AUC: {auc:.3f}")
        wf = model_data.get("wf_results", [])
        for i, r in enumerate(wf):
            lines.append(
                f"  WF-{i+1}/4: AUC={r.get('auc',0):.3f}, "
                f"Lift={r.get('lift',0):.1f}x, "
                f"Top적중={r.get('top_rate',0):.1%}"
            )
        lines.append("")
    elif version == "v8.0":
        # v8.0 Base Breakout 리포트
        n_hard_neg = model_data.get("n_hard_neg", 0)
        lstm_data = model_data.get("lstm_data")
        lines.append("=== AI 정밀학습 v8.0 — Base Breakout ===")
        lines.append(
            f"{n_stocks}종목 3년치 분석 | "
            f"총 {n_samples:,}건 | "
            f"양성(바닥→25%↑) {n_positive:,}건 ({pos_rate:.2f}%)"
        )
        lines.append(f"Hard Negative(피크후) {n_hard_neg:,}건 (가중치 1.5x)")
        if auc > 0:
            lines.append(f"Walk-Forward AUC: {auc:.3f}")
        if lstm_data:
            lstm_auc = lstm_data.get("val_auc", 0)
            lines.append(f"BiLSTM+Attention AUC: {lstm_auc:.3f}")
            lines.append("앙상블: HGB 55% + LSTM 45%")
        lines.append("")
    elif version == "v7.0":
        # v7.0 호환 리포트
        n_hard_neg = model_data.get("n_hard_neg", 0)
        lstm_data = model_data.get("lstm_data")
        lines.append("=== AI 정밀학습 v7.0 — 고요한 폭풍 전야 ===")
        lines.append(
            f"{n_stocks}종목 3년치 분석 | "
            f"총 {n_samples:,}건 | "
            f"양성(고요+30%↑) {n_positive:,}건 ({pos_rate:.2f}%)"
        )
        lines.append(f"Hard Negative(이미급등) {n_hard_neg:,}건 (가중치 2x)")
        if auc > 0:
            lines.append(f"Walk-Forward AUC: {auc:.3f}")
        if lstm_data:
            lstm_auc = lstm_data.get("val_auc", 0)
            lines.append(f"BiLSTM+Attention AUC: {lstm_auc:.3f}")
            lines.append("앙상블: HGB 60% + LSTM 40%")
        lines.append("")
    else:
        lines.append("=== AI 매수패턴 학습 결과 ===")
        lines.append(
            f"{n_stocks}종목 2년치 분석 | "
            f"총 {n_samples:,}건 | "
            f"양성(종가30%↑ 시작점) {n_positive:,}건 ({pos_rate:.1f}%)"
        )
        if auc > 0:
            lines.append(f"검증 AUC: {auc:.3f}")
        lines.append("")

    # 상승폭 상관 패턴 TOP 5
    lines.append("-- 1. 상승폭 상관 패턴 (상관계수 높은 순) --")
    for p in summary[:5]:
        corr_str = f"{p['corr']:+.2f}"
        lines.append(
            f"  {p['name_kr']:12s} ↔ 수익률 상관: {corr_str} " f"({p['direction']})"
        )

    lines.append("")
    lines.append("-- 2. 급등 시점 공통 프로필 (전체 대비) --")
    for p in summary[:5]:
        pm = p.get("pos_mean", 0)
        am = p.get("all_mean", 0)
        cov = p.get("coverage", 0)
        if abs(pm) < 1 and abs(am) < 1:
            lines.append(
                f"  {p['name_kr']:12s}: 급등시 {pm:.2f} "
                f"(전체 {am:.2f}) | 커버리지 {cov:.0%}"
            )
        else:
            lines.append(
                f"  {p['name_kr']:12s}: 급등시 {pm:.1f} "
                f"(전체 {am:.1f}) | 커버리지 {cov:.0%}"
            )

    if auc > 0:
        lines.append("")
        lines.append("-- 3. 모델 예측 정확도 --")
        lines.append(f"  검증 AUC: {auc:.3f}")

    # v7 피처 중요도 Top-5
    importances = model_data.get("feature_importances", {})
    if importances:
        lines.append("")
        lines.append("-- 4. Top-5 중요 피처 --")
        feat_kr = (
            ACCUM_FEATURE_NAMES_KR
            if version.startswith("accum")
            else SURGE_FEATURE_NAMES_KR
            if version.startswith("surge")
            else V8_FEATURE_NAMES_KR
            if version == "v8.0"
            else V7_FEATURE_NAMES_KR
            if version == "v7.0"
            else FEATURE_NAMES_KR
        )
        sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[
            :5
        ]
        for i, (feat, imp) in enumerate(sorted_imp, 1):
            kr = feat_kr.get(feat, feat)
            lines.append(f"  {i}. {kr} — {imp:.3f}")

    description = "\n".join(lines)

    return {
        "description": description,
        "top_features": summary[:5],
        "validation_auc": auc,
        "n_samples": n_samples,
        "n_positive": n_positive,
    }


# ──────────────────────────────────────────────
# 방안 3: 데이터 캐싱 활용 재학습
# ──────────────────────────────────────────────


def train_with_cache(
    universe_df: pd.DataFrame, date_str: str, progress_callback=None
) -> dict:
    """방안 3: 데이터 캐싱 + 전체 재학습 (속도 6배 향상).

    전략:
    1. 7일 전 ~ 2년 전 데이터는 캐시에서 로드 (20분 절약)
    2. 최근 7일 데이터만 새로 수집 (1분)
    3. 전체 데이터로 모델 재학습 (3분)

    결과:
    - 방안 1 대비: 동일한 품질, 6배 빠름 (25분 → 4분)
    - 방안 2 대비: 항상 최신 데이터 반영

    Args:
        universe_df: 유니버스 DataFrame
        date_str: 기준일 (YYYYMMDD)
        progress_callback: 진행상황 콜백

    Returns:
        학습된 모델 데이터 dict
    """
    import importlib

    from . import data_fetcher

    importlib.reload(data_fetcher)

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    # ── 캐시 디렉토리 생성 ──
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 종목 필터링 ──
    _cb("급등주 v2 종목 필터링 (시총≥1000억)...", 1)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    mcap = mcap.fillna(0)
    mask = mcap >= 100_000_000_000  # 1000억
    tickers = universe_df[mask].index.tolist()
    _cb(f"학습 대상: {len(tickers)}종목 (시총≥1000억)", 3)

    if len(tickers) < 10:
        raise ValueError(f"학습 대상 종목 부족: {len(tickers)}개 (최소 10개)")

    # ── KOSPI 지수 수집 ──
    _cb("KOSPI 지수 데이터 수집...", 4)
    kospi_close = None
    try:
        kospi_df = data_fetcher.fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
            _cb(f"  KOSPI 지수: {len(kospi_df)}일")
    except Exception as e:
        _cb(f"  KOSPI 수집 실패 (무시): {e}")

    # ── 날짜 계산 ──
    last_week = _get_last_week_date(date_str)
    two_years_ago = _get_2years_ago(date_str)

    # ── 1단계: 캐시된 구간 (2년 전 ~ 7일 전) 로드 ──
    _cb(
        f"캐시 로드 중 ({two_years_ago[:4]}/{two_years_ago[4:6]} ~ {last_week[:4]}/{last_week[4:6]})...",
        5,
    )

    cache_file = FEATURE_CACHE_DIR / f"surge_features_{two_years_ago}_{last_week}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

            all_X_cached = cached_data["X"]
            all_y_cached = cached_data["y"]
            all_hard_cached = cached_data["hard"]
            n_cached = len(all_y_cached)

            _cb(f"  ✅ 캐시 로드 완료: {n_cached:,}건 (20분 절약)", 15)
        except Exception as e:
            _cb(f"  ⚠️ 캐시 로드 실패, 전체 수집으로 전환: {e}", 15)
            all_X_cached = []
            all_y_cached = []
            all_hard_cached = []
            cache_file = None
    else:
        _cb("  ℹ️ 캐시 없음, 전체 구간 수집 시작...", 15)
        all_X_cached = []
        all_y_cached = []
        all_hard_cached = []

        # 캐시가 없으면 구간 데이터를 수집하여 캐시 생성
        done = 0
        total = len(tickers)
        n_skipped_tv = 0

        for ticker in tickers:
            done += 1
            if done % 50 == 0 or done == total:
                pct = 15 + int(done / total * 30)
                _cb(f"  [{done}/{total}] 캐시 생성 중...", pct)
            try:
                ohlcv = data_fetcher.fetch_ohlcv_3year(ticker, last_week)
                if ohlcv is None or len(ohlcv) < 200:
                    continue

                # 거래대금 필터 (30억 이상)
                if "거래대금" in ohlcv.columns:
                    _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
                else:
                    _avg_tv = (
                        (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                        .iloc[-20:]
                        .mean()
                    )
                if _avg_tv < 3_000_000_000:
                    n_skipped_tv += 1
                    continue

                lbl, fwd_ret, hard = compute_breakout_labels(ohlcv)
                feats = compute_ml_features_surge(ohlcv, kospi_close)
                if feats.empty:
                    continue

                # 피처 컬럼 정렬
                for col in SURGE_FEATURE_NAMES:
                    if col not in feats.columns:
                        feats[col] = 0.0
                feats = feats[SURGE_FEATURE_NAMES]

                # 유효 행 (라벨 ≠ NaN)
                valid = lbl.notna()
                if valid.sum() < 10:
                    continue

                all_X_cached.append(feats[valid])
                all_y_cached.append(lbl[valid])
                all_hard_cached.append(hard[valid])
            except Exception:
                continue

        # 캐시 저장
        if all_X_cached:
            X_cached = pd.concat(all_X_cached, ignore_index=True)
            y_cached = pd.concat(all_y_cached, ignore_index=True).values
            hard_cached = pd.concat(all_hard_cached, ignore_index=True).values

            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "X": X_cached,
                            "y": y_cached,
                            "hard": hard_cached,
                            "created_at": date_str,
                        },
                        f,
                    )
                _cb(f"  ✅ 캐시 생성 완료: {len(y_cached):,}건", 45)
            except Exception as e:
                _cb(f"  ⚠️ 캐시 저장 실패 (학습은 계속): {e}", 45)

            all_X_cached = [X_cached]
            all_y_cached = [pd.Series(y_cached)]
            all_hard_cached = [pd.Series(hard_cached)]
        else:
            _cb("  ⚠️ 캐시 생성 실패: 유효 데이터 없음", 45)

    # ── 2단계: 최근 7일 데이터 수집 ──
    _cb(f"최근 7일 데이터 수집 중 ({last_week[:4]}/{last_week[4:6]} ~ 현재)...", 50)

    all_X_recent, all_y_recent, all_hard_recent = [], [], []
    done = 0
    total = len(tickers)
    n_skipped_tv_recent = 0

    for ticker in tickers:
        done += 1
        if done % 50 == 0 or done == total:
            pct = 50 + int(done / total * 20)
            _cb(f"  [{done}/{total}] 최근 데이터 수집 중...", pct)
        try:
            # 최근 7일 + 피처 계산용 추가 데이터
            ohlcv = data_fetcher.fetch_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 200:
                continue

            # 최근 7일 데이터만 필터링
            from datetime import datetime

            last_week_dt = datetime.strptime(last_week, "%Y%m%d")
            ohlcv_recent = ohlcv[ohlcv.index > last_week_dt]

            if len(ohlcv_recent) == 0:
                continue

            # 거래대금 필터
            if "거래대금" in ohlcv.columns:
                _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
            else:
                _avg_tv = (
                    (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                    .iloc[-20:]
                    .mean()
                )
            if _avg_tv < 3_000_000_000:
                n_skipped_tv_recent += 1
                continue

            lbl, fwd_ret, hard = compute_breakout_labels(ohlcv)
            feats = compute_ml_features_surge(ohlcv, kospi_close)
            if feats.empty:
                continue

            # 피처 컬럼 정렬
            for col in SURGE_FEATURE_NAMES:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[SURGE_FEATURE_NAMES]

            # 최근 7일 데이터만 필터링
            feats_recent = feats[feats.index > last_week_dt]
            lbl_recent = lbl[lbl.index > last_week_dt]
            hard_recent = hard[hard.index > last_week_dt]

            # 유효 행
            valid = lbl_recent.notna()
            if valid.sum() > 0:
                all_X_recent.append(feats_recent[valid])
                all_y_recent.append(lbl_recent[valid])
                all_hard_recent.append(hard_recent[valid])
        except Exception:
            continue

    _cb(f"최근 7일 수집 완료: {len(all_X_recent)}종목", 70)

    # ── 3단계: 캐시 + 최근 데이터 병합 ──
    _cb("전체 데이터 병합 중...", 72)

    all_X = all_X_cached + all_X_recent
    all_y = all_y_cached + all_y_recent
    all_hard = all_hard_cached + all_hard_recent

    if not all_X:
        raise ValueError("피처/라벨 생성 실패: 유효 종목 없음")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True).values
    hard_neg_arr = pd.concat(all_hard, ignore_index=True).values

    n_pos = int((y == 1).sum())
    n_hard = int(hard_neg_arr.sum())
    n_total = len(y)
    pos_rate = n_pos / n_total if n_total > 0 else 0
    _cb(
        f"전체: {n_total:,}건 | 양성: {n_pos:,}건 ({pos_rate:.1%}) | "
        f"Hard Neg: {n_hard:,}건",
        75,
    )

    if n_pos < 20:
        raise ValueError(f"양성 샘플 부족: {n_pos}건 (최소 20건)")

    # ── 4단계: 모델 학습 ──
    _cb("LightGBM/HGB 학습 + Walk-Forward 4-window 검증...", 78)
    _sk_ok, sk = _try_import_sklearn()
    if not _sk_ok:
        raise ImportError("sklearn 필요: pip install scikit-learn")

    model_data = _train_sklearn_surge(X, y, hard_neg_arr, sk, _cb)

    # ── 결과 업데이트 ──
    model_data.update(
        {
            "train_date": date_str,
            "n_tickers": len(tickers),
            "n_samples": n_total,
            "n_positive": n_pos,
            "n_hard_neg": n_hard,
            "pos_rate": pos_rate,
            "feature_names": SURGE_FEATURE_NAMES,
            "validation_auc": model_data.get(
                "wf_avg_auc", model_data.get("train_auc", 0)
            ),
            "cached": True,
            "cache_size": len(all_y_cached[0]) if all_y_cached else 0,
            "recent_size": len(pd.concat(all_y_recent)) if all_y_recent else 0,
        }
    )

    engine = "LightGBM" if model_data.get("use_lgbm") else "HGB"
    _cb(
        f"급등주 v2 학습 완료 (캐싱): {engine} | AUC={model_data['validation_auc']:.3f} | "
        f"양성={n_pos:,}건 ({pos_rate:.1%})",
        100,
    )

    return model_data


def train_multi_period_surge(
    universe_df: pd.DataFrame,
    date_str: str,
    progress_callback=None,
    use_gpu: bool = False,
    horizon_days: int = 15,
    target_pct: float = 0.20,
    stop_pct: float = 0.08,
) -> dict:
    """기간별(2Y/1Y/6M/3M) 4개 급등주 모델을 한 번에 학습.

    데이터 수집은 1회만 하고, 날짜 기준 필터링으로 4개 기간별 학습.
    추가 소요시간: ~2분 (4 × ~30초/모델).

    Args:
        use_gpu: True이면 LightGBM GPU 학습 활성화.
        horizon_days: 급등 판정 기간 (거래일). 기본 15 (3주).
        target_pct: 목표수익률. 기본 0.20 (20%).
        stop_pct: 라벨 손절 기준. 기본 0.08 (8%).

    Returns:
        {"surge": model_data, "surge_1y": ..., "surge_6m": ..., "surge_3m": ...}
    """
    import importlib

    from . import data_fetcher

    importlib.reload(data_fetcher)

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    # ── 종목 필터링 (가장 낮은 min_mcap 기준으로 수집) ──
    _min_mcap_all = min(v[2] for v in SURGE_PERIODS.values())
    _min_mcap_label = f"{_min_mcap_all / 100_000_000:,.0f}억"
    _cb(f"급등주 멀티기간 학습: 종목 필터링 (시총≥{_min_mcap_label})...", 1)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    mcap = mcap.fillna(0)
    mask = mcap >= _min_mcap_all
    tickers = universe_df[mask].index.tolist()
    # 종목별 시총 저장 (기간별 mcap 필터링에 사용)
    ticker_mcap = {t: mcap[t] for t in tickers}
    _cb(f"학습 대상: {len(tickers)}종목 (시총≥{_min_mcap_label})", 3)

    if len(tickers) < 10:
        raise ValueError(f"학습 대상 종목 부족: {len(tickers)}개 (최소 10개)")

    # ── KOSPI 지수 수집 ──
    _cb("KOSPI 지수 데이터 수집...", 4)
    kospi_close = None
    try:
        kospi_df = data_fetcher.fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
            _cb(f"  KOSPI 지수: {len(kospi_df)}일")
    except Exception as e:
        _cb(f"  KOSPI 수집 실패 (무시): {e}")

    # ── 2년 OHLCV + 피처/라벨 생성 (날짜 보존) ──
    _cb("2년 OHLCV 수집 + Triple-Barrier 라벨링...", 5)

    # per-ticker 데이터: (mcap, X_df, y_series, hard_series) — DatetimeIndex 보존
    ticker_data = []  # list of (mcap_value, X_df, y_series, hard_series)
    done = 0
    total = len(tickers)
    n_skipped_tv = 0

    for ticker in tickers:
        done += 1
        if done % 50 == 0 or done == total:
            pct = 5 + int(done / total * 45)
            _cb(f"  [{done}/{total}] 종목 처리 중...", pct)
        try:
            ohlcv = data_fetcher.fetch_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 200:
                continue

            # 거래대금 필터 (30억 이상)
            if "거래대금" in ohlcv.columns:
                _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
            else:
                _avg_tv = (
                    (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                    .iloc[-20:]
                    .mean()
                )
            if _avg_tv < 3_000_000_000:
                n_skipped_tv += 1
                continue

            lbl, _fwd_ret, hard = compute_breakout_labels(
                ohlcv,
                target_pct=target_pct,
                stop_pct=stop_pct,
                horizon_days=horizon_days,
            )
            feats = compute_ml_features_surge(ohlcv, kospi_close)
            if feats.empty:
                continue

            # 피처 컬럼 정렬
            for col in SURGE_FEATURE_NAMES:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[SURGE_FEATURE_NAMES]

            # 유효 행 (라벨 ≠ NaN) — DatetimeIndex 보존
            valid = lbl.notna()
            if valid.sum() < 10:
                continue

            _mcap_val = ticker_mcap.get(ticker, 0)
            ticker_data.append((_mcap_val, feats[valid], lbl[valid], hard[valid]))
        except Exception:
            continue

    if not ticker_data:
        raise ValueError("피처/라벨 생성 실패: 유효 종목 없음")

    n_tickers_valid = len(ticker_data)
    _cb(f"데이터 수집 완료: {n_tickers_valid}종목 (스킵: {n_skipped_tv})", 52)

    # ── 기간별 모델 학습 ──
    _sk_ok, sk = _try_import_sklearn()
    if not _sk_ok:
        raise ImportError("sklearn 필요: pip install scikit-learn")

    from datetime import datetime, timedelta

    base_dt = datetime.strptime(date_str, "%Y%m%d")
    models = {}
    period_items = list(SURGE_PERIODS.items())

    for pidx, (mode, (period_label, lookback_days, period_mcap, lgb_ov)) in enumerate(
        period_items
    ):
        pct_base = 55 + int(pidx / len(period_items) * 40)
        _mcap_label = f"{period_mcap / 100_000_000:,.0f}억"
        _cb(
            f"[{pidx + 1}/{len(period_items)}] {period_label} 모델 학습 중 "
            f"(시총≥{_mcap_label})...",
            pct_base,
        )

        cutoff_dt = base_dt - timedelta(days=lookback_days)

        # 날짜 + 시총 기준 필터링
        period_X, period_y, period_hard = [], [], []
        n_tickers_period = 0
        for t_mcap, t_X, t_y, t_hard in ticker_data:
            if t_mcap < period_mcap:
                continue
            date_mask = t_X.index >= cutoff_dt
            if date_mask.sum() > 0:
                period_X.append(t_X[date_mask])
                period_y.append(t_y[date_mask])
                period_hard.append(t_hard[date_mask])
                n_tickers_period += 1

        if not period_X:
            _cb(f"  ⚠️ {period_label}: 유효 데이터 없음, 건너뜀")
            continue

        X = pd.concat(period_X, ignore_index=True)
        y = pd.concat(period_y, ignore_index=True).values
        hard_neg_arr = pd.concat(period_hard, ignore_index=True).values

        n_pos = int((y == 1).sum())
        n_hard = int(hard_neg_arr.sum())
        n_total = len(y)
        pos_rate = n_pos / n_total if n_total > 0 else 0

        _cb(
            f"  {period_label}: {n_tickers_period}종목 {n_total:,}건 | "
            f"양성: {n_pos:,}건 ({pos_rate:.1%}) | Hard Neg: {n_hard:,}건",
            pct_base + 2,
        )

        if n_pos < 10:
            _cb(f"  ⚠️ {period_label}: 양성 샘플 부족 ({n_pos}건), 건너뜀")
            continue

        try:
            _ov = dict(lgb_ov) if lgb_ov else {}
            if use_gpu:
                _ov.update({"device_type": "gpu", "gpu_use_dp": False})
            model_data = _train_sklearn_surge(
                X, y, hard_neg_arr, sk, _cb, lgb_overrides=_ov or None
            )
            model_data.update(
                {
                    "train_date": date_str,
                    "n_tickers": n_tickers_period,
                    "n_samples": n_total,
                    "n_positive": n_pos,
                    "n_hard_neg": n_hard,
                    "pos_rate": pos_rate,
                    "feature_names": SURGE_FEATURE_NAMES,
                    "validation_auc": model_data.get(
                        "wf_avg_auc", model_data.get("train_auc", 0)
                    ),
                    "period": period_label,
                    "lookback_days": lookback_days,
                    "min_mcap": period_mcap,
                }
            )
            engine = "LightGBM" if model_data.get("use_lgbm") else "HGB"
            _lgb_info = f" leaves={lgb_ov.get('num_leaves', 63)}" if lgb_ov else ""
            _cb(
                f"  ✅ {period_label}: {engine} AUC={model_data['validation_auc']:.3f}"
                f"{_lgb_info}",
                pct_base + 8,
            )
            models[mode] = model_data
        except Exception as e:
            _cb(f"  ⚠️ {period_label} 학습 실패: {e}")

    if not models:
        raise ValueError("모든 기간의 모델 학습 실패")

    _cb(
        f"멀티기간 학습 완료: {len(models)}/{len(SURGE_PERIODS)}개 모델 성공",
        100,
    )
    return models


def train_multi_period_with_cache(
    universe_df: pd.DataFrame,
    date_str: str,
    progress_callback=None,
    use_gpu: bool = False,
    horizon_days: int = 15,
    target_pct: float = 0.20,
    stop_pct: float = 0.08,
    min_avg_tv: float = 3_000_000_000,
) -> dict:
    """기간별 4개 급등주 모델 학습 — 데이터 캐싱 버전.

    데이터 캐싱 전략:
    1. 7일 전 ~ 2년 전 per-ticker 데이터는 캐시에서 로드
    2. 최근 7일 데이터만 새로 수집
    3. 병합 후 기간별 4개 모델 학습

    Args:
        use_gpu: True이면 LightGBM GPU 학습 활성화.
        horizon_days: 급등 판정 기간 (거래일). 기본 15 (3주).
        target_pct: 목표수익률. 기본 0.20 (20%).
        stop_pct: 라벨 손절 기준. 기본 0.08 (8%).
        min_avg_tv: 최소 평균거래대금 (원). 기본 3_000_000_000 (30억).

    Returns:
        {"surge": model_data, "surge_1y": ..., "surge_6m": ..., "surge_3m": ...}
    """
    import importlib

    from . import data_fetcher

    importlib.reload(data_fetcher)

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 종목 필터링 (가장 낮은 min_mcap 기준으로 수집) ──
    _min_mcap_all = min(v[2] for v in SURGE_PERIODS.values())
    _min_mcap_label = f"{_min_mcap_all / 100_000_000:,.0f}억"
    _cb(f"급등주 멀티기간 학습 (캐싱): 종목 필터링 (시총≥{_min_mcap_label})...", 1)
    mcap = universe_df.get("MarketCap", pd.Series(0, index=universe_df.index))
    mcap = mcap.fillna(0)
    mask = mcap >= _min_mcap_all
    tickers = universe_df[mask].index.tolist()
    ticker_mcap = {t: mcap[t] for t in tickers}
    _cb(f"학습 대상: {len(tickers)}종목 (시총≥{_min_mcap_label})", 3)

    if len(tickers) < 10:
        raise ValueError(f"학습 대상 종목 부족: {len(tickers)}개 (최소 10개)")

    # ── KOSPI 지수 수집 ──
    _cb("KOSPI 지수 데이터 수집...", 4)
    kospi_close = None
    try:
        kospi_df = data_fetcher.fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
            _cb(f"  KOSPI 지수: {len(kospi_df)}일")
    except Exception as e:
        _cb(f"  KOSPI 수집 실패 (무시): {e}")

    # ── Rolling 피처 캐시 로드 + 증분 업데이트 ──
    from datetime import datetime, timedelta

    _cb("Rolling 피처 캐시 로드 중...", 5)
    rolling = _load_rolling_cache(horizon_days, target_pct, stop_pct)
    is_incremental = False

    if rolling and rolling.get("end_date", "") < date_str:
        cached_end = rolling["end_date"]
        cached_end_dt = datetime.strptime(cached_end, "%Y%m%d")
        tickers_data = rolling["tickers_data"]
        n_cached = len(tickers_data)
        _cb(
            f"  ✅ Rolling 캐시 로드: {n_cached}종목 "
            f"(~{cached_end[:4]}/{cached_end[4:6]}/{cached_end[6:]}까지)",
            8,
        )
        is_incremental = True
    elif rolling and rolling.get("end_date", "") >= date_str:
        tickers_data = rolling["tickers_data"]
        _cb(f"  ✅ Rolling 캐시 최신: {len(tickers_data)}종목", 40)
        is_incremental = True
    else:
        tickers_data = {}
        _cb("  ℹ️ Rolling 캐시 없음, 전체 수집 시작...", 8)

    # ── Step 1: 단일 패스 — OHLCV 증분 수집 + 피처/라벨 계산 ──
    done = 0
    total = len(tickers)
    n_skipped_tv = 0
    n_updated = 0
    n_new = 0

    for ticker in tickers:
        done += 1
        if done % 50 == 0 or done == total:
            pct = 8 + int(done / total * 40)
            mode_str = "증분 업데이트" if is_incremental else "전체 수집"
            _cb(f"  [{done}/{total}] {mode_str} 중...", pct)
        try:
            ohlcv = data_fetcher.fetch_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 200:
                continue
            if "거래대금" in ohlcv.columns:
                _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
            else:
                _avg_tv = (
                    (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                    .iloc[-20:]
                    .mean()
                )
            if _avg_tv < min_avg_tv:
                n_skipped_tv += 1
                continue

            # 라벨: 항상 전체 재계산 (이전 NaN → valid 반영)
            lbl, _fwd_ret, hard = compute_breakout_labels(
                ohlcv,
                target_pct=target_pct,
                stop_pct=stop_pct,
                horizon_days=horizon_days,
            )
            valid = lbl.notna()
            if valid.sum() < 10:
                continue

            if is_incremental and ticker in tickers_data:
                # 증분: 기존 피처에 새 날짜분만 추가
                _old_mcap, old_feats, _old_lbl, _old_hard = tickers_data[ticker]
                feats_all = compute_ml_features_surge(ohlcv, kospi_close)
                if feats_all.empty:
                    continue
                for col in SURGE_FEATURE_NAMES:
                    if col not in feats_all.columns:
                        feats_all[col] = 0.0
                feats_all = feats_all[SURGE_FEATURE_NAMES]
                new_mask = feats_all.index > cached_end_dt
                if new_mask.any():
                    merged = pd.concat([old_feats, feats_all[new_mask]])
                    merged = merged[~merged.index.duplicated(keep="last")]
                else:
                    merged = old_feats
                # 라벨과 피처의 공통 인덱스 (valid만)
                common_idx = merged.index.intersection(lbl[valid].index)
                if len(common_idx) < 10:
                    continue
                tickers_data[ticker] = (
                    ticker_mcap.get(ticker, 0),
                    merged.loc[common_idx],
                    lbl.loc[common_idx],
                    hard.loc[common_idx],
                )
                n_updated += 1
            else:
                # 신규 종목 또는 콜드 스타트: 전체 계산
                feats = compute_ml_features_surge(ohlcv, kospi_close)
                if feats.empty:
                    continue
                for col in SURGE_FEATURE_NAMES:
                    if col not in feats.columns:
                        feats[col] = 0.0
                feats = feats[SURGE_FEATURE_NAMES]
                tickers_data[ticker] = (
                    ticker_mcap.get(ticker, 0),
                    feats[valid],
                    lbl[valid],
                    hard[valid],
                )
                n_new += 1
        except Exception:
            continue

    # 상장폐지 종목 제거 (universe에 없는 종목)
    ticker_set = set(tickers)
    n_removed = 0
    for t in list(tickers_data):
        if t not in ticker_set:
            del tickers_data[t]
            n_removed += 1

    # ── Step 2: 윈도우 트리밍 (2년 + 1주 초과분 제거) ──
    trim_dt = pd.Timestamp(
        datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365 * 2 + 7)
    )
    n_trimmed = 0
    for ticker in list(tickers_data):
        mcap, feats, lbl, hard_s = tickers_data[ticker]
        mask = feats.index >= trim_dt
        if mask.sum() < 10:
            del tickers_data[ticker]
            n_trimmed += 1
            continue
        if not mask.all():
            tickers_data[ticker] = (mcap, feats[mask], lbl[mask], hard_s[mask])

    # Rolling 캐시 저장
    if tickers_data:
        try:
            _save_rolling_cache(
                tickers_data, date_str, horizon_days, target_pct, stop_pct
            )
            if is_incremental:
                _cb(
                    f"  ✅ Rolling 캐시 업데이트: {len(tickers_data)}종목 "
                    f"(증분 {n_updated}, 신규 {n_new}, 제거 {n_removed})",
                    50,
                )
            else:
                _cb(f"  ✅ Rolling 캐시 생성: {len(tickers_data)}종목", 50)
        except Exception as e:
            _cb(f"  ⚠️ Rolling 캐시 저장 실패 (학습은 계속): {e}", 50)

    # ── 3단계: 기간별 모델 학습 ──
    all_ticker_data = list(tickers_data.values())
    if not all_ticker_data:
        raise ValueError("피처/라벨 생성 실패: 유효 종목 없음")

    _sk_ok, sk = _try_import_sklearn()
    if not _sk_ok:
        raise ImportError("sklearn 필요: pip install scikit-learn")

    base_dt = datetime.strptime(date_str, "%Y%m%d")
    models = {}
    period_items = list(SURGE_PERIODS.items())

    for pidx, (mode, (period_label, lookback_days, period_mcap, lgb_ov)) in enumerate(
        period_items
    ):
        pct_base = 55 + int(pidx / len(period_items) * 40)
        _mcap_label = f"{period_mcap / 100_000_000:,.0f}억"
        _cb(
            f"[{pidx + 1}/{len(period_items)}] {period_label} 모델 학습 중 "
            f"(시총≥{_mcap_label})...",
            pct_base,
        )

        cutoff_dt = base_dt - timedelta(days=lookback_days)

        # 날짜 + 시총 기준 필터링
        period_X, period_y, period_hard = [], [], []
        n_tickers_period = 0
        for t_mcap, t_X, t_y, t_hard in all_ticker_data:
            if t_mcap < period_mcap:
                continue
            date_mask = t_X.index >= cutoff_dt
            if date_mask.sum() > 0:
                period_X.append(t_X[date_mask])
                period_y.append(t_y[date_mask])
                period_hard.append(t_hard[date_mask])
                n_tickers_period += 1

        if not period_X:
            _cb(f"  ⚠️ {period_label}: 유효 데이터 없음, 건너뜀")
            continue

        X = pd.concat(period_X, ignore_index=True)
        y = pd.concat(period_y, ignore_index=True).values
        hard_neg_arr = pd.concat(period_hard, ignore_index=True).values

        n_pos = int((y == 1).sum())
        n_hard = int(hard_neg_arr.sum())
        n_total = len(y)
        pos_rate = n_pos / n_total if n_total > 0 else 0

        _cb(
            f"  {period_label}: {n_tickers_period}종목 {n_total:,}건 | "
            f"양성: {n_pos:,}건 ({pos_rate:.1%})",
            pct_base + 2,
        )

        if n_pos < 10:
            _cb(f"  ⚠️ {period_label}: 양성 샘플 부족 ({n_pos}건), 건너뜀")
            continue

        try:
            _ov = dict(lgb_ov) if lgb_ov else {}
            if use_gpu:
                _ov.update({"device_type": "gpu", "gpu_use_dp": False})
            model_data = _train_sklearn_surge(
                X, y, hard_neg_arr, sk, _cb, lgb_overrides=_ov or None
            )
            model_data.update(
                {
                    "train_date": date_str,
                    "n_tickers": n_tickers_period,
                    "n_samples": n_total,
                    "n_positive": n_pos,
                    "n_hard_neg": n_hard,
                    "pos_rate": pos_rate,
                    "feature_names": SURGE_FEATURE_NAMES,
                    "validation_auc": model_data.get(
                        "wf_avg_auc", model_data.get("train_auc", 0)
                    ),
                    "period": period_label,
                    "lookback_days": lookback_days,
                    "min_mcap": period_mcap,
                    "cached": True,
                }
            )
            engine = "LightGBM" if model_data.get("use_lgbm") else "HGB"
            _lgb_info = f" leaves={lgb_ov.get('num_leaves', 63)}" if lgb_ov else ""
            _cb(
                f"  ✅ {period_label}: {engine} AUC={model_data['validation_auc']:.3f}"
                f"{_lgb_info}",
                pct_base + 8,
            )
            models[mode] = model_data
        except Exception as e:
            _cb(f"  ⚠️ {period_label} 학습 실패: {e}")

    if not models:
        raise ValueError("모든 기간의 모델 학습 실패")

    _cb(
        f"멀티기간 학습 완료 (캐싱): {len(models)}/{len(SURGE_PERIODS)}개 모델 성공",
        100,
    )
    return models


def _get_last_week_date(date_str: str) -> str:
    """7일 전 날짜 문자열 반환 (YYYYMMDD).

    Args:
        date_str: 기준일 (YYYYMMDD)

    Returns:
        7일 전 날짜 문자열 (YYYYMMDD)
    """
    from datetime import datetime, timedelta

    dt = datetime.strptime(date_str, "%Y%m%d")
    last_week_dt = dt - timedelta(days=7)
    return last_week_dt.strftime("%Y%m%d")


def _get_2years_ago(date_str: str) -> str:
    """2년 전 날짜 문자열 반환 (YYYYMMDD).

    Args:
        date_str: 기준일 (YYYYMMDD)

    Returns:
        2년 전 날짜 문자열 (YYYYMMDD)
    """
    from datetime import datetime, timedelta

    dt = datetime.strptime(date_str, "%Y%m%d")
    two_years_ago_dt = dt - timedelta(days=365 * 2)
    return two_years_ago_dt.strftime("%Y%m%d")


# ══════════════════════════════════════════════════════
# AI ETF 학습/예측 — 주식 급등주와 완전 분리된 파이프라인
# ══════════════════════════════════════════════════════


def train_multi_period_etf_surge(
    etf_tickers: list,
    date_str: str,
    progress_callback=None,
    use_gpu: bool = False,
    horizon_days: int = 20,
    target_pct: float = 0.10,
    stop_pct: float = 0.05,
) -> dict:
    """기간별(2Y/1Y/6M/3M) 4개 ETF 급등 모델을 한 번에 학습.

    주식 급등주 모델(train_multi_period_surge)과 완전 분리:
    - ETF 전용 유니버스 (stock.get_etf_ticker_list)
    - 낮은 목표수익률 (기본 10%)
    - 거래대금 기반 필터 (시총 대신)
    - 동일한 55개 피처 + Triple-Barrier 라벨 재사용

    Args:
        etf_tickers: ETF 티커 리스트 (사전 필터 완료).
        date_str: 기준일 (YYYYMMDD).
        horizon_days: 급등 판정 기간 (거래일). 기본 20 (4주).
        target_pct: 목표수익률. 기본 0.10 (10%).
        stop_pct: 라벨 손절 기준. 기본 0.05 (5%).

    Returns:
        {"etf_surge": model_data, "etf_surge_1y": ..., ...}
    """
    import importlib

    from . import data_fetcher

    importlib.reload(data_fetcher)

    def _cb(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    _cb(f"ETF 멀티기간 학습: {len(etf_tickers)}종목 대상", 1)

    if len(etf_tickers) < 5:
        raise ValueError(f"ETF 학습 대상 부족: {len(etf_tickers)}개 (최소 5개)")

    # ── KOSPI 지수 수집 ──
    _cb("KOSPI 지수 데이터 수집...", 3)
    kospi_close = None
    try:
        kospi_df = data_fetcher.fetch_kospi_index_3year(date_str)
        if kospi_df is not None and len(kospi_df) > 0:
            kospi_close = kospi_df["종가"].astype(float)
            _cb(f"  KOSPI 지수: {len(kospi_df)}일")
    except Exception as e:
        _cb(f"  KOSPI 수집 실패 (무시): {e}")

    # ── 3년 OHLCV + 피처/라벨 생성 ──
    _cb("ETF 3년 OHLCV 수집 + Triple-Barrier 라벨링...", 5)

    ticker_data = []  # (avg_tv, X_df, y_series, hard_series)
    done = 0
    total = len(etf_tickers)
    n_skipped_tv = 0

    for ticker in etf_tickers:
        done += 1
        if done % 50 == 0 or done == total:
            pct = 5 + int(done / total * 45)
            _cb(f"  [{done}/{total}] ETF 처리 중...", pct)
        try:
            ohlcv = data_fetcher.fetch_etf_ohlcv_3year(ticker, date_str)
            if ohlcv is None or len(ohlcv) < 120:
                continue

            # 거래대금 필터 (최소 기간 기준 _ETF_MIN_TV_ALL)
            if "거래대금" in ohlcv.columns:
                _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
            else:
                _avg_tv = (
                    (ohlcv["종가"].astype(float) * ohlcv["거래량"].astype(float))
                    .iloc[-20:]
                    .mean()
                )
            if _avg_tv < _ETF_MIN_TV_ALL:
                n_skipped_tv += 1
                continue

            lbl, _fwd_ret, hard = compute_breakout_labels(
                ohlcv,
                target_pct=target_pct,
                stop_pct=stop_pct,
                horizon_days=horizon_days,
            )
            feats = compute_ml_features_surge(ohlcv, kospi_close)
            if feats.empty:
                continue

            for col in SURGE_FEATURE_NAMES:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[SURGE_FEATURE_NAMES]

            valid = lbl.notna()
            if valid.sum() < 10:
                continue

            ticker_data.append((_avg_tv, feats[valid], lbl[valid], hard[valid]))
        except Exception:
            continue

    if not ticker_data:
        raise ValueError("ETF 피처/라벨 생성 실패: 유효 종목 없음")

    n_tickers_valid = len(ticker_data)
    _cb(f"ETF 데이터 수집 완료: {n_tickers_valid}종목 (스킵: {n_skipped_tv})", 52)

    # ── 기간별 모델 학습 ──
    _sk_ok, sk = _try_import_sklearn()
    if not _sk_ok:
        raise ImportError("sklearn 필요: pip install scikit-learn")

    from datetime import datetime, timedelta

    base_dt = datetime.strptime(date_str, "%Y%m%d")
    models = {}
    period_items = list(ETF_SURGE_PERIODS.items())

    for pidx, (mode, (period_label, lookback_days, period_min_tv, lgb_ov)) in enumerate(
        period_items
    ):
        pct_base = 55 + int(pidx / len(period_items) * 40)
        _tv_label = f"{period_min_tv / 100_000_000:,.0f}억"
        _cb(
            f"[{pidx + 1}/{len(period_items)}] ETF {period_label} 모델 학습 중 "
            f"(거래대금≥{_tv_label})...",
            pct_base,
        )

        cutoff_dt = base_dt - timedelta(days=lookback_days)

        # 날짜 + 거래대금 기준 필터링
        period_X, period_y, period_hard = [], [], []
        n_tickers_period = 0
        for t_avg_tv, t_X, t_y, t_hard in ticker_data:
            if t_avg_tv < period_min_tv:
                continue
            date_mask = t_X.index >= cutoff_dt
            if date_mask.sum() > 0:
                period_X.append(t_X[date_mask])
                period_y.append(t_y[date_mask])
                period_hard.append(t_hard[date_mask])
                n_tickers_period += 1

        if not period_X:
            _cb(f"  ⚠️ ETF {period_label}: 유효 데이터 없음, 건너뜀")
            continue

        X = pd.concat(period_X, ignore_index=True)
        y = pd.concat(period_y, ignore_index=True).values
        hard_neg_arr = pd.concat(period_hard, ignore_index=True).values

        n_pos = int((y == 1).sum())
        n_hard = int(hard_neg_arr.sum())
        n_total = len(y)
        pos_rate = n_pos / n_total if n_total > 0 else 0

        _cb(
            f"  ETF {period_label}: {n_tickers_period}종목 {n_total:,}건 | "
            f"양성: {n_pos:,}건 ({pos_rate:.1%}) | Hard Neg: {n_hard:,}건",
            pct_base + 2,
        )

        if n_pos < 10:
            _cb(f"  ⚠️ ETF {period_label}: 양성 샘플 부족 ({n_pos}건), 건너뜀")
            continue

        try:
            _ov = dict(lgb_ov) if lgb_ov else {}
            if use_gpu:
                _ov.update({"device_type": "gpu", "gpu_use_dp": False})
            model_data = _train_sklearn_surge(
                X, y, hard_neg_arr, sk, _cb, lgb_overrides=_ov or None
            )
            model_data.update(
                {
                    "train_date": date_str,
                    "asset_type": "ETF",
                    "n_tickers": n_tickers_period,
                    "n_samples": n_total,
                    "n_positive": n_pos,
                    "n_hard_neg": n_hard,
                    "pos_rate": pos_rate,
                    "feature_names": SURGE_FEATURE_NAMES,
                    "validation_auc": model_data.get(
                        "wf_avg_auc", model_data.get("train_auc", 0)
                    ),
                    "period": period_label,
                    "lookback_days": lookback_days,
                    "min_avg_tv": period_min_tv,
                }
            )
            engine = "LightGBM" if model_data.get("use_lgbm") else "HGB"
            _lgb_info = f" leaves={lgb_ov.get('num_leaves', 63)}" if lgb_ov else ""
            _cb(
                f"  ✅ ETF {period_label}: {engine} "
                f"AUC={model_data['validation_auc']:.3f}{_lgb_info}",
                pct_base + 8,
            )
            models[mode] = model_data
        except Exception as e:
            _cb(f"  ⚠️ ETF {period_label} 학습 실패: {e}")

    if not models:
        raise ValueError("모든 기간의 ETF 모델 학습 실패")

    _cb(
        f"ETF 멀티기간 학습 완료: {len(models)}/{len(ETF_SURGE_PERIODS)}개 모델 성공",
        100,
    )
    return models


def predict_current_etf_surge(
    model_data: dict,
    ohlcv_dict: dict,
    tickers: list,
    kospi_close: pd.Series = None,
    **kwargs,
) -> pd.DataFrame:
    """AI_추천ETF: 전체 ETF 스코어링 → Top 20 리포트.

    주식 급등주 예측(predict_current_surge)과 동일한 구조,
    ETF 맞춤 필터 임계값 적용:
    - 거래대금 ≥ 1억 (주식: 30억)
    - 과열 배제: 5일 수익률 ≥ 20% (주식: 30%)
    - 후처리 감점: 60일 +20% 이상 → prob × 0.5 (주식: 30%)
    """
    results = []
    scored = []

    model = model_data.get("model")
    if model is None:
        for t in tickers:
            results.append(
                {
                    "ticker": t,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
        return pd.DataFrame(results).set_index("ticker")

    use_lgb = model_data.get("use_lgbm", False)

    for ticker in tickers:
        ohlcv = ohlcv_dict.get(ticker) if ohlcv_dict else None
        if ohlcv is None or len(ohlcv) < 60:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
            continue

        close = ohlcv["종가"].astype(float)
        volume = ohlcv["거래량"].astype(float)
        daily_ret = close.pct_change()

        # ── 거래대금 필터 (1억 이상) ──
        if "거래대금" in ohlcv.columns:
            _avg_tv = ohlcv["거래대금"].astype(float).iloc[-20:].mean()
        else:
            _avg_tv = (close * volume).iloc[-20:].mean()
        if _avg_tv < 100_000_000:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "거래대금부족",
                }
            )
            continue

        # ── 과열 필터 (ETF 맞춤) ──
        ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else 0
        max_daily_5d = daily_ret.iloc[-5:].max() if len(close) >= 5 else 0
        if max_daily_5d >= 0.15 or ret_5d >= 0.20:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "과열",
                }
            )
            continue

        # 피처 계산
        feats = compute_ml_features_surge(ohlcv, kospi_close)
        if feats.empty:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
            continue

        for col in SURGE_FEATURE_NAMES:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[SURGE_FEATURE_NAMES]

        last_feats = feats.iloc[-1:]
        if last_feats.isna().all(axis=1).iloc[0]:
            results.append(
                {
                    "ticker": ticker,
                    "AI_Pass": False,
                    "AI_Score": 0.0,
                    "AI_Timing": "",
                    "AI_Reason": "",
                }
            )
            continue

        # 모델 예측
        X_pred = np.nan_to_num(
            last_feats.values, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64)

        if use_lgb:
            prob = float(model.predict(X_pred)[0])
        else:
            prob = float(model.predict_proba(X_pred)[:, 1][0])

        prob = max(0.0, min(1.0, prob))

        # 후처리: 60일 +20% 이상이면 감점 (ETF는 20% 기준)
        ret_60d = close.iloc[-1] / close.iloc[-61] - 1 if len(close) >= 61 else 0
        if ret_60d >= 0.20:
            prob *= 0.5

        ai_reason = _generate_ai_reason_surge(last_feats.iloc[0], model_data)
        scored.append((ticker, prob, ai_reason))

    # ── Top 20 랭킹 ──
    scored.sort(key=lambda x: -x[1])
    for rank, (ticker, prob, ai_reason) in enumerate(scored, 1):
        if rank <= 5:
            timing = "즉시매수"
            ai_pass = True
        elif rank <= 10:
            timing = "매수"
            ai_pass = True
        elif rank <= 20:
            timing = "관심"
            ai_pass = True
        else:
            timing = ""
            ai_pass = False

        results.append(
            {
                "ticker": ticker,
                "AI_Pass": ai_pass,
                "AI_Score": round(prob * 100, 1),
                "AI_Timing": timing,
                "AI_Reason": ai_reason,
            }
        )

    return pd.DataFrame(results).set_index("ticker")
