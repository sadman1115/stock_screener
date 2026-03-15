"""KOSPI/KOSDAQ 지수 3년 예측 모듈.

학술 리서치 기반 설계:
- LightGBM 분위수 회귀 (Direct MIMO: 36개월 × 3분위수 = 108 모델)
- Walk-Forward 3-fold 검증
- 67개 피처: 가격/추세, 변동성/리짐, 밸류에이션(기본+심화),
  투자자수급, 외부시장, KB전략
- 밸류에이션 심화 피처 12개:
  EY, EYG, BEER, PER/PBR 퍼센타일, PBR 평균회귀, 복합 z-score 등
- 상승장 패턴/조정 예측 피처 10개:
  급등→조정 사이클, PBR/PER 밴드 위치, 조정 위험 점수
- 월간 리샘플 기반 (일별 OHLCV → 월간 피처)
- 증분 캐싱 전략 (OHLCV/펀더멘털/투자자 데이터)
- KB 1월 전략 기반 분기별 알파/베타 전략 오버레이 + 샘플가중치
- 주별 예측 보간 (월간 예측 → 주별 등락 구체화)

References:
- KJFS: "Stock Return Prediction Using Macroeconomic Drivers: KOSPI"
- PLOS One: "Predictability of ML techniques for Korean stock markets"
- Financial Innovation 2024: Elastic Net / LASSO for KOSPI prediction
- KB증권 2025.12.30: "1월 전략_1분기에 알파 드리븐, 2분기엔 베타 컨트롤"
- Pacific-Basin Finance 2024: PER/PBR/CAPE predict KOSPI equity premium
- Boucher et al. 2023: Conditional CAPE mean-reversion → 10x out-of-sample R²
- ECB WP: Earnings Yield Gap forecasts excess market returns
- Vanguard ML-VAR 2019: CAPE decomposition, 56% forecast improvement
"""

import pickle
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate as _scipy_interp

from core.config import INDEX_PREDICTION_DIR

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

INDEX_PRED_DIR = INDEX_PREDICTION_DIR
INDEX_PRED_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CODES = {"KOSPI": "1001", "KOSDAQ": "2001"}

# 역사적 리짐 이벤트 (차트 시각화용, 학습 피처 아님)
REGIME_EVENTS = [
    {"start": "1986-01", "end": "1989-12", "label": "3저호황", "type": "bull"},
    {"start": "1997-11", "end": "1998-12", "label": "IMF 위기", "type": "bear"},
    {"start": "2003-01", "end": "2007-10", "label": "중국호황", "type": "bull"},
    {"start": "2008-09", "end": "2009-03", "label": "글로벌 금융위기", "type": "bear"},
    {"start": "2011-08", "end": "2011-12", "label": "유럽재정위기", "type": "bear"},
    {"start": "2020-02", "end": "2020-03", "label": "COVID 폭락", "type": "bear"},
    {
        "start": "2020-04",
        "end": "2021-06",
        "label": "동학개미 유동성장",
        "type": "bull",
    },
    {"start": "2022-01", "end": "2022-10", "label": "금리인상 약세", "type": "bear"},
]

# 피처명 (67개)
FEATURE_NAMES = [
    # A. 가격/추세 (8)
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",
    "ma3_ma6_ratio",
    "ma6_ma12_ratio",
    "pct_from_12m_high",
    "pct_from_12m_low",
    # B. 변동성/리짐 (6)
    "vol_3m",
    "vol_12m",
    "vol_regime",
    "max_dd_6m",
    "bb_width",
    "rsi_monthly",
    # C. 밸류에이션 기본 (6)
    "per_index",
    "pbr_index",
    "div_yield",
    "per_chg_3m",
    "pbr_chg_3m",
    "per_deviation",
    # C2. 밸류에이션 심화 (12) — CAPE, EYG, BEER, 퍼센타일, 평균회귀
    "earnings_yield",
    "eyg",
    "eyg_zscore",
    "beer",
    "per_percentile_5y",
    "pbr_percentile_5y",
    "pbr_zscore_10y",
    "pbr_mr_signal",
    "div_yield_zscore",
    "val_composite_z",
    "bond_yield_10y",
    "real_earnings_yield",
    # D. 투자자 수급 (6)
    "foreign_net_1m",
    "foreign_net_3m",
    "inst_net_1m",
    "inst_net_3m",
    "individual_net_1m",
    "foreign_inst_ratio",
    # E. 외부 시장 (9)
    "sp500_ret_1m",
    "sp500_ret_3m",
    "vix_level",
    "vix_chg_1m",
    "usdkrw_level",
    "usdkrw_chg_1m",
    "oil_ret_1m",
    "kospi_kosdaq_spread",
    "month_of_year",
    # F. KB 전략 기반 (9) — 어닝스 모멘텀 계절성, 분기 레짐, 조정 리스크, 알파/베타
    "kb_earnings_season",
    "kb_quarter_regime",
    "kb_correction_risk",
    "kb_cycle_sin",
    "kb_cycle_cos",
    "kb_alpha_beta_regime",
    "kb_macro_regime",
    "kb_sector_rotation_phase",
    "kb_earnings_mmt_raw",
    # G. 상승장 패턴/조정 예측 (10) — 급등→조정 사이클, PBR/PER 밴드
    "rally_from_trough",
    "rally_velocity",
    "trend_acceleration",
    "trend_deviation",
    "consecutive_up_months",
    "overshoot_intensity",
    "pbr_band_position",
    "per_band_position",
    "pbr_fair_value_gap",
    "correction_risk_score",
    # H. 큰 조정 후 회복 추세 (5) — 조정 깊이→회복 지속 패턴
    "correction_depth",
    "months_since_trough",
    "drawdown_recovery_ratio",
    "post_correction_momentum",
    "recovery_phase",
]

# ── KB 1월 전략 기반 분기별 오버레이 ──
# (KB증권 2025.12.30 "1분기 알파 드리븐, 2분기 베타 컨트롤")
#
# 핵심 근거:
# 1. 3저 호황 패턴: Q1/Q3 상승, Q2/Q4 조정 (3~4개월 상승 → 2개월 조정)
# 2. 통화정책 변화가 강세장 조정의 주요 원인
# 3. 코스닥 1~2월 계절성 최고, 어닝스 모멘텀 연초 강화
# 4. 반도체 재고/매출 비율 50%대 → 추가 하락 여지 있으나 바닥 경계
#
# 월별 조정 계수: 모델 예측에 가산되는 미세 조정 (return basis)
# Q1(1~3월): 알파 드리븐 overweight → 약간 상향
# Q2(4~6월): 베타 컨트롤 → 하향 (금리인하 종료, 변동성 확대)
# Q3(7~9월): 회복 랠리 → 상향
# Q4(10~12월): 어닝스 시즌/조정 → 중립~소폭 상향
# 월별 어닝스 모멘텀 1M 팩터 수익률 (annualized %p, KOSPI WMI500, 2010~)
# 출처: KB 1월전략 Table 5, p31-32
KB_EARNINGS_MMT_1M = {
    1: +8.3,
    2: +16.3,
    3: +14.9,
    4: +25.8,
    5: +20.2,
    6: +10.4,
    7: +0.9,
    8: +19.5,
    9: +0.1,
    10: -8.5,
    11: -0.5,
    12: +6.4,
}
# 월별 어닝스 모멘텀 1M Hit Ratio (%)
KB_EARNINGS_HIT_RATIO = {
    1: 62.5,
    2: 81.3,
    3: 56.3,
    4: 68.8,
    5: 62.5,
    6: 68.8,
    7: 43.8,
    8: 75.0,
    9: 43.8,
    10: 31.3,
    11: 37.5,
    12: 66.7,
}
# 3저 호황 조정 집중월 (Mar-Apr, Oct-Nov): -10% ~ -18% 조정
KB_CORRECTION_MONTHS = {3, 4, 10, 11}
# 3저 호황 분기 레짐: Q1/Q3=상승(+1), Q2/Q4=조정(-1)
KB_QUARTER_REGIME = {1: +1, 2: -1, 3: +1, 4: -1}
# 월별 조정 계수 — Hit Ratio 기반 + 3저 호황 패턴 합산
# hit_bias = (hit_ratio - 50) / 50 → ±0.63 스케일 → ×0.012 = return basis
KB_MONTHLY_OVERLAY = {
    1: +0.004,  # HR 62.5%, 어닝스 모멘텀 시작
    2: +0.009,  # HR 81.3%, 최강 어닝스 월
    3: +0.002,  # HR 56.3%, 3저 조정 경계 (Mar)
    4: +0.006,  # HR 68.8%, 어닝스 피크 BUT 조정 경계 (Apr)
    5: +0.004,  # HR 62.5%, Q2 중반
    6: +0.006,  # HR 68.8%, Q2 후반
    7: -0.002,  # HR 43.8%, Q3 초반 약세
    8: +0.008,  # HR 75.0%, 회복 랠리
    9: -0.002,  # HR 43.8%, Q3 후반 약세
    10: -0.006,  # HR 31.3%, 최약 어닝스 월 + 조정
    11: -0.004,  # HR 37.5%, 조정 지속
    12: +0.005,  # HR 66.7%, 연말 랠리
}

# KB 업종별 월간 어닝스 모멘텀 (연환산 %p, KOSPI 대비)
# 출처: KB 1월전략 Table 4 (p31-32), Table 5 기반 재구성 — 18개 업종 확장
KB_SECTOR_MONTHLY = {
    "반도체": {
        1: +15,
        2: +20,
        3: +18,
        4: +12,
        5: +5,
        6: -2,
        7: -8,
        8: -5,
        9: -10,
        10: -12,
        11: -8,
        12: +3,
    },
    "소프트웨어": {
        1: +8,
        2: +12,
        3: +10,
        4: +5,
        5: +3,
        6: -1,
        7: -3,
        8: +2,
        9: +5,
        10: +8,
        11: +10,
        12: +5,
    },
    "증권": {
        1: +12,
        2: +15,
        3: +10,
        4: +5,
        5: +3,
        6: -2,
        7: -4,
        8: +2,
        9: +1,
        10: -5,
        11: -3,
        12: +8,
    },
    "은행": {
        1: +5,
        2: +8,
        3: +5,
        4: +3,
        5: +5,
        6: +2,
        7: 0,
        8: +2,
        9: +1,
        10: -2,
        11: +3,
        12: +5,
    },
    "보험": {
        1: +3,
        2: +5,
        3: +3,
        4: +2,
        5: +4,
        6: +2,
        7: -1,
        8: +1,
        9: 0,
        10: -2,
        11: +2,
        12: +3,
    },
    "금융(통합)": {
        1: +10,
        2: +12,
        3: +8,
        4: +6,
        5: +8,
        6: +2,
        7: -1,
        8: +3,
        9: +1,
        10: -3,
        11: +2,
        12: +5,
    },
    "헬스케어": {
        1: +3,
        2: +5,
        3: -2,
        4: +5,
        5: +8,
        6: +3,
        7: -5,
        8: +2,
        9: -3,
        10: -2,
        11: +3,
        12: +5,
    },
    "자동차": {
        1: +5,
        2: +8,
        3: +3,
        4: -2,
        5: +2,
        6: +5,
        7: -3,
        8: +1,
        9: +3,
        10: -5,
        11: -3,
        12: +2,
    },
    "IT하드웨어": {
        1: -2,
        2: +3,
        3: +1,
        4: -3,
        5: -5,
        6: -2,
        7: +1,
        8: +5,
        9: +8,
        10: +12,
        11: +10,
        12: +8,
    },
    "유틸리티": {
        1: -5,
        2: -8,
        3: +2,
        4: +8,
        5: +5,
        6: +10,
        7: +8,
        8: -2,
        9: +5,
        10: +12,
        11: +8,
        12: -3,
    },
    "소재/화학": {
        1: +2,
        2: +5,
        3: -3,
        4: -8,
        5: -5,
        6: -10,
        7: +3,
        8: +8,
        9: +10,
        10: +6,
        11: +4,
        12: +2,
    },
    "배터리": {
        1: +3,
        2: +5,
        3: -5,
        4: -8,
        5: -10,
        6: -5,
        7: -3,
        8: +2,
        9: -5,
        10: -8,
        11: -5,
        12: +2,
    },
    "철강": {
        1: -3,
        2: -5,
        3: -8,
        4: -10,
        5: -8,
        6: -5,
        7: +2,
        8: +5,
        9: +8,
        10: -10,
        11: -8,
        12: -5,
    },
    "건설": {
        1: +2,
        2: +3,
        3: 0,
        4: -3,
        5: -2,
        6: -5,
        7: -3,
        8: +1,
        9: +2,
        10: -3,
        11: -2,
        12: +1,
    },
    "운송": {
        1: +1,
        2: +3,
        3: +2,
        4: 0,
        5: -2,
        6: -3,
        7: -2,
        8: +2,
        9: +3,
        10: -2,
        11: -1,
        12: +2,
    },
    "미디어": {
        1: +3,
        2: +5,
        3: +2,
        4: 0,
        5: -3,
        6: -2,
        7: -5,
        8: -3,
        9: +2,
        10: +3,
        11: +5,
        12: +3,
    },
    "지주사": {
        1: +8,
        2: +10,
        3: +5,
        4: +3,
        5: +2,
        6: 0,
        7: -2,
        8: +1,
        9: -1,
        10: -3,
        11: -2,
        12: +5,
    },
    "통신": {
        1: -3,
        2: -5,
        3: -2,
        4: +2,
        5: +3,
        6: +5,
        7: +3,
        8: +1,
        9: +2,
        10: +5,
        11: +3,
        12: 0,
    },
}

# KB 모델 포트폴리오 업종별 over/underweight (bp) — Table 19 (p51)
KB_MODEL_PORTFOLIO = {
    "반도체": +300,
    "금융(통합)": +250,
    "증권": +150,
    "은행": +100,
    "보험": +50,
    "소프트웨어": +100,
    "지주사": +100,
    "헬스케어": +50,
    "자동차": 0,
    "IT하드웨어": 0,
    "유틸리티": 0,
    "건설": 0,
    "운송": 0,
    "미디어": 0,
    "소재/화학": -200,
    "배터리": -150,
    "통신": -200,
    "철강": -100,
}

# KB 분기별 기대수익률 (연환산 → 월환산 시 /12)
KB_Q_RETURN_EXPECTATION = {1: +0.05, 2: -0.02, 3: +0.03, 4: +0.01}

# ── 과거 상승장 패턴 분석 (Stooq 실제 데이터 기반) ──
# 4개 역사적 상승장의 시작점과 가중치 정의
_BULL_PATTERN_DEFS = {
    "3저호황(1986-89)": {
        "weight": 0.40,
        "start": "19860801",
        "end": "19890901",
    },
    "IMF회복(1998-00)": {
        "weight": 0.20,
        "start": "19980901",
        "end": "20010901",
    },
    "유동성장세(2003-07)": {
        "weight": 0.25,
        "start": "20030401",
        "end": "20060501",
    },
    "코로나반등(2020-21)": {
        "weight": 0.15,
        "start": "20200301",
        "end": "20230401",
    },
}

# 캐시: 한 번만 Stooq 호출
_bull_patterns_cache: dict | None = None


def _load_bull_patterns() -> dict:
    """Stooq에서 실제 KOSPI 월별 종가를 가져와 36개월 누적수익률 계산."""
    global _bull_patterns_cache
    if _bull_patterns_cache is not None:
        return _bull_patterns_cache

    import pandas as pd

    result = {}
    for name, cfg in _BULL_PATTERN_DEFS.items():
        w = cfg["weight"]
        url = (
            f"https://stooq.com/q/d/l/?s=^kospi"
            f"&d1={cfg['start']}&d2={cfg['end']}&i=d"
        )
        try:
            df = pd.read_csv(url)
            if df is None or df.empty or "Close" not in df.columns:
                raise ValueError(f"Stooq 데이터 없음: {name}")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            # 월말 리샘플링
            monthly = df["Close"].resample("ME").last().dropna()
            if len(monthly) < 2:
                raise ValueError(f"월별 데이터 부족: {name}")
            base = float(monthly.iloc[0])
            # M0=시작월(cum=0), M1~M36 = 이후 월별 누적수익률
            cum = []
            for i in range(1, min(len(monthly), 37)):
                cum.append((float(monthly.iloc[i]) - base) / base)
            # 36개월 미만이면 마지막 값으로 패딩
            while len(cum) < 36:
                cum.append(cum[-1] if cum else 0.0)
            result[name] = {"weight": w, "monthly_cum": cum}
        except Exception:
            # Stooq 접속 실패 시 해당 패턴 제외
            pass

    _bull_patterns_cache = result
    return result


def _bull_pattern_return(month: int) -> float:
    """과거 상승장 패턴 가중 평균 수익률 (월 단위, 1-36). Stooq 실제 데이터 기반."""
    if month < 1:
        return 0.0
    patterns = _load_bull_patterns()
    if not patterns:
        return 0.0
    idx = min(month, 36) - 1  # 0-indexed
    weighted_sum = 0.0
    total_weight = 0.0
    for _name, patt in patterns.items():
        w = patt["weight"]
        cum = patt["monthly_cum"]
        weighted_sum += w * cum[idx]
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ── 중동전쟁 패턴 분석 (Stooq 실제 데이터 기반) ──
# 이란전쟁 발발일 (2026-02-28)
IRAN_WAR_START = "2026-02-28"

_WAR_PATTERN_DEFS = {
    "걸프전(1990-91)": {
        "weight": 0.35,
        "start": "19900801",  # 이라크의 쿠웨이트 침공
        "end": "19920301",
    },
    "이라크전(2003)": {
        "weight": 0.40,
        "start": "20030301",  # 미국 이라크 침공 개시
        "end": "19920301",  # 18개월 후까지
    },
    "이란긴장(2020)": {
        "weight": 0.25,
        "start": "20200101",  # 솔레이마니 제거
        "end": "20210701",
    },
}

# end 날짜 보정 (이라크전)
_WAR_PATTERN_DEFS["이라크전(2003)"]["end"] = "20040901"

_war_patterns_cache: dict | None = None


def _load_war_patterns() -> dict:
    """Stooq에서 실제 KOSPI 데이터로 중동전쟁 시 18개월 누적수익률 계산."""
    global _war_patterns_cache
    if _war_patterns_cache is not None:
        return _war_patterns_cache

    import pandas as pd

    result = {}
    for name, cfg in _WAR_PATTERN_DEFS.items():
        w = cfg["weight"]
        url = (
            f"https://stooq.com/q/d/l/?s=^kospi"
            f"&d1={cfg['start']}&d2={cfg['end']}&i=d"
        )
        try:
            df = pd.read_csv(url)
            if df is None or df.empty or "Close" not in df.columns:
                raise ValueError(f"Stooq 데이터 없음: {name}")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            monthly = df["Close"].resample("ME").last().dropna()
            if len(monthly) < 2:
                raise ValueError(f"월별 데이터 부족: {name}")
            base = float(monthly.iloc[0])
            # M1~M18: 전쟁 발발 후 월별 누적수익률
            cum = []
            for i in range(1, min(len(monthly), 19)):
                cum.append((float(monthly.iloc[i]) - base) / base)
            while len(cum) < 18:
                cum.append(cum[-1] if cum else 0.0)
            result[name] = {"weight": w, "monthly_cum": cum}
        except Exception:
            pass

    _war_patterns_cache = result
    return result


def _war_pattern_return(month: int) -> float:
    """중동전쟁 패턴 가중 평균 누적수익률 (월 단위, 1-18). Stooq 실제 데이터 기반."""
    if month < 1:
        return 0.0
    patterns = _load_war_patterns()
    if not patterns:
        return 0.0
    idx = min(month, 18) - 1
    weighted_sum = 0.0
    total_weight = 0.0
    for _name, patt in patterns.items():
        w = patt["weight"]
        cum = patt["monthly_cum"]
        weighted_sum += w * cum[idx]
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ── KB Research Coverage 종목 추천 (PDF pp.56-80, Table 19 기반) ──
KB_STOCK_PICKS = {
    "반도체": {
        "weight": +300,
        "stocks": [
            {"name": "SK하이닉스", "opinion": "Buy", "reason": "HBM/ASIC 수혜"},
            {"name": "삼성전자", "opinion": "Buy", "reason": "파운드리+메모리 회복"},
            {"name": "한미반도체", "opinion": "Buy", "reason": "HBM 장비 독점"},
        ],
        "quarters": [1, 2, 3, 4],
    },
    "증권": {
        "weight": +150,
        "stocks": [
            {"name": "미래에셋증권", "opinion": "Buy", "reason": "거래대금 증가 수혜"},
            {"name": "키움증권", "opinion": "Buy", "reason": "리테일 점유율 1위"},
            {"name": "한국금융지주", "opinion": "Buy", "reason": "종합금융 시너지"},
        ],
        "quarters": [1, 2],
    },
    "은행": {
        "weight": +100,
        "stocks": [
            {"name": "KB금융", "opinion": "Buy", "reason": "실적 안정+주주환원"},
            {"name": "신한지주", "opinion": "Buy", "reason": "배당 성장+ROE 개선"},
        ],
        "quarters": [1, 2, 3, 4],
    },
    "소프트웨어": {
        "weight": +100,
        "stocks": [
            {"name": "카카오", "opinion": "Buy", "reason": "플랫폼 반등+AI"},
            {"name": "NAVER", "opinion": "Buy", "reason": "검색+커머스+AI"},
        ],
        "quarters": [1, 3],
    },
    "지주사": {
        "weight": +100,
        "stocks": [
            {"name": "한화", "opinion": "Buy", "reason": "방산+에너지 지주 할인"},
            {"name": "SK", "opinion": "Buy", "reason": "반도체+바이오 NAV할인"},
            {"name": "LG", "opinion": "Buy", "reason": "전자+화학 NAV할인 해소"},
        ],
        "quarters": [1, 2],
    },
    "보험": {
        "weight": +50,
        "stocks": [
            {"name": "삼성화재", "opinion": "Buy", "reason": "손해율 개선+배당"},
            {"name": "DB손해보험", "opinion": "Buy", "reason": "실적 성장+배당"},
        ],
        "quarters": [1, 2, 4],
    },
    "헬스케어": {
        "weight": +50,
        "stocks": [
            {"name": "삼성바이오로직스", "opinion": "Buy", "reason": "CMO 수주 확대"},
            {"name": "셀트리온", "opinion": "Buy", "reason": "바이오시밀러 성장"},
        ],
        "quarters": [1, 3],
    },
    "내수/소비재": {
        "weight": 0,
        "stocks": [
            {"name": "CJ제일제당", "opinion": "Buy", "reason": "음식료 안정+환율방어"},
            {"name": "아모레퍼시픽", "opinion": "Hold", "reason": "중국 회복 기대"},
        ],
        "quarters": [2, 4],
    },
    "자동차": {
        "weight": 0,
        "stocks": [
            {"name": "현대차", "opinion": "Buy", "reason": "글로벌 점유율+EV"},
            {"name": "기아", "opinion": "Buy", "reason": "수익성+주주환원"},
        ],
        "quarters": [1, 3],
    },
}

# KB 알파/베타 레짐 — Q1/Q3=alpha(1), Q2/Q4=beta(0)
KB_ALPHA_BETA_REGIME = {1: 1, 2: 0, 3: 1, 4: 0}

# ── 확장 투자유형 분류체계 (KB 1월 전략 기반) ──
KB_INVEST_TYPE_CATALOG = {
    "성장주": {
        "label": "성장주",
        "sub_types": ["AI/반도체", "플랫폼", "바이오"],
        "icon": "\U0001f680",
        "quarters": [1, 3],
    },
    "저PER주": {
        "label": "저PER주",
        "sub_types": ["실적턴어라운드", "업종저평가"],
        "icon": "\U0001f48e",
        "quarters": [1, 2, 3, 4],
    },
    "저PBR주": {
        "label": "저PBR주",
        "sub_types": ["자산가치주", "구조조정기대"],
        "icon": "\U0001f3e6",
        "quarters": [2, 4],
    },
    "고배당주": {
        "label": "고배당주",
        "sub_types": ["배당성장", "고정배당"],
        "icon": "\U0001f4b0",
        "quarters": [4],
    },
    "대형주": {
        "label": "대형주",
        "sub_types": ["시총상위", "우량주"],
        "icon": "\U0001f3e2",
        "quarters": [1, 2, 3, 4],
    },
    "중소형주": {
        "label": "중소형주",
        "sub_types": ["코스닥성장", "니치마켓"],
        "icon": "\U0001f331",
        "quarters": [1, 3],
    },
    "ETF": {
        "label": "ETF",
        "sub_types": ["인덱스ETF", "섹터ETF", "해외ETF", "레버리지ETF"],
        "icon": "\U0001f4ca",
        "quarters": [1, 2, 3, 4],
    },
    "지주사": {
        "label": "지주사",
        "sub_types": ["순수지주", "사업지주"],
        "icon": "\U0001f3db",
        "quarters": [1, 2],
    },
    "증권주": {
        "label": "증권주",
        "sub_types": ["대형증권", "중소형증권"],
        "icon": "\U0001f4c8",
        "quarters": [1],
    },
    "금융주": {
        "label": "금융주",
        "sub_types": ["은행", "증권", "보험"],
        "icon": "\U0001f3e7",
        "quarters": [1, 2],
    },
    "내수주": {
        "label": "내수주",
        "sub_types": ["소비재", "유통", "음식료"],
        "icon": "\U0001f3ea",
        "quarters": [2, 4],
    },
    "수출주": {
        "label": "수출주",
        "sub_types": ["자동차", "조선", "기계"],
        "icon": "\U0001f30d",
        "quarters": [1, 3],
    },
}

# 분기별 투자유형 우선순위
KB_Q_INVEST_PRIORITY = {
    1: ["성장주", "증권주", "금융주", "지주사", "수출주", "ETF"],
    2: ["저PER주", "저PBR주", "내수주", "대형주", "고배당주", "ETF"],
    3: ["성장주", "중소형주", "수출주", "ETF"],
    4: ["고배당주", "저PBR주", "내수주", "대형주", "ETF"],
}

# 분기별 스타일 (확장)
KB_Q_STYLE = {
    1: "성장주(AI반도체)",
    2: "가치주(저PER/PBR)",
    3: "성장주(회복)",
    4: "배당주(고배당)",
}

# 분기별 기본 주식비중
KB_Q_WEIGHT = {1: 0.95, 2: 0.70, 3: 0.85, 4: 0.75}

# KOSDAQ 월별 시즌성 (2000-2025 평균 %)
KB_KOSDAQ_SEASON = {
    1: +3.5,
    2: +2.8,
    3: -0.5,
    4: +1.2,
    5: -0.3,
    6: -1.0,
    7: -1.5,
    8: +1.5,
    9: -0.8,
    10: -2.0,
    11: -1.2,
    12: +1.0,
}

# ── 예측 기간 상수 ──
N_PRED_MONTHS = 36  # 향후 예측 개월 수 (3년)
N_PRED_WEEKS = N_PRED_MONTHS * 52 // 12  # 156주

# 주별 보간에 사용되는 주간 변동성 계수 (월간 예측 사이의 잔차 변동)
WEEKLY_VOL_SCALE = 0.012  # 주간 수익률 표준편차 (약 1.2%)


# ──────────────────────────────────────────────
# 캐시 유틸리티 (data_fetcher.py 패턴 재사용)
# ──────────────────────────────────────────────


def _idx_cache_path(name: str) -> Path:
    return INDEX_PRED_DIR / f"{name}.pkl"


def _load_idx_cache(name: str):
    p = _idx_cache_path(name)
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_idx_cache(name: str, data):
    INDEX_PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(_idx_cache_path(name), "wb") as f:
        pickle.dump(data, f)


def _is_krx_nontrading_error(e: Exception) -> bool:
    """pykrx 내부 KeyError가 비거래일(주말/공휴일)로 인한 것인지 판별."""
    if isinstance(e, KeyError):
        key_str = str(e)
        if any(
            k in key_str for k in ("시장", "지수명", "종목", "종목명", "BPS", "PER")
        ):
            return True
    if isinstance(e, RuntimeError | ValueError):
        msg = str(e)
        if "빈 응답" in msg or "비거래일" in msg:
            return True
    return False


def _krx_api_failure_msg(func_name: str, e: Exception) -> str:
    """KRX API 실패 시 원인을 포함한 명확한 로그 메시지 생성."""
    if _is_krx_nontrading_error(e):
        reason = "비거래일(주말/공휴일) 데이터 미제공"
    elif isinstance(e, TimeoutError):
        reason = "API 응답 타임아웃"
    else:
        reason = f"API 오류({type(e).__name__}: {e})"
    return f"기본 KRX API({func_name})가 {reason}으로 실패"


def _pykrx_retry(func, *args, retries=3, delay=2, timeout=120, **kwargs):
    """pykrx API 호출 with retry + timeout."""
    import concurrent.futures

    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(func, *args, **kwargs)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise TimeoutError(f"pykrx API 타임아웃 ({timeout}초)") from None
        except Exception as e:
            if _is_krx_nontrading_error(e):
                # 비거래일 오류는 재시도 무의미 → 즉시 전파
                raise RuntimeError(_krx_api_failure_msg(func.__name__, e)) from e
            if attempt < retries - 1:
                time.sleep(delay * (2**attempt))
            else:
                raise e


_INDEX_YAHOO = {"1001": "^KS11", "2001": "^KQ11"}
_INDEX_STOOQ = {"1001": "^kospi", "2001": "^kosdaq"}

# pykrx 실제 데이터 시작일 (이전 데이터는 Stooq로 보충)
_PYKRX_EARLIEST = {"1001": "19970101", "2001": "20001001"}


def _stooq_index_ohlcv(
    index_code: str,
    start_str: str,
    end_str: str,
    progress_callback=None,
) -> pd.DataFrame | None:
    """Stooq.com에서 지수 OHLCV 수집 (1983~ 과거 데이터용)."""
    stooq_symbol = _INDEX_STOOQ.get(index_code)
    if stooq_symbol is None:
        return None
    try:
        s = start_str.replace("-", "")
        e = end_str.replace("-", "")
        s_fmt = f"{s[:4]}{s[4:6]}{s[6:8]}"
        e_fmt = f"{e[:4]}{e[4:6]}{e[6:8]}"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}" f"&d1={s_fmt}&d2={e_fmt}&i=d"
        if progress_callback:
            progress_callback(f"  Stooq.com에서 과거 데이터 수집: {s[:4]}~{e[:4]}")
        df = pd.read_csv(url)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        rename_map = {
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가",
            "Volume": "거래량",
        }
        df = df.rename(columns=rename_map)
        cols = [
            c for c in ["시가", "고가", "저가", "종가", "거래량"] if c in df.columns
        ]
        result = df[cols]
        # Stooq 일부 초기 데이터에서 OHLC가 같은 값 → 종가만 있어도 OK
        if progress_callback:
            progress_callback(
                f"  Stooq 수집 완료: {len(result)}일 "
                f"({result.index[0].strftime('%Y-%m-%d')}~"
                f"{result.index[-1].strftime('%Y-%m-%d')})"
            )
        return result
    except Exception:
        return None


def _fdr_index_ohlcv_fallback(
    index_code: str,
    start_str: str,
    end_str: str,
) -> pd.DataFrame | None:
    """FinanceDataReader로 지수 OHLCV 수집 (pykrx 실패 시 fallback)."""
    yahoo_code = _INDEX_YAHOO.get(index_code)
    if yahoo_code is None:
        return None
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return None
    try:
        s = start_str.replace("-", "")
        e = end_str.replace("-", "")
        s_fmt = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        e_fmt = f"{e[:4]}-{e[4:6]}-{e[6:8]}"
        df = fdr.DataReader(yahoo_code, s_fmt, e_fmt)
        if df is None or df.empty:
            return None
        rename_map = {
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가",
            "Volume": "거래량",
        }
        df = df.rename(columns=rename_map)
        cols = [
            c for c in ["시가", "고가", "저가", "종가", "거래량"] if c in df.columns
        ]
        return df[cols]
    except Exception:
        return None


# ──────────────────────────────────────────────
# 1단계: 데이터 수집 함수
# ──────────────────────────────────────────────


def fetch_index_ohlcv_long(
    index_code: str,
    end_date: str,
    years: int = 12,
    progress_callback=None,
) -> pd.DataFrame:
    """지수 OHLCV 장기 수집 (증분 캐싱).

    Args:
        index_code: "1001" (KOSPI) or "2001" (KOSDAQ)
        end_date: YYYYMMDD
        years: 수집 연수

    Returns:
        DataFrame (DatetimeIndex, 시가/고가/저가/종가/거래량)
    """
    from pykrx import stock

    cache_name = f"idx_{index_code}_ohlcv_{years}y"

    # 1. 기존 캐시 확인
    cached = _load_idx_cache(cache_name)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index)
        last_cached = cached.index[-1].strftime("%Y%m%d")
        if last_cached >= end_date:
            return cached

        # 캐시가 너무 오래된 경우 (end_date와 10년+ 차이) → 캐시 무시, 전체 재수집
        _gap_days = (
            datetime.strptime(end_date, "%Y%m%d") - cached.index[-1].to_pydatetime()
        ).days
        if _gap_days > 365 * 10:
            if progress_callback:
                progress_callback(
                    f"  캐시가 {cached.index[-1].strftime('%Y-%m-%d')}까지만 "
                    f"존재 ({_gap_days // 365}년 차이) → 전체 재수집"
                )
            cached = None  # 아래 전체 수집 로직으로 진행
        else:
            # 증분 수집
            next_day = (cached.index[-1] + timedelta(days=1)).strftime("%Y%m%d")
            if next_day <= end_date:
                if progress_callback:
                    progress_callback(f"지수 OHLCV 증분 수집: {next_day}~{end_date}")
                try:
                    inc = _pykrx_retry(
                        stock.get_index_ohlcv_by_date,
                        next_day,
                        end_date,
                        index_code,
                        name_display=False,
                    )
                    if inc is not None and not inc.empty:
                        if not isinstance(inc.index, pd.DatetimeIndex):
                            inc.index = pd.to_datetime(inc.index)
                        merged = pd.concat([cached, inc])
                        merged = merged[~merged.index.duplicated(keep="last")]
                        cutoff = pd.Timestamp(end_date) - pd.DateOffset(
                            years=years, days=30
                        )
                        merged = merged[merged.index >= cutoff]
                        _save_idx_cache(cache_name, merged)
                        return merged
                except Exception as e:
                    if progress_callback:
                        api_msg = _krx_api_failure_msg("get_index_ohlcv_by_date", e)
                        progress_callback(f"  ⚠️ 증분 수집: {api_msg} → FDR 시도")
                    fdr_inc = _fdr_index_ohlcv_fallback(index_code, next_day, end_date)
                    if fdr_inc is not None and not fdr_inc.empty:
                        if not isinstance(fdr_inc.index, pd.DatetimeIndex):
                            fdr_inc.index = pd.to_datetime(fdr_inc.index)
                        merged = pd.concat([cached, fdr_inc])
                        merged = merged[~merged.index.duplicated(keep="last")]
                        cutoff = pd.Timestamp(end_date) - pd.DateOffset(
                            years=years, days=30
                        )
                        merged = merged[merged.index >= cutoff]
                        _save_idx_cache(cache_name, merged)
                        if progress_callback:
                            progress_callback(f"  FDR 증분 수집 성공({len(fdr_inc)}일)")
                        return merged
                    if progress_callback:
                        progress_callback("  ⚠️ FDR도 실패 → 기존 캐시 사용")
            return cached

    # cached가 None이면 여기로 도달 (전체 수집)

    # 2. 전체 수집
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=years * 365 + 60)
    start_str = start_dt.strftime("%Y%m%d")

    if progress_callback:
        progress_callback(f"지수 OHLCV 전체 수집: {start_str}~{end_date}")

    # 2-a. pykrx로 수집 (pykrx는 ~1997년부터만 제공)
    pykrx_start = _PYKRX_EARLIEST.get(index_code, "19970101")
    pykrx_query_start = max(start_str, pykrx_start)

    df = None
    try:
        df = _pykrx_retry(
            stock.get_index_ohlcv_by_date,
            pykrx_query_start,
            end_date,
            index_code,
            name_display=False,
        )
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if progress_callback:
                progress_callback(
                    f"  pykrx: {len(df)}일 "
                    f"({df.index[0].strftime('%Y-%m-%d')}~"
                    f"{df.index[-1].strftime('%Y-%m-%d')})"
                )
    except Exception as e:
        if progress_callback:
            api_msg = _krx_api_failure_msg("get_index_ohlcv_by_date", e)
            progress_callback(f"  ⚠️ {api_msg}")
        # pykrx 실패 시 FDR fallback
        fdr_df = _fdr_index_ohlcv_fallback(index_code, pykrx_query_start, end_date)
        if fdr_df is not None and not fdr_df.empty:
            if not isinstance(fdr_df.index, pd.DatetimeIndex):
                fdr_df.index = pd.to_datetime(fdr_df.index)
            df = fdr_df
            if progress_callback:
                progress_callback(f"  FDR 대체 수집: {len(df)}일")

    # 2-b. 요청 시작일이 pykrx 범위 이전이면 Stooq로 과거 데이터 보충
    if start_str < pykrx_start:
        stooq_end = pykrx_start  # pykrx 시작 직전까지
        stooq_df = _stooq_index_ohlcv(
            index_code, start_str, stooq_end, progress_callback
        )
        if stooq_df is not None and not stooq_df.empty:
            if df is not None and not df.empty:
                # Stooq(과거) + pykrx(최근) 병합
                df = pd.concat([stooq_df, df])
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()
                if progress_callback:
                    progress_callback(
                        f"  Stooq+pykrx 병합: {len(df)}일 "
                        f"({df.index[0].strftime('%Y-%m-%d')}~"
                        f"{df.index[-1].strftime('%Y-%m-%d')})"
                    )
            else:
                # Stooq만 있고 pykrx/FDR 실패 → 최근 데이터 없음
                if progress_callback:
                    progress_callback(
                        "  ⚠️ pykrx/FDR 모두 실패, Stooq 과거 데이터만 사용 가능"
                    )
                df = stooq_df

    if df is not None and not df.empty:
        # Stooq-only 등으로 end_date보다 10년 이상 오래된 경우 캐시 저장 안 함
        _last_dt = df.index[-1]
        _end_dt = datetime.strptime(end_date, "%Y%m%d")
        _data_gap = (_end_dt - _last_dt.to_pydatetime()).days
        if _data_gap > 365 * 10:
            if progress_callback:
                progress_callback(
                    f"  ⚠️ 수집 데이터가 "
                    f"{_last_dt.strftime('%Y-%m-%d')}까지만 존재 "
                    f"(요청 종료: {end_date}). 캐시 미저장."
                )
            # 캐시 저장하지 않고 반환 (다음 실행 시 재시도)
            return df
        _save_idx_cache(cache_name, df)
        return df

    return pd.DataFrame()


def fetch_index_fundamentals_long(
    index_code: str,
    end_date: str,
    years: int = 12,
    progress_callback=None,
) -> pd.DataFrame:
    """지수 PER/PBR/배당수익률 장기 수집 (연도별 chunk).

    Returns:
        DataFrame (DatetimeIndex, PER/PBR/배당수익률 등)
    """
    from pykrx import stock

    cache_name = f"idx_{index_code}_fund_{years}y"

    cached = _load_idx_cache(cache_name)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index)
        last_cached = cached.index[-1].strftime("%Y%m%d")
        if last_cached >= end_date:
            return cached

        # 증분 수집
        next_day = (cached.index[-1] + timedelta(days=1)).strftime("%Y%m%d")
        if next_day <= end_date:
            if progress_callback:
                progress_callback(f"지수 펀더멘털 증분: {next_day}~{end_date}")
            try:
                inc = _pykrx_retry(
                    stock.get_index_fundamental_by_date,
                    next_day,
                    end_date,
                    index_code,
                )
                if inc is not None and not inc.empty:
                    if not isinstance(inc.index, pd.DatetimeIndex):
                        inc.index = pd.to_datetime(inc.index)
                    merged = pd.concat([cached, inc])
                    merged = merged[~merged.index.duplicated(keep="last")]
                    cutoff = pd.Timestamp(end_date) - pd.DateOffset(
                        years=years, days=30
                    )
                    merged = merged[merged.index >= cutoff]
                    _save_idx_cache(cache_name, merged)
                    return merged
            except Exception as e:
                if progress_callback:
                    api_msg = _krx_api_failure_msg("get_index_fundamental_by_date", e)
                    progress_callback(f"  ⚠️ 펀더멘털 증분: {api_msg} → 기존 캐시 사용")
        return cached

    # 전체 수집 (연도별 chunk — 타임아웃 방지)
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    frames = []
    first_error_logged = False
    for y_offset in range(years, 0, -1):
        yr_start = end_dt - timedelta(days=y_offset * 365 + 30)
        yr_end = end_dt - timedelta(days=(y_offset - 1) * 365)
        if yr_end > end_dt:
            yr_end = end_dt
        s = yr_start.strftime("%Y%m%d")
        e = yr_end.strftime("%Y%m%d")
        if progress_callback:
            progress_callback(f"지수 펀더멘털: {s[:4]}년...")
        try:
            chunk = _pykrx_retry(
                stock.get_index_fundamental_by_date,
                s,
                e,
                index_code,
            )
            if chunk is not None and not chunk.empty:
                frames.append(chunk)
        except Exception as exc:
            if not first_error_logged and progress_callback:
                api_msg = _krx_api_failure_msg("get_index_fundamental_by_date", exc)
                progress_callback(f"  ⚠️ {api_msg}")
                first_error_logged = True
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    _save_idx_cache(cache_name, df)
    return df


def fetch_market_investor_long(
    market: str,
    end_date: str,
    years: int = 12,
    progress_callback=None,
) -> pd.DataFrame:
    """시장 전체 투자자별 순매수 일별 추이 (연도별 chunk).

    Args:
        market: "KOSPI" or "KOSDAQ"

    Returns:
        DataFrame (DatetimeIndex, 기관합계/기타법인/개인/외국인합계/전체 등)
    """
    from pykrx import stock

    cache_name = f"idx_{market.lower()}_investor_{years}y"

    cached = _load_idx_cache(cache_name)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index)
        last_cached = cached.index[-1].strftime("%Y%m%d")
        if last_cached >= end_date:
            return cached

        next_day = (cached.index[-1] + timedelta(days=1)).strftime("%Y%m%d")
        if next_day <= end_date:
            if progress_callback:
                progress_callback(f"투자자 데이터 증분: {next_day}~{end_date}")
            try:
                inc = _pykrx_retry(
                    stock.get_market_trading_value_by_date,
                    next_day,
                    end_date,
                    market,
                    timeout=60,
                )
                if inc is not None and not inc.empty:
                    if not isinstance(inc.index, pd.DatetimeIndex):
                        inc.index = pd.to_datetime(inc.index)
                    merged = pd.concat([cached, inc])
                    merged = merged[~merged.index.duplicated(keep="last")]
                    cutoff = pd.Timestamp(end_date) - pd.DateOffset(
                        years=years, days=30
                    )
                    merged = merged[merged.index >= cutoff]
                    _save_idx_cache(cache_name, merged)
                    return merged
            except Exception as e:
                if progress_callback:
                    api_msg = _krx_api_failure_msg(
                        "get_market_trading_value_by_date", e
                    )
                    progress_callback(f"  ⚠️ 투자자 증분: {api_msg} → 기존 캐시 사용")
        return cached

    # 전체 수집 (연도별 chunk)
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    frames = []
    for y_offset in range(years, 0, -1):
        yr_start = end_dt - timedelta(days=y_offset * 365 + 30)
        yr_end = end_dt - timedelta(days=(y_offset - 1) * 365)
        if yr_end > end_dt:
            yr_end = end_dt
        s = yr_start.strftime("%Y%m%d")
        e = yr_end.strftime("%Y%m%d")
        if progress_callback:
            progress_callback(f"투자자 매매: {s[:4]}년...")
        try:
            chunk = _pykrx_retry(
                stock.get_market_trading_value_by_date,
                s,
                e,
                market,
                timeout=60,
            )
            if chunk is not None and not chunk.empty:
                frames.append(chunk)
        except Exception as exc:
            if progress_callback:
                api_msg = _krx_api_failure_msg("get_market_trading_value_by_date", exc)
                progress_callback(f"  ⚠️ {api_msg}")
            break  # 비거래일이면 나머지 chunk도 실패하므로 중단
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    _save_idx_cache(cache_name, df)
    return df


def fetch_external_features(
    end_date: str,
    years: int = 12,
    progress_callback=None,
) -> pd.DataFrame | None:
    """외부 시장 데이터 수집 (FinanceDataReader — 선택적).

    FDR 미설치 시 None 반환 → 모델은 축소 피처로 동작.
    """
    cache_name = f"external_features_{years}y"

    cached = _load_idx_cache(cache_name)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index)
        last_cached = cached.index[-1].strftime("%Y%m%d")
        if last_cached >= end_date:
            return cached

    try:
        import FinanceDataReader as fdr
    except ImportError:
        return None

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=years * 365 + 60)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    if progress_callback:
        progress_callback(
            "외부 시장 데이터 수집 (S&P 500, VIX, USD/KRW, WTI, KR국채10Y)..."
        )

    result = pd.DataFrame()
    symbols = {
        "SP500": "US500",
        "VIX": "VIX",
        "USDKRW": "USD/KRW",
        "WTI": "CL=F",
    }

    for col_name, symbol in symbols.items():
        try:
            df = fdr.DataReader(symbol, start_str, end_str)
            if df is not None and not df.empty and "Close" in df.columns:
                result[col_name] = df["Close"]
        except Exception:
            pass
        time.sleep(0.3)

    # 한국 10년 국채금리 (EYG/BEER 피처용)
    for kr_bond_sym in ("KR10YT=RR", "KR10Y", "IRLTLT01KRM156N"):
        try:
            df_bond = fdr.DataReader(kr_bond_sym, start_str, end_str)
            if df_bond is not None and not df_bond.empty:
                _bond_col = (
                    "Close" if "Close" in df_bond.columns else df_bond.columns[0]
                )
                result["KR10Y"] = df_bond[_bond_col]
                break
        except Exception:
            pass
        time.sleep(0.3)

    if result.empty:
        return None

    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    result = result.sort_index()

    _save_idx_cache(cache_name, result)
    return result


# ──────────────────────────────────────────────
# 2단계: 피처 엔지니어링
# ──────────────────────────────────────────────


def _monthly_resample(df: pd.DataFrame, col: str = "종가") -> pd.Series:
    """일별 → 월말 리샘플 (마지막 거래일 종가)."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].resample("ME").last().dropna()


def _rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    """롤링 최대 낙폭 계산."""
    result = pd.Series(np.nan, index=series.index)
    for i in range(window, len(series)):
        window_data = series.iloc[i - window : i + 1]
        peak = window_data.expanding().max()
        dd = (window_data - peak) / peak
        result.iloc[i] = dd.min()
    return result


def _rsi_monthly(close: pd.Series, period: int = 14) -> pd.Series:
    """월간 RSI 계산."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_index_features(
    index_ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    investor: pd.DataFrame | None = None,
    external: pd.DataFrame | None = None,
    other_index_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """지수 예측 피처 계산 (~35개, 월간 리샘플).

    모든 피처는 과거 데이터만 사용 (미래 참조 없음).
    """
    # 월간 리샘플
    monthly_close = _monthly_resample(index_ohlcv, "종가")
    monthly_high = _monthly_resample(index_ohlcv, "고가")
    monthly_low = _monthly_resample(index_ohlcv, "저가")

    if monthly_close.empty or len(monthly_close) < 24:
        return pd.DataFrame()

    feats = pd.DataFrame(index=monthly_close.index)

    # ── A. 가격/추세 (8개) ──
    feats["ret_1m"] = monthly_close.pct_change(1)
    feats["ret_3m"] = monthly_close.pct_change(3)
    feats["ret_6m"] = monthly_close.pct_change(6)
    feats["ret_12m"] = monthly_close.pct_change(12)

    ma3 = monthly_close.rolling(3).mean()
    ma6 = monthly_close.rolling(6).mean()
    ma12 = monthly_close.rolling(12).mean()
    feats["ma3_ma6_ratio"] = ma3 / ma6.replace(0, np.nan) - 1
    feats["ma6_ma12_ratio"] = ma6 / ma12.replace(0, np.nan) - 1

    high_12m = monthly_high.rolling(12).max()
    low_12m = monthly_low.rolling(12).min()
    feats["pct_from_12m_high"] = monthly_close / high_12m.replace(0, np.nan) - 1
    feats["pct_from_12m_low"] = monthly_close / low_12m.replace(0, np.nan) - 1

    # ── B. 변동성/리짐 (6개) ──
    monthly_ret = monthly_close.pct_change()
    feats["vol_3m"] = monthly_ret.rolling(3).std()
    feats["vol_12m"] = monthly_ret.rolling(12).std()
    feats["vol_regime"] = feats["vol_3m"] / feats["vol_12m"].replace(0, np.nan)
    feats["max_dd_6m"] = _rolling_max_drawdown(monthly_close, 6)

    # 월간 볼린저 밴드 폭
    bb_ma = monthly_close.rolling(12).mean()
    bb_std = monthly_close.rolling(12).std()
    feats["bb_width"] = (2 * bb_std) / bb_ma.replace(0, np.nan)

    feats["rsi_monthly"] = _rsi_monthly(monthly_close, 14) / 100.0  # 0~1 정규화

    # ── C. 밸류에이션 (6개) ──
    if fundamentals is not None and not fundamentals.empty:
        fund_monthly = fundamentals.resample("ME").last()
        # PER/PBR/배당수익률 컬럼명 탐색
        per_col = next((c for c in fund_monthly.columns if "PER" in c.upper()), None)
        pbr_col = next((c for c in fund_monthly.columns if "PBR" in c.upper()), None)
        div_col = next(
            (c for c in fund_monthly.columns if "배당" in c or "DIV" in c.upper()),
            None,
        )

        per_s = (
            fund_monthly[per_col].reindex(feats.index, method="ffill")
            if per_col
            else pd.Series(np.nan, index=feats.index)
        )
        pbr_s = (
            fund_monthly[pbr_col].reindex(feats.index, method="ffill")
            if pbr_col
            else pd.Series(np.nan, index=feats.index)
        )
        div_s = (
            fund_monthly[div_col].reindex(feats.index, method="ffill")
            if div_col
            else pd.Series(np.nan, index=feats.index)
        )

        feats["per_index"] = per_s
        feats["pbr_index"] = pbr_s
        feats["div_yield"] = div_s
        feats["per_chg_3m"] = per_s.pct_change(3)
        feats["pbr_chg_3m"] = pbr_s.pct_change(3)
        # PER vs 5년 롤링 평균 편차 (밸류에이션 극단 감지)
        per_5y_mean = per_s.rolling(60, min_periods=24).mean()
        per_5y_std = per_s.rolling(60, min_periods=24).std()
        feats["per_deviation"] = (per_s - per_5y_mean) / per_5y_std.replace(0, np.nan)
    else:
        for col in [
            "per_index",
            "pbr_index",
            "div_yield",
            "per_chg_3m",
            "pbr_chg_3m",
            "per_deviation",
        ]:
            feats[col] = np.nan
        per_s = pd.Series(np.nan, index=feats.index)
        pbr_s = pd.Series(np.nan, index=feats.index)
        div_s = pd.Series(np.nan, index=feats.index)

    # ── C2. 밸류에이션 심화 (12개) ──
    # References:
    #   - Pacific-Basin Finance 2024: PER/PBR/CAPE predict KOSPI equity premium
    #   - Boucher et al. 2023: Conditional mean-reversion of CAPE → 10x R²
    #   - ECB WP: Earnings Yield Gap → excess return predictor
    #   - Vanguard ML-VAR 2019: CAPE decomposition for 10-yr forecast

    # 1) Earnings Yield (1/PER) — CAPE의 핵심 입력
    ey = (1.0 / per_s.replace(0, np.nan)).clip(-1, 1)
    feats["earnings_yield"] = ey

    # 2-3) Earnings Yield Gap (EYG) = EY - 10년 국채금리
    #       → 주식이 채권 대비 저평가/고평가 신호
    bond_10y = pd.Series(np.nan, index=feats.index)
    if external is not None and not external.empty and "KR10Y" in external.columns:
        bond_monthly = external["KR10Y"].resample("ME").last()
        bond_10y = bond_monthly.reindex(feats.index, method="ffill") / 100.0
    feats["bond_yield_10y"] = bond_10y

    feats["eyg"] = ey - bond_10y
    eyg_mean = feats["eyg"].rolling(60, min_periods=24).mean()
    eyg_std = feats["eyg"].rolling(60, min_periods=24).std()
    feats["eyg_zscore"] = (feats["eyg"] - eyg_mean) / eyg_std.replace(0, np.nan)

    # 4) BEER (Bond-Equity Earnings Yield Ratio) = 국채금리 / EY
    feats["beer"] = bond_10y / ey.replace(0, np.nan)
    feats["beer"] = feats["beer"].clip(-5, 5)  # 극단값 방지

    # 5-6) PER/PBR 5년 퍼센타일 순위 (0~1)
    #       → 현재 밸류에이션이 최근 5년 역사 대비 어디인지
    feats["per_percentile_5y"] = per_s.rolling(60, min_periods=24).apply(
        lambda x: np.searchsorted(np.sort(x[:-1]), x[-1]) / max(len(x) - 1, 1),
        raw=True,
    )
    feats["pbr_percentile_5y"] = pbr_s.rolling(60, min_periods=24).apply(
        lambda x: np.searchsorted(np.sort(x[:-1]), x[-1]) / max(len(x) - 1, 1),
        raw=True,
    )

    # 7) PBR 10년 z-score → 극단적 저평가/고평가 감지
    pbr_10y_mean = pbr_s.rolling(120, min_periods=36).mean()
    pbr_10y_std = pbr_s.rolling(120, min_periods=36).std()
    feats["pbr_zscore_10y"] = (pbr_s - pbr_10y_mean) / pbr_10y_std.replace(0, np.nan)

    # 8) PBR 평균회귀 신호: ln(장기평균 / 현재) → 양수면 저평가(상승 기대)
    #    (Ornstein-Uhlenbeck 기반, Half-life 약 3~5년)
    feats["pbr_mr_signal"] = np.log(pbr_10y_mean / pbr_s.replace(0, np.nan)).clip(-2, 2)

    # 9) 배당수익률 z-score (5년) → 높으면 저평가, 낮으면 고평가
    div_5y_mean = div_s.rolling(60, min_periods=24).mean()
    div_5y_std = div_s.rolling(60, min_periods=24).std()
    feats["div_yield_zscore"] = (div_s - div_5y_mean) / div_5y_std.replace(0, np.nan)

    # 10) 복합 밸류에이션 z-score (PER deviation + PBR z-score + DY z-score 평균)
    #     → 개별 지표보다 노이즈 감소, 모든 구간에서 예측력 강화
    _z_per = feats["per_deviation"].fillna(0)
    _z_pbr = feats["pbr_zscore_10y"].fillna(0)
    _z_div = -feats["div_yield_zscore"].fillna(0)  # 배당높으면 저평가 → 부호반전
    feats["val_composite_z"] = (_z_per + _z_pbr + _z_div) / 3.0

    # 11) bond_yield_10y — 이미 위에서 계산됨

    # 12) 실질 Earnings Yield ≈ EY - (전년동월 대비 지수수익률 근사 인플레)
    feats["real_earnings_yield"] = ey - feats["ret_12m"].clip(-0.5, 0.5).fillna(0)

    # ── D. 투자자 수급 (6개) ──
    if investor is not None and not investor.empty:
        # 외국인/기관/개인 컬럼 탐색
        foreign_col = next((c for c in investor.columns if "외국인" in c), None)
        inst_col = next((c for c in investor.columns if "기관" in c), None)
        indiv_col = next((c for c in investor.columns if "개인" in c), None)

        if foreign_col:
            foreign_daily = investor[foreign_col]
            foreign_monthly = foreign_daily.resample("ME").sum()
            feats["foreign_net_1m"] = foreign_monthly.reindex(
                feats.index, method="ffill"
            )
            feats["foreign_net_3m"] = (
                foreign_monthly.rolling(3).sum().reindex(feats.index, method="ffill")
            )
        else:
            feats["foreign_net_1m"] = np.nan
            feats["foreign_net_3m"] = np.nan

        if inst_col:
            inst_daily = investor[inst_col]
            inst_monthly = inst_daily.resample("ME").sum()
            feats["inst_net_1m"] = inst_monthly.reindex(feats.index, method="ffill")
            feats["inst_net_3m"] = (
                inst_monthly.rolling(3).sum().reindex(feats.index, method="ffill")
            )
        else:
            feats["inst_net_1m"] = np.nan
            feats["inst_net_3m"] = np.nan

        if indiv_col:
            indiv_daily = investor[indiv_col]
            indiv_monthly = indiv_daily.resample("ME").sum()
            feats["individual_net_1m"] = indiv_monthly.reindex(
                feats.index, method="ffill"
            )
        else:
            feats["individual_net_1m"] = np.nan

        # 외국인/기관 비율
        if "foreign_net_1m" in feats and "inst_net_1m" in feats:
            denom = feats["inst_net_1m"].abs().replace(0, np.nan)
            feats["foreign_inst_ratio"] = feats["foreign_net_1m"] / denom
        else:
            feats["foreign_inst_ratio"] = np.nan
    else:
        for col in [
            "foreign_net_1m",
            "foreign_net_3m",
            "inst_net_1m",
            "inst_net_3m",
            "individual_net_1m",
            "foreign_inst_ratio",
        ]:
            feats[col] = np.nan

    # ── E. 외부 시장 (9개) ──
    if external is not None and not external.empty:
        ext_monthly = external.resample("ME").last()

        if "SP500" in ext_monthly.columns:
            sp = ext_monthly["SP500"].reindex(feats.index, method="ffill")
            feats["sp500_ret_1m"] = sp.pct_change(1)
            feats["sp500_ret_3m"] = sp.pct_change(3)
        else:
            feats["sp500_ret_1m"] = np.nan
            feats["sp500_ret_3m"] = np.nan

        if "VIX" in ext_monthly.columns:
            vix = ext_monthly["VIX"].reindex(feats.index, method="ffill")
            feats["vix_level"] = vix
            feats["vix_chg_1m"] = vix.pct_change(1)
        else:
            feats["vix_level"] = np.nan
            feats["vix_chg_1m"] = np.nan

        if "USDKRW" in ext_monthly.columns:
            fx = ext_monthly["USDKRW"].reindex(feats.index, method="ffill")
            feats["usdkrw_level"] = fx
            feats["usdkrw_chg_1m"] = fx.pct_change(1)
        else:
            feats["usdkrw_level"] = np.nan
            feats["usdkrw_chg_1m"] = np.nan

        if "WTI" in ext_monthly.columns:
            oil = ext_monthly["WTI"].reindex(feats.index, method="ffill")
            feats["oil_ret_1m"] = oil.pct_change(1)
        else:
            feats["oil_ret_1m"] = np.nan
    else:
        for col in [
            "sp500_ret_1m",
            "sp500_ret_3m",
            "vix_level",
            "vix_chg_1m",
            "usdkrw_level",
            "usdkrw_chg_1m",
            "oil_ret_1m",
        ]:
            feats[col] = np.nan

    # KOSPI-KOSDAQ 상대 수익률
    if other_index_ohlcv is not None and not other_index_ohlcv.empty:
        other_monthly = _monthly_resample(other_index_ohlcv, "종가")
        other_ret = other_monthly.pct_change(1).reindex(feats.index, method="ffill")
        feats["kospi_kosdaq_spread"] = feats["ret_1m"] - other_ret
    else:
        feats["kospi_kosdaq_spread"] = np.nan

    # 계절성
    feats["month_of_year"] = feats.index.month

    # ── F. KB 전략 기반 (5개) ──
    # (KB증권 '1월 전략' 정량 데이터 기반)
    _months = feats.index.month

    # 1) 어닝스 모멘텀 계절성 — Hit Ratio를 ±1 스케일로 정규화
    #    HR=81.3% → (81.3-50)/50 = +0.626, HR=31.3% → -0.374
    feats["kb_earnings_season"] = _months.map(
        lambda m: (KB_EARNINGS_HIT_RATIO[m] - 50.0) / 50.0
    )

    # 2) 분기 레짐 — 3저 호황 Q1/Q3=+1, Q2/Q4=-1
    feats["kb_quarter_regime"] = _months.map(
        lambda m: KB_QUARTER_REGIME[(m - 1) // 3 + 1]
    ).astype(float)

    # 3) 조정 리스크 — 3저 호황 조정 집중월 (Mar,Apr,Oct,Nov) = 1
    feats["kb_correction_risk"] = _months.map(
        lambda m: 1.0 if m in KB_CORRECTION_MONTHS else 0.0
    )

    # 4,5) 6개월 주기 사이클 (sin/cos) — Q1/Q3 상승, Q2/Q4 조정
    #       period=6 months, phase: peak at Feb (month=2)
    feats["kb_cycle_sin"] = np.sin(2 * np.pi * (_months - 2) / 6)
    feats["kb_cycle_cos"] = np.cos(2 * np.pi * (_months - 2) / 6)

    # 6) 알파/베타 레짐 — Q1/Q3=1(alpha-driven), Q2/Q4=0(beta-control)
    feats["kb_alpha_beta_regime"] = _months.map(
        lambda m: float(KB_ALPHA_BETA_REGIME[(m - 1) // 3 + 1])
    )

    # 7) 매크로 레짐 — Q1 통화완화(+1), Q2 긴축위험(-1), Q3 회복(+0.5), Q4 중립(0)
    _macro_map = {1: +1.0, 2: -1.0, 3: +0.5, 4: 0.0}
    feats["kb_macro_regime"] = _months.map(
        lambda m: _macro_map[(m - 1) // 3 + 1]
    ).astype(float)

    # 8) 업종순환 위상 — sin(2π(m-1)/12), 성장주 피크=Q1, 방어주 전환=Q3
    feats["kb_sector_rotation_phase"] = np.sin(2 * np.pi * (_months - 1) / 12)

    # 9) 원시 어닝스 모멘텀 1M — 연환산 %p를 정규화 (max abs ~25.8)
    _max_mmt = max(abs(v) for v in KB_EARNINGS_MMT_1M.values())
    feats["kb_earnings_mmt_raw"] = _months.map(
        lambda m: KB_EARNINGS_MMT_1M[m] / _max_mmt
    )

    # ── G. 상승장 패턴 / 조정 예측 (10개) ──
    # 과거 KOSPI 상승장(3저호황, 중국호황, 동학개미)에서 반복된 패턴:
    #   "저점 → 급등 → 과열 → 조정(-10~18%) → 재상승"
    # 이 사이클을 모델이 학습할 수 있도록 다면적 피처 구성.

    # 1) 저점 대비 상승률 — 최근 12개월 저점에서 얼마나 올랐나
    #    3저호황: +200%, 동학개미: +120% 수준의 랠리를 포착
    trough_12m = monthly_close.rolling(12, min_periods=3).min()
    feats["rally_from_trough"] = (
        monthly_close / trough_12m.replace(0, np.nan) - 1
    ).clip(0, 5)

    # 2) 상승 속도 — 3개월 수익률의 연율화 (급등 감지)
    #    정상 상승: ~10-15%/년, 급등: >40%/년 → 조정 확률 급증
    feats["rally_velocity"] = ((1 + feats["ret_3m"].fillna(0)) ** 4 - 1).clip(-1, 5)

    # 3) 추세 가속도 — 최근 3개월 수익률 vs 이전 3개월 수익률 차이
    #    양수 → 가속 상승(과열 초기), 음수 → 감속(조정 전조)
    prev_3m_ret = monthly_close.pct_change(3).shift(3)
    feats["trend_acceleration"] = (feats["ret_3m"] - prev_3m_ret).clip(-1, 1)

    # 4) 추세선 대비 편차 — 12개월 선형추세에서 얼마나 벗어났나
    #    +2σ 이상: 과열 오버슈팅, -2σ 이하: 과매도
    def _trend_deviation(series, window=12):
        """선형추세 잔차를 표준편차 단위로 반환."""
        result = pd.Series(np.nan, index=series.index)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        for i in range(window, len(series)):
            y = series.iloc[i - window : i].values
            if np.any(np.isnan(y)):
                continue
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / max(x_var, 1e-10)
            trend_end = y_mean + slope * (window - 1 - x_mean)
            residuals = y - (y_mean + slope * (x - x_mean))
            res_std = residuals.std()
            if res_std > 1e-10:
                result.iloc[i] = (series.iloc[i] - trend_end) / res_std
        return result

    feats["trend_deviation"] = _trend_deviation(monthly_close, 12).clip(-4, 4)

    # 5) 연속 상승 개월 수 — 길수록 조정 임박 가능성 증가
    #    역사적 패턴: 5~7개월 연속 상승 후 조정 빈번
    _up = (monthly_ret > 0).astype(int)
    _consec = pd.Series(0, index=monthly_ret.index)
    for i in range(1, len(_up)):
        if _up.iloc[i] == 1:
            _consec.iloc[i] = _consec.iloc[i - 1] + 1
        else:
            _consec.iloc[i] = 0
    feats["consecutive_up_months"] = _consec.astype(float)

    # 6) 오버슈팅 강도 — RSI과매수 + 추세편차 + 급등속도 복합
    #    3저호황/동학개미 상승장 정점에서 이 값이 극대화
    _rsi_overshoot = (feats["rsi_monthly"] - 0.7).clip(0, 0.3) / 0.3  # RSI>70 과매수
    _trend_overshoot = feats["trend_deviation"].clip(0, 4) / 4.0  # 추세 상방이탈
    _vel_overshoot = (feats["rally_velocity"] - 0.3).clip(0, 2) / 2.0  # 연30%↑ 급등
    feats["overshoot_intensity"] = (
        _rsi_overshoot.fillna(0) + _trend_overshoot.fillna(0) + _vel_overshoot.fillna(0)
    ) / 3.0

    # 7-8) PBR/PER 밴드 내 위치 (10년 min-max 기준, 0~1)
    #    0에 가까우면 밴드 하단(우상향 여력 큼), 1이면 밴드 상단(조정 위험)
    #    KOSPI PBR이 0.84로 역대 최저 → 밴드 하단 ≈ 장기 우상향 근거
    pbr_10y_min = pbr_s.rolling(120, min_periods=36).min()
    pbr_10y_max = pbr_s.rolling(120, min_periods=36).max()
    pbr_range = (pbr_10y_max - pbr_10y_min).replace(0, np.nan)
    feats["pbr_band_position"] = ((pbr_s - pbr_10y_min) / pbr_range).clip(0, 1)

    per_10y_min = per_s.rolling(120, min_periods=36).min()
    per_10y_max = per_s.rolling(120, min_periods=36).max()
    per_range = (per_10y_max - per_10y_min).replace(0, np.nan)
    feats["per_band_position"] = ((per_s - per_10y_min) / per_range).clip(0, 1)

    # 9) PBR 적정가 대비 괴리율
    #    = (PBR중앙값 - 현재PBR) / 현재PBR → 양수면 저평가(우상향 근거)
    pbr_10y_median = pbr_s.rolling(120, min_periods=36).median()
    feats["pbr_fair_value_gap"] = (
        (pbr_10y_median - pbr_s) / pbr_s.replace(0, np.nan)
    ).clip(-2, 2)

    # 10) 조정 위험 점수 — 상승장 내 급등 후 조정 가능성 종합 (0~1)
    #    구성: 오버슈팅 강도 40% + 연속상승 정규화 20% + PBR밴드상단 20% + 변동성축소 20%
    #    변동성 축소(vol_regime<0.7)는 폭풍전야 패턴 (3저호황, 동학개미 공통)
    _consec_norm = (feats["consecutive_up_months"] / 8.0).clip(0, 1)
    _pbr_top = feats["pbr_band_position"].fillna(0.5)
    _vol_complacency = (1.0 - feats["vol_regime"].clip(0.5, 1.5)).clip(0, 1)
    feats["correction_risk_score"] = (
        0.4 * feats["overshoot_intensity"].fillna(0)
        + 0.2 * _consec_norm
        + 0.2 * _pbr_top
        + 0.2 * _vol_complacency.fillna(0)
    ).clip(0, 1)

    # ── H. 큰 조정 후 회복 추세 (5개) ──
    # 역사적 패턴: 큰 조정(-20%↓) 이후 상승 추세가 수년간 지속
    #   IMF(-70%) → 3년 상승, GFC(-55%) → 7년 상승, COVID(-35%) → 1.5년 상승
    #   조정이 깊을수록 회복이 길고 강하다 → 모델이 이 관계를 학습

    # 1) 조정 깊이 — 24개월 고점 대비 최대 낙폭 (0 ~ -1 스케일)
    #    값이 -0.5 이하면 대형 조정 경험 → 이후 장기 상승 기대
    peak_24m = monthly_close.rolling(24, min_periods=6).max()
    feats["correction_depth"] = (
        (monthly_close.rolling(24, min_periods=6).min() - peak_24m)
        / peak_24m.replace(0, np.nan)
    ).clip(-1, 0)

    # 2) 저점 이후 경과 개월 수 — 0이면 지금이 저점, 큰 값이면 회복 중반/후반
    #    IMF: 저점 후 12개월 = 초기 회복(가장 강한 상승), 24개월 = 중기, 36+= 성숙기
    def _months_since_rolling_trough(close, window=24):
        """24개월 롤링 저점으로부터 경과 개월 수."""
        result = pd.Series(np.nan, index=close.index)
        for i in range(window, len(close)):
            window_data = close.iloc[max(0, i - window) : i + 1]
            trough_idx = window_data.idxmin()
            months_passed = (close.index[i] - trough_idx).days / 30.44
            result.iloc[i] = months_passed
        return result

    feats["months_since_trough"] = _months_since_rolling_trough(monthly_close, 24).clip(
        0, 36
    )

    # 3) 낙폭 회복 비율 — 고점→저점 낙폭의 몇 % 를 회복했나
    #    0=아직 저점, 1=고점 완전 회복, >1=신고가 돌파
    #    0.3~0.7 구간이 상승 추세 '달리는 구간'
    _peak = monthly_close.rolling(24, min_periods=6).max()
    _trough = monthly_close.rolling(24, min_periods=6).min()
    _drop = (_peak - _trough).replace(0, np.nan)
    feats["drawdown_recovery_ratio"] = ((monthly_close - _trough) / _drop).clip(0, 2)

    # 4) 조정 후 모멘텀 — 저점 대비 상승률 × (1 - 조정깊이의 절대값)
    #    큰 조정 후 초기 회복은 모멘텀이 매우 강함 (V자 반등)
    #    GFC 저점 후 6개월: +50%, COVID 후 6개월: +60%
    feats["post_correction_momentum"] = (
        feats["rally_from_trough"] * (-feats["correction_depth"]).fillna(0)
    ).clip(0, 3)

    # 5) 회복 국면 — 0=하락중, 1=초기회복, 2=중기회복, 3=성숙기/신고가
    #    모델이 "지금 어느 국면인가"를 직접 학습
    _recovery = feats["drawdown_recovery_ratio"].fillna(0)
    _mst = feats["months_since_trough"].fillna(0)
    feats["recovery_phase"] = pd.Series(0.0, index=feats.index)
    # 하락 중 (고점 대비 -10% 이상 + 최근 수익 음)
    _falling = (feats["pct_from_12m_high"] < -0.10) & (feats["ret_3m"] < 0)
    # 초기 회복 (저점 후 0~6개월, 회복 < 50%)
    _early = (_mst <= 6) & (_mst > 0) & (_recovery < 0.5)
    # 중기 회복 (회복 50~100%)
    _mid = (_recovery >= 0.5) & (_recovery < 1.0)
    # 성숙기 (고점 회복 이상)
    _mature = _recovery >= 1.0
    feats.loc[_falling, "recovery_phase"] = 0.0
    feats.loc[_early, "recovery_phase"] = 1.0
    feats.loc[_mid, "recovery_phase"] = 2.0
    feats.loc[_mature, "recovery_phase"] = 3.0

    # 피처 순서 정렬 (누락 피처는 NaN)
    for col in FEATURE_NAMES:
        if col not in feats.columns:
            feats[col] = np.nan
    feats = feats[FEATURE_NAMES]

    # 수급 피처 정규화 (억원 단위 → z-score)
    flow_cols = [
        "foreign_net_1m",
        "foreign_net_3m",
        "inst_net_1m",
        "inst_net_3m",
        "individual_net_1m",
    ]
    for col in flow_cols:
        if col in feats.columns:
            s = feats[col]
            mean = s.rolling(60, min_periods=12).mean()
            std = s.rolling(60, min_periods=12).std().replace(0, np.nan)
            feats[col] = (s - mean) / std

    return feats


# ──────────────────────────────────────────────
# 3단계: 라벨 생성
# ──────────────────────────────────────────────


def compute_index_labels(monthly_close: pd.Series) -> pd.DataFrame:
    """N_PRED_MONTHS개월 전방 누적 수익률 라벨 생성.

    fwd_ret_hm = close[t+h] / close[t] - 1
    """
    labels = pd.DataFrame(index=monthly_close.index)
    for h in range(1, N_PRED_MONTHS + 1):
        labels[f"fwd_ret_{h}m"] = monthly_close.shift(-h) / monthly_close - 1
    return labels


# ──────────────────────────────────────────────
# 4단계: 모델 학습
# ──────────────────────────────────────────────


def _try_import_lightgbm():
    try:
        import lightgbm as lgb

        return True, lgb
    except ImportError:
        return False, None


def _train_lgb_quantile(X, y, alpha, lgb, num_round=200, w=None):
    """단일 LightGBM 분위수 회귀 모델 학습."""
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "min_data_in_leaf": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "verbose": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X, label=y, weight=w)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
    )
    return model


def _train_sklearn_quantile(X, y, alpha, w=None):
    """sklearn HistGradientBoosting 분위수 회귀 (fallback)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=alpha,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(X, y, sample_weight=w)
    return model


def train_index_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    index_name: str = "KOSPI",
    progress_callback=None,
) -> dict:
    """지수 예측 모델 학습 (LightGBM 분위수 회귀 + Walk-Forward 검증).

    12 예측 구간 × 3 분위수 = 36 모델.

    Returns:
        dict with models, validation, feature_names, etc.
    """
    if progress_callback is None:

        def progress_callback(msg, pct=None):
            pass

    lgb_ok, lgb = _try_import_lightgbm()

    # 유효 데이터 (features + labels 모두 있는 행)
    common_idx = features.index.intersection(labels.index)
    feats = features.loc[common_idx].copy()
    lbls = labels.loc[common_idx].copy()

    # NaN 라벨 행 제거 (마지막 12개월은 fwd_ret_12m이 NaN)
    valid_mask = lbls.notna().all(axis=1)
    feats_valid = feats[valid_mask]
    lbls_valid = lbls[valid_mask]

    n_samples = len(feats_valid)
    if n_samples < 48:
        raise ValueError(f"학습 데이터 부족: {n_samples}개월 (최소 48개월 필요)")

    progress_callback(
        f"{index_name} 학습 시작: {n_samples}개월, {len(FEATURE_NAMES)}피처", 50
    )

    X_all = feats_valid.values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = list(feats_valid.columns)

    # KB 80% 가중치 체계: Hit Ratio 높은 월에 4~8배 가중
    # 기존 모델 기여 ~20% (base=0.30), KB 유리 월 최대 2.50
    _months_arr = feats_valid.index.month

    def _kb_sample_weight(month: int) -> float:
        hr = KB_EARNINGS_HIT_RATIO[month]
        base = 0.30
        if hr >= 75:
            return base + 2.20  # ~2.50 (Feb=81.3%, Aug=75%)
        elif hr >= 65:
            return base + 1.50  # ~1.80 (Apr, Jun, Dec)
        elif hr >= 55:
            return base + 0.80  # ~1.10 (Jan, Mar, May)
        elif hr >= 45:
            return base + 0.30  # ~0.60 (Jul, Sep)
        else:
            return base  # ~0.30 (Oct=31.3%, Nov=37.5%)

    w_all = np.array(
        [_kb_sample_weight(m) for m in _months_arr],
        dtype=np.float64,
    )

    # Walk-Forward 검증 (3 folds)
    fold_results = []
    fold_splits = [
        (0, int(n_samples * 0.60), int(n_samples * 0.75)),
        (0, int(n_samples * 0.75), int(n_samples * 0.88)),
        (0, int(n_samples * 0.88), n_samples),
    ]

    for fi, (_, train_end, test_end) in enumerate(fold_splits):
        X_train = X_all[:train_end]
        X_test = X_all[train_end:test_end]

        fold_mae = []
        fold_dir_acc = []
        fold_coverage = []

        # 검증은 대표 구간만 (1,3,6,12,24,36) — 전체 36개월 하면 과다
        _val_horizons = [h for h in [1, 3, 6, 12, 24, 36] if h <= N_PRED_MONTHS]
        for h in _val_horizons:
            y_col = f"fwd_ret_{h}m"
            y_train = lbls_valid[y_col].values[:train_end]
            y_test = lbls_valid[y_col].values[train_end:test_end]

            if len(X_test) == 0:
                continue

            w_train = w_all[:train_end]
            if lgb_ok:
                m_med = _train_lgb_quantile(X_train, y_train, 0.50, lgb, w=w_train)
                m_low = _train_lgb_quantile(X_train, y_train, 0.10, lgb, w=w_train)
                m_up = _train_lgb_quantile(X_train, y_train, 0.90, lgb, w=w_train)
                pred_med = m_med.predict(X_test)
                pred_low = m_low.predict(X_test)
                pred_up = m_up.predict(X_test)
            else:
                m_med = _train_sklearn_quantile(X_train, y_train, 0.50, w=w_train)
                m_low = _train_sklearn_quantile(X_train, y_train, 0.10, w=w_train)
                m_up = _train_sklearn_quantile(X_train, y_train, 0.90, w=w_train)
                pred_med = m_med.predict(X_test)
                pred_low = m_low.predict(X_test)
                pred_up = m_up.predict(X_test)

            # MAE
            mae = np.mean(np.abs(pred_med - y_test))
            fold_mae.append(mae)

            # 방향 정확도
            actual_dir = np.sign(y_test)
            pred_dir = np.sign(pred_med)
            dir_acc = np.mean(actual_dir == pred_dir)
            fold_dir_acc.append(dir_acc)

            # CI 커버리지
            covered = np.mean((y_test >= pred_low) & (y_test <= pred_up))
            fold_coverage.append(covered)

        fold_results.append(
            {
                "mae": np.mean(fold_mae) if fold_mae else np.nan,
                "direction_accuracy": np.mean(fold_dir_acc) if fold_dir_acc else np.nan,
                "coverage": np.mean(fold_coverage) if fold_coverage else np.nan,
            }
        )

        progress_callback(
            f"Walk-Forward Fold {fi + 1}/3: MAE={np.mean(fold_mae):.4f}, "
            f"방향={np.mean(fold_dir_acc):.1%}, 커버리지={np.mean(fold_coverage):.1%}",
            55 + fi * 10,
        )

    # 최종 모델 (전체 데이터로 학습)
    _n_models = N_PRED_MONTHS * 3
    progress_callback(f"최종 모델 학습 ({_n_models}개)...", 85)
    models = {}
    for h in range(1, N_PRED_MONTHS + 1):
        y_col = f"fwd_ret_{h}m"
        y_all = lbls_valid[y_col].values

        if lgb_ok:
            models[h] = {
                "median": _train_lgb_quantile(X_all, y_all, 0.50, lgb, w=w_all),
                "lower": _train_lgb_quantile(X_all, y_all, 0.10, lgb, w=w_all),
                "upper": _train_lgb_quantile(X_all, y_all, 0.90, lgb, w=w_all),
            }
        else:
            models[h] = {
                "median": _train_sklearn_quantile(X_all, y_all, 0.50, w=w_all),
                "lower": _train_sklearn_quantile(X_all, y_all, 0.10, w=w_all),
                "upper": _train_sklearn_quantile(X_all, y_all, 0.90, w=w_all),
            }

    # 피처 중요도 (median 모델 기준)
    importance = {}
    _imp_horizons = [h for h in [1, 3, 6, 12, 24, 36] if h <= N_PRED_MONTHS]
    for h in _imp_horizons:
        m = models[h]["median"]
        if lgb_ok:
            imp = m.feature_importance(importance_type="gain")
        else:
            imp = getattr(m, "feature_importances_", np.zeros(len(feature_names)))
        importance[f"{h}m"] = dict(zip(feature_names, imp, strict=False))

    # 검증 결과 집계
    avg_mae = np.mean([f["mae"] for f in fold_results])
    avg_dir = np.mean([f["direction_accuracy"] for f in fold_results])
    avg_cov = np.mean([f["coverage"] for f in fold_results])

    progress_callback(
        f"학습 완료! MAE={avg_mae:.4f}, 방향정확도={avg_dir:.1%}, "
        f"CI커버리지={avg_cov:.1%}",
        95,
    )

    return {
        "models": models,
        "feature_names": feature_names,
        "validation": {
            "avg_mae": avg_mae,
            "avg_direction_accuracy": avg_dir,
            "avg_coverage": avg_cov,
            "folds": fold_results,
        },
        "importance": importance,
        "train_date": datetime.now().strftime("%Y%m%d"),
        "index_name": index_name,
        "n_train_samples": n_samples,
        "use_lgbm": lgb_ok,
    }


# ──────────────────────────────────────────────
# 5단계: 예측
# ──────────────────────────────────────────────


def predict_index_future(
    model_data: dict,
    latest_features: pd.Series,
    current_price: float,
) -> dict:
    """N_PRED_MONTHS개월 미래 지수 예측.

    Args:
        model_data: train_index_model() 반환값
        latest_features: 최신 월 피처 (Series or 1-row array)
        current_price: 현재 지수 종가

    Returns:
        dict with predictions, current_price, prediction_date
    """
    models = model_data["models"]
    use_lgb = model_data.get("use_lgbm", False)

    if isinstance(latest_features, pd.Series):
        X = latest_features.values.reshape(1, -1)
    else:
        X = np.array(latest_features).reshape(1, -1)

    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    predictions = []
    for h in range(1, N_PRED_MONTHS + 1):
        m = models[h]
        if use_lgb:
            med = float(m["median"].predict(X)[0])
            low = float(m["lower"].predict(X)[0])
            up = float(m["upper"].predict(X)[0])
        else:
            med = float(m["median"].predict(X)[0])
            low = float(m["lower"].predict(X)[0])
            up = float(m["upper"].predict(X)[0])

        predictions.append(
            {
                "month": h,
                "return_median": med,
                "return_lower": low,
                "return_upper": up,
                "price_median": current_price * (1 + med),
                "price_lower": current_price * (1 + low),
                "price_upper": current_price * (1 + up),
            }
        )

    # ── 3중 블렌딩: 50% AI모델 + 25% KB전략 + 25% 상승장패턴 ──
    # AI모델(50%): PER/PBR/매크로 기반 LightGBM 예측
    # KB전략(25%): KB증권 PDF 분기별 기대수익률 + 월별 오버레이
    # 상승장패턴(25%): 3저호황·IMF회복·유동성장세·코로나반등 가중평균
    _model_w = 0.50
    _kb_w = 0.25
    _bull_w = 0.25
    _now_month = datetime.now().month
    for p in predictions:
        h = p["month"]
        _future_m = (_now_month + h - 1) % 12 + 1
        _future_q = (_future_m - 1) // 3 + 1
        # (A) KB 분기별 기대수익률 × 기간 비례 + 월별 오버레이 누적
        _q_ret = KB_Q_RETURN_EXPECTATION[_future_q]
        _kb_cum = _q_ret * (h / 12.0)
        _mmt_adj = sum(
            KB_MONTHLY_OVERLAY.get((_now_month + j - 1) % 12 + 1, 0.0)
            for j in range(1, h + 1)
        )
        _kb_adj = _kb_cum + _mmt_adj
        # (B) 과거 상승장 패턴 가중평균 수익률
        _bull_ret = _bull_pattern_return(h)
        # 블렌딩
        raw_med = p["return_median"]
        raw_low = p["return_lower"]
        raw_up = p["return_upper"]
        p["return_median"] = _model_w * raw_med + _kb_w * _kb_adj + _bull_w * _bull_ret
        # CI: 패턴 확신도 반영으로 폭 조정
        _ci_w = raw_up - raw_low
        _shrunk = _ci_w * (_model_w + _kb_w * 0.4 + _bull_w * 0.3)
        p["return_lower"] = p["return_median"] - _shrunk / 2
        p["return_upper"] = p["return_median"] + _shrunk / 2
        # 가격 재계산 (전쟁 오버레이 적용 전 기본 블렌딩)
        p["price_median"] = current_price * (1 + p["return_median"])
        p["price_lower"] = current_price * (1 + p["return_lower"])
        p["price_upper"] = current_price * (1 + p["return_upper"])

    # ── 전쟁패턴 오버레이: 이란전쟁(2026-02-28) 발발 이후 중동전쟁 패턴 반영 ──
    # 과거 걸프전(1990), 이라크전(2003), 이란긴장(2020) 실제 KOSPI 데이터 기반
    _war_start = pd.Timestamp(IRAN_WAR_START)
    _pred_dt = pd.Timestamp(datetime.now().strftime("%Y%m%d"))
    _war_months_elapsed = max(
        0,
        (_pred_dt.year - _war_start.year) * 12 + _pred_dt.month - _war_start.month,
    )
    _war_weight = 0.15  # 전쟁 오버레이 가중치
    _war_cum_now = (
        _war_pattern_return(_war_months_elapsed) if _war_months_elapsed > 0 else 0.0
    )
    for p in predictions:
        h = p["month"]
        _war_m = _war_months_elapsed + h
        if _war_m > 18:
            continue  # 18개월 이후는 전쟁 영향 소멸
        # 예측 시점까지의 누적수익률 - 현재까지 반영된 수익률 = 증분
        _war_cum_future = _war_pattern_return(_war_m)
        _war_incr = _war_cum_future - _war_cum_now
        # 시간 감쇠 (18개월에 걸쳐 선형 감소)
        _decay = max(0.0, 1.0 - _war_m / 24.0)
        _war_adj = _war_incr * _decay * _war_weight
        p["return_median"] += _war_adj
        p["return_lower"] += _war_adj * 0.7  # 하방 리스크 더 크게 반영
        p["return_upper"] += _war_adj * 1.3
        p["price_median"] = current_price * (1 + p["return_median"])
        p["price_lower"] = current_price * (1 + p["return_lower"])
        p["price_upper"] = current_price * (1 + p["return_upper"])
        p["war_overlay"] = _war_adj  # 전쟁 오버레이 크기 저장

    return {
        "predictions": predictions,
        "current_price": current_price,
        "prediction_date": datetime.now().strftime("%Y%m%d"),
        "index_name": model_data.get("index_name", ""),
        "iran_war_start": IRAN_WAR_START,
        "war_months_elapsed": _war_months_elapsed,
        "kb_blend_ratio": {
            "model": _model_w,
            "kb": _kb_w,
            "bull_pattern": _bull_w,
        },
    }


_FINE_MONTHS = 3  # 세밀 예측 구간 (예측 기준일 이후 3개월)
_FINE_INTERVAL_DAYS = 2  # 세밀 구간 내 간격 (2일마다 = 약 45 포인트)


def interpolate_weekly_predictions(
    pred_data: dict,
    current_price: float | None = None,
    apply_kb_overlay: bool = True,
    seed: int = 42,
) -> dict:
    """월간 예측을 보간: 첫 3개월 일별(2일 간격), 이후 주간.

    KB 1월 전략의 분기별 알파/베타 전략을 반영:
    - Q1(1~3월): 알파 드리븐 → 상향 조정
    - Q2(4~6월): 베타 컨트롤 → 하향 조정
    - Q3(7~9월): 회복 랠리 → 상향
    - Q4(10~12월): 어닝스/조정 → 중립

    + 전쟁 패턴 오버레이 (이란전쟁 2026-02-28 이후)

    Args:
        pred_data: predict_index_future() 결과
        current_price: 현재가 (None이면 pred_data에서 읽음)
        apply_kb_overlay: KB 전략 분기 조정 적용 여부
        seed: 재현성을 위한 랜덤 시드

    Returns:
        dict with weekly_predictions (첫 3개월 일별 + 이후 주간), 원본 monthly 보존
    """
    if current_price is None:
        current_price = pred_data["current_price"]

    pred_date = pd.Timestamp(
        pred_data.get("prediction_date", datetime.now().strftime("%Y%m%d"))
    )

    n_months = len(pred_data["predictions"])
    n_weeks = n_months * 52 // 12  # 36개월 → 156주
    weeks_per_month = n_weeks / n_months

    # 월간 예측 수익률 추출 (0=현재, 1~N=미래)
    monthly_returns = [0.0]
    for p in pred_data["predictions"]:
        monthly_returns.append(p["return_median"])

    monthly_returns_lower = [0.0]
    monthly_returns_upper = [0.0]
    for p in pred_data["predictions"]:
        monthly_returns_lower.append(p["return_lower"])
        monthly_returns_upper.append(p["return_upper"])

    # 월 → 주 매핑 (0~N월 → 0~N_WEEKS주)
    month_points = np.array([m * weeks_per_month for m in range(n_months + 1)])

    # Cubic spline 보간 (전체 기간 공통)
    cs_median = _scipy_interp.CubicSpline(
        month_points, monthly_returns, bc_type="natural"
    )
    cs_lower = _scipy_interp.CubicSpline(
        month_points, monthly_returns_lower, bc_type="natural"
    )
    cs_upper = _scipy_interp.CubicSpline(
        month_points, monthly_returns_upper, bc_type="natural"
    )

    # ── 샘플 포인트 생성: 첫 3개월=2일 간격, 이후=주간 ──
    fine_weeks = _FINE_MONTHS * weeks_per_month  # ~13주
    fine_days = int(_FINE_MONTHS * 30.44)  # ~91일
    n_fine = fine_days // _FINE_INTERVAL_DAYS  # ~45 포인트

    sample_points = []
    sample_is_daily = []
    # 첫 3개월: 2일 간격 (주 단위로 환산)
    for i in range(1, n_fine + 1):
        day_offset = i * _FINE_INTERVAL_DAYS
        week_equiv = day_offset / 7.0
        if week_equiv <= fine_weeks:
            sample_points.append(week_equiv)
            sample_is_daily.append(True)
    # 이후: 주간 (fine_weeks 이후)
    _first_weekly = int(fine_weeks) + 1
    for w in range(_first_weekly, n_weeks + 1):
        sample_points.append(float(w))
        sample_is_daily.append(False)

    sample_arr = np.array(sample_points)

    # spline 보간
    ret_median = cs_median(sample_arr)
    ret_lower = cs_lower(sample_arr)
    ret_upper = cs_upper(sample_arr)

    # 미세 변동 추가 (Brownian bridge)
    rng = np.random.RandomState(seed)
    noise = np.zeros(len(sample_arr))
    for i, wp in enumerate(sample_arr):
        month_idx = int(wp * n_months / n_weeks)
        month_frac = (wp * n_months / n_weeks) - month_idx
        damping = 4 * month_frac * (1 - month_frac)
        # 일별 포인트는 변동 폭 줄임 (주간의 1/3)
        vol_scale = WEEKLY_VOL_SCALE * (0.3 if sample_is_daily[i] else 1.0)
        noise[i] = rng.normal(0, vol_scale * damping)

    # 노이즈 스무딩
    kernel = np.array([0.15, 0.25, 0.35, 0.25])
    for _ in range(2):
        smoothed = np.convolve(noise, kernel / kernel.sum(), mode="same")
        noise[1:-1] = smoothed[1:-1]
    noise[0] = 0
    ret_median = ret_median + noise

    # KB 전략 분기별 오버레이
    if apply_kb_overlay:
        for i, wp in enumerate(sample_arr):
            days_offset = wp * 7
            future_date = pred_date + pd.DateOffset(days=int(days_offset))
            month = future_date.month
            overlay = KB_MONTHLY_OVERLAY.get(month, 0.0) * 2.0
            # 일별 포인트는 비례 축소
            interval_weeks = _FINE_INTERVAL_DAYS / 7.0 if sample_is_daily[i] else 1.0
            scaled_overlay = overlay / weeks_per_month * interval_weeks
            ret_median[i] += scaled_overlay
            ret_lower[i] += scaled_overlay
            ret_upper[i] += scaled_overlay

    # 전쟁 패턴 오버레이 (일별/주간 레벨)
    _war_start = pd.Timestamp(IRAN_WAR_START)
    for i, wp in enumerate(sample_arr):
        days_offset = wp * 7
        future_date = pred_date + pd.DateOffset(days=int(days_offset))
        _war_days = (future_date - _war_start).days
        _war_m = max(0, _war_days / 30.44)  # 전쟁 발발 후 경과 개월
        if 0 < _war_m <= 18:
            # 전쟁 시작 시점의 누적수익률
            _war_now_m = max(0, (pred_date - _war_start).days / 30.44)
            _war_cum_now = (
                _war_pattern_return(int(_war_now_m)) if _war_now_m > 0 else 0.0
            )
            _war_cum_future = _war_pattern_return(int(_war_m))
            _war_incr = _war_cum_future - _war_cum_now
            _decay = max(0.0, 1.0 - _war_m / 24.0)
            _war_adj = _war_incr * _decay * 0.10  # 보간 레벨에선 10%
            ret_median[i] += _war_adj
            ret_lower[i] += _war_adj * 0.7
            ret_upper[i] += _war_adj * 1.3

    # 결과 생성
    weekly_predictions = []
    for i, wp in enumerate(sample_arr):
        days_offset = wp * 7
        future_date = pred_date + pd.DateOffset(days=int(days_offset))
        # week 값: 일별 포인트는 소수점, 주간은 정수
        week_val = round(wp, 2) if sample_is_daily[i] else int(wp)
        weekly_predictions.append(
            {
                "week": week_val,
                "date": future_date.strftime("%Y-%m-%d"),
                "return_median": float(ret_median[i]),
                "return_lower": float(ret_lower[i]),
                "return_upper": float(ret_upper[i]),
                "price_median": current_price * (1 + float(ret_median[i])),
                "price_lower": current_price * (1 + float(ret_lower[i])),
                "price_upper": current_price * (1 + float(ret_upper[i])),
                "is_daily": sample_is_daily[i],
            }
        )

    return {
        "weekly_predictions": weekly_predictions,
        "predictions": pred_data["predictions"],
        "current_price": current_price,
        "prediction_date": pred_data.get("prediction_date"),
        "index_name": pred_data.get("index_name", ""),
        "kb_overlay_applied": apply_kb_overlay,
        "iran_war_start": pred_data.get("iran_war_start"),
        "war_months_elapsed": pred_data.get("war_months_elapsed"),
    }


# ──────────────────────────────────────────────
# 5-B단계: 월별 투자전략 생성
# ──────────────────────────────────────────────

_Q_STRATEGY_NAME = {
    1: "알파 드리븐",
    2: "베타 컨트롤",
    3: "회복 랠리",
    4: "어닝스/연말",
}
_KOSDAQ_FOCUS_SECTORS = ["바이오", "IT서비스", "게임", "반도체"]


def generate_monthly_strategy(
    pred_result: dict,
    index_name: str = "KOSPI",
    recommend_results: dict | None = None,
) -> list[dict]:
    """36개월 월별 투자전략 생성 (KB 전략 80% 기반 확장판).

    Returns list of 36 dicts, each containing: invest_types (list[dict]),
    buy_priority, specific_conditions, risk_notes, sector_detail, kb_conviction.
    """
    predictions = pred_result.get("predictions", [])
    if not predictions:
        return []

    pred_date = pred_result.get("prediction_date")
    if isinstance(pred_date, str):
        for _fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
            try:
                pred_date = datetime.strptime(pred_date, _fmt).date()
                break
            except ValueError:
                continue
        else:
            pred_date = date.today()
    elif isinstance(pred_date, datetime):
        pred_date = pred_date.date()
    elif pred_date is None:
        pred_date = date.today()

    is_kosdaq = "KOSDAQ" in index_name.upper()
    strategies: list[dict] = []

    for i, p in enumerate(predictions):
        # ── 기본 날짜/분기 ──
        month_offset = i + 1
        target_year = pred_date.year + (pred_date.month + month_offset - 1) // 12
        target_month = (pred_date.month + month_offset - 1) % 12 + 1
        quarter = (target_month - 1) // 3 + 1
        date_label = f"{target_year}-{target_month:02d}"

        # ── 1단계: 분기 기본패턴 ──
        base_weight = KB_Q_WEIGHT.get(quarter, 0.80)
        style = KB_Q_STYLE.get(quarter, "성장주(AI반도체)")
        q_regime = KB_QUARTER_REGIME.get(quarter, 0)
        alpha_beta = KB_ALPHA_BETA_REGIME.get(quarter, 0)

        # ── 2단계: 월별 HR + 모멘텀 조정 ──
        hr = KB_EARNINGS_HIT_RATIO.get(target_month, 50.0)
        mmt = KB_EARNINGS_MMT_1M.get(target_month, 0.0)
        weight = base_weight
        if hr > 70:
            weight += 0.08
        elif hr > 60:
            weight += 0.05
        elif hr < 40:
            weight -= 0.08
        elif hr < 50:
            weight -= 0.05
        if target_month in KB_CORRECTION_MONTHS:
            weight -= 0.05 if q_regime >= 0 else 0.10

        # ── 3단계: 예측 반영 (7단계 방향 + 6단계 포지션) ──
        ret_median = p.get("return_median", 0.0)
        ret_lower = p.get("return_lower", ret_median - 0.05)
        ret_upper = p.get("return_upper", ret_median + 0.05)
        price_median = p.get("price_median", 0)

        if ret_median >= 0.05:
            direction = "강세상승"
            weight += 0.12
        elif ret_median >= 0.03:
            direction = "상승"
            weight += 0.10
        elif ret_median <= -0.05:
            direction = "강세하락"
            weight -= 0.18
        elif ret_median <= -0.03:
            direction = "하락"
            weight -= 0.15
        elif abs(ret_median) <= 0.01:
            direction = "횡보"
        elif ret_median > 0:
            direction = "소폭상승"
            weight += 0.03
        else:
            direction = "소폭하락"
            weight -= 0.05

        weight = max(0.20, min(1.0, weight))

        if weight >= 0.92:
            position = "적극매수"
        elif weight >= 0.80:
            position = "매수"
        elif weight >= 0.65:
            position = "소폭매수"
        elif weight >= 0.50:
            position = "관망"
        elif weight >= 0.35:
            position = "비중축소"
        else:
            position = "방어"

        # ── 4단계: 상세 투자유형 선정 ──
        invest_types: list[dict] = []
        q_priorities = KB_Q_INVEST_PRIORITY.get(quarter, ["ETF"])

        for type_key in q_priorities:
            type_info = KB_INVEST_TYPE_CATALOG.get(type_key)
            if type_info is None:
                continue
            include = True
            if type_key == "성장주" and direction in ("하락", "강세하락"):
                include = False
            elif type_key == "저PBR주" and direction == "강세상승":
                include = False
            elif type_key == "고배당주" and quarter != 4:
                include = False
            elif (
                type_key == "중소형주"
                and not is_kosdaq
                and direction in ("하락", "강세하락")
            ):
                include = False
            elif type_key == "증권주" and alpha_beta == 0:
                include = False
            elif type_key == "지주사" and quarter > 2:
                include = False
            if include:
                invest_types.append(
                    {
                        "key": type_key,
                        "label": type_info["label"],
                        "icon": type_info["icon"],
                        "sub_types": type_info.get("sub_types", []),
                        "condition": type_info.get("condition", ""),
                    }
                )

        if len(invest_types) < 2:
            invest_types.append(
                {
                    "key": "ETF",
                    "label": "ETF",
                    "icon": "\U0001f4ca",
                    "sub_types": ["인덱스ETF"],
                    "condition": "분산투자",
                }
            )
            if direction not in ("하락", "강세하락"):
                invest_types.append(
                    {
                        "key": "대형주",
                        "label": "대형주",
                        "icon": "\U0001f3e2",
                        "sub_types": ["우량주"],
                        "condition": "안전마진",
                    }
                )

        # ── 5단계: 업종 로테이션 (18개 업종) ──
        if is_kosdaq:
            top_sectors = list(_KOSDAQ_FOCUS_SECTORS)
            neutral_sectors: list[str] = []
            avoid_sectors: list[str] = []
            season_adj = KB_KOSDAQ_SEASON.get(target_month, 0.0)
            if season_adj > 1.0:
                weight = min(1.0, weight + 0.03)
            elif season_adj < -1.0:
                weight = max(0.20, weight - 0.03)
        else:
            month_scores = {
                s: v.get(target_month, 0) for s, v in KB_SECTOR_MONTHLY.items()
            }
            sorted_sectors = sorted(
                month_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_sectors = [s[0] for s in sorted_sectors[:4]]
            neutral_sectors = [s[0] for s in sorted_sectors[4:-3]]
            avoid_sectors = [s[0] for s in sorted_sectors[-3:]]

        sector_detail = {
            "overweight": top_sectors,
            "neutral": neutral_sectors,
            "underweight": avoid_sectors,
        }

        # ── 6단계: 매수우선순위 문자열 ──
        buy_parts = []
        for it in invest_types[:3]:
            sub = it["sub_types"][0] if it["sub_types"] else ""
            buy_parts.append(f"{it['label']}({sub})" if sub else it["label"])
        buy_priority = " > ".join(buy_parts)

        # ── 7단계: 구체적 매수조건 ──
        specific_conditions: list[str] = []
        _type_keys = {it["key"] for it in invest_types}
        if "성장주" in _type_keys:
            specific_conditions.append("PER 15배 이하 저평가 AI/반도체주 중심")
        if "저PER주" in _type_keys:
            specific_conditions.append("PER 10배 이하 실적턴어라운드 후보")
        if "저PBR주" in _type_keys:
            specific_conditions.append("PBR 1.0 이하 자산가치 저평가 종목")
        if "고배당주" in _type_keys:
            specific_conditions.append("배당수익률 4% 이상 안정배당 종목")
        if "증권주" in _type_keys:
            specific_conditions.append("거래대금 증가 추세 증권주 (Q1 alpha)")
        if "금융주" in _type_keys:
            specific_conditions.append("KB overweight +250bp, 은행/증권 실적개선")
        if "지주사" in _type_keys:
            specific_conditions.append("NAV 대비 할인율 50%+ 자본시장 활성화 수혜")
        if "수출주" in _type_keys:
            specific_conditions.append("환율 안정 시 자동차/조선/기계 수출 증가")
        if "내수주" in _type_keys:
            specific_conditions.append("환율 상승 방어, 소비재/유통/음식료")

        # ── 8단계: 리스크 노트 ──
        risk_notes: list[str] = []
        correction_risk = target_month in KB_CORRECTION_MONTHS and ret_median < 0
        if correction_risk:
            risk_notes.append(f"{target_month}월 3저호황 조정 경계 구간")
        if alpha_beta == 0 and ret_median > 0.03:
            risk_notes.append("Q2/Q4 베타 컨트롤: 상승폭 제한 가능")
        if hr < 45:
            risk_notes.append(f"어닝스 HR {hr:.0f}%: 수익 변동성 확대")
        if direction in ("하락", "강세하락"):
            risk_notes.append("하락 예측: 현금비중 확대 또는 인버스ETF 고려")
        if quarter == 2:
            risk_notes.append("금리인하 종료 우려, 매크로 불확실성")

        # ── 9단계: KB 확신도 ──
        kb_conviction = 0.5
        kb_conviction += (hr - 50) / 100
        if alpha_beta == 1:
            kb_conviction += 0.10
        if mmt > 10:
            kb_conviction += 0.10
        elif mmt < 0:
            kb_conviction -= 0.10
        kb_conviction = max(0.0, min(1.0, kb_conviction))

        # ── 10단계: KB 종목 추천 (PDF Table 19 + Research Coverage) ──
        stock_picks = []
        # KB_STOCK_PICKS에서 해당 분기 활성 업종의 종목 수집
        _top_sector_set = set(top_sectors)
        for _sec_name, _sec_info in KB_STOCK_PICKS.items():
            # 해당 분기에 활성이고 overweight 업종인 종목 우선
            if quarter not in _sec_info.get("quarters", []):
                continue
            if _sec_info.get("weight", 0) <= 0 and _sec_name not in _top_sector_set:
                continue
            for _stk in _sec_info["stocks"]:
                if _stk.get("opinion") == "Buy":
                    stock_picks.append(
                        {
                            "name": _stk["name"],
                            "sector": _sec_name,
                            "opinion": _stk["opinion"],
                            "reason": _stk["reason"],
                            "weight_bp": _sec_info["weight"],
                        }
                    )
        # 업종 가중치 높은 순 정렬, 최대 8종목
        stock_picks.sort(key=lambda x: -x["weight_bp"])
        stock_picks = stock_picks[:8]
        # AI추천주 결과가 있으면 병합 (상위 3개)
        if recommend_results and i < 6:
            ai_picks = recommend_results.get("top_picks", [])
            if ai_picks:
                for _ap in ai_picks[:3]:
                    _ap_name = _ap.get("name", _ap.get("종목명", ""))
                    if _ap_name and not any(p["name"] == _ap_name for p in stock_picks):
                        stock_picks.append(
                            {
                                "name": _ap_name,
                                "sector": _ap.get("sector", _ap.get("업종", "")),
                                "opinion": "AI",
                                "reason": f"AI스크리닝 점수 {_ap.get('score', _ap.get('점수', ''))}",
                                "weight_bp": 0,
                            }
                        )

        # ── 전략 노트 (확장판) ──
        q_name = _Q_STRATEGY_NAME.get(quarter, "")
        sector_str = "/".join(top_sectors[:3])
        note_parts = [
            f"Q{quarter} {q_name} | {buy_priority}.",
            f"예측 {ret_median:+.1%} {direction}.",
            f"HR {hr:.0f}%, 모멘텀 {mmt:+.1f}%p.",
            f"비중확대: {sector_str}.",
        ]
        if specific_conditions:
            note_parts.append(f"조건: {specific_conditions[0]}.")
        if risk_notes:
            note_parts.append(f"리스크: {risk_notes[0]}.")
        if is_kosdaq:
            season_val = KB_KOSDAQ_SEASON.get(target_month, 0)
            note_parts.append(f"코스닥 시즌성 {season_val:+.1f}%.")
        strategy_note = " ".join(note_parts)

        strategies.append(
            {
                "date_label": date_label,
                "quarter": quarter,
                "predicted_return": ret_median,
                "return_lower": ret_lower,
                "return_upper": ret_upper,
                "direction": direction,
                "position": position,
                "equity_weight": round(weight, 2),
                "style": style,
                "top_sectors": top_sectors,
                "avoid_sectors": avoid_sectors,
                "sector_detail": sector_detail,
                "invest_types": invest_types,
                "buy_priority": buy_priority,
                "specific_conditions": specific_conditions,
                "risk_notes": risk_notes,
                "kb_conviction": round(kb_conviction, 2),
                "strategy_note": strategy_note,
                "correction_risk": correction_risk,
                "hit_ratio": hr,
                "earnings_mmt": mmt,
                "price_median": price_median,
                "stock_picks": stock_picks,
            }
        )

    return strategies


# ──────────────────────────────────────────────
# 6단계: 모델 저장/로드
# ──────────────────────────────────────────────


def save_index_model(model_data: dict, date_str: str, index_name: str) -> Path:
    """학습 모델 저장 (날짜별 + latest)."""
    INDEX_PRED_DIR.mkdir(parents=True, exist_ok=True)

    path = INDEX_PRED_DIR / f"index_model_{index_name}_{date_str}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    latest = INDEX_PRED_DIR / f"index_model_{index_name}_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(model_data, f)

    return path


def load_index_model(index_name: str, date_str: str = None) -> dict | None:
    """학습 모델 로드."""
    if date_str:
        path = INDEX_PRED_DIR / f"index_model_{index_name}_{date_str}.pkl"
    else:
        path = INDEX_PRED_DIR / f"index_model_{index_name}_latest.pkl"

    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_index_prediction(pred_data: dict, date_str: str, index_name: str) -> Path:
    """예측 결과 저장."""
    INDEX_PRED_DIR.mkdir(parents=True, exist_ok=True)

    path = INDEX_PRED_DIR / f"index_pred_{index_name}_{date_str}.pkl"
    with open(path, "wb") as f:
        pickle.dump(pred_data, f)

    latest = INDEX_PRED_DIR / f"index_pred_{index_name}_latest.pkl"
    with open(latest, "wb") as f:
        pickle.dump(pred_data, f)

    return path


def load_index_prediction(index_name: str) -> dict | None:
    """최신 예측 결과 로드."""
    path = INDEX_PRED_DIR / f"index_pred_{index_name}_latest.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
