"""설정/임계치 관리 모듈.

11개 카테고리별 게이트, 스코어링, 타이밍 설정.
프리셋 로드/저장 기능 제공.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent.parent / "cache" / "presets"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ──────────────────────────────────────────────
# 22개 스코어링 조건 키 — 기본값 0 (사용 안 함)
# ──────────────────────────────────────────────
_ZERO_SCORING = {
    # 수급
    "fi_net_buy_1m": 0,
    "inst_net_buy_1m": 0,
    "fi_inst_both": 0,
    "fi_top30": 0,
    "inst_top30": 0,
    "fi_inst_both_top": 0,
    # 기술
    "macd_cross": 0,
    "rsi_turn_up": 0,
    "stoch_golden": 0,
    "ma20_cross": 0,
    "rsi40_cross": 0,
    # 자산/현금
    "cash_ratio_mid": 0,
    "cash_ratio_high": 0,
    "realestate_mid": 0,
    "realestate_high": 0,
    "both_asset_bonus": 0,
    # 펀더멘털
    "roe_positive": 0,
    "roe_above5": 0,
    "roe_above10": 0,
    "roa_above2": 0,
    # 배당/안전
    "div_above3": 0,
    "op_profit_positive": 0,
    # 영업이익률
    "op_margin_mid": 0,
    "op_margin_high": 0,
    "op_margin_improving": 0,
}


def _sc(**overrides):
    """스코어링 기본값에 override 적용."""
    base = _ZERO_SCORING.copy()
    base.update(overrides)
    return base


# ──────────────────────────────────────────────
# 11개 카테고리 설정
# ──────────────────────────────────────────────

CATEGORIES_CONFIG = {
    # ═══════════════════════════════════════════════════
    # 1. 자산괴리주 💎 — 숨은 자산가치가 시총을 압도
    # ═══════════════════════════════════════════════════
    # 투자논거: 부동산/현금이 시총의 120%↑. 시장이 자산가치를 반영하지 못한 상태.
    # 핵심필터: DART strict (재무확인 불가 종목 = 자산가치 모름 = 제외)
    #          자산비율 높은 임계치 + 수급 동반 = 촉매 확인
    # 타겟: 공장부지/토지 보유 제조업, 현금부자 지주사. 8~15종목.
    "자산괴리주": {
        "emoji": "💎",
        "description": "부동산/현금 자산 > 시총 120% + 영업흑자 + 수급동반",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.30},
            "roe": {"min": 0.03},
            "profit": {"strict": True},
            "asset": {"re_min": 1.2, "cash_min": 0.50, "strict": True},
            "supply_or": {"days_min": 5},
            "signal": {"min_count": 2, "rsi_max": 50},
            "breach": {},
        },
        "scoring": {
            "detail": _sc(
                cash_ratio_mid=6,
                cash_ratio_high=18,
                realestate_mid=8,
                realestate_high=22,
                both_asset_bonus=12,
                roe_positive=4,
                roe_above5=8,
                op_profit_positive=4,
                fi_net_buy_1m=8,
                inst_net_buy_1m=5,
                fi_inst_both=5,
                macd_cross=6,
                rsi_turn_up=4,
                stoch_golden=4,
            ),
        },
        "timing": {"strong_buy_score": 70, "buy_score": 50},
    },
    # ═══════════════════════════════════════════════════
    # 2. 고배당_가치주 💰 — 지속 가능한 고배당 + 시세차익
    # ═══════════════════════════════════════════════════
    # 투자논거: DIV 4%↑ 배당을 유지할 수 있는 수익성(ROE 5%↑) + 저평가.
    # 핵심필터: 적자기업 고배당 = 배당컷 위험 → DART strict 필수
    #          PBR ≤ 0.60 (진짜 저평가 배당주만)
    # 타겟: 유틸리티, 금융, 성숙 산업재. 8~15종목.
    "고배당_가치주": {
        "emoji": "💰",
        "description": "DIV 4%↑ + PBR≤0.6 + ROE 5%↑ + 영업흑자 확인",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.60},
            "roe": {"min": 0.05},
            "dividend": {"min": 4.0},
            "profit": {"strict": True},
            "supply_or": {"days_min": 5},
            "breach": {},
        },
        "scoring": {
            "detail": _sc(
                div_above3=18,
                roe_above5=10,
                roe_above10=15,
                roa_above2=8,
                op_profit_positive=5,
                fi_net_buy_1m=8,
                inst_net_buy_1m=6,
                fi_inst_both=5,
                macd_cross=6,
                ma20_cross=5,
                rsi40_cross=4,
                cash_ratio_mid=5,
                cash_ratio_high=8,
            ),
        },
        "timing": {"strong_buy_score": 70, "buy_score": 50},
    },
    # ═══════════════════════════════════════════════════
    # 3. 실적_턴어라운드 📈 — 실적 회복 초기 + 저평가
    # ═══════════════════════════════════════════════════
    # 투자논거: ROE 5%↑, ROA 2%↑ 으로 실적이 회복되기 시작한 종목.
    #          아직 시장이 실적 개선을 주가에 반영하지 못한 상태.
    # 핵심필터: 종합_밸런스형보다 낮은 ROE/ROA 임계치 (회복 초기 포착)
    #          DART strict로 실적 확인은 필수. PBR 0.55까지 허용(회복 초기)
    # 타겟: 경기순환주(철강, 화학, 해운) 턴어라운드. 5~15종목.
    "실적_턴어라운드": {
        "emoji": "📈",
        "description": "ROE 5%↑ + ROA 2%↑ + PBR≤0.55 + 수급동반 → 턴어라운드",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.55},
            "roe": {"min": 0.05},
            "roa": {"min": 0.02, "strict": True},
            "profit": {"strict": True},
            "supply_or": {"days_min": 6},
            "breach": {},
        },
        "scoring": {
            "detail": _sc(
                roe_above5=10,
                roe_above10=22,
                roa_above2=14,
                op_profit_positive=5,
                roe_positive=2,
                fi_net_buy_1m=8,
                inst_net_buy_1m=6,
                fi_inst_both=6,
                fi_top30=5,
                macd_cross=8,
                rsi_turn_up=5,
                ma20_cross=5,
            ),
        },
        "timing": {"strong_buy_score": 70, "buy_score": 50},
    },
    # ═══════════════════════════════════════════════════
    # 4. 외인기관_쌍끌이 🌍 — 최고 확신도 수급
    # ═══════════════════════════════════════════════════
    # 투자논거: 외인+기관이 동시에(AND) 7일 이상 순매수.
    #          두 주체 모두 확신 → 가장 높은 정보 우위.
    # 핵심필터: supply_and는 이미 극도로 선별적 (3~10종목).
    #          PBR/ROE는 적당한 하한선만.
    # 타겟: 중대형주 확신 수급. 3~10종목.
    "외인기관_쌍끌이": {
        "emoji": "🌍",
        "description": "외인+기관 동시 7일↑ AND순매수 → 최고확신 수급",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.50},
            "roe": {"min": 0.03},
            "profit": {},
            "supply_and": {"days_min": 7},
            "breach": {},
        },
        "scoring": {
            "detail": _sc(
                fi_net_buy_1m=12,
                inst_net_buy_1m=10,
                fi_inst_both=12,
                fi_top30=12,
                inst_top30=8,
                fi_inst_both_top=10,
                roe_above5=8,
                roe_above10=12,
                roa_above2=5,
                macd_cross=5,
                ma20_cross=3,
            ),
        },
        "timing": {"strong_buy_score": 70, "buy_score": 50},
    },
    # ═══════════════════════════════════════════════════
    # 5. 기술적_바닥반전 ⚡ — 52주 바닥권 + 복합 반등 신호
    # ═══════════════════════════════════════════════════
    # 투자논거: 52주 저점 근접(10%이내) + MACD 필수 + 2개↑ 기술 신호.
    #          바닥 확인 후 반등 초기 포착. 단기~중기 보유.
    # 핵심필터: 기술적 반전 중심 → PBR 0.80까지 허용(가격 패턴이 핵심)
    # 타겟: 급락 후 바닥 반등 시그널 발생 종목. 5~12종목.
    "기술적_바닥반전": {
        "emoji": "⚡",
        "description": "52주저점 근접 + MACD필수 + 복합반등 2개↑ + PBR≤0.80",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.80},
            "drawdown": {"min": 0.25},
            "near_low": {"max": 0.12},
            "signal": {"min_count": 2, "rsi_max": 48, "require_macd": True},
            "roe": {"min": -0.10},
        },
        "scoring": {
            "detail": _sc(
                macd_cross=15,
                rsi_turn_up=12,
                stoch_golden=12,
                ma20_cross=8,
                rsi40_cross=5,
                fi_net_buy_1m=10,
                inst_net_buy_1m=8,
                fi_inst_both=5,
                roe_positive=5,
                roe_above5=8,
            ),
        },
        "timing": {"strong_buy_score": 55, "buy_score": 35},
    },
    # ═══════════════════════════════════════════════════
    # 6. 초저PBR_안전망 🔥 — PBR≤0.25 극저평가 + 3중 안전장치
    # ═══════════════════════════════════════════════════
    # 투자논거: PBR ≤ 0.25 극단적 저평가 + ROE↑ + 영업흑자 + 52주안전.
    #          극저평가 + 안전장치 = 하방 리스크 극소.
    # 핵심필터: PBR 극저 자체가 강력한 필터. 약간 완화(0.25).
    # 타겟: 극저평가 안전 자산주. 3~10종목.
    "초저PBR_안전망": {
        "emoji": "🔥",
        "description": "PBR≤0.25 극저평가 + ROE/영업이익/52주 3중 안전",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.25},
            "roe": {"min": 0.005},
            "profit": {},
            "asset": {"re_min": 0.50, "cash_min": 0.20},
            "signal": {"min_count": 1, "rsi_max": 55},
            "breach": {},
        },
        "scoring": {
            "detail": _sc(
                cash_ratio_mid=10,
                cash_ratio_high=18,
                realestate_mid=10,
                realestate_high=18,
                both_asset_bonus=6,
                roe_positive=5,
                roe_above5=10,
                roa_above2=5,
                op_profit_positive=5,
                fi_net_buy_1m=8,
                inst_net_buy_1m=5,
                fi_inst_both=5,
                macd_cross=5,
                stoch_golden=5,
            ),
        },
        "timing": {"strong_buy_score": 60, "buy_score": 40},
    },
    # ═══════════════════════════════════════════════════
    # 7. 수급전환_모멘텀 📊 — 극단적 집중 순매수
    # ═══════════════════════════════════════════════════
    # 투자논거: 외인 또는 기관이 8/10일 순매수 + 기술 2개↑ 확인.
    #          단일 주체의 극단적 집중 매수 = 강한 정보 우위.
    # 핵심필터: days_min=8 + signal≥2 + ROE 3%↑. PBR 0.60까지 허용(수급이 핵심)
    # 타겟: 단중기 모멘텀 플레이. 5~15종목.
    "수급전환_모멘텀": {
        "emoji": "📊",
        "description": "외인/기관 8일↑ 극단적 집중순매수 + PBR≤0.60 + 기술확인",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.60},
            "roe": {"min": 0.03},
            "profit": {},
            "supply_or": {"days_min": 8},
            "signal": {"min_count": 2, "rsi_max": 50},
            "breach": {},
        },
        "scoring": {
            "detail": _sc(
                fi_net_buy_1m=15,
                inst_net_buy_1m=12,
                fi_inst_both=10,
                fi_top30=10,
                inst_top30=8,
                macd_cross=10,
                rsi_turn_up=6,
                ma20_cross=6,
                stoch_golden=5,
                roe_positive=3,
                roe_above5=5,
                roa_above2=3,
            ),
        },
        "timing": {"strong_buy_score": 70, "buy_score": 50},
    },
    # ═══════════════════════════════════════════════════
    # 8. 저PER_가치주 💲 — 초저PER 가치주 (수익 대비 극저평가)
    # ═══════════════════════════════════════════════════
    # 투자논거: PER ≤ 5 극저평가, 영업이익률 3%↑, ROE 3%↑.
    # 핵심필터: PER=0(적자) 제외, PBR≤0.60, 영업이익률 게이트+스코어링.
    # 타겟: 수익 대비 시장 관심 부족 + 실제 수익성 확인 종목. 5~20종목.
    "저PER_가치주": {
        "emoji": "💲",
        "description": "0<PER≤5 + 영업이익률≥3% + ROE 3%↑ + PBR≤0.60",
        "gates": {
            "pbr": {"min": 0.0, "max": 0.60},
            "per": {"min": 0.01, "max": 5.0},
            "roe": {"min": 0.03},
            "profit": {},
            "operating_margin": {"min": 3.0},  # B: 영업이익률 ≥ 3%
            "breach": {},
        },
        "scoring": {
            "detail": _sc(
                roe_positive=5,
                roe_above5=10,
                roe_above10=15,
                roa_above2=8,
                op_profit_positive=8,
                op_margin_mid=8,  # A: 영업이익률 5~10% → 8점
                op_margin_high=14,  # A: 영업이익률 ≥10% → 14점
                op_margin_improving=8,  # C: 전기 대비 개선 → 8점
                div_above3=12,
                fi_net_buy_1m=8,
                inst_net_buy_1m=6,
                fi_inst_both=6,
                macd_cross=8,
                rsi_turn_up=5,
                stoch_golden=4,
                ma20_cross=5,
                cash_ratio_mid=5,
                cash_ratio_high=8,
            ),
        },
        "timing": {"strong_buy_score": 65, "buy_score": 45},
    },
    # ═══════════════════════════════════════════════════
    # 9. 성장우량주A ⭐ — DART 기반 성장+수익+안전 (A등급)
    # ═══════════════════════════════════════════════════
    # 투자논거: 매출↑10%+이익률8%+ROE12%+ROA5%+안전핀.
    #          성장+수익+효율+안전성이 모두 충족된 우량 가치주.
    # 핵심: DART 재무데이터 필수. quality_category="A" 플래그.
    "성장우량주A": {
        "emoji": "⭐",
        "description": "매출↑10%+이익률8%+ROE12%+ROA5%+안전핀",
        "quality_category": "A",
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 80, "buy_score": 60},
    },
    # ═══════════════════════════════════════════════════
    # 10. 성장우량주B 🅱️ — DART 기반 중간 성장 (B등급)
    # ═══════════════════════════════════════════════════
    # 투자논거: 매출↑5%+이익률5%+ROE8%+부채<200%.
    #          A만큼 뛰어나지 않지만 성장+안전 기본기 확보.
    "성장우량주B": {
        "emoji": "🅱️",
        "description": "매출↑5%+이익률5%+ROE8%+부채<200%+안전핀",
        "quality_category": "B",
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 55, "buy_score": 40},
    },
    "AI_빠른학습": {
        "emoji": "⚡",
        "description": "ML 빠른학습 4주내 종가30%↑ 확률 예측",
        "ml_category": True,
        "ml_mode": "fast",
        "ml_pre_filter": {
            "min_mcap": 70_000_000_000,  # 시가총액 ≥ 700억
            "per_min": 0.0,  # PER > 0 (적자기업 제외)
            "per_max": 12.0,  # PER ≤ 12
            "pbr_min": 0.0,  # PBR > 0
            "pbr_max": 2.0,  # PBR ≤ 2
            "roe_min": 0.0,  # ROE > 0
        },
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_매집주학습": {
        "emoji": "🔬",
        "description": "매집 완료 직전 종목 탐지: 2년학습+BiLSTM OBV/거래량수축/BB수축 분석",
        "ml_category": True,
        "ml_mode": "accum",
        "ml_pre_filter": {
            "min_mcap": 100_000_000_000,  # 시총 ≥ 1000억
            "min_avg_trading_value": 7_000_000_000,  # 거래대금 ≥ 70억
        },
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    # ── AI 25% 목표 ──
    "AI_25%급등주": {
        "emoji": "🚀",
        "description": "25% 목표 2년 전체 학습 급등 후보 Top20",
        "ml_category": True,
        "ml_mode": "surge",
        "label_target_pct": 0.25,
        "ml_pre_filter": {
            "min_mcap": 100_000_000_000,
            "min_avg_trading_value": 3_000_000_000,
        },
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_25%급등주_1Y": {
        "emoji": "🚀",
        "description": "25% 목표 최근 1년 패턴 급등주 Top20",
        "ml_category": True,
        "ml_mode": "surge_1y",
        "label_target_pct": 0.25,
        "ml_pre_filter": {"min_mcap": 100_000_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_25%급등주_6M": {
        "emoji": "🚀",
        "description": "25% 목표 최근 6개월 패턴 급등주 Top20",
        "ml_category": True,
        "ml_mode": "surge_6m",
        "label_target_pct": 0.25,
        "ml_pre_filter": {"min_mcap": 70_000_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_25%급등주_3M": {
        "emoji": "🚀",
        "description": "25% 목표 최근 3개월 패턴 급등주 Top20",
        "ml_category": True,
        "ml_mode": "surge_3m",
        "label_target_pct": 0.25,
        "ml_pre_filter": {"min_mcap": 50_000_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    # ── AI 30% 목표 ──
    "AI_30%급등주": {
        "emoji": "🚀",
        "description": "30% 목표 2년 전체 학습 급등 후보 Top20",
        "ml_category": True,
        "ml_mode": "surge",
        "label_target_pct": 0.30,
        "ml_pre_filter": {
            "min_mcap": 100_000_000_000,
            "min_avg_trading_value": 3_000_000_000,
        },
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_30%급등주_1Y": {
        "emoji": "🚀",
        "description": "30% 목표 최근 1년 패턴 급등주 Top20",
        "ml_category": True,
        "ml_mode": "surge_1y",
        "label_target_pct": 0.30,
        "ml_pre_filter": {"min_mcap": 100_000_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_30%급등주_6M": {
        "emoji": "🚀",
        "description": "30% 목표 최근 6개월 패턴 급등주 Top20",
        "ml_category": True,
        "ml_mode": "surge_6m",
        "label_target_pct": 0.30,
        "ml_pre_filter": {"min_mcap": 70_000_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_30%급등주_3M": {
        "emoji": "🚀",
        "description": "30% 목표 최근 3개월 패턴 급등주 Top20",
        "ml_category": True,
        "ml_mode": "surge_3m",
        "label_target_pct": 0.30,
        "ml_pre_filter": {"min_mcap": 50_000_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    # ── AI ETF 10% 목표 ──
    "AI_10%추천ETF": {
        "emoji": "📦",
        "description": "10% 목표 2년 전체 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge",
        "label_target_pct": 0.10,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_10%추천ETF_1Y": {
        "emoji": "📦",
        "description": "10% 목표 최근 1년 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge_1y",
        "label_target_pct": 0.10,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_10%추천ETF_6M": {
        "emoji": "📦",
        "description": "10% 목표 최근 6개월 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge_6m",
        "label_target_pct": 0.10,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_10%추천ETF_3M": {
        "emoji": "📦",
        "description": "10% 목표 최근 3개월 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge_3m",
        "label_target_pct": 0.10,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    # ── AI ETF 15% 목표 ──
    "AI_15%추천ETF": {
        "emoji": "📦",
        "description": "15% 목표 2년 전체 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge",
        "label_target_pct": 0.15,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_15%추천ETF_1Y": {
        "emoji": "📦",
        "description": "15% 목표 최근 1년 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge_1y",
        "label_target_pct": 0.15,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_15%추천ETF_6M": {
        "emoji": "📦",
        "description": "15% 목표 최근 6개월 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge_6m",
        "label_target_pct": 0.15,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
    "AI_15%추천ETF_3M": {
        "emoji": "📦",
        "description": "15% 목표 최근 3개월 ETF학습 Top20",
        "ml_category": True,
        "ml_mode": "etf_surge_3m",
        "label_target_pct": 0.15,
        "asset_type": "ETF",
        "ml_pre_filter": {"min_avg_trading_value": 100_000_000},
        "gates": {},
        "scoring": {"detail": _ZERO_SCORING.copy()},
        "timing": {"strong_buy_score": 60, "buy_score": 45},
    },
}

# ──────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "min_mcap_default": 50_000_000_000,  # 500억
    "internal_pbr_scope": 1.0,
    "categories": CATEGORIES_CONFIG,
}


# ──────────────────────────────────────────────
# 프리셋 로드/저장
# ──────────────────────────────────────────────


def get_default_config():
    return deepcopy(DEFAULT_CONFIG)


def get_categories_config(target_pct_list=None, etf_target_pct_list=None):
    """카테고리 설정 반환. target_pct_list로 AI 카테고리 필터링 가능.

    Args:
        target_pct_list: [25, 30] 등 정수 % (주식). None이면 전체 반환.
        etf_target_pct_list: [10, 15] 등 정수 % (ETF). None이면 전체 반환.
    """
    cfg = deepcopy(CATEGORIES_CONFIG)
    if target_pct_list is not None:
        to_remove = [
            k
            for k, v in cfg.items()
            if "label_target_pct" in v
            and v.get("asset_type") != "ETF"
            and int(v["label_target_pct"] * 100) not in target_pct_list
        ]
        for k in to_remove:
            del cfg[k]
    if etf_target_pct_list is not None:
        to_remove = [
            k
            for k, v in cfg.items()
            if v.get("asset_type") == "ETF"
            and int(v["label_target_pct"] * 100) not in etf_target_pct_list
        ]
        for k in to_remove:
            del cfg[k]
    return cfg


def save_preset(name, config):
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESET_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


def load_preset(name):
    path = PRESET_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"프리셋 '{name}'을 찾을 수 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_presets():
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in sorted(PRESET_DIR.glob("*.json"))]


INDEX_PREDICTION_DIR = CACHE_DIR / "index_prediction"

# ──────────────────────────────────────────────
# KRX Open API 인증키
# ──────────────────────────────────────────────
# openapi.krx.co.kr 에서 무료 발급 (관리자 승인 후 사용 가능)
# 환경변수 KRX_OPENAPI_AUTH_KEY 로 오버라이드 가능
KRX_OPENAPI_AUTH_KEY = os.environ.get("KRX_OPENAPI_AUTH_KEY", "")

# ──────────────────────────────────────────────
# KRX 정보데이터시스템 로그인 (data.krx.co.kr)
# ──────────────────────────────────────────────
# 2025-12-27부터 회원제 전환 → 로그인 필수
# 환경변수 KRX_MBR_ID / KRX_PW 로 오버라이드 가능
KRX_MBR_ID = ""  # data.krx.co.kr 회원 ID
KRX_PW = ""  # data.krx.co.kr 비밀번호

# ──────────────────────────────────────────────
# AI 결과 시총 필터 (모든 top_k 테이블 공통)
# ──────────────────────────────────────────────
# 최소값: 500억 (이 이하로 설정 불가)
# 기본값: 800억
RESULT_MCAP_MIN_BILLION = 500  # 슬라이더 하한 (억)
RESULT_MCAP_DEFAULT_BILLION = 800  # 기본값 (억)


def ensure_dirs():
    for d in [CACHE_DIR, PRESET_DIR, OUTPUT_DIR, INDEX_PREDICTION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
