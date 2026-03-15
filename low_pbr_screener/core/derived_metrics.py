"""파생지표 및 수급 지표 계산 모듈."""

import numpy as np
import pandas as pd


def calc_52w_high_low(ohlcv_1y: pd.DataFrame) -> dict:
    """1년치 OHLCV에서 52주 최고/최저가 계산."""
    if ohlcv_1y is None or ohlcv_1y.empty:
        return {"High52": np.nan, "Low52": np.nan}

    return {
        "High52": float(ohlcv_1y["고가"].max()),
        "Low52": float(ohlcv_1y["저가"].min()),
    }


def calc_near_52w_low(close: float, low_52w: float) -> float:
    """52주 저점 대비 근접도 = (Close - Low52) / Low52."""
    if pd.isna(low_52w) or low_52w <= 0:
        return np.nan
    return (close - low_52w) / low_52w


def calc_drawdown_from_52w_high(close: float, high_52w: float) -> float:
    """52주 고점 대비 하락폭 = (High52 - Close) / High52."""
    if pd.isna(high_52w) or high_52w <= 0:
        return np.nan
    return (high_52w - close) / high_52w


def calc_cash_ratio(cash: float, st_financial: float,
                    market_cap: float) -> float:
    """현금비율 = (현금 + 단기금융상품) / 시가총액."""
    if pd.isna(market_cap) or market_cap <= 0:
        return np.nan
    total_cash = (cash or 0) + (st_financial or 0)
    if total_cash <= 0:
        return 0.0
    return total_cash / market_cap


def calc_realestate_ratio(land: float, building: float,
                          invest_property: float,
                          market_cap: float) -> float:
    """부동산비율 = (토지 + 건물 + 투자부동산) / 시가총액."""
    if pd.isna(market_cap) or market_cap <= 0:
        return np.nan
    total_re = (land or 0) + (building or 0) + (invest_property or 0)
    if total_re <= 0:
        return 0.0
    return total_re / market_cap


def calc_roe_from_pbr_per(pbr: float, per: float) -> float:
    """ROE ≈ PBR / PER = EPS / BPS."""
    if pd.isna(per) or per == 0 or pd.isna(pbr):
        return np.nan
    if per < 0:
        return -abs(pbr / per)
    return pbr / per


def calc_roa(net_income: float, total_assets: float) -> float:
    """ROA = 당기순이익 / 총자산."""
    if pd.isna(total_assets) or total_assets <= 0:
        return np.nan
    if pd.isna(net_income):
        return np.nan
    return net_income / total_assets


def calc_flow_metrics(investor_df: pd.DataFrame) -> dict:
    """투자자별 거래 데이터에서 수급 지표 산출.

    Args:
        investor_df: get_market_trading_value_by_date 결과.
                     컬럼: 기관합계, 기타법인, 개인, 외국인합계

    Returns:
        dict with flow metrics.
    """
    result = {
        "Flow1M_FI_KRW": 0.0,       # 외인 1개월 순매수 금액
        "Flow1M_INST_KRW": 0.0,     # 기관 1개월 순매수 금액
        "NetBuyDays_FI_10d": 0,      # 최근 10일 외인 순매수 일수
        "NetBuyDays_INST_10d": 0,    # 최근 10일 기관 순매수 일수
    }

    if investor_df is None or investor_df.empty:
        return result

    # 외국인 순매수
    fi_col = None
    for col in ["외국인합계", "외국인"]:
        if col in investor_df.columns:
            fi_col = col
            break

    # 기관 순매수
    inst_col = None
    for col in ["기관합계", "기관"]:
        if col in investor_df.columns:
            inst_col = col
            break

    if fi_col:
        fi_series = investor_df[fi_col].astype(float)
        result["Flow1M_FI_KRW"] = float(fi_series.sum())
        last_10 = fi_series.iloc[-10:] if len(fi_series) >= 10 else fi_series
        result["NetBuyDays_FI_10d"] = int((last_10 > 0).sum())

    if inst_col:
        inst_series = investor_df[inst_col].astype(float)
        result["Flow1M_INST_KRW"] = float(inst_series.sum())
        last_10 = inst_series.iloc[-10:] if len(inst_series) >= 10 else inst_series
        result["NetBuyDays_INST_10d"] = int((last_10 > 0).sum())

    return result


def calc_revenue_growth(revenue: float, revenue_prev: float) -> float:
    """매출 YoY 성장률 (%). 데이터 부족 시 NaN."""
    if pd.isna(revenue) or pd.isna(revenue_prev):
        return np.nan
    if revenue_prev <= 0:
        return np.nan
    return (revenue - revenue_prev) / revenue_prev * 100


def calc_operating_margin(operating_profit: float, revenue: float) -> float:
    """영업이익률 (%). 데이터 부족 시 NaN."""
    if pd.isna(operating_profit) or pd.isna(revenue) or revenue <= 0:
        return np.nan
    return operating_profit / revenue * 100


def calc_debt_ratio(total_liabilities: float, total_equity: float) -> float:
    """부채비율 = 부채총계 / 자본총계. 자본 음수 → NaN."""
    if pd.isna(total_liabilities) or pd.isna(total_equity):
        return np.nan
    if total_equity <= 0:
        return np.nan
    return total_liabilities / total_equity


# ── v3 신규 함수 ──────────────────────────────────────────

def calc_revenue_cagr_3y(revenue: float, revenue_prev2: float) -> float:
    """3년 매출 CAGR (%). rev=당기, rev_prev2=전전기."""
    if pd.isna(revenue) or pd.isna(revenue_prev2):
        return np.nan
    if revenue_prev2 <= 0 or revenue <= 0:
        return np.nan
    return ((revenue / revenue_prev2) ** 0.5 - 1) * 100


def calc_margin_improvement(op_margin: float,
                            op_margin_prev: float) -> float:
    """영업이익률 전기 대비 변동폭 (pp). 양수=개선."""
    if pd.isna(op_margin) or pd.isna(op_margin_prev):
        return np.nan
    return op_margin - op_margin_prev


def calc_interest_coverage(operating_profit: float,
                           interest_expense: float) -> float:
    """이자보상배율 = 영업이익 / 이자비용."""
    if pd.isna(operating_profit) or pd.isna(interest_expense):
        return np.nan
    if interest_expense <= 0:
        return np.nan  # 이자비용 0 → 무차입, 별도 처리
    return operating_profit / interest_expense


def calc_ebitda(operating_profit: float, depreciation: float) -> float:
    """EBITDA = 영업이익 + 감가상각비."""
    op = operating_profit if not pd.isna(operating_profit) else 0.0
    dep = depreciation if not pd.isna(depreciation) else 0.0
    if op == 0 and dep == 0:
        return np.nan
    return op + dep


def calc_net_debt_to_ebitda(short_b: float, long_b: float,
                            cash: float, st_fin: float,
                            ebitda: float) -> float:
    """순차입금/EBITDA. 순차입금 = 차입금 - 현금성자산."""
    if pd.isna(ebitda) or ebitda <= 0:
        return np.nan
    net_debt = ((short_b or 0) + (long_b or 0)
                - (cash or 0) - (st_fin or 0))
    if net_debt <= 0:
        return 0.0  # 순현금 상태 → 안전
    return net_debt / ebitda


def calc_eps_growth(net_income: float, net_income_prev: float,
                    total_equity: float) -> float:
    """EPS 성장률 (%). 순이익 변동 기반 (자본 대비 정규화)."""
    if pd.isna(net_income) or pd.isna(net_income_prev):
        return np.nan
    if pd.isna(total_equity) or total_equity <= 0:
        return np.nan
    if net_income_prev <= 0:
        # 전기 적자 → 흑자전환 시 +100, 아니면 NaN
        if net_income > 0:
            return 100.0
        return np.nan
    return (net_income - net_income_prev) / net_income_prev * 100


def calc_peg_ratio(per: float, eps_growth: float) -> float:
    """PEG = PER / EPS성장률. 저평가 < 1 < 고평가."""
    if pd.isna(per) or pd.isna(eps_growth):
        return np.nan
    if per <= 0 or eps_growth <= 0:
        return np.nan
    return per / eps_growth


# ── v3 등급 판정 ──────────────────────────────────────────

def _ok(v):
    """값이 유효한지 (None/NaN 아닌지) 확인."""
    return v is not None and not pd.isna(v)


def _val(row, key, default=0.0):
    """row에서 안전하게 값 추출. NaN → default."""
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return v


def compute_quality_detail(row) -> dict:
    """우량주 등급 상세 판정 (v3).

    Returns:
        dict with keys:
            grade: "A+"/"A"/"B"/"C"/""
            tags: list[str] — 판정 사유 태그
            warnings: list[str] — C등급 위험 경고
            peg: float or None
            peg_verdict: "저평가"/"적정"/"고평가"/""
    """
    # ── 지표 수집 ──
    rev_g = row.get("RevenueGrowth")      # YoY %
    rev_cagr = row.get("RevCAGR3Y")       # 3년 CAGR %
    op_m = row.get("OperatingMargin")      # 영업이익률 %
    op_m_prev = row.get("OpMarginPrev")    # 전기 영업이익률 %
    roe = row.get("ROE")                   # 비율 (0.12 = 12%)
    roa = row.get("ROA")
    ocf = _val(row, "OpCashFlow")
    ocf_prev = _val(row, "OpCashFlowPrev")
    ni = _val(row, "net_income")
    debt_r = row.get("DebtRatio")          # 비율 (2.0 = 200%)
    icr = row.get("ICR")
    nd_ebitda = row.get("NetDebtEBITDA")
    peg = row.get("PEG")
    eps_g = row.get("EPSGrowth")

    tags = []
    warnings = []

    # ── 안전핀 체크 (A/B 등급 차단 가능) ──
    safety_fail = False
    safety_reasons = []

    # 안전핀 1: ICR < 2
    if _ok(icr) and icr < 2:
        safety_fail = True
        safety_reasons.append(f"ICR={icr:.1f}<2")

    # 안전핀 2: 순차입금/EBITDA > 5
    if _ok(nd_ebitda) and nd_ebitda > 5:
        safety_fail = True
        safety_reasons.append(f"순차입금/EBITDA={nd_ebitda:.1f}>5")

    # 안전핀 3: CFO 당기·전기 모두 마이너스
    if ocf < 0 and ocf_prev < 0:
        safety_fail = True
        safety_reasons.append("CFO 2기연속 적자")

    # ── C등급 위험 패턴 감지 ──
    c_grade = False

    # C-1: 매출 성장 + 영업이익률 하락 (외형 성장 함정)
    if (_ok(rev_g) and rev_g >= 5
            and _ok(op_m) and _ok(op_m_prev)
            and op_m < op_m_prev and (op_m_prev - op_m) >= 2):
        c_grade = True
        warnings.append("외형성장함정(매출↑이익률↓)")

    # C-2: 레버리지 착시 (ROE높+ROA낮+부채높)
    if (_ok(roe) and roe >= 0.10
            and _ok(roa) and roa < 0.03
            and _ok(debt_r) and debt_r > 3.0):
        c_grade = True
        warnings.append("레버리지착시(ROE↑ROA↓부채↑)")

    # C-3: 현금 없는 성장 (매출↑ + CFO 2기 마이너스)
    if (_ok(rev_g) and rev_g >= 5
            and ocf < 0 and ocf_prev < 0):
        c_grade = True
        warnings.append("현금없는성장(매출↑CFO↓)")

    # ── PEG 판정 ──
    peg_verdict = ""
    if _ok(peg):
        if peg < 1.0:
            peg_verdict = "저평가"
        elif peg <= 2.0:
            peg_verdict = "적정"
        else:
            peg_verdict = "고평가"

    # ── A등급 판정 ──
    # 성장: YoY ≥10% OR 3년 CAGR ≥10%
    a_rev = ((_ok(rev_g) and rev_g >= 10)
             or (_ok(rev_cagr) and rev_cagr >= 10))
    # 수익성: 이익률 ≥8% OR (≥5% + 전기 대비 개선)
    margin_improving = (_ok(op_m) and _ok(op_m_prev)
                        and op_m >= 5 and op_m > op_m_prev)
    a_margin = (_ok(op_m) and op_m >= 8) or margin_improving
    # 효율: ROE ≥12% AND ROA ≥5%
    a_roe = _ok(roe) and roe >= 0.12
    a_roa = _ok(roa) and roa >= 0.05
    # 안전핀 + CFO 당기 양수
    a_safe = not safety_fail and ocf > 0

    a_pass = a_rev and a_margin and a_roe and a_roa and a_safe

    if a_pass:
        # 태그 생성
        if _ok(rev_g) and rev_g >= 10:
            tags.append(f"매출↑{rev_g:.0f}%")
        elif _ok(rev_cagr):
            tags.append(f"CAGR↑{rev_cagr:.0f}%")
        if _ok(op_m):
            tags.append(f"이익률{op_m:.0f}%")
        if _ok(roe):
            tags.append(f"ROE{roe*100:.0f}%")
        tags.append("CFO양호")

        # A+ 보너스: OCF ≥ NI × 0.8 AND ICR ≥ 5
        aplus = (ocf > 0 and ni > 0 and ocf >= ni * 0.8
                 and _ok(icr) and icr >= 5)
        grade = "A+" if aplus else "A"
        if aplus:
            tags.append(f"ICR{icr:.0f}")

        return {"grade": grade, "tags": tags, "warnings": [],
                "peg": peg if _ok(peg) else None,
                "peg_verdict": peg_verdict}

    # ── C등급 (위험 패턴) — B보다 우선 ──
    if c_grade:
        return {"grade": "C", "tags": [], "warnings": warnings,
                "peg": peg if _ok(peg) else None,
                "peg_verdict": peg_verdict}

    # ── B등급 판정 ──
    b_rev = _ok(rev_g) and rev_g >= 5
    b_margin = _ok(op_m) and op_m >= 5
    # 이익률 개선 중이면 4%도 통과
    if not b_margin and margin_improving and _ok(op_m) and op_m >= 4:
        b_margin = True
    b_roe_or_roa = ((_ok(roe) and roe >= 0.08)
                    or (_ok(roa) and roa >= 0.03))
    b_debt = (not _ok(debt_r)) or debt_r < 2.0
    b_safe = not safety_fail

    b_pass = b_rev and b_margin and b_roe_or_roa and b_debt and b_safe

    if b_pass:
        if _ok(rev_g):
            tags.append(f"매출↑{rev_g:.0f}%")
        if _ok(op_m):
            tags.append(f"이익률{op_m:.0f}%")
        if _ok(roe) and roe >= 0.08:
            tags.append(f"ROE{roe*100:.0f}%")
        elif _ok(roa):
            tags.append(f"ROA{roa*100:.0f}%")

        return {"grade": "B", "tags": tags, "warnings": [],
                "peg": peg if _ok(peg) else None,
                "peg_verdict": peg_verdict}

    # ── 안전핀 실패 시 경고만 ──
    if safety_fail:
        return {"grade": "", "tags": [], "warnings": safety_reasons,
                "peg": peg if _ok(peg) else None,
                "peg_verdict": peg_verdict}

    # ── 데이터 부족 또는 미달 ──
    return {"grade": "", "tags": [], "warnings": [],
            "peg": peg if _ok(peg) else None,
            "peg_verdict": peg_verdict}


def compute_quality_grade(row) -> str:
    """v2 호환 래퍼 — compute_quality_detail의 grade만 반환."""
    return compute_quality_detail(row)["grade"]


def check_52w_low_breach(ohlcv_100d: pd.DataFrame,
                         low_52w: float) -> bool:
    """최근 20일 최저가가 52주 저점 하회 여부."""
    if ohlcv_100d is None or ohlcv_100d.empty or pd.isna(low_52w):
        return True  # 데이터 없으면 보수적으로 breach 처리

    recent_20 = ohlcv_100d.iloc[-20:]
    if "저가" in recent_20.columns:
        min_low = recent_20["저가"].min()
        return float(min_low) < low_52w
    return True
