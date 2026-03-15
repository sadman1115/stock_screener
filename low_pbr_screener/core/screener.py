"""통합 게이트 레지스트리 스크리닝 엔진.

14개 플러그인 게이트 + 22개 스코어링 조건 + 매수 타이밍 신호.
모든 임계치와 배점은 카테고리별 config dict로 외부에서 주입받는다.
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────


def _safe_col(df, col, default):
    """컬럼이 없거나 전부 NaN이면 default 값으로 채운 Series 반환."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def _safe_bool_col(df, col):
    """bool 컬럼 안전 접근. 없거나 NaN이면 False."""
    if col in df.columns:
        s = df[col].copy()
        s[s.isna()] = False
        return s.astype(bool)
    return pd.Series(False, index=df.index)


# ──────────────────────────────────────────────
# 14개 플러그인 게이트
# ──────────────────────────────────────────────


def _gate_pbr(df, params):
    """PBR 범위 게이트. params: min(기본0), max(필수)."""
    pbr = _safe_col(df, "PBR", np.nan)
    pbr_min = params.get("min", 0.0)
    pbr_max = params.get("max", 1.0)
    return (pbr > pbr_min) & (pbr <= pbr_max)


def _gate_per(df, params):
    """PER 범위 게이트. params: min, max. PER=0/NaN 제외."""
    per = _safe_col(df, "PER", np.nan)
    per_min = params.get("min", 0.0)
    per_max = params.get("max", 9999)
    return (per > 0) & (per >= per_min) & (per <= per_max)


def _gate_roe(df, params):
    """ROE 최소값 게이트. NaN→fillna(-999)→실패 처리."""
    roe_min = params.get("min", 0.0)
    roe = _safe_col(df, "ROE", -999)
    return roe >= roe_min


def _gate_roa(df, params):
    """ROA 최소값 게이트. DART없으면(컬럼 자체 없음) skip(통과).
    strict=True: 개별 종목 NaN → 실패 처리."""
    strict = params.get("strict", False)
    if "ROA" not in df.columns or not df["ROA"].notna().any():
        return pd.Series(not strict, index=df.index)
    roa_min = params.get("min", 0.0)
    roa = df["ROA"]
    if strict:
        return (roa >= roa_min) & roa.notna()
    return (roa >= roa_min) | roa.isna()


def _gate_drawdown(df, params):
    """52주 고점 대비 하락폭 최소. params: min."""
    dd_min = params.get("min", 0.0)
    dd = _safe_col(df, "DrawdownFrom52wHigh", np.nan)
    return dd >= dd_min


def _gate_near_low(df, params):
    """52주 저점 근접도 최대. params: max."""
    nl_max = params.get("max", 1.0)
    nl = _safe_col(df, "Near52wLow", np.nan)
    return nl <= nl_max


def _gate_asset(df, params):
    """부동산/현금 비율 게이트 (OR 논리). DART없으면 skip.
    strict=True: 개별 종목 DART 데이터 없으면 실패 처리."""
    re_min = params.get("re_min", 0.6)
    cash_min = params.get("cash_min", 0.25)
    strict = params.get("strict", False)

    has_rr = "RealEstateRatio" in df.columns and df["RealEstateRatio"].notna().any()
    has_cr = "CashRatio" in df.columns and df["CashRatio"].notna().any()

    if not has_rr and not has_cr:
        return pd.Series(not strict, index=df.index)

    rr_ok = (
        (df["RealEstateRatio"] >= re_min)
        if has_rr
        else pd.Series(False, index=df.index)
    )
    cr_ok = (
        (df["CashRatio"] >= cash_min) if has_cr else pd.Series(False, index=df.index)
    )

    asset_pass = rr_ok.fillna(False) | cr_ok.fillna(False)

    if strict:
        return asset_pass

    # 개별 종목이 둘 다 NaN이면 → DART 데이터 없음 → skip(통과)
    both_nan = pd.Series(True, index=df.index)
    if has_rr:
        both_nan = both_nan & df["RealEstateRatio"].isna()
    if has_cr:
        both_nan = both_nan & df["CashRatio"].isna()

    return asset_pass | both_nan


def _gate_dividend(df, params):
    """배당수익률 Gate. DIV >= min (%)."""
    div_min = params.get("min", 0.0)
    div = _safe_col(df, "DIV", 0.0)
    return div >= div_min


def _gate_profit(df, params):
    """영업이익 Gate. operating_profit > 0. DART없으면 skip.
    strict=True: 개별 종목 DART 데이터 없으면 실패 처리."""
    strict = params.get("strict", False)
    if "operating_profit" not in df.columns or not df["operating_profit"].notna().any():
        return pd.Series(not strict, index=df.index)
    op = df["operating_profit"]
    if strict:
        return (op > 0) & op.notna()
    return (op > 0) | op.isna()


def _gate_operating_margin(df, params):
    """영업이익률 Gate. OperatingMargin >= min (%). DART없으면 skip.
    strict=True: DART 데이터 없으면 실패 처리."""
    om_min = params.get("min", 0.0)
    strict = params.get("strict", False)
    if "OperatingMargin" not in df.columns or not df["OperatingMargin"].notna().any():
        return pd.Series(not strict, index=df.index)
    om = df["OperatingMargin"]
    if strict:
        return (om >= om_min) & om.notna()
    return (om >= om_min) | om.isna()


def _gate_supply_or(df, params):
    """외인 OR 기관 순매수 일수 게이트. params: days_min."""
    days_min = params.get("days_min", 5)
    fi_days = _safe_col(df, "NetBuyDays_FI_10d", 0)
    inst_days = _safe_col(df, "NetBuyDays_INST_10d", 0)
    fi = _safe_col(df, "Flow1M_FI_KRW", 0)
    inst = _safe_col(df, "Flow1M_INST_KRW", 0)
    return ((fi > 0) & (fi_days >= days_min)) | ((inst > 0) & (inst_days >= days_min))


def _gate_supply_and(df, params):
    """외인 AND 기관 순매수 일수 게이트. params: days_min."""
    days_min = params.get("days_min", 6)
    fi_days = _safe_col(df, "NetBuyDays_FI_10d", 0)
    inst_days = _safe_col(df, "NetBuyDays_INST_10d", 0)
    fi = _safe_col(df, "Flow1M_FI_KRW", 0)
    inst = _safe_col(df, "Flow1M_INST_KRW", 0)
    return ((fi > 0) & (fi_days >= days_min)) & ((inst > 0) & (inst_days >= days_min))


def _gate_signal(df, params):
    """기술적 신호 N개 이상 게이트 (v1 스타일). params: min_count, rsi_max, require_macd."""
    min_count = params.get("min_count", 2)
    rsi_max = params.get("rsi_max", 48)
    require_macd = params.get("require_macd", False)

    signal_count = pd.Series(0, index=df.index)

    # RSI 신호
    rsi_cond = _safe_bool_col(df, "RSI_TurnUp") & (
        _safe_col(df, "RSI14", 100) <= rsi_max
    )
    signal_count += rsi_cond.astype(int)

    # MACD 신호
    macd_cond = _safe_bool_col(df, "MACD_Hist_3dUp") | _safe_bool_col(
        df, "MACD_Cross_5d"
    )
    signal_count += macd_cond.astype(int)

    # Stoch 신호
    signal_count += _safe_bool_col(df, "Stoch_Golden_Low").astype(int)

    # 수급 신호 — 5일 이상 순매수인 경우에만 카운트 (단순 순매수 양은 너무 느슨)
    fi_flow = _safe_col(df, "Flow1M_FI_KRW", 0)
    inst_flow = _safe_col(df, "Flow1M_INST_KRW", 0)
    fi_days = _safe_col(df, "NetBuyDays_FI_10d", 0)
    inst_days = _safe_col(df, "NetBuyDays_INST_10d", 0)
    flow_cond = ((fi_flow > 0) & (fi_days >= 5)) | ((inst_flow > 0) & (inst_days >= 5))
    signal_count += flow_cond.astype(int)

    result = signal_count >= min_count

    # MACD 필수 옵션
    if require_macd:
        macd_ok = (
            _safe_bool_col(df, "MACD_Hist_3dUp")
            | _safe_bool_col(df, "MACD_Cross_5d")
            | _safe_bool_col(df, "MACD_Cross_10d")
        )
        result = result & macd_ok

    return result


def _gate_tech(df, params):
    """확정 기술 신호 1개 이상 게이트 (v2 스타일)."""
    return (
        _safe_bool_col(df, "MACD_Cross_10d")
        | _safe_bool_col(df, "MA20_Cross")
        | _safe_bool_col(df, "RSI_Cross40_10d")
    )


def _gate_breach(df, params):
    """52주 신저점 비돌파 게이트."""
    return ~_safe_bool_col(df, "Low52_Breach_20d")


# ──────────────────────────────────────────────
# 게이트 레지스트리
# ──────────────────────────────────────────────

GATE_REGISTRY = {
    "pbr": _gate_pbr,
    "per": _gate_per,
    "roe": _gate_roe,
    "roa": _gate_roa,
    "drawdown": _gate_drawdown,
    "near_low": _gate_near_low,
    "asset": _gate_asset,
    "dividend": _gate_dividend,
    "profit": _gate_profit,
    "operating_margin": _gate_operating_margin,
    "supply_or": _gate_supply_or,
    "supply_and": _gate_supply_and,
    "signal": _gate_signal,
    "tech": _gate_tech,
    "breach": _gate_breach,
}


# ──────────────────────────────────────────────
# 통합 게이트 적용
# ──────────────────────────────────────────────


def apply_gates(df, gates_config):
    """게이트 목록을 AND로 결합. bool Series 반환."""
    result = pd.Series(True, index=df.index)
    for gate_name, params in gates_config.items():
        if gate_name in GATE_REGISTRY:
            gate_result = GATE_REGISTRY[gate_name](df, params)
            result = result & gate_result
    return result


# ──────────────────────────────────────────────
# 통합 스코어링 (22개 조건)
# ──────────────────────────────────────────────


def apply_scoring(df, scoring_detail):
    """통합 스코어링. scoring_detail의 키별 점수가 0이면 해당 조건 skip.
    float Series (0~100) 반환."""
    sc = scoring_detail
    score = pd.Series(0.0, index=df.index)

    # ── 수급 (6개) ──
    fi = _safe_col(df, "Flow1M_FI_KRW", 0)
    inst = _safe_col(df, "Flow1M_INST_KRW", 0)

    if sc.get("fi_net_buy_1m", 0):
        score += (fi > 0).astype(float) * sc["fi_net_buy_1m"]
    if sc.get("inst_net_buy_1m", 0):
        score += (inst > 0).astype(float) * sc["inst_net_buy_1m"]
    if sc.get("fi_inst_both", 0):
        score += ((fi > 0) & (inst > 0)).astype(float) * sc["fi_inst_both"]
    if sc.get("fi_top30", 0) and "FI_NetBuy_Top30" in df.columns:
        score += _safe_bool_col(df, "FI_NetBuy_Top30").astype(float) * sc["fi_top30"]
    if sc.get("inst_top30", 0) and "INST_NetBuy_Top30" in df.columns:
        score += (
            _safe_bool_col(df, "INST_NetBuy_Top30").astype(float) * sc["inst_top30"]
        )
    if sc.get("fi_inst_both_top", 0):
        if "FI_NetBuy_Top30" in df.columns and "INST_NetBuy_Top30" in df.columns:
            both = _safe_bool_col(df, "FI_NetBuy_Top30") & _safe_bool_col(
                df, "INST_NetBuy_Top30"
            )
            score += both.astype(float) * sc["fi_inst_both_top"]

    # ── 기술 (5개) ──
    if sc.get("macd_cross", 0):
        score += _safe_bool_col(df, "MACD_Cross_10d").astype(float) * sc["macd_cross"]
    if sc.get("rsi_turn_up", 0):
        rsi14 = _safe_col(df, "RSI14", 100)
        rsi_turn = _safe_bool_col(df, "RSI_TurnUp") & (rsi14 >= 35) & (rsi14 <= 45)
        score += rsi_turn.astype(float) * sc["rsi_turn_up"]
    if sc.get("stoch_golden", 0):
        score += (
            _safe_bool_col(df, "Stoch_Golden_Low").astype(float) * sc["stoch_golden"]
        )
    if sc.get("ma20_cross", 0):
        score += _safe_bool_col(df, "MA20_Cross").astype(float) * sc["ma20_cross"]
    if sc.get("rsi40_cross", 0):
        score += _safe_bool_col(df, "RSI_Cross40_10d").astype(float) * sc["rsi40_cross"]

    # ── 자산/현금 (5개) ──
    if "CashRatio" in df.columns:
        cr = df["CashRatio"].fillna(0)
        if sc.get("cash_ratio_mid", 0):
            score += ((cr >= 0.25) & (cr < 0.4)).astype(float) * sc["cash_ratio_mid"]
        if sc.get("cash_ratio_high", 0):
            score += (cr >= 0.4).astype(float) * sc["cash_ratio_high"]

    if "RealEstateRatio" in df.columns:
        rr = df["RealEstateRatio"].fillna(0)
        if sc.get("realestate_mid", 0):
            score += ((rr >= 0.6) & (rr < 1.0)).astype(float) * sc["realestate_mid"]
        if sc.get("realestate_high", 0):
            score += (rr >= 1.0).astype(float) * sc["realestate_high"]

    if sc.get("both_asset_bonus", 0):
        if "CashRatio" in df.columns and "RealEstateRatio" in df.columns:
            both_high = (df["CashRatio"].fillna(0) >= 0.4) & (
                df["RealEstateRatio"].fillna(0) >= 1.0
            )
            score += both_high.astype(float) * sc["both_asset_bonus"]

    # ── 펀더멘털 (4개) ──
    if "ROE" in df.columns:
        roe = df["ROE"].fillna(0)
        if sc.get("roe_positive", 0):
            score += ((roe > 0) & (roe < 0.05)).astype(float) * sc["roe_positive"]
        if sc.get("roe_above5", 0):
            score += ((roe >= 0.05) & (roe < 0.10)).astype(float) * sc["roe_above5"]
        if sc.get("roe_above10", 0):
            score += (roe >= 0.10).astype(float) * sc["roe_above10"]

    if sc.get("roa_above2", 0) and "ROA" in df.columns:
        score += (df["ROA"].fillna(0) >= 0.02).astype(float) * sc["roa_above2"]

    # ── 배당/안전 (2개) ──
    if sc.get("div_above3", 0):
        div = _safe_col(df, "DIV", 0.0)
        score += (div >= 3.0).astype(float) * sc["div_above3"]

    if sc.get("op_profit_positive", 0) and "operating_profit" in df.columns:
        op = df["operating_profit"]
        score += ((op > 0) & op.notna()).astype(float) * sc["op_profit_positive"]

    # ── 영업이익률 (3개) ──
    if "OperatingMargin" in df.columns:
        om = df["OperatingMargin"].fillna(0)
        if sc.get("op_margin_mid", 0):
            score += ((om >= 5.0) & (om < 10.0)).astype(float) * sc["op_margin_mid"]
        if sc.get("op_margin_high", 0):
            score += (om >= 10.0).astype(float) * sc["op_margin_high"]

    if sc.get("op_margin_improving", 0):
        if "OperatingMargin" in df.columns and "OpMarginPrev" in df.columns:
            om_cur = df["OperatingMargin"].fillna(0)
            om_prev = df["OpMarginPrev"].fillna(0)
            improving = (om_cur > om_prev) & (om_cur > 0)
            score += improving.astype(float) * sc["op_margin_improving"]

    return score


# ──────────────────────────────────────────────
# 매수 타이밍 신호
# ──────────────────────────────────────────────


def compute_timing(df, pass_mask, score, timing_cfg, ohlcv_dict=None):
    """매수 타이밍 신호 계산.

    Returns:
        Series[str]: "STRONG_BUY", "BUY_NOW", "WATCH", "" (미통과)
    """
    strong_threshold = timing_cfg.get("strong_buy_score", 70)
    buy_threshold = timing_cfg.get("buy_score", 50)

    # 기술적 신호 개수 합산
    tech_count = pd.Series(0, index=df.index)
    tech_count += _safe_bool_col(df, "MACD_Cross_10d").astype(int)
    tech_count += _safe_bool_col(df, "MACD_Cross_5d").astype(int)
    tech_count += _safe_bool_col(df, "RSI_TurnUp").astype(int)
    tech_count += _safe_bool_col(df, "Stoch_Golden_Low").astype(int)
    tech_count += _safe_bool_col(df, "MA20_Cross").astype(int)

    # 찐 신호 여부 (ohlcv_dict가 있으면 사용)
    has_real = pd.Series(False, index=df.index)
    if ohlcv_dict is not None:
        from .signals import has_recent_real_signal

        for ticker in df[pass_mask].index:
            ohlcv = ohlcv_dict.get(ticker)
            if ohlcv is not None:
                has_real[ticker] = has_recent_real_signal(ohlcv, 20)

    timing = pd.Series("", index=df.index)

    strong_mask = pass_mask & (score >= strong_threshold) & has_real & (tech_count >= 3)
    buy_mask = pass_mask & ~strong_mask & (score >= buy_threshold) & (tech_count >= 2)
    watch_mask = pass_mask & ~strong_mask & ~buy_mask

    timing[strong_mask] = "STRONG_BUY"
    timing[buy_mask] = "BUY_NOW"
    timing[watch_mask] = "WATCH"

    return timing


# ──────────────────────────────────────────────
# 발굴 사유 태그
# ──────────────────────────────────────────────


def generate_reason_tags(row):
    """단일 종목의 발굴 사유 태그 문자열 생성."""
    tags = []

    # PBR
    pbr = row.get("PBR", np.nan)
    if pd.notna(pbr) and pbr > 0:
        if pbr <= 0.15:
            tags.append(f"PBR{pbr:.2f}")
        elif pbr <= 0.30:
            tags.append(f"PBR{pbr:.2f}")

    # 자산/현금
    cr = row.get("CashRatio", 0) or 0
    if cr >= 0.4:
        tags.append(f"현금{cr:.0%}")
    elif cr >= 0.25:
        tags.append(f"현금{cr:.0%}")

    rr = row.get("RealEstateRatio", 0) or 0
    if rr >= 0.8:
        tags.append("부동산자산")

    # 배당
    div = row.get("DIV", 0) or 0
    if div >= 2.0:
        tags.append(f"DIV{div:.1f}%")

    # 영업이익
    op = row.get("operating_profit", 0) or 0
    if op > 0:
        tags.append("영업이익+")

    # 기술
    if row.get("MACD_Cross_10d") or row.get("MACD_Cross_5d"):
        tags.append("MACD상향")
    elif row.get("MACD_Hist_3dUp"):
        tags.append("MACDHist")
    if row.get("Stoch_Golden_Low"):
        tags.append("Stoch골든")
    if row.get("MA20_Cross"):
        tags.append("MA20돌파")
    if row.get("RSI_Cross40_10d"):
        tags.append("RSI40돌파")

    # 수급
    fi = row.get("Flow1M_FI_KRW", 0) or 0
    inst = row.get("Flow1M_INST_KRW", 0) or 0
    if fi > 0 and inst > 0:
        tags.append("외인+기관")
    elif fi > 0:
        tags.append("외인순매수")
    elif inst > 0:
        tags.append("기관순매수")

    # ROE
    roe = row.get("ROE", 0) or 0
    if roe >= 0.10:
        tags.append("ROE10%")
    elif roe >= 0.05:
        tags.append("ROE5%")

    return " + ".join(tags) if tags else "-"


# ──────────────────────────────────────────────
# 카테고리별 실행
# ──────────────────────────────────────────────


def run_category(df, category_cfg, ohlcv_dict=None, ml_model=None):
    """단일 카테고리 실행.

    Args:
        df: universe DataFrame (지표 계산 완료)
        category_cfg: 카테고리 config dict (gates, scoring, timing)
        ohlcv_dict: {ticker: ohlcv_100d_df} — 타이밍 신호용
        ml_model: AI학습 모델 데이터 (ml_category용)

    Returns:
        dict: {"pass": Series[bool], "score": Series[float],
               "timing": Series[str], "reason": Series[str]}
    """
    # 성장우량주 카테고리 분기 (quality_category)
    if category_cfg.get("quality_category"):
        from . import derived_metrics as dm

        target = category_cfg["quality_category"]  # "A" or "B"

        _qd = df.apply(dm.compute_quality_detail, axis=1)
        grades = _qd.apply(lambda d: d["grade"])

        if target == "A":
            pass_mask = grades.isin(["A", "A+"])
            score = grades.map({"A+": 90.0, "A": 75.0}).fillna(0.0)
        else:  # "B"
            pass_mask = grades == "B"
            score = pass_mask.astype(float) * 60.0

        timing = compute_timing(
            df,
            pass_mask,
            score,
            category_cfg.get("timing", {}),
            ohlcv_dict,
        )

        reason = pd.Series("", index=df.index)
        for t in df[pass_mask].index:
            d = _qd[t]
            reason[t] = " ".join(d["tags"])
            if d.get("peg_verdict"):
                reason[t] += f" PEG:{d['peg_verdict']}"

        return {"pass": pass_mask, "score": score, "timing": timing, "reason": reason}

    # ML 카테고리 분기
    if category_cfg.get("ml_category"):
        empty = {
            "pass": pd.Series(False, index=df.index),
            "score": pd.Series(0.0, index=df.index),
            "timing": pd.Series("", index=df.index),
            "reason": pd.Series("", index=df.index),
        }
        if ml_model is None:
            return empty

        # ml_model: dict {"fast": model, "surge_t25": model, ...} 또는 단일 model dict
        ml_mode = category_cfg.get("ml_mode", "fast")
        # label_target_pct가 있으면 복합 키(예: "surge_t25") 사용
        _tp = category_cfg.get("label_target_pct")
        if _tp is not None:
            model_key = f"{ml_mode}_t{int(_tp * 100)}"
        else:
            model_key = ml_mode
        if isinstance(ml_model, dict) and "model_type" in ml_model:
            # 단일 모델 dict (이전 호환) → fast로 간주
            actual_model = ml_model if ml_mode == "fast" else None
        else:
            actual_model = (
                ml_model.get(model_key) if isinstance(ml_model, dict) else None
            )

        if actual_model is None:
            return empty

        try:
            if ml_mode in ("surge", "surge_1y", "surge_6m", "surge_3m"):
                from .ml_pattern import predict_current_surge as _predict_fn
            elif ml_mode in (
                "etf_surge",
                "etf_surge_1y",
                "etf_surge_6m",
                "etf_surge_3m",
            ):
                from .ml_pattern import predict_current_etf_surge as _predict_fn
            elif ml_mode == "accum":
                from .ml_pattern import predict_current_accum as _predict_fn
            elif ml_mode == "precise":
                from .ml_pattern import predict_current_precise as _predict_fn
            else:
                from .ml_pattern import predict_current as _predict_fn

            # ml_pre_filter: 학습과 동일한 기본조건 적용
            pf = category_cfg.get("ml_pre_filter", {})
            eligible_mask = pd.Series(True, index=df.index)
            if pf:
                mcap = _safe_col(df, "MarketCap", 0)
                per = _safe_col(df, "PER", 0)
                pbr = _safe_col(df, "PBR", 0)
                if "min_mcap" in pf:
                    eligible_mask = eligible_mask & (mcap >= pf["min_mcap"])
                if "per_min" in pf:
                    eligible_mask = eligible_mask & (per > pf["per_min"])
                if "per_max" in pf:
                    eligible_mask = eligible_mask & (per <= pf["per_max"])
                if "pbr_min" in pf:
                    eligible_mask = eligible_mask & (pbr > pf["pbr_min"])
                if "pbr_max" in pf:
                    eligible_mask = eligible_mask & (pbr <= pf["pbr_max"])
                if "roe_min" in pf:
                    roe = pbr / per.replace(0, np.nan)
                    eligible_mask = eligible_mask & (roe.fillna(-999) > pf["roe_min"])

            eligible_tickers = df[eligible_mask].index.tolist()

            if ml_mode in (
                "precise",
                "surge",
                "accum",
                "surge_1y",
                "surge_6m",
                "surge_3m",
            ):
                # KOSPI 지수 (상대강도 피처)
                _kospi_close = None
                try:
                    from .data_fetcher import fetch_kospi_index_3year

                    _kdf = fetch_kospi_index_3year(None)
                    if _kdf is not None and len(_kdf) > 0:
                        _kospi_close = _kdf["종가"].astype(float)
                except Exception:
                    pass
                ml_result = _predict_fn(
                    actual_model, ohlcv_dict, eligible_tickers, kospi_close=_kospi_close
                )
            else:
                ml_result = _predict_fn(actual_model, ohlcv_dict, eligible_tickers)

            pass_s = pd.Series(False, index=df.index)
            score_s = pd.Series(0.0, index=df.index)
            timing_s = pd.Series("", index=df.index)
            reason_s = pd.Series("", index=df.index)
            for t in eligible_tickers:
                if t in ml_result.index:
                    pass_s[t] = bool(ml_result.at[t, "AI_Pass"])
                    score_s[t] = float(ml_result.at[t, "AI_Score"])
                    timing_s[t] = str(ml_result.at[t, "AI_Timing"])
                    reason_s[t] = str(ml_result.at[t, "AI_Reason"])
            return {
                "pass": pass_s,
                "score": score_s,
                "timing": timing_s,
                "reason": reason_s,
            }
        except Exception:
            return empty

    pass_mask = apply_gates(df, category_cfg["gates"])
    score = apply_scoring(df, category_cfg["scoring"]["detail"])
    score[~pass_mask] = 0

    timing = compute_timing(
        df,
        pass_mask,
        score,
        category_cfg.get("timing", {}),
        ohlcv_dict,
    )

    # 사유 태그
    reason = pd.Series("", index=df.index)
    for idx in df[pass_mask].index:
        reason[idx] = generate_reason_tags(df.loc[idx])

    return {
        "pass": pass_mask,
        "score": score,
        "timing": timing,
        "reason": reason,
    }


def run_all_categories(df, categories_config, ohlcv_dict=None, ml_model=None):
    """12개 카테고리 동시 실행.

    Args:
        df: universe DataFrame
        categories_config: {name: category_cfg} dict
        ohlcv_dict: {ticker: ohlcv_100d_df}
        ml_model: {"fast": model_data, "precise": model_data} 또는 단일 model dict

    Returns:
        DataFrame: 원본 + Pass_/Score_/Timing_/Reason_ 컬럼 추가
    """
    result = df.copy()

    # 수급 상위 30% 사전 계산
    fi_amt = _safe_col(result, "Flow1M_FI_KRW", 0)
    inst_amt = _safe_col(result, "Flow1M_INST_KRW", 0)
    fi_threshold = fi_amt[fi_amt > 0].quantile(0.7) if (fi_amt > 0).any() else 0
    inst_threshold = inst_amt[inst_amt > 0].quantile(0.7) if (inst_amt > 0).any() else 0
    result["FI_NetBuy_Top30"] = fi_amt >= fi_threshold
    result["INST_NetBuy_Top30"] = inst_amt >= inst_threshold

    for name, cat_cfg in categories_config.items():
        cat_result = run_category(result, cat_cfg, ohlcv_dict, ml_model=ml_model)
        result[f"Pass_{name}"] = cat_result["pass"]
        result[f"Score_{name}"] = cat_result["score"]
        result[f"Timing_{name}"] = cat_result["timing"]
        result[f"Reason_{name}"] = cat_result["reason"]

    # Pass_any: 어느 카테고리든 하나라도 통과
    pass_cols = [f"Pass_{name}" for name in categories_config]
    result["Pass_any"] = result[pass_cols].any(axis=1)

    return result


# ──────────────────────────────────────────────
# 쉬운 말 발굴이유 / 타이밍이유
# ──────────────────────────────────────────────


def generate_simple_reason(row):
    """발굴 이유를 쉬운 말로 요약 (Excel/UI용)."""
    reasons = []

    pbr = row.get("PBR", 0) or 0
    if 0 < pbr <= 0.2:
        reasons.append(f"주가가 순자산의 {pbr:.0%}에 불과(극저평가)")
    elif 0 < pbr <= 0.4:
        reasons.append(f"주가가 순자산의 {pbr:.0%}(저평가)")
    elif 0 < pbr <= 0.7:
        reasons.append(f"PBR {pbr:.2f}(순자산 대비 저평가)")

    cr = row.get("CashRatio", 0) or 0
    if cr >= 0.4:
        reasons.append(f"보유 현금이 시총의 {cr:.0%}(현금부자)")
    elif cr >= 0.25:
        reasons.append(f"현금비율 {cr:.0%}")

    rr = row.get("RealEstateRatio", 0) or 0
    if rr >= 1.0:
        reasons.append(f"부동산가치가 시총 초과({rr:.0%})")
    elif rr >= 0.6:
        reasons.append(f"상당한 부동산 보유(시총의 {rr:.0%})")

    div_val = row.get("DIV", 0) or 0
    if div_val >= 5:
        reasons.append(f"배당수익률 {div_val:.1f}%(초고배당)")
    elif div_val >= 3:
        reasons.append(f"배당수익률 {div_val:.1f}%(고배당)")

    roe = row.get("ROE", 0) or 0
    if roe >= 0.10:
        reasons.append(f"ROE {roe*100:.1f}%(수익성 우수)")
    elif roe >= 0.05:
        reasons.append(f"ROE {roe*100:.1f}%(수익성 양호)")

    fi = row.get("Flow1M_FI_KRW", 0) or 0
    inst = row.get("Flow1M_INST_KRW", 0) or 0
    if fi > 0 and inst > 0:
        reasons.append("외국인+기관 동시 순매수 중")
    elif fi > 0:
        reasons.append("외국인 순매수 중")
    elif inst > 0:
        reasons.append("기관 순매수 중")

    dd = row.get("DrawdownFrom52wHigh", 0) or 0
    if dd >= 0.5:
        reasons.append(f"52주 고점 대비 {dd*100:.0f}%↓(낙폭 과대)")
    elif dd >= 0.3:
        reasons.append(f"고점 대비 {dd*100:.0f}%↓(반등 여지)")

    op = row.get("operating_profit", 0) or 0
    if op > 0:
        reasons.append("영업이익 흑자(실적 안전망)")

    return " / ".join(reasons) if reasons else "저평가 구간, 추가 분석 필요"


def generate_timing_explanation(row, category_names_icons):
    """매수 타이밍 이유를 쉬운 말로 요약 (Excel/UI용).

    Args:
        row: DataFrame row
        category_names_icons: [(name, icon), ...] 카테고리 목록
    """
    TIMING_DESC = {
        "STRONG_BUY": "높은 점수 + 복합 매수신호 + 기술적 반등이 동시에 나타남 → 즉시 매수 고려",
        "BUY_NOW": "양호한 점수 + 기술적 반등 조짐 확인 → 매수 시점",
        "WATCH": "게이트 통과했으나 기술적 신호 부족 → 신호 발생 시 매수",
    }
    TIMING_KR = {"STRONG_BUY": "즉시매수", "BUY_NOW": "매수", "WATCH": "관심"}

    parts = []
    for name, icon in category_names_icons:
        timing = row.get(f"Timing_{name}", "")
        if timing and timing in TIMING_KR:
            score = row.get(f"Score_{name}", 0) or 0
            kr = TIMING_KR[timing]
            desc = TIMING_DESC[timing]
            parts.append(f"[{icon}{name}] {kr}({score:.0f}점): {desc}")
    return "\n".join(parts) if parts else "-"
