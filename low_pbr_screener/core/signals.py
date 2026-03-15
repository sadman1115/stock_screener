"""찐 신호 (Real Buy Signal) 감지 모듈.

3거래일 내 2개 이상 기술적 매수 신호가 동시 발생하면 '찐 신호'로 판정한다.
4가지 신호: MACD 골든크로스, RSI 30 상향돌파, Stoch 저점 골든크로스, MA20 돌파.
"""

import numpy as np
import pandas as pd

from . import indicators


def detect_individual_signals(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """일별 4개 매수 신호 bool DataFrame 반환.

    Returns:
        DataFrame with columns: sig_macd, sig_rsi, sig_stoch, sig_ma20
    """
    if ohlcv_df is None or len(ohlcv_df) < 35:
        return pd.DataFrame()

    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    low = ohlcv_df["저가"].astype(float)

    rsi = indicators.calc_rsi(close, 14)
    macd_df = indicators.calc_macd(close, 12, 26, 9)
    stoch_df = indicators.calc_stochastic(high, low, close, 14, 3)
    ma20 = indicators.calc_ma(close, 20)

    n = len(ohlcv_df)
    dates = ohlcv_df.index

    sig_macd = pd.Series(False, index=dates)
    sig_rsi = pd.Series(False, index=dates)
    sig_stoch = pd.Series(False, index=dates)
    sig_ma20 = pd.Series(False, index=dates)

    macd_line = macd_df["MACD"]
    macd_sig = macd_df["MACD_Signal"]
    stoch_k = stoch_df["StochK"]
    stoch_d = stoch_df["StochD"]

    for i in range(1, n):
        # MACD 골든크로스
        m_now, m_prev = macd_line.iloc[i], macd_line.iloc[i - 1]
        s_now, s_prev = macd_sig.iloc[i], macd_sig.iloc[i - 1]
        if pd.notna(m_now) and pd.notna(s_now) and pd.notna(m_prev) and pd.notna(s_prev):
            if m_now > s_now and m_prev <= s_prev:
                sig_macd.iloc[i] = True

        # RSI 30 상향돌파 (과매도 탈출)
        r_now, r_prev = rsi.iloc[i], rsi.iloc[i - 1]
        if pd.notna(r_now) and pd.notna(r_prev):
            if r_now > 30 and r_prev <= 30:
                sig_rsi.iloc[i] = True

        # Stochastic 저점 골든크로스 (K > D, K <= 25 — 진정한 과매도만)
        k_now, k_prev = stoch_k.iloc[i], stoch_k.iloc[i - 1]
        d_now, d_prev = stoch_d.iloc[i], stoch_d.iloc[i - 1]
        if pd.notna(k_now) and pd.notna(d_now) and pd.notna(k_prev) and pd.notna(d_prev):
            if k_now > d_now and k_prev <= d_prev and k_now <= 25:
                sig_stoch.iloc[i] = True

        # MA20 상향돌파
        c_now, c_prev = close.iloc[i], close.iloc[i - 1]
        ma_now, ma_prev = ma20.iloc[i], ma20.iloc[i - 1]
        if pd.notna(ma_now) and pd.notna(ma_prev):
            if c_now > ma_now and c_prev <= ma_prev:
                sig_ma20.iloc[i] = True

    return pd.DataFrame({
        "sig_macd": sig_macd,
        "sig_rsi": sig_rsi,
        "sig_stoch": sig_stoch,
        "sig_ma20": sig_ma20,
    }, index=dates)


def detect_real_signals(ohlcv_df: pd.DataFrame,
                        window: int = 5,
                        min_count: int = 2,
                        min_dedup_gap: int = 10) -> list[dict]:
    """찐 신호 감지: window일 내 min_count개 이상 신호 동시 발생.

    품질 필터: MACD 또는 RSI 중 최소 1개 필수 (Stoch+MA20 단독 불가).
    중복 방지: 직전 신호 후 min_dedup_gap 거래일 경과해야 새 신호 인정.

    Returns:
        List of {date, price, signals: [str], count: int}
    """
    sig_df = detect_individual_signals(ohlcv_df)
    if sig_df.empty:
        return []

    close = ohlcv_df["종가"].astype(float)
    signal_cols = ["sig_macd", "sig_rsi", "sig_stoch", "sig_ma20"]
    signal_names = ["MACD", "RSI", "Stoch", "MA20"]
    quality_signals = {"MACD", "RSI"}  # 노이즈가 적은 핵심 신호

    real_signals = []
    last_signal_idx = -min_dedup_gap - 1  # 첫 신호 허용

    for i in range(window - 1, len(sig_df)):
        # 중복 방지: 직전 신호로부터 충분한 간격 확보
        if i - last_signal_idx < min_dedup_gap:
            continue

        start = max(0, i - window + 1)
        window_slice = sig_df.iloc[start:i + 1]

        active = []
        for col, name in zip(signal_cols, signal_names):
            if window_slice[col].any():
                active.append(name)

        if len(active) >= min_count:
            # 품질 필터: MACD 또는 RSI 하나 이상 포함 필수
            if not quality_signals.intersection(active):
                continue

            real_signals.append({
                "date": sig_df.index[i],
                "price": float(close.iloc[i]),
                "signals": active,
                "count": len(active),
            })
            last_signal_idx = i

    return real_signals


def has_recent_real_signal(ohlcv_df: pd.DataFrame, days: int = 20) -> bool:
    """최근 N거래일 내 찐 신호가 있는지 빠르게 확인."""
    sigs = detect_real_signals(ohlcv_df)
    if not sigs or ohlcv_df is None or ohlcv_df.empty:
        return False
    cutoff = ohlcv_df.index[-min(days, len(ohlcv_df))]
    return any(s["date"] >= cutoff for s in sigs)


def get_signal_recency_badge(ohlcv_df: pd.DataFrame) -> str:
    """신호 근접도에 따른 종 뱃지 반환.

    2거래일 이내: 🔴🔔🔔🔔🔔 (긴급)
    5거래일 이내: 🔔🔔🔔
    10거래일 이내: 🔔🔔
    20거래일 이내: 🔔
    그 외: ""
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return ""
    sigs = detect_real_signals(ohlcv_df)
    if not sigs:
        return ""

    n = len(ohlcv_df)
    tiers = [
        (2, "🔴🔔🔔🔔🔔"),
        (5, "🔔🔔🔔"),
        (10, "🔔🔔"),
        (20, "🔔"),
    ]
    for days, badge in tiers:
        cutoff = ohlcv_df.index[-min(days, n)]
        if any(s["date"] >= cutoff for s in sigs):
            return badge

    return ""
