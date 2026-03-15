"""기술지표 계산 모듈.

100일치 OHLCV에서 RSI(14), MACD(12,26,9), Stochastic(14,3,3), MA20을 계산한다.
앞 40일은 EMA 워밍업으로 사용하고, 실제 신호 판정은 최근 60일 기준.
"""

import numpy as np
import pandas as pd


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD, Signal, Histogram 반환."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": histogram,
    }, index=close.index)


def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic %K, %D."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    denom = highest_high - lowest_low
    denom = denom.replace(0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return pd.DataFrame({
        "StochK": stoch_k,
        "StochD": stoch_d,
    }, index=close.index)


def calc_ma(close: pd.Series, period: int = 20) -> pd.Series:
    """단순이동평균."""
    return close.rolling(window=period).mean()


def compute_all_indicators(ohlcv_df: pd.DataFrame) -> dict:
    """100일치 OHLCV DataFrame에서 모든 기술지표 + BOOL 신호를 계산.

    Args:
        ohlcv_df: 컬럼 = 시가, 고가, 저가, 종가, 거래량

    Returns:
        dict with all indicator values and bool signals for the last row.
    """
    if ohlcv_df is None or len(ohlcv_df) < 35:
        return _empty_indicators()

    close = ohlcv_df["종가"].astype(float)
    high = ohlcv_df["고가"].astype(float)
    low = ohlcv_df["저가"].astype(float)

    # 계산
    rsi = calc_rsi(close, 14)
    macd_df = calc_macd(close, 12, 26, 9)
    stoch_df = calc_stochastic(high, low, close, 14, 3)
    ma20 = calc_ma(close, 20)

    # 최근값
    rsi_now = rsi.iloc[-1] if len(rsi) > 0 else np.nan
    rsi_prev = rsi.iloc[-2] if len(rsi) > 1 else np.nan
    macd_now = macd_df["MACD"].iloc[-1] if len(macd_df) > 0 else np.nan
    macd_sig_now = macd_df["MACD_Signal"].iloc[-1] if len(macd_df) > 0 else np.nan
    macd_hist_now = macd_df["MACD_Hist"].iloc[-1] if len(macd_df) > 0 else np.nan
    stoch_k_now = stoch_df["StochK"].iloc[-1] if len(stoch_df) > 0 else np.nan
    stoch_d_now = stoch_df["StochD"].iloc[-1] if len(stoch_df) > 0 else np.nan
    ma20_now = ma20.iloc[-1] if len(ma20) > 0 else np.nan
    close_now = close.iloc[-1] if len(close) > 0 else np.nan

    # BOOL 신호 판정
    result = {
        "RSI14": rsi_now,
        "MACD": macd_now,
        "MACD_Signal": macd_sig_now,
        "MACD_Hist": macd_hist_now,
        "StochK": stoch_k_now,
        "StochD": stoch_d_now,
        "MA20": ma20_now,
    }

    # RSI 상승 전환
    result["RSI_TurnUp"] = (
        pd.notna(rsi_now) and pd.notna(rsi_prev) and rsi_now > rsi_prev
    )

    # MACD Histogram 3일 연속 증가
    if len(macd_df) >= 4:
        hist = macd_df["MACD_Hist"].iloc[-4:]
        result["MACD_Hist_3dUp"] = bool(
            (hist.diff().iloc[-3:] > 0).all()
        )
    else:
        result["MACD_Hist_3dUp"] = False

    # MACD 시그널 상향 돌파 (최근 N일 이내)
    result["MACD_Cross_10d"] = _check_cross_up(
        macd_df["MACD"], macd_df["MACD_Signal"], lookback=10
    )

    # MACD 시그널 상향 돌파 (최근 5일 이내) — v1 Signal용
    result["MACD_Cross_5d"] = _check_cross_up(
        macd_df["MACD"], macd_df["MACD_Signal"], lookback=5
    )

    # Stoch 저점 골든크로스 (%K > %D 상향돌파 AND %K <= 30 — 과매도 영역)
    result["Stoch_Golden_Low"] = _check_stoch_golden(
        stoch_df["StochK"], stoch_df["StochD"], k_max=30
    )

    # MA20 상향 돌파 (최근 10일 이내)
    result["MA20_Cross"] = _check_cross_up(close, ma20, lookback=10)

    # RSI 40 상향 돌파 (최근 10일 이내)
    rsi_40_line = pd.Series(40.0, index=rsi.index)
    result["RSI_Cross40_10d"] = _check_cross_up(rsi, rsi_40_line, lookback=10)

    return result


def _check_cross_up(fast: pd.Series, slow: pd.Series,
                    lookback: int = 10) -> bool:
    """fast가 slow를 상향 돌파했는지 (최근 lookback일 이내)."""
    if len(fast) < lookback + 1 or len(slow) < lookback + 1:
        return False

    recent_fast = fast.iloc[-(lookback + 1):]
    recent_slow = slow.iloc[-(lookback + 1):]

    diff = recent_fast - recent_slow
    for i in range(1, len(diff)):
        if pd.notna(diff.iloc[i]) and pd.notna(diff.iloc[i - 1]):
            if diff.iloc[i] > 0 and diff.iloc[i - 1] <= 0:
                return True
    return False


def _check_stoch_golden(stoch_k: pd.Series, stoch_d: pd.Series,
                        k_max: float = 40) -> bool:
    """Stoch %K가 %D를 상향 돌파 AND %K <= k_max."""
    if len(stoch_k) < 3:
        return False

    k_now = stoch_k.iloc[-1]
    d_now = stoch_d.iloc[-1]
    k_prev = stoch_k.iloc[-2]
    d_prev = stoch_d.iloc[-2]

    if pd.isna(k_now) or pd.isna(d_now) or pd.isna(k_prev) or pd.isna(d_prev):
        return False

    crossed = (k_now > d_now) and (k_prev <= d_prev)
    return crossed and (k_now <= k_max)


def _empty_indicators() -> dict:
    return {
        "RSI14": np.nan,
        "MACD": np.nan,
        "MACD_Signal": np.nan,
        "MACD_Hist": np.nan,
        "StochK": np.nan,
        "StochD": np.nan,
        "MA20": np.nan,
        "RSI_TurnUp": False,
        "MACD_Hist_3dUp": False,
        "MACD_Cross_10d": False,
        "MACD_Cross_5d": False,
        "Stoch_Golden_Low": False,
        "MA20_Cross": False,
        "RSI_Cross40_10d": False,
    }
