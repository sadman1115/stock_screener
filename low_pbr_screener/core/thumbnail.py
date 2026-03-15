"""썸네일 차트 생성 모듈 (matplotlib).

종목 그리드용 경량 PNG 이미지를 생성한다.
Plotly 대신 matplotlib을 사용하여 수십 개 차트를 빠르게 렌더링.
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .signals import detect_real_signals


def generate_thumbnail(ohlcv_df: pd.DataFrame,
                       days: int = 60,
                       figsize: tuple = (3.5, 2.0),
                       dpi: int = 100) -> bytes | None:
    """3개월 라인 차트 썸네일 PNG bytes 생성.

    Args:
        ohlcv_df: 100일 OHLCV (종가 컬럼 필수)
        days: 표시할 거래일 수 (기본 60 = ~3개월)

    Returns:
        PNG bytes 또는 None
    """
    if ohlcv_df is None or len(ohlcv_df) < 20:
        return None

    df = ohlcv_df.iloc[-days:] if len(ohlcv_df) > days else ohlcv_df
    close = df["종가"].astype(float)
    dates = df.index

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 상승/하락 색상
    color = "#2ecc71" if close.iloc[-1] >= close.iloc[0] else "#e74c3c"
    ax.plot(dates, close, color=color, linewidth=1.3)
    ax.fill_between(dates, close, close.min() * 0.98, alpha=0.08, color=color)

    # 찐 신호 마커
    real_sigs = detect_real_signals(ohlcv_df)
    for sig in real_sigs:
        if sig["date"] in df.index:
            ax.scatter(sig["date"], sig["price"],
                       marker="^", color="#f1c40f", edgecolors="#e67e22",
                       s=50, zorder=5, linewidths=0.8)

    # 미니멀 스타일
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    ax.tick_params(axis="x", labelsize=7, length=2)
    ax.tick_params(axis="y", labelsize=0, length=0)
    ax.yaxis.set_visible(False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)

    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_thumbnails_batch(ohlcv_dict: dict,
                              days: int = 60) -> dict[str, bytes]:
    """여러 종목의 썸네일 일괄 생성.

    Args:
        ohlcv_dict: {ticker: ohlcv_df}

    Returns:
        {ticker: png_bytes}
    """
    result = {}
    for ticker, ohlcv_df in ohlcv_dict.items():
        img = generate_thumbnail(ohlcv_df, days=days)
        if img:
            result[ticker] = img
    return result
