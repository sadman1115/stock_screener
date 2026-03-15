"""Page 3: 종목 상세 — 12개 전략별 썸네일 그리드 + 상세 팝업 + 네이버 금융."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data_fetcher as fetcher
from core import indicators
from core.signals import (
    detect_real_signals,
    get_signal_recency_badge,
)
from core.styles import apply_mobile_styles
from core.thumbnail import generate_thumbnails_batch

st.set_page_config(page_title="종목 상세", page_icon="📋", layout="wide")
apply_mobile_styles()

st.title("📋 종목 상세")

if "screened" not in st.session_state or "preset_results" not in st.session_state:
    st.warning("먼저 **🎯 스크리닝** 페이지에서 스크리닝을 실행하세요.")
    st.stop()

df = st.session_state["screened"]
date_str = st.session_state.get("date_str", "20260204")
ohlcv_dict = st.session_state.get("candidates_ohlcv_100d", {})
# 1년 OHLCV (썸네일용) — 없으면 100d로 폴백
ohlcv_1y_dict = st.session_state.get("candidates_ohlcv_1y", {})

# 카테고리 정의 (Page 2와 동일 — 12개 전략)
# (데이터키, 아이콘, 탭 짧은 이름)
PRESET_LIST = [
    ("자산괴리주", "💎", "자산"),
    ("고배당_가치주", "💰", "배당"),
    ("실적_턴어라운드", "📈", "턴어라운드"),
    ("외인기관_쌍끌이", "🌍", "외인기관"),
    ("기술적_바닥반전", "⚡", "기술적바닥"),
    ("초저PBR_안전망", "🔥", "저PBR"),
    ("수급전환_모멘텀", "📊", "수급"),
    ("저PER_가치주", "💲", "저PER"),
    ("AI_25%급등주", "🤖", "25%2Y"),
    ("AI_25%급등주_1Y", "🤖", "25%1Y"),
    ("AI_25%급등주_6M", "🤖", "25%6M"),
    ("AI_25%급등주_3M", "🤖", "25%3M"),
    ("AI_30%급등주", "🤖", "30%2Y"),
    ("AI_30%급등주_1Y", "🤖", "30%1Y"),
    ("AI_30%급등주_6M", "🤖", "30%6M"),
    ("AI_30%급등주_3M", "🤖", "30%3M"),
]

TIMING_BADGE = {
    "STRONG_BUY": "🔴",
    "BUY_NOW": "🟢",
    "WATCH": "🟡",
}


# ══════════════════════════════════════════════════════
# 분석 코멘트 생성
# ══════════════════════════════════════════════════════
def _generate_analysis(row, ticker):
    """썸네일 아래 표시할 상승 전망 분석 텍스트 생성.

    원칙:
    - 고점 근접(5%이내)은 상승 이유가 아니라 주의 요인
    - PBR 목표는 상승여력 20%이상일 때만 표시
    - 52주 회복 목표는 낙폭 25%이상일 때만 표시
    - 주의 요인(고점근접, 적자, 과매수)을 명시적으로 경고
    """
    lines = []
    cautions = []  # 주의 요인

    pbr = row.get("PBR", np.nan)
    dd = row.get("DrawdownFrom52wHigh", np.nan)

    # ── 1. 상승 이유 (Why) ──
    reasons = []
    if pd.notna(pbr) and pbr > 0:
        if pbr <= 0.2:
            reasons.append(f"PBR {pbr:.2f} 극저평가")
        elif pbr <= 0.4:
            reasons.append(f"PBR {pbr:.2f} 저평가")
        elif pbr <= 0.7:
            reasons.append(f"PBR {pbr:.2f}")
        # PBR > 0.7: 특별한 저평가 신호가 아님 → 생략

    div = row.get("DIV", 0) or 0
    if div >= 3.0:
        reasons.append(f"배당 {div:.1f}%")

    # 고점대비 낙폭: 15%이상일 때만 반등 여력으로 표시
    if pd.notna(dd):
        if dd >= 0.40:
            reasons.append(f"고점대비 -{dd*100:.0f}% 대폭낙폭")
        elif dd >= 0.25:
            reasons.append(f"고점대비 -{dd*100:.0f}% 낙폭")
        elif dd >= 0.15:
            reasons.append(f"고점대비 -{dd*100:.0f}%")
        # 5%미만: 고점 근접 → 주의 요인으로 분류
        if dd < 0.05:
            cautions.append(f"52주 고점 근접(-{dd*100:.0f}%)")

    cr = row.get("CashRatio", np.nan)
    if pd.notna(cr) and cr >= 0.3:
        reasons.append(f"현금비율 {cr:.0%}")

    rr = row.get("RealEstateRatio", np.nan)
    if pd.notna(rr) and rr >= 0.8:
        reasons.append(f"부동산비율 {rr:.0%}")

    fi = row.get("Flow1M_FI_KRW", 0) or 0
    inst = row.get("Flow1M_INST_KRW", 0) or 0
    if fi > 0 and inst > 0:
        reasons.append("외인+기관 동반매수")
    elif fi > 0:
        reasons.append("외인 순매수")
    elif inst > 0:
        reasons.append("기관 순매수")

    if reasons:
        lines.append(" / ".join(reasons[:4]))

    # ── 2. 기술적 신호 (When) ──
    signals = []
    ohlcv = ohlcv_dict.get(ticker)
    has_sig = False
    if ohlcv is not None and len(ohlcv) >= 35:
        real_sigs = detect_real_signals(ohlcv)
        recent_sigs = (
            [s for s in real_sigs if s["date"] >= ohlcv.index[-20]] if real_sigs else []
        )
        if recent_sigs:
            has_sig = True
            latest = recent_sigs[-1]
            _sig_kr = {
                "MACD": "중기추세전환",
                "RSI": "과매도탈출",
                "Stoch": "단기바닥확인",
                "MA20": "20일평균돌파",
            }
            desc = "+".join(_sig_kr.get(s, s) for s in latest["signals"])
            signals.append(f"복합 매수신호 ({desc})")
    else:
        real_sigs = []

    rsi_val = row.get("RSI14", np.nan)
    if pd.notna(rsi_val):
        if rsi_val <= 30:
            signals.append(f"과매도 반등 임박 (RSI {rsi_val:.0f})")
        elif rsi_val <= 40:
            signals.append(f"매도세 약화 (RSI {rsi_val:.0f})")
        elif rsi_val >= 70:
            cautions.append(f"과매수 구간(RSI {rsi_val:.0f})")

    if row.get("MACD_Cross_5d") or row.get("MACD_Cross_10d"):
        signals.append("중기 추세 상승 전환")
    elif row.get("MACD_Hist_3dUp"):
        signals.append("하락세 약화")

    if row.get("Stoch_Golden_Low"):
        signals.append("단기 바닥 전환")

    if signals:
        prefix = "▲" if has_sig else "△"
        lines.append(f"{prefix} {' · '.join(signals[:3])}")

    # ── 3. 예상 상승폭 (How much) ──
    target_parts = []

    # PBR 기반 목표: PBR < 0.4이고 상승여력 20%이상일 때만
    if pd.notna(pbr) and 0 < pbr < 0.4:
        target_pbr = min(0.6, pbr * 1.8)  # 보수적 목표
        upside_pbr = (target_pbr / pbr - 1) * 100
        if upside_pbr >= 20:
            target_parts.append(f"PBR {target_pbr:.1f} 회귀 시 +{upside_pbr:.0f}%")

    # 52주 고점 회복: 낙폭 25%이상일 때만
    high52 = row.get("High52", 0) or 0
    close = row.get("Close", 0) or 0
    if high52 > 0 and close > 0 and pd.notna(dd) and dd >= 0.25:
        recovery_half = (high52 + close) / 2  # 고점 50% 회복
        upside_52 = (recovery_half / close - 1) * 100
        if upside_52 >= 10:
            target_parts.append(f"고점 50%회복 시 +{upside_52:.0f}%")

    if target_parts:
        period = "단기~중기" if has_sig else ("중기" if signals else "중장기")
        lines.append(f"{period} {target_parts[0]}")

    # ── 4. 주의 요인 ──
    roe = row.get("ROE", np.nan)
    if pd.notna(roe) and roe < 0:
        cautions.append("적자 기업")

    if cautions:
        lines.append(f"⚠ {' · '.join(cautions)}")

    if not lines:
        lines.append("저평가 구간, 모니터링 필요")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 썸네일 그리드 렌더링
# ══════════════════════════════════════════════════════
def _ensure_thumbnails(stocks_df, tab_key):
    """썸네일 이미지 캐시 생성/반환 (1년 차트)."""
    cache_key = f"thumbnails_{tab_key}"
    if cache_key not in st.session_state:
        tickers = stocks_df.index.tolist()
        # 1년 OHLCV 우선, 없으면 100d 폴백
        subset = {}
        for t in tickers:
            if t in ohlcv_1y_dict:
                subset[t] = ohlcv_1y_dict[t]
            elif t in ohlcv_dict:
                subset[t] = ohlcv_dict[t]
        st.session_state[cache_key] = generate_thumbnails_batch(subset, days=250)
    return st.session_state[cache_key]


def render_stock_grid(stocks_df, tab_key):
    """종목 썸네일 그리드 렌더링."""
    if stocks_df.empty:
        st.info("해당 전략 통과 종목이 없습니다.")
        return

    thumbs = _ensure_thumbnails(stocks_df, tab_key)

    cols = st.columns(4)
    for i, (ticker, row) in enumerate(stocks_df.iterrows()):
        col = cols[i % 4]
        with col:
            name = row.get("Name", ticker)
            pbr = row.get("PBR", 0)
            score_key = f"Score_{tab_key}"
            score = row.get(score_key, 0) or 0

            sig_badge_str = (
                get_signal_recency_badge(ohlcv_dict.get(ticker))
                if ticker in ohlcv_dict
                else ""
            )
            sig_badge = f" {sig_badge_str}" if sig_badge_str else ""

            # 타이밍 뱃지
            timing_key = f"Timing_{tab_key}"
            timing_val = row.get(timing_key, "")
            timing_badge = TIMING_BADGE.get(timing_val, "")
            if timing_badge:
                sig_badge = f" {timing_badge}" + sig_badge

            # 헤더: 종목명(클릭→상세) + 네이버(작은 링크)
            naver_url = f"https://finance.naver.com/item/main.naver?code={ticker}"
            hc1, hc2 = st.columns([5, 1])
            with hc1:
                if st.button(
                    f"{name}{sig_badge}",
                    key=f"d_{tab_key}_{ticker}",
                    use_container_width=True,
                ):
                    st.session_state["detail_ticker"] = ticker
                    st.session_state["detail_trigger"] = True
            with hc2:
                st.markdown(
                    f'<a href="{naver_url}" target="_blank" '
                    f'style="font-size:0.75em;color:#888;text-decoration:none">'
                    f"N</a>",
                    unsafe_allow_html=True,
                )

            st.caption(f"PBR {pbr:.2f} | 점수 {score:.0f}")

            # 썸네일
            if ticker in thumbs:
                st.image(thumbs[ticker], use_container_width=True)
            else:
                st.caption("차트 없음")

            # 분석 코멘트
            analysis = _generate_analysis(row, ticker)
            st.markdown(
                f'<div style="font-size:0.72em;color:#555;line-height:1.4;'
                f'padding:2px 0 6px 0">{analysis.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")


# ══════════════════════════════════════════════════════
# 정렬 옵션
# ══════════════════════════════════════════════════════
TIMING_RANK = {"STRONG_BUY": 3, "BUY_NOW": 2, "WATCH": 1, "": 0}

sort_options = {
    "score": "점수 (높은순)",
    "timing": "타이밍 (즉시매수 우선)",
    "pbr": "PBR (낮은순)",
}
if "DIV" in df.columns:
    sort_options["div"] = "배당률 (높은순)"

sort_choice = st.radio(
    "정렬 기준",
    list(sort_options.keys()),
    format_func=lambda x: sort_options[x],
    horizontal=True,
    key="p3_sort",
)

# ══════════════════════════════════════════════════════
# 탭: 전략별 + AI/타이밍별 (2줄로 분리)
# ══════════════════════════════════════════════════════


def _sort_preset_df(preset_df, tab_key, sort_key):
    """정렬 기준에 따라 DataFrame 정렬."""
    score_col = f"Score_{tab_key}"
    timing_col = f"Timing_{tab_key}"
    if sort_key == "timing" and timing_col in preset_df.columns:
        preset_df["_TR"] = preset_df[timing_col].map(TIMING_RANK).fillna(0)
        preset_df = preset_df.sort_values(["_TR", score_col], ascending=[False, False])
        preset_df.drop(columns=["_TR"], inplace=True)
    elif sort_key == "pbr" and "PBR" in preset_df.columns:
        preset_df = preset_df.sort_values("PBR", ascending=True)
    elif sort_key == "div" and "DIV" in preset_df.columns:
        preset_df = preset_df.sort_values("DIV", ascending=False)
    elif score_col in preset_df.columns:
        preset_df = preset_df.sort_values(score_col, ascending=False)
    return preset_df


def _render_preset_tab(tab, name):
    """전략 탭 내부 렌더링."""
    with tab:
        pass_key = f"Pass_{name}"
        if pass_key in df.columns:
            preset_df = df[df[pass_key]].copy()
            preset_df = _sort_preset_df(preset_df, name, sort_choice)
            st.caption(f"{len(preset_df)}종목")
            render_stock_grid(preset_df, name)
        else:
            st.info(f"{name} 결과 없음")


def _render_timing_tab(tab, timing_val, timing_label):
    """타이밍 탭 내부 렌더링."""
    with tab:
        timing_cols = [
            f"Timing_{name}"
            for name, _, _ in PRESET_LIST
            if f"Timing_{name}" in df.columns
        ]
        if timing_cols:
            mask = df[timing_cols].apply(lambda row: timing_val in row.values, axis=1)
            timing_df = df[
                mask & df.get("Pass_any", pd.Series(True, index=df.index))
            ].copy()
            score_cols = [
                f"Score_{name}"
                for name, _, _ in PRESET_LIST
                if f"Score_{name}" in timing_df.columns
            ]
            if score_cols:
                timing_df["_BestScore"] = timing_df[score_cols].max(axis=1)
                timing_df = timing_df.sort_values("_BestScore", ascending=False)
            st.caption(f"{len(timing_df)}종목")
            render_stock_grid(timing_df, f"timing_{timing_val}")
        else:
            st.info("타이밍 데이터 없음")


# ── 12개 전략 + 타이밍 3개 = 15개 탭 (한 줄) ──
all_labels = [short for _, _, short in PRESET_LIST]
_timing_defs = [
    ("STRONG_BUY", "🔴즉시"),
    ("BUY_NOW", "🟢매수"),
    ("WATCH", "🟡관심"),
]
all_labels += [lbl for _, lbl in _timing_defs]
all_tabs = st.tabs(all_labels)

# 12개 전략 탭
for tab, (name, _icon, _short) in zip(
    all_tabs[: len(PRESET_LIST)], PRESET_LIST, strict=False
):
    _render_preset_tab(tab, name)

# 3개 타이밍 탭
for tab, (timing_val, _) in zip(
    all_tabs[len(PRESET_LIST) :], _timing_defs, strict=False
):
    _render_timing_tab(tab, timing_val, timing_val)


# ══════════════════════════════════════════════════════
# 상세 팝업 (st.dialog)
# ══════════════════════════════════════════════════════
@st.dialog("종목 상세 분석", width="large")
def show_stock_detail(ticker: str):
    row = df.loc[ticker]
    name = row.get("Name", ticker)

    # ── 요약 메트릭 ──
    st.subheader(f"{name} ({ticker})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재가", f"{row.get('Close', 0):,.0f}원")
    m2.metric("PBR", f"{row.get('PBR', 0):.2f}")
    m3.metric("ROE", f"{(row.get('ROE', 0) or 0)*100:.1f}%")
    m4.metric("시가총액", f"{row.get('MarketCap', 0)/1e8:,.0f}억")

    m5, m6, m7 = st.columns(3)
    m5.metric("52주 저가", f"{row.get('Low52', 0):,.0f}")
    m6.metric("52주 고가", f"{row.get('High52', 0):,.0f}")
    div_val = row.get("DIV", 0) or 0
    m7.metric("배당률", f"{div_val:.1f}%")

    # 통과 전략 표시 (타이밍 뱃지 포함)
    passed = []
    for pname, picon, _ in PRESET_LIST:
        if row.get(f"Pass_{pname}", False):
            score = row.get(f"Score_{pname}", 0) or 0
            timing = row.get(f"Timing_{pname}", "")
            badge = TIMING_BADGE.get(timing, "")
            label = f"{picon}{pname}({score:.0f})"
            if badge:
                label += f" {badge}"
            passed.append(label)
    if passed:
        st.success("통과 전략: " + " · ".join(passed))

    # 발굴 사유
    for pname, _, _ in PRESET_LIST:
        reason = row.get(f"Reason_{pname}", "")
        if reason and reason != "-" and str(reason) != "nan":
            st.caption(f"발굴사유: {reason}")
            break

    # ── 상승 전망 분석 ──
    analysis = _generate_analysis(row, ticker)
    st.info(f"📈 **상승 전망**\n\n{analysis}")

    # ── 2년 차트 ──
    st.markdown("---")
    ohlcv_2y = fetcher._load_cache(f"ohlcv_{ticker}_2y", date_str)
    if ohlcv_2y is None:
        with st.spinner("2년 데이터 로딩..."):
            try:
                ohlcv_2y = fetcher.fetch_ohlcv_2year(ticker, date_str)
            except Exception:
                ohlcv_2y = None

    if ohlcv_2y is not None and not ohlcv_2y.empty:
        ohlcv_chart = ohlcv_2y.copy()
        close = ohlcv_chart["종가"].astype(float)
        high = ohlcv_chart["고가"].astype(float)
        low = ohlcv_chart["저가"].astype(float)
        opn = ohlcv_chart["시가"].astype(float)
        vol = ohlcv_chart["거래량"].astype(float)
        dates = ohlcv_chart.index

        rsi = indicators.calc_rsi(close, 14)
        macd_df = indicators.calc_macd(close, 12, 26, 9)
        ma20 = indicators.calc_ma(close, 20)
        ma60 = indicators.calc_ma(close, 60)

        # 찐 신호
        real_sigs = detect_real_signals(ohlcv_chart)

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.45, 0.12, 0.20, 0.20],
            vertical_spacing=0.02,
        )

        # 캔들
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=opn,
                high=high,
                low=low,
                close=close,
                name="OHLC",
                increasing_line_color="red",
                decreasing_line_color="blue",
            ),
            row=1,
            col=1,
        )

        # 이평선
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ma20,
                mode="lines",
                name="MA20",
                line={"color": "orange", "width": 1.5},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ma60,
                mode="lines",
                name="MA60",
                line={"color": "green", "width": 1},
            ),
            row=1,
            col=1,
        )

        # 찐 신호 마커 (금색 별)
        if real_sigs:
            fig.add_trace(
                go.Scatter(
                    x=[s["date"] for s in real_sigs],
                    y=[s["price"] for s in real_sigs],
                    mode="markers+text",
                    name="찐 신호",
                    marker={
                        "symbol": "star",
                        "size": 16,
                        "color": "gold",
                        "line": {"color": "darkorange", "width": 2},
                    },
                    text=[f"매수({s['count']})" for s in real_sigs],
                    textposition="top center",
                    textfont={"size": 9, "color": "darkorange"},
                ),
                row=1,
                col=1,
            )

        # 거래량
        colors = ["red" if c >= o else "blue" for c, o in zip(close, opn, strict=False)]
        fig.add_trace(
            go.Bar(x=dates, y=vol, name="거래량", marker_color=colors), row=2, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rsi,
                mode="lines",
                name="RSI(14)",
                line={"color": "purple", "width": 1.5},
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hrect(
            y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0, row=3, col=1
        )

        # MACD
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=macd_df["MACD"],
                mode="lines",
                name="MACD",
                line={"color": "blue", "width": 1.5},
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=macd_df["MACD_Signal"],
                mode="lines",
                name="Signal",
                line={"color": "red", "width": 1},
            ),
            row=4,
            col=1,
        )
        hist_colors = ["green" if h >= 0 else "red" for h in macd_df["MACD_Hist"]]
        fig.add_trace(
            go.Bar(
                x=dates, y=macd_df["MACD_Hist"], name="Hist", marker_color=hist_colors
            ),
            row=4,
            col=1,
        )
        fig.add_hline(y=0, line_color="gray", line_width=0.5, row=4, col=1)

        fig.update_layout(
            height=700,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            margin={"l": 40, "r": 20, "t": 20, "b": 20},
        )
        st.plotly_chart(fig, use_container_width=True)

        # 최근 찐 신호 요약
        if real_sigs and len(dates) >= 20:
            _sig_kr = {
                "MACD": "추세전환",
                "RSI": "과매도탈출",
                "Stoch": "바닥확인",
                "MA20": "평균돌파",
            }
            recent = [s for s in real_sigs if s["date"] >= dates[-60]]
            if recent:
                st.info(
                    "🔔 **복합 매수신호**: "
                    + ", ".join(
                        [
                            f"{s['date'].strftime('%m/%d')} "
                            f"({'+'.join(_sig_kr.get(x, x) for x in s['signals'])})"
                            for s in recent[-5:]
                        ]
                    )
                )
    else:
        st.caption("차트 데이터 없음")

    # ── 자산/수급 상세 ──
    st.markdown("---")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        cr = row.get("CashRatio", np.nan)
        if pd.notna(cr):
            st.metric("현금비율", f"{cr:.2f}")
        else:
            st.caption("현금비율: DART 없음")
    with ac2:
        rr = row.get("RealEstateRatio", np.nan)
        if pd.notna(rr):
            st.metric("부동산비율", f"{rr:.2f}")
        else:
            st.caption("부동산비율: DART 없음")
    with ac3:
        roe = row.get("ROE", np.nan)
        roa = row.get("ROA", np.nan)
        if pd.notna(roe):
            st.metric("ROE", f"{roe*100:.1f}%")
        if pd.notna(roa):
            st.metric("ROA", f"{roa*100:.1f}%")

    # 수급
    sc1, sc2 = st.columns(2)
    with sc1:
        fi_amt = row.get("Flow1M_FI_KRW", 0) or 0
        fi_days = int(row.get("NetBuyDays_FI_10d", 0) or 0)
        st.metric("외인 1M", f"{fi_amt/1e8:+,.1f}억")
        st.caption(f"10일 중 {fi_days}일 순매수")
    with sc2:
        inst_amt = row.get("Flow1M_INST_KRW", 0) or 0
        inst_days = int(row.get("NetBuyDays_INST_10d", 0) or 0)
        st.metric("기관 1M", f"{inst_amt/1e8:+,.1f}억")
        st.caption(f"10일 중 {inst_days}일 순매수")


# ── 상세 팝업 트리거 ──
if st.session_state.get("detail_trigger"):
    st.session_state["detail_trigger"] = False
    ticker = st.session_state.get("detail_ticker")
    if ticker and ticker in df.index:
        show_stock_detail(ticker)
