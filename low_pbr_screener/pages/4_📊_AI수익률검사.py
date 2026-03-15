"""Page 4: AI 수익률검사 — 롤링 백테스트로 AI급등주 모델 수익률 검증.

설정된 보유기간 기준 수익률 산출 + KOSPI/KOSDAQ 지수 비교.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest_engine import (  # noqa: E402
    HOLD_PERIODS,
    clear_gs_checkpoint,
    get_checkpoint_info,
    get_gs_checkpoint_info,
    has_checkpoint,
    has_gs_checkpoint,
    load_backtest_result,
    load_cached_ohlcv,
    optimize_trade_params,
    run_grid_search,
    run_rolling_backtest,
    save_backtest_result,
)
from core.config import CACHE_DIR  # noqa: E402
from core.styles import apply_mobile_styles  # noqa: E402

_PERIOD_COLORS = {
    3: "#E91E63",  # 3주 pink
    4: "#FF9800",  # 4주 orange
    6: "#2196F3",  # 6주 blue
    8: "#4CAF50",  # 8주 green
    10: "#9C27B0",  # 10주 purple
    12: "#795548",  # 12주 brown
}


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────


def _parse_dates(daily):
    """YYYYMMDD 문자열을 datetime으로 변환."""
    from datetime import datetime

    out = []
    for d in daily:
        raw = d["date"]
        try:
            out.append(datetime.strptime(str(raw), "%Y%m%d"))
        except (ValueError, TypeError):
            out.append(raw)
    return out


# ──────────────────────────────────────────────
# 차트 함수
# ──────────────────────────────────────────────


def _plot_cumulative_return_multi(summary, daily):
    """기간별 실투자 누적 수익률 곡선 + KODEX 200/코스닥150 비교."""
    from datetime import datetime

    import plotly.graph_objects as go

    sbp = summary.get("summary_by_period", {})
    initial_capital = 100_000_000

    fig = go.Figure()

    # ── 기간별 실투자 수익률 곡선 ──
    for months in HOLD_PERIODS:
        sp = sbp.get(months, {})
        wv = sp.get("realistic_weekly_values", [])
        if not wv:
            continue

        dates = []
        pct_vals = []
        for w in wv:
            d = w.get("date", "")
            if len(d) == 8:
                dates.append(datetime.strptime(d, "%Y%m%d"))
            tv = w.get("total_value", initial_capital)
            pct_vals.append((tv / initial_capital - 1) * 100)

        if dates:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=pct_vals,
                    mode="lines+markers",
                    name=f"{months}주 ({sp.get('n_slots', 0)}슬롯)",
                    line={"color": _PERIOD_COLORS.get(months, "#888"), "width": 2},
                    marker={"size": 3},
                )
            )

    # ── KODEX 200 (KOSPI) / KODEX 코스닥150 (KOSDAQ) ──
    parsed_dates = _parse_dates(daily)
    kospi_base, kosdaq_base = None, None
    kospi_dates, kospi_vals = [], []
    kosdaq_dates, kosdaq_vals = [], []

    for i, d in enumerate(daily):
        k_buy = d.get("kospi_buy")
        if k_buy:
            if kospi_base is None:
                kospi_base = k_buy
            kospi_dates.append(parsed_dates[i])
            kospi_vals.append((k_buy / kospi_base - 1) * 100)

        kd_buy = d.get("kosdaq_buy")
        if kd_buy:
            if kosdaq_base is None:
                kosdaq_base = kd_buy
            kosdaq_dates.append(parsed_dates[i])
            kosdaq_vals.append((kd_buy / kosdaq_base - 1) * 100)

    if kospi_vals:
        fig.add_trace(
            go.Scatter(
                x=kospi_dates,
                y=kospi_vals,
                mode="lines",
                name="KODEX 200",
                line={"color": "#888888", "width": 1.5, "dash": "dash"},
            )
        )
    if kosdaq_vals:
        fig.add_trace(
            go.Scatter(
                x=kosdaq_dates,
                y=kosdaq_vals,
                mode="lines",
                name="KODEX 코스닥150",
                line={"color": "#BBBBBB", "width": 1.5, "dash": "dot"},
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="지정일",
        yaxis_title="누적 수익률 (%)",
        height=450,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_weekly_returns(daily, selected_period=3):
    """선택 기간의 주별 수익률 바 차트."""
    import plotly.graph_objects as go

    dates = _parse_dates(daily)
    returns = []
    valid_dates = []
    for i, d in enumerate(daily):
        pr = d.get("portfolio_returns", {}).get(selected_period)
        if pr is not None:
            returns.append(pr * 100)
            valid_dates.append(dates[i])

    if not returns:
        st.info(f"{selected_period}주 수익률 데이터가 없습니다.")
        return

    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in returns]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=valid_dates,
            y=returns,
            marker_color=colors,
            name=f"{selected_period}주 수익률",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="지정일",
        yaxis_title=f"{selected_period}주 수익률 (%)",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_index_comparison(summary, daily, config_lh=4):
    """실투자 수익률 vs KODEX 200 / KODEX 코스닥150 비교.

    AI: 수익목표기간의 실투자 누적 수익률 (슬롯 모델).
    지수: 최초 지정일 기준 변동률(%). 호버 시 실제 지수 표시.
    """
    from datetime import datetime as _dt

    import plotly.graph_objects as go

    sbp = summary.get("summary_by_period", {})
    initial_capital = 100_000_000

    fig = go.Figure()

    # ── AI 실투자 수익률 (수익목표기간 기준) ──
    sp = sbp.get(config_lh, {})
    wv = sp.get("realistic_weekly_values", [])
    if wv:
        ai_dates, ai_vals = [], []
        for w in wv:
            d = w.get("date", "")
            if len(d) == 8:
                ai_dates.append(_dt.strptime(d, "%Y%m%d"))
            tv = w.get("total_value", initial_capital)
            ai_vals.append((tv / initial_capital - 1) * 100)
        fig.add_trace(
            go.Scatter(
                x=ai_dates,
                y=ai_vals,
                mode="lines",
                name=f"AI 분할매수({config_lh}주, {sp.get('n_slots', 0)}슬롯)",
                line={"color": "#2196F3", "width": 2.5},
                hovertemplate="분할: %{y:+.1f}%<extra></extra>",
            )
        )

    # ── AI 일괄매수 수익률 ──
    lump_wv = sp.get("lumpsum_weekly_values", [])
    if lump_wv:
        lump_dates, lump_vals = [], []
        for w in lump_wv:
            d = w.get("date", "")
            if len(d) == 8:
                lump_dates.append(_dt.strptime(d, "%Y%m%d"))
            tv = w.get("total_value", initial_capital)
            lump_vals.append((tv / initial_capital - 1) * 100)
        fig.add_trace(
            go.Scatter(
                x=lump_dates,
                y=lump_vals,
                mode="lines",
                name=f"AI 일괄매수({config_lh}주)",
                line={"color": "#4CAF50", "width": 2, "dash": "dash"},
                hovertemplate="일괄: %{y:+.1f}%<extra></extra>",
            )
        )

    # ── KODEX 200 / KODEX 코스닥150 ──
    parsed_dates = _parse_dates(daily)
    kospi_base, kosdaq_base = None, None
    kospi_vals, kospi_dates, kospi_actuals = [], [], []
    kosdaq_vals, kosdaq_dates, kosdaq_actuals = [], [], []

    for i, d in enumerate(daily):
        k_buy = d.get("kospi_buy")
        if k_buy:
            if kospi_base is None:
                kospi_base = k_buy
            kospi_vals.append((k_buy / kospi_base - 1) * 100)
            kospi_actuals.append(k_buy)
            kospi_dates.append(parsed_dates[i])

        kd_buy = d.get("kosdaq_buy")
        if kd_buy:
            if kosdaq_base is None:
                kosdaq_base = kd_buy
            kosdaq_vals.append((kd_buy / kosdaq_base - 1) * 100)
            kosdaq_actuals.append(kd_buy)
            kosdaq_dates.append(parsed_dates[i])

    if kospi_vals:
        hover_kospi = [
            f"KODEX 200: {chg:+.1f}% (지수 {act:,.0f})"
            for chg, act in zip(kospi_vals, kospi_actuals, strict=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=kospi_dates,
                y=kospi_vals,
                mode="lines",
                name=f"KODEX 200 (기준 {kospi_base:,.0f})"
                if kospi_base
                else "KODEX 200",
                line={"color": "#FF5722", "width": 1.5, "dash": "dot"},
                text=hover_kospi,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if kosdaq_vals:
        hover_kosdaq = [
            f"KODEX 코스닥150: {chg:+.1f}% (지수 {act:,.0f})"
            for chg, act in zip(kosdaq_vals, kosdaq_actuals, strict=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=kosdaq_dates,
                y=kosdaq_vals,
                mode="lines",
                name=f"KODEX 코스닥150 (기준 {kosdaq_base:,.0f})"
                if kosdaq_base
                else "KODEX 코스닥150",
                line={"color": "#9C27B0", "width": 1.5, "dash": "dot"},
                text=hover_kosdaq,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="지정일",
        yaxis_title="수익률 (%)",
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_auc_trend(daily):
    """모델 AUC 추이."""
    import plotly.graph_objects as go

    dates = _parse_dates(daily)
    aucs = [d.get("model_auc", 0.5) for d in daily]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=aucs,
            mode="lines+markers",
            name="AUC",
            line={"color": "#FF9800", "width": 2},
            marker={"size": 4},
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5)
    avg_auc = float(np.mean(aucs))
    fig.add_hline(
        y=avg_auc,
        line_dash="dot",
        line_color="blue",
        opacity=0.5,
        annotation_text=f"평균 {avg_auc:.3f}",
    )
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title="AUC",
        yaxis_range=[0.4, 0.9],
        height=300,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# 거래 내역 테이블
# ──────────────────────────────────────────────


def _show_trade_details(daily, selected_period=3, fund_info=None):
    """종목별 거래 내역 테이블 (선택 기간, 재무정보 포함)."""
    rows = []
    for d in daily:
        for s in d["stocks"]:
            rbp = s.get("returns_by_period", {}).get(selected_period)
            if rbp is None:
                continue
            _ticker = s["ticker"]
            _fi = (fund_info or {}).get(_ticker, {})
            rows.append(
                {
                    "지정일": d["date"],
                    "매수일": d["buy_date"],
                    "종목코드": _ticker,
                    "종목명": s.get("name", _ticker),
                    "시총(조)": _fi.get("시총(조)", ""),
                    "PBR": _fi.get("PBR", ""),
                    "PER": _fi.get("PER", ""),
                    "EPS": _fi.get("EPS", ""),
                    "ROE": _fi.get("ROE", ""),
                    "배당률": _fi.get("배당률", ""),
                    "영업이익률": _fi.get("영업이익률", ""),
                    "예측확률": f"{s['predicted_prob']:.2%}",
                    "매수가": f"{s['buy_price']:,.0f}",
                    "매도가": f"{rbp['sell_price']:,.0f}",
                    f"{selected_period}주수익률": f"{rbp['return_pct'] * 100:+.2f}%",
                    "매도사유": rbp["exit_reason"],
                    "보유일(거래일)": rbp["hold_days"],
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=min(400, len(rows) * 35 + 38),
        )
    else:
        st.info(f"{selected_period}주 거래 내역이 없습니다.")


# ──────────────────────────────────────────────
# 엑셀 다운로드 (3시트)
# ──────────────────────────────────────────────


def _download_section(res, fund_info=None):
    """Excel 다운로드 — 3개 시트."""
    daily = res.get("daily_results", [])
    summary = res.get("summary", {})
    config = res.get("config", {})
    if not daily:
        return

    _hold_periods = config.get("hold_periods", HOLD_PERIODS)

    # === Sheet 1: 종목별 거래내역 ===
    trade_rows = []
    for d in daily:
        for s in d["stocks"]:
            _ticker = s["ticker"]
            _fi = (fund_info or {}).get(_ticker, {})
            row = {
                "지정일": d["date"],
                "매수일": d["buy_date"],
                "종목코드": _ticker,
                "종목명": s.get("name", ""),
                "시총(조)": _fi.get("시총(조)", ""),
                "PBR": _fi.get("PBR", ""),
                "PER": _fi.get("PER", ""),
                "EPS": _fi.get("EPS", ""),
                "ROE": _fi.get("ROE", ""),
                "배당률": _fi.get("배당률", ""),
                "예측확률": s["predicted_prob"],
                "매수가": s["buy_price"],
                "AUC": d.get("model_auc", 0),
            }
            rbp = s.get("returns_by_period", {})
            for m in _hold_periods:
                p = rbp.get(m)
                if p:
                    row[f"{m}주매도가"] = p["sell_price"]
                    row[f"{m}주수익률"] = p["return_pct"]
                    row[f"{m}주매도사유"] = p["exit_reason"]
                    row[f"{m}주보유일(거래일)"] = p["hold_days"]
                else:
                    row[f"{m}주매도가"] = None
                    row[f"{m}주수익률"] = None
                    row[f"{m}주매도사유"] = None
                    row[f"{m}주보유일(거래일)"] = None
            trade_rows.append(row)
    df_trades = pd.DataFrame(trade_rows)

    # === Sheet 2: 주별 요약 ===
    weekly_rows = []
    for d in daily:
        row = {"지정일": d["date"], "매수일": d["buy_date"]}
        prs = d.get("portfolio_returns", {})
        for m in _hold_periods:
            row[f"{m}주수익률"] = prs.get(m)

        row["KOSPI매수일"] = d.get("kospi_buy")
        row["KOSDAQ매수일"] = d.get("kosdaq_buy")

        kospi_after = d.get("kospi_after", {})
        kosdaq_after = d.get("kosdaq_after", {})
        for m in _hold_periods:
            row[f"KOSPI_{m}주후"] = (
                kospi_after.get(m) if isinstance(kospi_after, dict) else None
            )
            row[f"KOSDAQ_{m}주후"] = (
                kosdaq_after.get(m) if isinstance(kosdaq_after, dict) else None
            )

        row["AUC"] = d.get("model_auc")
        row["학습데이터수"] = d.get("n_samples")
        row["양성수"] = d.get("n_positive")
        weekly_rows.append(row)
    df_weekly = pd.DataFrame(weekly_rows)

    # === Sheet 3: 누적 수익률 ===
    cum_rows = []
    cums = {m: 1.0 for m in _hold_periods}
    lump_cums = {m: 1.0 for m in _hold_periods}
    lump_idx = {m: 0 for m in _hold_periods}
    kospi_base_xl, kosdaq_base_xl = None, None
    for d in daily:
        row = {"지정일": d["date"]}
        prs = d.get("portfolio_returns", {})
        for m in _hold_periods:
            pr = prs.get(m)
            if pr is not None:
                cums[m] *= 1.0 + pr
            row[f"{m}주누적수익률"] = round((cums[m] - 1) * 100, 2)
            # 일괄매수: 매 m주마다만 수익률 적용
            if lump_idx[m] % m == 0 and pr is not None:
                lump_cums[m] *= 1.0 + pr
            row[f"{m}주일괄누적수익률"] = round((lump_cums[m] - 1) * 100, 2)
            lump_idx[m] += 1

        # KODEX 200 / KODEX 코스닥150: 최초일 대비 상승률
        k_buy = d.get("kospi_buy")
        if k_buy:
            if kospi_base_xl is None:
                kospi_base_xl = k_buy
            row["KODEX200_상승률"] = round((k_buy / kospi_base_xl - 1) * 100, 2)
        else:
            row["KODEX200_상승률"] = None

        kd_buy = d.get("kosdaq_buy")
        if kd_buy:
            if kosdaq_base_xl is None:
                kosdaq_base_xl = kd_buy
            row["KODEX코스닥150_상승률"] = round((kd_buy / kosdaq_base_xl - 1) * 100, 2)
        else:
            row["KODEX코스닥150_상승률"] = None

        # KOSPI/KOSDAQ 시점별 변동률 (개별 매수일 기준)
        kospi_buy = d.get("kospi_buy")
        kosdaq_buy = d.get("kosdaq_buy")
        kospi_after = d.get("kospi_after", {})
        kosdaq_after = d.get("kosdaq_after", {})
        for m in _hold_periods:
            ka = kospi_after.get(m) if isinstance(kospi_after, dict) else None
            if kospi_buy and ka:
                row[f"KOSPI_{m}주변동률"] = round((ka / kospi_buy - 1) * 100, 2)
            else:
                row[f"KOSPI_{m}주변동률"] = None
            kda = kosdaq_after.get(m) if isinstance(kosdaq_after, dict) else None
            if kosdaq_buy and kda:
                row[f"KOSDAQ_{m}주변동률"] = round((kda / kosdaq_buy - 1) * 100, 2)
            else:
                row[f"KOSDAQ_{m}주변동률"] = None

        cum_rows.append(row)
    df_cum = pd.DataFrame(cum_rows)

    # === Excel 파일 생성 ===
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_trades.to_excel(writer, sheet_name="종목별거래내역", index=False)
        df_weekly.to_excel(writer, sheet_name="주별요약", index=False)
        df_cum.to_excel(writer, sheet_name="누적수익률", index=False)
    buf.seek(0)

    # 조건 포함 파일명 생성
    _tw = config.get("train_window", "2Y")
    _lhw = config.get("label_horizon_weeks", 4)
    _nb = config.get("num_boost_round", 400)
    _a = config.get("min_profit", 0)
    _b = config.get("trailing_stop", 0)
    _c = config.get("stop_loss", 0)
    _topk = config.get("top_k", 20)
    _sd = config.get("start_date", "")
    _ed = config.get("end_date", "")
    _fname_base = (
        f"backtest_{_sd}_{_ed}_"
        f"win{_tw}_goal{_lhw}w_boost{_nb}_"
        f"A{_a * 100:.0f}_B{_b * 100:.0f}_C{_c * 100:.0f}_top{_topk}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 백테스트 결과 (Excel)",
            buf.getvalue(),
            f"{_fname_base}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col2:
        # 요약 텍스트
        sbp = summary.get("summary_by_period", {})
        lines = [f"[백테스트 요약] {_sd} ~ {_ed}"]
        lines.append(f"학습윈도우: {_tw}")
        lines.append(f"수익목표기간: {_lhw}주 ({_lhw * 5}영업일)")
        lines.append(f"Boost Rounds: {_nb}")
        lines.append(f"Top K: {_topk}")
        lines.append(f"최소수익률(A): {_a * 100:.0f}%")
        lines.append(f"트레일링스탑(B): {_b * 100:.0f}%")
        lines.append(f"손절(C): {_c * 100:.0f}%")
        _bpt = config.get("buy_price_type", "close")
        lines.append(f"매수가기준: {'시가' if _bpt == 'open' else '종가'}")
        _ltp = config.get("label_target_pct", 0.20)
        _lsp = config.get("label_stop_pct", 0.08)
        lines.append(f"학습목표수익률(target_pct): {_ltp * 100:.0f}%")
        lines.append(f"학습손절기준(stop_pct): {_lsp * 100:.0f}%")
        lines.append("")
        lines.append("[기간별 누적 수익률 (초기 1억, 분할매수 슬롯 순환)]")
        for m in _hold_periods:
            sp = sbp.get(m, {})
            real_r = sp.get("realistic_cum_return", 1.0)
            wr = sp.get("win_rate", 0)
            n_slots = sp.get("n_slots", 0)
            nw = sp.get("n_weeks", 0)
            _wavg = sp.get("avg_return", 0)
            lines.append(
                f"{m:2d}주: {(real_r - 1) * 100:+.1f}% | "
                f"주간평균 {_wavg * 100:+.2f}% | "
                f"승률 {wr * 100:.1f}% | {n_slots}슬롯 | {nw}주"
            )
        lines.append("")
        lines.append("[기간별 누적 수익률 (초기 1억, 일괄매수)]")
        for m in _hold_periods:
            sp = sbp.get(m, {})
            lump_r = sp.get("lumpsum_cum_return", 1.0)
            nw = sp.get("n_weeks", 0)
            lines.append(f"{m:2d}주: {(lump_r - 1) * 100:+.1f}% | {nw}주")

        # KODEX 200 / KODEX 코스닥150 비교
        _k_base, _kd_base = None, None
        _k_last, _kd_last = None, None
        for d in daily:
            kb = d.get("kospi_buy")
            if kb:
                if _k_base is None:
                    _k_base = kb
                _k_last = kb
            kdb = d.get("kosdaq_buy")
            if kdb:
                if _kd_base is None:
                    _kd_base = kdb
                _kd_last = kdb
        lines.append("")
        lines.append("[지수 비교 (최초 → 최종 지정일)]")
        if _k_base and _k_last:
            lines.append(
                f"KODEX 200:     {_k_base:.0f} → {_k_last:.0f} "
                f"({(_k_last / _k_base - 1) * 100:+.1f}%)"
            )
        if _kd_base and _kd_last:
            lines.append(
                f"KODEX 코스닥150: {_kd_base:.0f} → {_kd_last:.0f} "
                f"({(_kd_last / _kd_base - 1) * 100:+.1f}%)"
            )
        st.download_button(
            "📥 요약 TXT",
            "\n".join(lines),
            f"{_fname_base}_summary.txt",
            "text/plain",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# 결과 렌더링
# ──────────────────────────────────────────────


def _render_backtest_results(res):
    """백테스트 결과를 차트/표/통계로 렌더링 (설정 기간 기준)."""
    summary = res.get("summary", {})
    daily = res.get("daily_results", [])

    if not daily:
        st.warning("결과 데이터가 비어있습니다.")
        return

    st.divider()

    config = res.get("config", {})
    _goal_w = config.get("label_horizon_weeks", 4)
    sbp = summary.get("summary_by_period", {})
    _goal_sp = sbp.get(_goal_w, {})

    # ── 누적 수익률 상세 통계 (설정 기간 기준, 기본 표시) ──
    st.subheader(f"📈 {_goal_w}주 기준 수익률 상세 통계")

    # 분할매수(슬롯) 수익률
    st.caption("원금 매주분할 매수 (N슬롯 순환투자)")
    c1, c2, c3, c4, c5 = st.columns(5)
    real_ret = _goal_sp.get("realistic_cum_return", 1.0)
    n_slots = _goal_sp.get("n_slots", 0)
    c1.metric(
        f"분할투자 수익률({_goal_w}주)",
        f"{(real_ret - 1) * 100:+.1f}%",
        delta=f"{n_slots}슬롯 순환",
    )
    _weekly_avg = _goal_sp.get("avg_return", 0)
    c2.metric("평균 주간수익률", f"{_weekly_avg * 100:+.2f}%")
    c3.metric("승률", f"{_goal_sp.get('win_rate', 0) * 100:.1f}%")
    c4.metric("최대 낙폭", f"{_goal_sp.get('max_drawdown', 0) * 100:.1f}%")
    c5.metric("샤프 비율", f"{_goal_sp.get('sharpe_ratio', 0):.2f}")

    # 일괄매수 수익률
    lump_ret = _goal_sp.get("lumpsum_cum_return", 1.0)
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric(
        f"일괄투자 수익률({_goal_w}주)",
        f"{(lump_ret - 1) * 100:+.1f}%",
        delta="전액매수→매도→재투자",
    )
    lc2.metric("총 거래주", f"{_goal_sp.get('n_weeks', 0)}주")
    lc3.metric("평균 수익(양)", f"{_goal_sp.get('avg_win', 0) * 100:+.2f}%")
    lc4.metric("평균 손실(음)", f"{_goal_sp.get('avg_loss', 0) * 100:.2f}%")

    exit_stats = _goal_sp.get("exit_stats", {})
    if exit_stats:
        exit_rows = []
        for reason, data in exit_stats.items():
            exit_rows.append(
                {
                    "매도사유": reason,
                    "건수": data["count"],
                    "평균수익률": f"{data['avg_return'] * 100:+.2f}%",
                }
            )
        st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

    # ── 서브기간 통계 (최근 1년, 6개월, 3개월) ──
    _sub_periods = summary.get("summary_by_subperiod", {})
    if _sub_periods:
        st.subheader("📊 기간별 성과 비교")
        _sub_rows = []
        # 전체 기간
        _sub_rows.append(
            {
                "기간": "전체 (2년)",
                "분할투자수익률": f"{(real_ret - 1) * 100:+.1f}%",
                "일괄투자수익률": f"{(lump_ret - 1) * 100:+.1f}%",
                "평균 주간수익률": f"{_weekly_avg * 100:+.2f}%",
                "승률": f"{_goal_sp.get('win_rate', 0) * 100:.1f}%",
                "샤프비율": f"{_goal_sp.get('sharpe_ratio', 0):.2f}",
                "최대낙폭": f"{_goal_sp.get('max_drawdown', 0) * 100:.1f}%",
                "거래주": _goal_sp.get("n_weeks", 0),
            }
        )
        for _sl, _sl_name in [
            ("1Y", "최근 1년"),
            ("6M", "최근 6개월"),
            ("3M", "최근 3개월"),
            ("1M", "최근 1개월"),
        ]:
            _sp_data = _sub_periods.get(_sl, {}).get(_goal_w, {})
            if _sp_data and _sp_data.get("n_weeks", 0) > 0:
                _sp_real = _sp_data.get("realistic_cum_return", 1.0)
                _sp_lump = _sp_data.get("lumpsum_cum_return", 1.0)
                _sp_wavg = _sp_data.get("avg_return", 0)
                _sub_rows.append(
                    {
                        "기간": _sl_name,
                        "분할투자수익률": f"{(_sp_real - 1) * 100:+.1f}%",
                        "일괄투자수익률": f"{(_sp_lump - 1) * 100:+.1f}%",
                        "평균 주간수익률": f"{_sp_wavg * 100:+.2f}%",
                        "승률": f"{_sp_data.get('win_rate', 0) * 100:.1f}%",
                        "샤프비율": f"{_sp_data.get('sharpe_ratio', 0):.2f}",
                        "최대낙폭": f"{_sp_data.get('max_drawdown', 0) * 100:.1f}%",
                        "거래주": _sp_data.get("n_weeks", 0),
                    }
                )
        if len(_sub_rows) > 1:
            st.dataframe(
                pd.DataFrame(_sub_rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── 차트: AI 실투자 vs 지수 비교 ──
    st.subheader("📊 AI 실투자 vs 지수 비교")
    _plot_index_comparison(summary, daily, config_lh=_goal_w)

    # ── 차트: 주별 수익률 ──
    st.subheader("📊 주별 포트폴리오 수익률")
    _plot_weekly_returns(daily, _goal_w)

    # ── AUC 추이 ──
    st.subheader("🎯 모델 AUC 추이")
    _plot_auc_trend(daily)

    # ── 거래 내역 ──
    st.subheader("📋 종목별 거래 내역")
    _show_trade_details(daily, _goal_w, fund_info=_build_fund_info(universe_df))

    # ── 다운로드 ──
    st.subheader("📥 다운로드")
    _download_section(res, fund_info=_build_fund_info(universe_df))


# ──────────────────────────────────────────────
# 메인 UI
# ──────────────────────────────────────────────

st.set_page_config(page_title="AI 수익률검사", page_icon="📊", layout="wide")
apply_mobile_styles()

st.title("📊 AI 수익률검사")

# ── 학습모델 선택 ──
_model_choice = st.radio(
    "학습모델",
    ["AI급등주", "AI추천ETF"],
    horizontal=True,
    help="AI급등주: 시총 1000억 이상 주식 | AI추천ETF: 국내 ETF 전종목",
)
_is_etf_mode = _model_choice == "AI추천ETF"
_asset_type = "ETF" if _is_etf_mode else "stock"
_model_desc = "AI추천ETF 모델" if _is_etf_mode else "AI급등주 모델"
st.caption(f"{_model_desc}의 과거 실제 수익률을 검증합니다 (주별 롤링)")

# ── 데이터 확인 ──
_REQUIRED_FILES = [
    "all_tickers.pkl",
    "all_ohlcv.pkl",
    "all_fundamentals.pkl",
    "all_market_cap.pkl",
]

date_str = st.session_state.get("date_str")
if not date_str:
    try:
        _cached = sorted(
            [
                d.name
                for d in CACHE_DIR.iterdir()
                if d.is_dir() and d.name.isdigit() and len(d.name) == 8
            ],
            reverse=True,
        )
    except FileNotFoundError:
        _cached = []
    for _cd in _cached:
        if all((CACHE_DIR / _cd / f).exists() for f in _REQUIRED_FILES):
            date_str = _cd
            st.session_state["date_str"] = date_str
            break
    if not date_str:
        st.warning("수집된 데이터가 없습니다. **데이터 수집** 페이지에서 수집하세요.")
        st.stop()

cache_dir = CACHE_DIR / date_str
missing = [f for f in _REQUIRED_FILES if not (cache_dir / f).exists()]
if missing:
    st.warning(f"필수 데이터 없음: {', '.join(missing)}")
    st.stop()

st.caption(f"기준일: **{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}**")


@st.cache_data(ttl=3600)
def _load_universe(ds):
    import pickle

    cd = CACHE_DIR / ds
    with open(cd / "all_tickers.pkl", "rb") as f:
        tickers_df = pickle.load(f)
    with open(cd / "all_market_cap.pkl", "rb") as f:
        cap_df = pickle.load(f)
    fund_path = cd / "all_fundamentals.pkl"
    fund_df = (
        pickle.load(open(fund_path, "rb")) if fund_path.exists() else pd.DataFrame()
    )

    universe = tickers_df.set_index("Ticker")
    if not cap_df.empty and "시가총액" in cap_df.columns:
        universe = universe.join(cap_df[["시가총액"]], how="left")
        universe.rename(columns={"시가총액": "MarketCap"}, inplace=True)
    if "MarketCap" not in universe.columns:
        universe["MarketCap"] = 0
    if not fund_df.empty:
        _fund_cols = [c for c in ["PBR", "PER", "DIV"] if c in fund_df.columns]
        if _fund_cols:
            universe = universe.join(fund_df[_fund_cols], how="left")
    if "PBR" in universe.columns and "PER" in universe.columns:
        _pbr = universe["PBR"].astype(float)
        _per = universe["PER"].astype(float)
        universe["ROE"] = np.where(
            (_per > 0) & (_pbr > 0), (_pbr / _per * 100).round(2), np.nan
        )
    return universe


def _build_fund_info(udf):
    """universe_df에서 종목별 재무정보 dict 생성 (거래내역용)."""
    info = {}
    for t in udf.index:
        _mcap = udf.at[t, "MarketCap"] if "MarketCap" in udf.columns else 0
        _pbr = udf.at[t, "PBR"] if "PBR" in udf.columns else np.nan
        _per = udf.at[t, "PER"] if "PER" in udf.columns else np.nan
        _eps = udf.at[t, "EPS"] if "EPS" in udf.columns else np.nan
        _roe = udf.at[t, "ROE"] if "ROE" in udf.columns else np.nan
        _div = udf.at[t, "DIV"] if "DIV" in udf.columns else np.nan
        info[t] = {
            "시총(조)": round(float(_mcap or 0) / 1e12, 2),
            "PBR": round(float(_pbr), 2) if pd.notna(_pbr) else "",
            "PER": round(float(_per), 2) if pd.notna(_per) else "",
            "EPS": round(float(_eps)) if pd.notna(_eps) else "",
            "ROE": round(float(_roe), 2) if pd.notna(_roe) else "",
            "배당률": round(float(_div), 2) if pd.notna(_div) else "",
            "영업이익률": "",
        }
    return info


@st.cache_data(ttl=3600)
def _load_etf_universe(ds):
    """ETF 전종목 유니버스 로드 (pykrx)."""
    from core.data_fetcher import fetch_etf_tickers as _fetch_etf_tk

    etf_df = _fetch_etf_tk(ds)
    if etf_df is None or etf_df.empty:
        return pd.DataFrame(columns=["Name", "Market"])
    etf_df = etf_df.set_index("Ticker")
    etf_df["MarketCap"] = 0  # ETF는 시총 필터 사용 안 함
    return etf_df


if _is_etf_mode:
    universe_df = _load_etf_universe(date_str)
    st.info(f"전체 ETF: **{len(universe_df)}**개")
else:
    universe_df = _load_universe(date_str)
    st.info(f"전체 종목: **{len(universe_df)}**개")

# ── 설정 패널 ──
with st.expander("설정", expanded=True):
    col1, col2, col3, col_lbl2, col_lbl3 = st.columns(5)
    with col1:
        train_window = st.selectbox(
            "학습 윈도우",
            [2, 1, 3, 4],
            format_func=lambda x: f"{x}년",
            help="학습에 사용할 과거 데이터 기간",
        )
    with col2:
        top_k = st.number_input("Top K 종목", min_value=5, max_value=50, value=20)
    with col3:
        label_horizon_weeks = st.selectbox(
            "수익목표기간 (horizon)",
            [3, 4, 6, 8, 10, 12],
            index=1,  # 기본 4주
            format_func=lambda w: f"{w}주 ({w * 5}영업일)",
            help="라벨링 horizon: 이 기간 내 목표수익 달성 여부로 학습",
        )
    with col_lbl2:
        _default_target = 10 if _is_etf_mode else 30
        label_target = st.number_input(
            "학습목표수익률 (target_pct) %",
            min_value=5,
            max_value=50,
            value=_default_target,
            help="이 수익률 이상 달성 시 '급등' 라벨 부여",
        )
    with col_lbl3:
        from core.ml_pattern import compute_recommended_stop

        _etf_mode_str = "etf" if _is_etf_mode else "stock"
        _stop_rec_bt = compute_recommended_stop(
            label_horizon_weeks, label_target, mode=_etf_mode_str
        )
        # horizon / target 변경 시 stop 자동갱신
        _prev_hw_bt = st.session_state.get("_bt_prev_hw", label_horizon_weeks)
        _prev_tp_bt = st.session_state.get("_bt_prev_tp", label_target)
        if label_horizon_weeks != _prev_hw_bt or label_target != _prev_tp_bt:
            st.session_state["_bt_stop_input"] = _stop_rec_bt
        st.session_state["_bt_prev_hw"] = label_horizon_weeks
        st.session_state["_bt_prev_tp"] = label_target
        _bt_stop_kw = {
            "label": f"학습손절기준 (stop_pct) % — 추천: {_stop_rec_bt}%",
            "min_value": 3 if _is_etf_mode else 5,
            "max_value": 30,
            "key": "_bt_stop_input",
            "help": "이 비율 이상 하락 시 'Hard Negative'로 강한 부정 라벨 (1.5배 가중)",
        }
        if "_bt_stop_input" not in st.session_state:
            _bt_stop_kw["value"] = _stop_rec_bt
        label_stop = st.number_input(**_bt_stop_kw)

    col4, col5 = st.columns(2)
    with col4:
        num_boost = st.slider(
            "모델강화 (Boost Rounds)",
            300,
            800,
            400,
            step=50,
            help="높을수록 정확도 향상, 실행시간 증가",
        )
    with col5:
        st.caption("300: 빠른실행 | 400: 균형(권장) | 800: 최고정확도")

    col6, col7, col8, col9 = st.columns(4)
    with col6:
        _default_a = 10 if _is_etf_mode else 30
        min_profit = st.number_input(
            "A: 최소수익률 (%)",
            min_value=0,
            max_value=50,
            value=_default_a,
            help="트레일링 발동 조건: 매수가 대비 최소 A% 이상 상승 경험",
        )
    with col7:
        trailing_stop = st.number_input(
            "B: 고점대비 하락 (%)",
            min_value=3,
            max_value=30,
            value=5,
            help="트레일링 스탑: 고점 대비 B% 하락 시 매도",
        )
    with col8:
        stop_loss = st.number_input(
            "손절 C (%)",
            min_value=5,
            max_value=30,
            value=10,
            help="매수가 대비 해당% 이상 하락 시 손절",
        )
    with col9:
        buy_price_type = st.selectbox(
            "매수가 기준",
            ["시가", "종가"],
            help="지정일 다음 거래일의 시가 또는 종가로 매수",
        )

    _buy_type_label = "시가" if buy_price_type == "시가" else "종가"
    st.caption(
        f"학습윈도우 {train_window}년 | 수익목표기간 {label_horizon_weeks}주({label_horizon_weeks * 5}일) | "
        f"목표수익률 {label_target}% | 손절기준 {label_stop}% | "
        f"Boost {num_boost} | Top {top_k}종목 | 매수가: {_buy_type_label} | "
        f"매도: A={min_profit}%이상→고점대비 B={trailing_stop}%하락, "
        f"손절 C={stop_loss}% | 보유기간: {label_horizon_weeks}주"
    )

# ── 현재 학습 목표 파라미터 표시 ──
st.info(
    f"학습 목표: **목표수익률 {label_target}%** | "
    f"**목표기간 {label_horizon_weeks}주 ({label_horizon_weeks * 5}영업일)** | "
    f"**손절 {label_stop}%**"
)

# ── 캐시된 결과 ──
cached_result = load_backtest_result()
if cached_result:
    cfg = cached_result.get("config", {})
    st.success(
        f"저장된 결과 감지: {cfg.get('start_date', '?')} ~ {cfg.get('end_date', '?')} | "
        f"Top {cfg.get('top_k', '?')} | {cached_result['summary'].get('total_weeks', 0)}주"
    )

# ── 체크포인트 확인 ──
_has_cp = has_checkpoint(date_str)
if _has_cp:
    _cp_info = get_checkpoint_info(date_str)
    if _cp_info:
        st.warning(
            f"중단된 백테스트 발견: "
            f"{_cp_info['completed_weeks']}/{_cp_info['total_weeks']}주 완료 | "
            f"누적 {(_cp_info['cum_return'] - 1) * 100:+.1f}% | "
            f"저장 시각: {_cp_info['saved_at'][:19]}"
        )

# ── 실행 버튼 ──
col_run, col_resume, col_load = st.columns(3)

with col_run:
    run_btn = st.button("수익률검사 실행", type="primary", use_container_width=True)
with col_resume:
    resume_btn = st.button(
        "이어서 실행",
        use_container_width=True,
        disabled=not _has_cp,
    )
with col_load:
    load_btn = st.button(
        "저장된 결과 보기",
        use_container_width=True,
        disabled=cached_result is None,
    )

_do_run = run_btn or resume_btn
_is_resume = resume_btn

if _do_run:
    import time as _time

    _run_start = _time.time()
    _mode_tag = f"[{_model_choice}] " if _is_etf_mode else ""
    _status_label = (
        "체크포인트에서 이어서 실행 중..."
        if _is_resume
        else f"{_mode_tag}수익률검사 실행 중... (보유기간: {label_horizon_weeks}주)"
    )

    with st.status(_status_label, expanded=True) as run_status:
        progress_bar = st.progress(0)
        log_container = st.empty()
        log_lines = []

        def _progress_cb(msg, pct=None):
            if pct is not None:
                progress_bar.progress(min(pct, 100))
            log_lines.append(msg)
            log_container.text("\n".join(log_lines[-50:]))

        try:
            _buy_type_param = "open" if buy_price_type == "시가" else "close"
            bt_result = run_rolling_backtest(
                universe_df=universe_df,
                date_str=date_str,
                progress_callback=_progress_cb,
                train_window_years=train_window,
                top_k=top_k,
                trailing_stop_pct=trailing_stop / 100.0,
                stop_loss_pct=stop_loss / 100.0,
                min_profit_pct=min_profit / 100.0,
                resume=_is_resume,
                num_boost_round=num_boost,
                label_horizon_weeks=label_horizon_weeks,
                buy_price_type=_buy_type_param,
                label_target_pct=label_target / 100.0,
                label_stop_pct=label_stop / 100.0,
                target_hold_periods=[label_horizon_weeks],
                asset_type=_asset_type,
            )

            save_path = save_backtest_result(bt_result, date_str)
            progress_bar.progress(100)
            n_weeks = len(bt_result.get("daily_results", []))
            total_min = (_time.time() - _run_start) / 60

            # 기간별 누적수익률 로그 (실투자 기준)
            sbp = bt_result.get("summary", {}).get("summary_by_period", {})
            period_log = " | ".join(
                f"{m}주: {(sbp.get(m, {}).get('realistic_cum_return', 1) - 1) * 100:+.1f}%"
                for m in HOLD_PERIODS
            )
            log_lines.append(
                f"{'=' * 50}\n"
                f"완료! {n_weeks}주 | {period_log} | "
                f"총 소요시간 {total_min:.1f}분\n"
                f"저장: {save_path.name}"
            )
            log_container.text("\n".join(log_lines[-50:]))
            st.session_state["backtest_result"] = bt_result

            _lhw_done = bt_result.get("config", {}).get("label_horizon_weeks", 4)
            _cum_goal = sbp.get(_lhw_done, {}).get("realistic_cum_return", 1.0)
            run_status.update(
                label=(
                    f"수익률검사 완료! ({n_weeks}주, "
                    f"{total_min:.1f}분, "
                    f"{_lhw_done}주누적 {(_cum_goal - 1) * 100:+.1f}%)"
                ),
                state="complete",
            )
        except Exception as e:
            total_min = (_time.time() - _run_start) / 60
            run_status.update(
                label=f"오류 ({total_min:.1f}분 경과): {e}", state="error"
            )
            import traceback

            st.code(traceback.format_exc())

elif load_btn and cached_result:
    st.session_state["backtest_result"] = cached_result

# ── 결과 표시 ──
result = st.session_state.get("backtest_result")
if result is None and cached_result:
    result = cached_result

if result:
    _render_backtest_results(result)

    # ── 최적 매도조건(A,B,C) 탐색 ──
    st.markdown("---")
    st.subheader("최적 매도조건(A,B,C) 탐색")
    st.caption(
        "이미 완료된 백테스트의 종목 선정 결과를 재사용하여, "
        "매도조건(A,B,C)만 변경한 시뮬레이션을 수행합니다. (학습 불필요, 수초 소요)"
    )

    # 설정의 수익목표기간(label_horizon_weeks)에 연동
    _cfg = result.get("config", {}) if result else {}
    opt_period = _cfg.get("label_horizon_weeks", 4)
    st.info(f"최적화 대상 기간: **{opt_period}주** (수익목표기간 설정에 연동)")
    opt_btn = st.button("최적 조건 탐색", type="secondary", use_container_width=True)

    if opt_btn:
        _opt_asset = _cfg.get("asset_type", "stock")
        ohlcv_data = load_cached_ohlcv(date_str, asset_type=_opt_asset)
        if ohlcv_data is None:
            st.error(
                "캐시된 OHLCV 데이터를 찾을 수 없습니다. 먼저 수익률검사를 실행해주세요."
            )
        else:
            opt_progress = st.progress(0)
            opt_log = st.empty()

            def _opt_cb(msg, pct=None):
                if pct is not None:
                    opt_progress.progress(min(pct, 100))
                opt_log.text(msg)

            with st.spinner(
                "A,B,C 그리드 서치 중... (1,386개 조합: " "A=20~40%, B=5~10%, C=10~20%)"
            ):
                opt_result = optimize_trade_params(
                    bt_result=result,
                    ohlcv_dict=ohlcv_data,
                    target_period=opt_period,
                    progress_callback=_opt_cb,
                    date_str=date_str,
                )

            opt_progress.progress(100)
            opt_log.empty()

            best = opt_result.get("best")
            if best:
                st.success(
                    f"**최적 조합(전체)**: A={best['A%']}% B={best['B%']}% C={best['C%']}% | "
                    f"분할 **{best['cum_pct']:+.2f}%** | "
                    f"일괄 **{best.get('lumpsum_pct', 0):+.2f}%** | "
                    f"승률 {best['win_rate']:.1f}% | 샤프 {best['sharpe']:.2f}"
                )

                # 기간별 최적조합 테이블
                _bbp = opt_result.get("best_by_period", {})
                if _bbp:
                    _bbp_rows = []
                    for _bl, _bn in [
                        ("전체", "전체"),
                        ("1Y", "최근 1년"),
                        ("6M", "최근 6개월"),
                        ("3M", "최근 3개월"),
                        ("1M", "최근 1개월"),
                    ]:
                        _bd = _bbp.get(_bl)
                        if _bd:
                            _ar_key = (
                                f"avg_return_{_bl}" if _bl != "전체" else "avg_return"
                            )
                            _bbp_rows.append(
                                {
                                    "기간": _bn,
                                    "A%": _bd["A%"],
                                    "B%": _bd["B%"],
                                    "C%": _bd["C%"],
                                    "주간평균%": _bd.get(
                                        _ar_key, _bd.get("avg_return", 0)
                                    ),
                                    "분할수익률%": _bd.get("cum_pct", 0),
                                    "승률%": _bd.get("win_rate", 0),
                                    "샤프": _bd.get("sharpe", 0),
                                }
                            )
                    if _bbp_rows:
                        st.caption("기간별 최적 A,B,C 조합")
                        st.dataframe(
                            pd.DataFrame(_bbp_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                # 상위 20개 결과 테이블
                grid_df = pd.DataFrame(opt_result["grid_results"][:20])
                st.dataframe(
                    grid_df,
                    column_config={
                        "A%": st.column_config.NumberColumn("A%"),
                        "B%": st.column_config.NumberColumn("B%"),
                        "C%": st.column_config.NumberColumn("C%"),
                        "cum_pct": st.column_config.NumberColumn(
                            "분할수익률%", format="%.2f"
                        ),
                        "lumpsum_pct": st.column_config.NumberColumn(
                            "일괄수익률%", format="%.2f"
                        ),
                        "avg_return": st.column_config.NumberColumn(
                            "주간평균%", format="%.2f"
                        ),
                        "win_rate": st.column_config.NumberColumn(
                            "승률%", format="%.1f"
                        ),
                        "sharpe": st.column_config.NumberColumn("샤프", format="%.2f"),
                        "n_trades": st.column_config.NumberColumn("거래수"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # 조건 포함 파일명
                _cfg = result.get("config", {})
                _tw = _cfg.get("train_window", "2Y")
                _lhw_cfg = _cfg.get(
                    "label_horizon_weeks",
                    _cfg.get("label_horizon_months", 4) * 4,
                )
                _nb = _cfg.get("num_boost_round", 400)
                _opt_fname = (
                    f"optimal_ABC_{date_str}_{opt_period}M_"
                    f"win{_tw}_goal{_lhw_cfg}w_boost{_nb}_"
                    f"best_A{best['A%']}_B{best['B%']}_C{best['C%']}"
                )

                # txt 다운로드 + 설정 저장
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "결과 다운로드 (txt)",
                        data=opt_result["settings_text"],
                        file_name=f"{_opt_fname}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with col_dl2:
                    # JSON 설정 저장
                    import json

                    _opt_settings_path = CACHE_DIR / "backtest" / f"{_opt_fname}.json"
                    _opt_settings_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(_opt_settings_path, "w") as f:
                        json.dump(best, f, indent=2, ensure_ascii=False)
                    st.info(f"최적 설정 저장됨: {_opt_settings_path.name}")
            else:
                st.warning("최적 조합을 찾을 수 없습니다.")

    # ── 설정조합별 수익률검사 (Grid Search) ──
    st.markdown("---")
    st.subheader("설정조합별 수익률검사")
    st.caption(
        "선택한 파라미터 값들의 모든 조합에 대해 수익률을 검사합니다. "
        "라벨링 파라미터가 같으면 모델 학습을 재사용하므로, "
        "매매 조건 조합은 빠르게 처리됩니다."
    )

    # Grid Search 파라미터 옵션 및 기본값 정의
    _GS_OPTS = {
        "gs_horizon": ([3, 4, 6, 8, 10, 12], [4]),
        "gs_target": ([20, 25, 30, 40, 50], [30]),
        "gs_stop": ([8, 10, 12, 15, 20], [10]),
        "gs_topk": (
            [5, 10, 15, 20, "5(>98)", "5(>97)", "5(>96)", "5(>95)", "5(>90)"],
            [20],
        ),
        "gs_buy": (["시가", "종가"], ["시가"]),
        "gs_invest": (["분할매수", "일괄매수"], ["분할매수"]),
        "gs_a": ([20, 25, 30, 40, 50], [30]),
        "gs_b": ([5, 7, 10, 12, 15], [5]),
        "gs_c": ([8, 10, 12, 15, 20], [10]),
    }
    # 기간연동 stop_pct: compute_recommended_stop() 사용 (2D 매핑)
    from core.ml_pattern import compute_recommended_stop as _gs_rec_stop

    def _parse_topk_option(opt):
        """TopK 옵션 파싱: int → int, 'K(>TH)' 문자열 → (K, TH/100)."""
        if isinstance(opt, int):
            return opt
        if isinstance(opt, str):
            import re

            m = re.match(r"(\d+)\(>(\d+)\)", opt)
            if m:
                return (int(m.group(1)), int(m.group(2)) / 100.0)
        return opt

    def _gs_toggle_all():
        """모두 선택 체크박스 콜백: session_state 직접 업데이트."""
        checked = st.session_state.get("gs_select_all", False)
        for k, (opts, defs) in _GS_OPTS.items():
            st.session_state[k] = list(opts) if checked else list(defs)

    def _gs_multi(label, key):
        """multiselect 헬퍼: session_state에 키가 있으면 default 생략."""
        opts, defs = _GS_OPTS[key]
        kw = {"label": label, "options": opts, "key": key}
        if key not in st.session_state:
            kw["default"] = defs
        return st.multiselect(**kw)

    with st.expander("파라미터 선택", expanded=False):
        # 모델 선택 (AI급등주, AI추천ETF 하나 또는 모두)
        _gs_model_sel = st.multiselect(
            "모델 선택",
            ["AI급등주", "AI추천ETF"],
            default=["AI급등주"],
            key="gs_model",
            help="하나 또는 모두 선택 가능. 각 모델별로 그리드 서치 실행",
        )
        st.checkbox("**모두 선택**", key="gs_select_all", on_change=_gs_toggle_all)

        st.markdown("##### 학습 파라미터 (라벨링)")
        _gs_c1, _gs_c2, _gs_c3 = st.columns(3)
        with _gs_c1:
            _hw_sel = _gs_multi("수익목표기간 (주)", "gs_horizon")
        with _gs_c2:
            _tp_sel = _gs_multi("목표수익률 (%)", "gs_target")
        with _gs_c3:
            _stop_linked = st.checkbox(
                "기간연동",
                value=True,
                key="gs_stop_linked",
                help="수익목표기간별 추천 손절기준을 자동 적용",
            )
            if _stop_linked:
                # 선택된 (horizon, target)별 추천값 표시
                _linked_pairs = [
                    f"({hw}주,{tp}%)→{_gs_rec_stop(hw, tp)}%"
                    for hw in sorted(_hw_sel)
                    for tp in sorted(_tp_sel)
                ]
                _sp_sel = []  # 기간연동 시 별도 선택 불필요
                st.caption(
                    "손절기준: " + (", ".join(_linked_pairs) if _linked_pairs else "—")
                )
            else:
                _sp_sel = _gs_multi("학습손절기준 (%)", "gs_stop")

        st.markdown("##### 매매 파라미터")
        _gs_c4, _gs_c5, _gs_c5b = st.columns(3)
        with _gs_c4:
            _topk_sel = _gs_multi("Top K 종목", "gs_topk")
        with _gs_c5:
            _buy_opts = {"시가": "open", "종가": "close"}
            _buy_sel = _gs_multi("매수가 기준", "gs_buy")
        with _gs_c5b:
            _invest_sel = _gs_multi("운용방식", "gs_invest")

        _gs_c6, _gs_c7, _gs_c8 = st.columns(3)
        with _gs_c6:
            _a_sel = _gs_multi("A-트레일링 목표가 (%)", "gs_a")
        with _gs_c7:
            _b_sel = _gs_multi("B-고점하락 매도 (%)", "gs_b")
        with _gs_c8:
            _c_sel = _gs_multi("C-손절가 (%)", "gs_c")

        # 조합 수 계산 및 표시
        if _stop_linked:
            _n_label = max(1, len(_hw_sel)) * max(1, len(_tp_sel))
            _stop_desc = "기간연동"
        else:
            _n_label = (
                max(1, len(_hw_sel)) * max(1, len(_tp_sel)) * max(1, len(_sp_sel))
            )
            _stop_desc = f"stop {len(_sp_sel)}"
        _n_trade = (
            max(1, len(_topk_sel))
            * max(1, len(_buy_sel))
            * max(1, len(_a_sel))
            * max(1, len(_b_sel))
            * max(1, len(_c_sel))
        )
        _n_models = max(1, len(_gs_model_sel))
        _n_total = _n_label * _n_trade * _n_models

        _model_desc_gs = ", ".join(_gs_model_sel) if _gs_model_sel else "없음"
        st.markdown(
            f"**모델**: {_model_desc_gs} ({_n_models}개)  \n"
            f"**학습 조합**: {_n_label}개 "
            f"(horizon {len(_hw_sel)} x target {len(_tp_sel)} x {_stop_desc})  \n"
            f"**매매 조합**: {_n_trade}개/학습  \n"
            f"**총 조합**: **{_n_total:,}**개"
        )
        if _n_label > 10:
            st.warning(
                f"학습 조합이 {_n_label}개입니다. "
                "각 조합마다 모델 학습이 필요하므로 시간이 오래 걸릴 수 있습니다."
            )

    # 실행 버튼
    _gs_ready = all(
        [
            _gs_model_sel,
            _hw_sel,
            _tp_sel,
            _sp_sel or _stop_linked,  # 기간연동이면 sp_sel 비어있어도 OK
            _topk_sel,
            _buy_sel,
            _a_sel,
            _b_sel,
            _c_sel,
        ]
    )

    if not _gs_ready:
        st.info("모델과 모든 파라미터에서 최소 1개 이상 선택하세요.")
    else:
        # 체크포인트 존재 여부 확인 + 현재 선택과 일치 여부
        _has_gs_cp = has_gs_checkpoint(date_str)
        _gs_cp_info = get_gs_checkpoint_info(date_str) if _has_gs_cp else None

        # 현재 선택 기반 라벨링 조합 생성 (체크포인트 비교용)
        import itertools as _it

        if _stop_linked:
            _cur_label_combos = [
                (hw, tp, _gs_rec_stop(hw, tp))
                for hw, tp in _it.product(sorted(_hw_sel), sorted(_tp_sel))
            ]
        else:
            _cur_label_combos = list(
                _it.product(sorted(_hw_sel), sorted(_tp_sel), sorted(_sp_sel))
            )

        _cp_match = False
        if _gs_cp_info:
            _cp_match = (
                _gs_cp_info.get("labeling_combos") == _cur_label_combos
                and _gs_cp_info.get("trading_combos_count") == _n_trade
            )

        _btn_c1, _btn_c2 = st.columns(2)
        with _btn_c1:
            gs_btn = st.button(
                f"설정조합별 수익률검사 실행 ({_n_total:,}개 조합)",
                type="primary",
                key="gs_run",
                use_container_width=True,
            )
        with _btn_c2:
            if _gs_cp_info and _cp_match:
                _cp_done = _gs_cp_info["completed_label_idx"]
                _cp_total = _gs_cp_info["n_label"]
                _cp_rows = _gs_cp_info["n_results"]
                gs_resume_btn = st.button(
                    f"이어서 실행 (학습 {_cp_done}/{_cp_total} 완료, {_cp_rows}행)",
                    type="secondary",
                    key="gs_resume",
                    use_container_width=True,
                )
            elif _gs_cp_info and not _cp_match:
                st.warning(
                    "이전 체크포인트와 현재 설정이 다릅니다. "
                    "'실행' 시 처음부터 시작됩니다."
                )
                if st.button("체크포인트 초기화", key="gs_cp_clear"):
                    clear_gs_checkpoint(date_str)
                    st.rerun()
                gs_resume_btn = False
            else:
                gs_resume_btn = False

        _gs_run = gs_btn or gs_resume_btn
        _gs_is_resume = bool(gs_resume_btn)

        if _gs_run:
            _buy_type_list = [_buy_opts[k] for k in _buy_sel]

            gs_progress = st.progress(0.0)
            gs_status = st.empty()
            gs_log_container = st.container(height=250)
            _gs_log_lines = []

            def _gs_cb(msg, pct=None):
                if pct is not None:
                    gs_progress.progress(min(float(pct), 1.0))
                gs_status.text(msg)

            def _gs_log(msg):
                _gs_log_lines.append(msg)
                gs_log_container.text("\n".join(_gs_log_lines[-50:]))

            # 모델별 순차 실행
            _gs_all_results = []
            _gs_model_map = {
                "AI급등주": ("stock", universe_df),
                "AI추천ETF": ("ETF", _load_etf_universe(date_str)),
            }

            with st.spinner(
                f"설정조합별 수익률검사 {'이어서 ' if _gs_is_resume else ''}"
                f"진행 중... ({_n_total:,}개 조합)"
            ):
                # 기간연동: horizon별 추천 stop으로 라벨링 조합 직접 생성
                import itertools as _it

                if _stop_linked:
                    _lc_override = [
                        (hw, tp, _gs_rec_stop(hw, tp))
                        for hw, tp in _it.product(sorted(_hw_sel), sorted(_tp_sel))
                    ]
                else:
                    _lc_override = None

                for _mi, _gs_model_name in enumerate(_gs_model_sel):
                    _gs_asset, _gs_univ = _gs_model_map[_gs_model_name]
                    _gs_log(
                        f"{'=' * 40}\n"
                        f"[모델 {_mi + 1}/{len(_gs_model_sel)}] "
                        f"{_gs_model_name} ({_gs_asset}) 시작"
                    )

                    # 진행률 오프셋 (모델별 비율 분배)
                    _pct_base = _mi / len(_gs_model_sel)
                    _pct_range = 1.0 / len(_gs_model_sel)

                    def _gs_cb_model(
                        msg, pct=None, _b=_pct_base, _r=_pct_range, _mn=_gs_model_name
                    ):
                        if pct is not None:
                            _adj = _b + float(pct) * _r
                            gs_progress.progress(min(_adj, 1.0))
                        gs_status.text(f"[{_mn}] {msg}")

                    gs_results_part = run_grid_search(
                        universe_df=_gs_univ,
                        date_str=date_str,
                        horizon_weeks_list=sorted(_hw_sel),
                        target_pct_list=sorted(_tp_sel),
                        stop_pct_list=(sorted(_sp_sel) if not _stop_linked else [10]),
                        top_k_list=[_parse_topk_option(x) for x in _topk_sel],
                        buy_type_list=_buy_type_list,
                        a_list=sorted(_a_sel),
                        b_list=sorted(_b_sel),
                        c_list=sorted(_c_sel),
                        train_window_years=train_window,
                        num_boost_round=num_boost,
                        progress_callback=_gs_cb_model,
                        log_callback=_gs_log,
                        resume=_gs_is_resume if _mi == 0 else False,
                        labeling_combos_override=_lc_override,
                        asset_type=_gs_asset,
                    )
                    # 결과에 모델 태그 추가
                    if gs_results_part:
                        for row in gs_results_part:
                            row["모델"] = _gs_model_name
                        _gs_all_results.extend(gs_results_part)

                    _gs_log(
                        f"  [{_gs_model_name}] 완료: "
                        f"{len(gs_results_part) if gs_results_part else 0}개 조합"
                    )

                gs_results = _gs_all_results

            gs_progress.progress(1.0)
            gs_status.empty()

            if gs_results:
                st.session_state["gs_results"] = gs_results
                st.success(f"완료! {len(gs_results):,}개 조합 검사 완료")
            else:
                st.warning("결과가 없습니다.")

    # 결과 표시 및 다운로드
    _gs_data = st.session_state.get("gs_results")
    if _gs_data:
        gs_df = pd.DataFrame(_gs_data)
        st.markdown(f"**검사 결과**: {len(gs_df):,}개 조합 (실투자수익률 내림차순)")

        # 운용방식 선택 (session_state에서 읽기)
        _gs_invest = st.session_state.get("gs_invest", ["분할매수"])

        # 스크린 표시용 컬럼 (기간평균수익률 및 서브기간 세부 열 숨기기)
        _gs_hide_cols = {
            "기간평균수익률",
            "실투자수익률_1Y",
            "실투자수익률_6M",
            "실투자수익률_3M",
            "실투자수익률_1M",
            "일괄투자수익률_1Y",
            "일괄투자수익률_6M",
            "일괄투자수익률_3M",
            "일괄투자수익률_1M",
            "승률_1Y",
            "승률_6M",
            "승률_3M",
            "승률_1M",
        }
        # 운용방식에 따라 열 숨기기
        if "분할매수" not in _gs_invest:
            _gs_hide_cols.add("실투자수익률")
        if "일괄매수" not in _gs_invest:
            _gs_hide_cols.add("일괄투자수익률")
        _gs_display_cols = [c for c in gs_df.columns if c not in _gs_hide_cols]
        _gs_col_cfg = {
            "모델": st.column_config.TextColumn("모델"),
            "수익목표기간": st.column_config.TextColumn("기간"),
            "목표수익률": st.column_config.TextColumn("목표수익"),
            "학습손절기준": st.column_config.TextColumn("학습손절"),
            "TopK": st.column_config.NumberColumn("TopK"),
            "매수가기준": st.column_config.TextColumn("매수가"),
            "A_트레일링목표": st.column_config.TextColumn("A%"),
            "B_고점하락매도": st.column_config.TextColumn("B%"),
            "C_손절": st.column_config.TextColumn("C%"),
            "실투자수익률": st.column_config.NumberColumn("분할수익률%", format="%.2f"),
            "일괄투자수익률": st.column_config.NumberColumn(
                "일괄수익률%", format="%.2f"
            ),
            "누적수익률": st.column_config.NumberColumn("누적수익률%", format="%.2f"),
            "평균 주간수익률": st.column_config.NumberColumn(
                "주간평균%", format="%.2f"
            ),
            "평균 주간수익률_1Y": st.column_config.NumberColumn(
                "주간%_1Y", format="%.2f"
            ),
            "평균 주간수익률_6M": st.column_config.NumberColumn(
                "주간%_6M", format="%.2f"
            ),
            "평균 주간수익률_3M": st.column_config.NumberColumn(
                "주간%_3M", format="%.2f"
            ),
            "평균 주간수익률_1M": st.column_config.NumberColumn(
                "주간%_1M", format="%.2f"
            ),
            "승률": st.column_config.NumberColumn("승률%", format="%.1f"),
            "샤프비율": st.column_config.NumberColumn("샤프", format="%.2f"),
            "최대낙폭": st.column_config.NumberColumn("최대낙폭%", format="%.2f"),
            "총거래수": st.column_config.NumberColumn("거래수"),
            "손절%": st.column_config.NumberColumn("손절%", format="%.1f"),
            "트레일링%": st.column_config.NumberColumn("트레일링%", format="%.1f"),
            "만기매도%": st.column_config.NumberColumn("만기%", format="%.1f"),
        }
        st.dataframe(
            gs_df[_gs_display_cols],
            column_config=_gs_col_cfg,
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        # Excel 다운로드
        from datetime import datetime as _dt

        _today_str = _dt.now().strftime("%Y-%m-%d")
        _gs_fname = f"설정조합별 수익률검사({_today_str}).xlsx"

        _gs_buf = io.BytesIO()
        with pd.ExcelWriter(_gs_buf, engine="openpyxl") as writer:
            gs_df.to_excel(writer, sheet_name="조합별수익률", index=False)
        _gs_buf.seek(0)

        st.download_button(
            f"Excel 다운로드 ({_gs_fname})",
            data=_gs_buf.getvalue(),
            file_name=_gs_fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
