"""Page 6: AI추천주 스크리닝 — 독립 모델 학습/예측 + 수익률검사 통합.

AI급등주와 동일한 모델로 시작하되, 캐시/모델을 완전 분리.
Phase B에서 24주 안정형 우상향 모델로 교체 예정.

핵심 원칙: 이 페이지를 수정해도 AI급등주/AI강화학습에 영향 0.
"""

import io
import pickle
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (  # noqa: E402
    CACHE_DIR,
    RESULT_MCAP_DEFAULT_BILLION,
    RESULT_MCAP_MIN_BILLION,
)
from core.recommend_model import (  # noqa: E402
    load_recommend_backtest,
    load_recommend_model,
    predict_recommend,
    run_recommend_backtest,
    train_recommend_model,
)
from core.styles import apply_mobile_styles  # noqa: E402

# ── 보유기간 상수 (backtest_engine 공유) ──
try:
    from core.backtest_engine import HOLD_PERIODS
except ImportError:
    HOLD_PERIODS = [3, 4, 6, 8, 10, 12]

_PERIOD_COLORS = {
    3: "#E91E63",
    4: "#FF9800",
    6: "#2196F3",
    8: "#4CAF50",
    10: "#9C27B0",
    12: "#795548",
}


# ──────────────────────────────────────────────────
# 백테스트 결과 렌더링 헬퍼 함수 (호출 전에 정의)
# ──────────────────────────────────────────────────


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


def _plot_cumulative_return(summary, daily, config_lh=4):
    """실투자 누적 수익률 곡선 + KODEX 200/코스닥150 비교."""
    from datetime import datetime as _dt

    import plotly.graph_objects as go

    sbp = summary.get("summary_by_period", {})
    initial_capital = 100_000_000

    fig = go.Figure()

    # AI 실투자 수익률
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
                name=f"AI추천주 분할매수({config_lh}주, {sp.get('n_slots', 0)}슬롯)",
                line={"color": "#2196F3", "width": 2.5},
                hovertemplate="분할: %{y:+.1f}%<extra></extra>",
            )
        )

    # 일괄매수
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
                name=f"AI추천주 일괄매수({config_lh}주)",
                line={"color": "#4CAF50", "width": 2, "dash": "dash"},
                hovertemplate="일괄: %{y:+.1f}%<extra></extra>",
            )
        )

    # KODEX 200 / 코스닥150
    parsed_dates = _parse_dates(daily)
    kospi_base, kosdaq_base = None, None
    kospi_vals, kospi_dates = [], []
    kosdaq_vals, kosdaq_dates = [], []

    for i, d in enumerate(daily):
        k_buy = d.get("kospi_buy")
        if k_buy:
            if kospi_base is None:
                kospi_base = k_buy
            kospi_vals.append((k_buy / kospi_base - 1) * 100)
            kospi_dates.append(parsed_dates[i])
        kd_buy = d.get("kosdaq_buy")
        if kd_buy:
            if kosdaq_base is None:
                kosdaq_base = kd_buy
            kosdaq_vals.append((kd_buy / kosdaq_base - 1) * 100)
            kosdaq_dates.append(parsed_dates[i])

    if kospi_vals:
        fig.add_trace(
            go.Scatter(
                x=kospi_dates,
                y=kospi_vals,
                mode="lines",
                name="KODEX 200",
                line={"color": "#FF5722", "width": 1.5, "dash": "dot"},
            )
        )
    if kosdaq_vals:
        fig.add_trace(
            go.Scatter(
                x=kosdaq_dates,
                y=kosdaq_vals,
                mode="lines",
                name="KODEX 코스닥150",
                line={"color": "#9C27B0", "width": 1.5, "dash": "dot"},
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


def _plot_weekly_returns(daily, selected_period=4):
    """주별 수익률 바 차트."""
    import plotly.graph_objects as go

    dates = _parse_dates(daily)
    returns, valid_dates = [], []
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
    fig.add_trace(go.Bar(x=valid_dates, y=returns, marker_color=colors))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="지정일",
        yaxis_title=f"{selected_period}주 수익률 (%)",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
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


def _show_trade_details(daily, selected_period=4, fund_info=None):
    """종목별 거래 내역 테이블 (재무정보 포함)."""
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
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=min(400, len(rows) * 35 + 38),
        )
    else:
        st.info(f"{selected_period}주 거래 내역이 없습니다.")


def _download_bt_excel(res, fund_info=None):
    """백테스트 결과 Excel 다운로드."""
    daily = res.get("daily_results", [])
    config = res.get("config", {})
    if not daily:
        return

    _hold_periods = config.get("hold_periods", HOLD_PERIODS)

    # Sheet 1: 종목별 거래내역
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
                else:
                    row[f"{m}주매도가"] = None
                    row[f"{m}주수익률"] = None
                    row[f"{m}주매도사유"] = None
            trade_rows.append(row)
    df_trades = pd.DataFrame(trade_rows)

    # Sheet 2: 주별 요약
    weekly_rows = []
    for d in daily:
        row = {"지정일": d["date"], "매수일": d["buy_date"]}
        prs = d.get("portfolio_returns", {})
        for m in _hold_periods:
            row[f"{m}주수익률"] = prs.get(m)
        row["AUC"] = d.get("model_auc")
        weekly_rows.append(row)
    df_weekly = pd.DataFrame(weekly_rows)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_trades.to_excel(writer, sheet_name="종목별거래내역", index=False)
        df_weekly.to_excel(writer, sheet_name="주별요약", index=False)
    buf.seek(0)

    _lhw = config.get("label_horizon_weeks", 4)
    _nb = config.get("num_boost_round", 400)
    _sd = config.get("start_date", "")
    _ed = config.get("end_date", "")
    _fname = f"recommend_bt_{_sd}_{_ed}_goal{_lhw}w_boost{_nb}.xlsx"

    st.download_button(
        "📥 AI추천주 백테스트 결과 (Excel)",
        buf.getvalue(),
        _fname,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def _render_recommend_bt(res):
    """AI추천주 백테스트 결과 렌더링."""
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

    # ── 수익률 상세 통계 ──
    st.subheader(f"📈 {_goal_w}주 기준 수익률 상세 통계")

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

    # 매도사유 통계
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

    # 서브기간 통계
    _sub_periods = summary.get("summary_by_subperiod", {})
    if _sub_periods:
        st.subheader("📊 기간별 성과 비교")
        _sub_rows = []
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

    # ── 차트 ──
    st.subheader("📊 AI추천주 실투자 vs 지수 비교")
    _plot_cumulative_return(summary, daily, config_lh=_goal_w)

    st.subheader("📊 주별 포트폴리오 수익률")
    _plot_weekly_returns(daily, _goal_w)

    st.subheader("🎯 모델 AUC 추이")
    _plot_auc_trend(daily)

    st.subheader("📋 종목별 거래 내역")
    _show_trade_details(daily, _goal_w, fund_info=_build_fund_info(universe_df))

    st.subheader("📥 다운로드")
    _download_bt_excel(res, fund_info=_build_fund_info(universe_df))


# ══════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════

st.set_page_config(page_title="AI추천주 스크리닝", page_icon="🤖", layout="wide")
apply_mobile_styles()

st.title("🤖 AI추천주 스크리닝")
st.caption(
    "24주 내 50% 안정형 우상향 모델 (기본). "
    "보수적 1차필터 (시총≥1000억, 거래대금≥50억)."
)

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

st.caption(f"📅 기준일: **{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}**")


# ══════════════════════════════════════════════════════
# 유니버스 로드
# ══════════════════════════════════════════════════════


@st.cache_data(ttl=3600)
def _load_universe(ds):
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
    # fundamentals (PBR, PER, DIV 등)
    if not fund_df.empty:
        _fund_cols = [c for c in ["PBR", "PER", "DIV"] if c in fund_df.columns]
        if _fund_cols:
            universe = universe.join(fund_df[_fund_cols], how="left")
    # ROE = PBR / PER (page 2와 동일 로직)
    if "PBR" in universe.columns and "PER" in universe.columns:
        import numpy as np

        _pbr = universe["PBR"].astype(float)
        _per = universe["PER"].astype(float)
        universe["ROE"] = np.where(
            (_per > 0) & (_pbr > 0), (_pbr / _per * 100).round(2), np.nan
        )
    return universe


universe_df = _load_universe(date_str)
st.info(f"전체 종목: **{len(universe_df)}**개")


def _build_fund_info(udf):
    """universe_df에서 종목별 재무정보 dict 생성 (거래내역용)."""
    import numpy as np

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
            "영업이익률": "",  # DART 데이터 필요 — 현재 미지원
        }
    return info


# ══════════════════════════════════════════════════════
# Section 1: AI추천주 학습
# ══════════════════════════════════════════════════════

st.markdown("---")
st.header("🚀 Section 1: AI추천주 학습")

# ── 학습 설정 ──
with st.expander("학습 설정", expanded=False):
    _rc1, _rc2, _rc3 = st.columns(3)
    with _rc1:
        _rec_horizon_weeks = st.selectbox(
            "수익목표기간 (horizon)",
            [3, 4, 6, 8, 10, 12, 16, 20, 24],
            index=8,
            format_func=lambda w: f"{w}주 ({w * 5}영업일)",
            key="rec_horizon_weeks",
            help="이 기간 내 목표수익률 달성 여부로 라벨링",
        )
    with _rc2:
        _rec_target_pct = st.number_input(
            "목표수익률 (target_pct) %",
            min_value=5,
            max_value=100,
            value=50,
            key="rec_target_pct",
            help="이 수익률 이상 달성 시 '급등' 라벨 부여",
        )
    with _rc3:
        from core.ml_pattern import compute_recommended_stop

        _rec_stop_default = compute_recommended_stop(
            _rec_horizon_weeks, _rec_target_pct
        )
        # horizon / target 변경 시 stop 자동갱신
        _prev_hw_s1 = st.session_state.get("_rec_prev_hw", _rec_horizon_weeks)
        _prev_tp_s1 = st.session_state.get("_rec_prev_tp", _rec_target_pct)
        if _rec_horizon_weeks != _prev_hw_s1 or _rec_target_pct != _prev_tp_s1:
            st.session_state["rec_stop_pct"] = _rec_stop_default
        st.session_state["_rec_prev_hw"] = _rec_horizon_weeks
        st.session_state["_rec_prev_tp"] = _rec_target_pct
        _rec_stop_kw = {
            "label": f"손절기준 (stop_pct) % — 추천: {_rec_stop_default}%",
            "min_value": 5,
            "max_value": 30,
            "key": "rec_stop_pct",
            "help": "Hard Negative 라벨 기준",
        }
        if "rec_stop_pct" not in st.session_state:
            _rec_stop_kw["value"] = _rec_stop_default
        _rec_stop_pct = st.number_input(**_rec_stop_kw)
    _rc4, _rc5, _rc6 = st.columns(3)
    with _rc4:
        _rec_use_gpu = st.checkbox(
            "GPU 사용", value=False, key="rec_use_gpu", help="LightGBM GPU 학습 모드"
        )
    with _rc5:
        _rec_min_mcap = st.number_input(
            "학습 최소 시총 (억)",
            min_value=RESULT_MCAP_MIN_BILLION,
            max_value=5000,
            value=1000,
            step=100,
            key="rec_min_mcap",
            help=f"학습 유니버스 시총 필터 (최소 {RESULT_MCAP_MIN_BILLION}억)",
        )
    with _rc6:
        _rec_min_tv = st.number_input(
            "최소 평균거래대금 (억)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="rec_min_tv",
            help="1차 필터: 20일 평균거래대금 최소 기준 (AI급등주: 30억)",
        )
    st.caption(
        f"학습: {_rec_horizon_weeks}주/{_rec_target_pct}%/손절{_rec_stop_pct}% | "
        f"1차필터: 시총≥{_rec_min_mcap}억 / 거래대금≥{_rec_min_tv}억"
    )

# ── 기존 모델 확인 ──
_surge_modes = ["surge", "surge_1y", "surge_6m", "surge_3m"]
_surge_labels = ["2Y 전체", "최근 1Y", "최근 6M", "최근 3M"]
_rec_target_f = _rec_target_pct / 100.0

_existing_models = {}
for _mode in _surge_modes:
    _m = load_recommend_model(mode=_mode, target_pct=_rec_target_f)
    if _m:
        _existing_models[_mode] = _m

if _existing_models:
    _model_tags = [
        f"{_surge_labels[_surge_modes.index(m)]}(AUC {d.get('validation_auc', 0):.3f})"
        for m, d in _existing_models.items()
    ]
    st.success(f"저장된 AI추천주 모델: {', '.join(_model_tags)}")
else:
    st.info("저장된 AI추천주 모델이 없습니다. 학습을 실행하세요.")

# ── 현재 학습 목표 파라미터 표시 ──
st.info(
    f"학습 목표: **목표수익률 {_rec_target_pct}%** | "
    f"**목표기간 {_rec_horizon_weeks}주 ({_rec_horizon_weeks * 5}영업일)** | "
    f"**손절 {_rec_stop_pct}%**"
)

# ── 학습 실행 버튼 ──
_train_label = "🔄 AI추천주 재학습" if _existing_models else "🚀 AI추천주 학습 실행"
train_recommend_btn = st.button(_train_label, type="primary", use_container_width=True)

if train_recommend_btn:
    _run_start = _time.time()
    with st.status(
        f"AI추천주 학습 중 (목표 {_rec_target_pct}%, {_rec_horizon_weeks}주)...",
        expanded=True,
    ) as train_status:
        train_progress = st.progress(0)

        try:

            def _train_cb(msg, pct=None):
                st.write(msg)
                if pct is not None:
                    train_progress.progress(min(int(pct), 100))

            result = train_recommend_model(
                universe_df=universe_df,
                date_str=date_str,
                progress_callback=_train_cb,
                use_gpu=_rec_use_gpu,
                horizon_days=_rec_horizon_weeks * 5,
                target_pct=_rec_target_f,
                stop_pct=_rec_stop_pct / 100.0,
                min_avg_tv=_rec_min_tv * 100_000_000,
            )

            if result:
                # 세션에 모델 저장
                for _mode, _model in result.items():
                    if _model:
                        _key = f"recommend_model_{_mode}_t{_rec_target_pct}"
                        st.session_state[_key] = _model
                        _auc = _model.get("validation_auc", 0)
                        _period = _model.get("period", _mode)
                        st.write(f"  ✅ {_period}: AUC={_auc:.3f}")

                # 예측 실행
                st.write("AI추천주 예측 적용 중...")
                _screened_results = {}
                _ohlcv_dict = st.session_state.get("candidates_ohlcv_100d", {})

                # OHLCV가 없으면 캐시에서 시도
                if not _ohlcv_dict:
                    _ohlcv_cache = cache_dir / "screening_ohlcv.pkl"
                    if _ohlcv_cache.exists():
                        with open(_ohlcv_cache, "rb") as f:
                            _ohlcv_dict = pickle.load(f)
                        st.write(f"  캐시 OHLCV 로드: {len(_ohlcv_dict)}종목")

                if _ohlcv_dict:
                    # KOSPI 지수 로드
                    _kospi_close = None
                    try:
                        from core.data_fetcher import fetch_kospi_index_3year

                        _kospi_df = fetch_kospi_index_3year(date_str)
                        if _kospi_df is not None and len(_kospi_df) > 0:
                            _kospi_close = _kospi_df["종가"].astype(float)
                    except Exception:
                        pass

                    _tickers = list(_ohlcv_dict.keys())
                    for _mode, _model in result.items():
                        if not _model:
                            continue
                        _pred = predict_recommend(
                            model_data=_model,
                            ohlcv_dict=_ohlcv_dict,
                            tickers=_tickers,
                            kospi_close=_kospi_close,
                            min_avg_tv=_rec_min_tv * 100_000_000,
                        )
                        if _pred is not None and not _pred.empty:
                            _n_pass = int(_pred["AI_Pass"].sum())
                            _period = _model.get("period", _mode)
                            st.write(f"  ✅ {_period}: **{_n_pass}종목** 추천")
                            _screened_results[_mode] = _pred

                    st.session_state["recommend_screening_results"] = _screened_results
                else:
                    st.warning(
                        "OHLCV 데이터 없음. 스크리닝 페이지에서 먼저 데이터를 수집하세요."
                    )

                _total_models = sum(1 for v in result.values() if v)
                train_progress.progress(100)
                _elapsed = (_time.time() - _run_start) / 60
                train_status.update(
                    label=f"✅ AI추천주 학습 완료! ({_total_models}개 모델, {_elapsed:.1f}분)",
                    state="complete",
                )
                st.rerun()
            else:
                train_status.update(label="❌ 학습 결과 없음", state="error")

        except Exception as e:
            _elapsed = (_time.time() - _run_start) / 60
            train_status.update(
                label=f"❌ 학습 실패 ({_elapsed:.1f}분): {e}", state="error"
            )
            import traceback

            st.code(traceback.format_exc())

# ── 예측 결과 표시 ──
# ── 메타 정보 ──
_meta_mcap_eok = st.session_state.get("rec_min_mcap", 1000)
_meta_tp_pct = st.session_state.get("rec_target_pct", 50)
_meta_hw_w = st.session_state.get("rec_horizon_weeks", 24)
_meta_sp_pct = st.session_state.get("rec_stop_pct", 15)
_meta_date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

_rec_results = st.session_state.get("recommend_screening_results", {})
if _rec_results:
    st.subheader("📊 AI추천주 예측 결과")

    # ── 결과 시총 필터 ──
    _result_mcap_b = st.slider(
        "결과 최소 시총 (억)",
        RESULT_MCAP_MIN_BILLION,
        5000,
        value=st.session_state.get(
            "result_min_mcap_billion", RESULT_MCAP_DEFAULT_BILLION
        ),
        step=100,
        key="rec_result_min_mcap",
        help=f"Top K 결과에서 시총 이 값 이상만 표시 (최소 {RESULT_MCAP_MIN_BILLION}억)",
    )
    st.session_state["result_min_mcap_billion"] = _result_mcap_b

    st.caption(
        f"📋 시총 {_result_mcap_b:,}억 이상 | "
        f"목표 {_meta_tp_pct}%/{_meta_hw_w}주 | "
        f"손절 {_meta_sp_pct}% | "
        f"{_meta_date_fmt} 종가 기준"
    )

    _mode_tabs = st.tabs(
        [f"{_surge_labels[_surge_modes.index(m)]}" for m in _rec_results.keys()]
    )
    for _tab, (_mode, _pred_df) in zip(_mode_tabs, _rec_results.items(), strict=False):
        with _tab:
            if _pred_df is None or _pred_df.empty:
                st.info("예측 결과 없음")
                continue

            _pass_df = _pred_df[_pred_df["AI_Pass"]].copy()
            if _pass_df.empty:
                st.info("추천 종목 없음")
                continue

            _pass_df = _pass_df.sort_values("AI_Score", ascending=False)

            # 종목명 + 재무정보 조인
            for _col in ["Name", "Market", "MarketCap", "PBR", "PER", "ROE", "DIV"]:
                if _col in universe_df.columns:
                    _pass_df[_col] = _pass_df.index.map(
                        lambda t, c=_col: universe_df.at[t, c]
                        if t in universe_df.index
                        else None
                    )

            # 시총 필터 적용
            _mcap_threshold = _result_mcap_b * 1e8
            if "MarketCap" in _pass_df.columns:
                _pass_df = _pass_df[_pass_df["MarketCap"].fillna(0) >= _mcap_threshold]

            _pass_df["종목명"] = _pass_df.get("Name", _pass_df.index)
            _pass_df["시장"] = _pass_df.get("Market", "")
            _pass_df["시총(조)"] = (
                _pass_df.get("MarketCap", 0).fillna(0) / 1e12
            ).round(2)
            # 현재가
            _ohlcv_dict_disp = st.session_state.get("candidates_ohlcv_100d", {})
            _pass_df["현재가"] = _pass_df.index.map(
                lambda t, _d=_ohlcv_dict_disp: int(_d[t]["종가"].astype(float).iloc[-1])
                if t in _d and len(_d[t]) > 0
                else 0
            )
            _pass_df["Naver"] = _pass_df.index.map(
                lambda t: f"https://finance.naver.com/item/main.naver?code={t}"
            )

            _display_cols = [
                "종목명",
                "Naver",
                "시장",
                "시총(조)",
                "현재가",
                "PBR",
                "PER",
                "ROE",
                "DIV",
                "AI_Score",
                "AI_Timing",
                "AI_Reason",
            ]
            _display_cols = [c for c in _display_cols if c in _pass_df.columns]

            st.caption(f"**{len(_pass_df)}종목** 추천 (Top 20 기준)")
            st.dataframe(
                _pass_df[_display_cols]
                .head(20)
                .rename(
                    columns={
                        "Naver": "Naver",
                        "DIV": "배당률",
                        "AI_Score": "점수",
                        "AI_Timing": "타이밍",
                        "AI_Reason": "근거",
                    }
                ),
                column_config={
                    "Naver": st.column_config.LinkColumn(
                        "Naver", display_text="📊 Naver"
                    ),
                },
                use_container_width=True,
                height=min(500, 35 * min(len(_pass_df), 20) + 38),
            )

            # ── Excel / CSV 다운로드 ──
            _dl_rename = {
                "DIV": "배당률",
                "AI_Score": "점수",
                "AI_Timing": "타이밍",
                "AI_Reason": "근거",
            }
            _dl_df = _pass_df[_display_cols].head(20).rename(columns=_dl_rename)
            _dl_df.index.name = "순위"
            _dl_df.index = range(1, len(_dl_df) + 1)
            try:
                from datetime import datetime as _dt_dl

                _now_str = _dt_dl.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                _now_str = ""
            _dl_meta = {
                "모델": f"AI추천주({_mode})",
                "시총필터": f"{_result_mcap_b}억",
                "목표수익률": f"{_meta_tp_pct}%",
                "수익목표기간": f"{_meta_hw_w}주",
                "손절기준": f"{_meta_sp_pct}%",
                "종가기준일": _meta_date_fmt,
                "Top_K": str(len(_dl_df)),
                "생성일시": _now_str,
            }
            _dl_c1, _dl_c2 = st.columns(2)
            with _dl_c1:
                _xl_buf = io.BytesIO()
                with pd.ExcelWriter(_xl_buf, engine="openpyxl") as xw:
                    pd.DataFrame(
                        list(_dl_meta.items()), columns=["항목", "값"]
                    ).to_excel(xw, sheet_name="META", index=False)
                    _dl_df.to_excel(xw, sheet_name="TOP_K", index=True)
                st.download_button(
                    "📥 Excel",
                    _xl_buf.getvalue(),
                    f"AI추천주_{_mode}_top{len(_dl_df)}_{date_str}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"rec_xl_{_mode}",
                )
            with _dl_c2:
                _csv_lines = [f"#META:{k}={v}" for k, v in _dl_meta.items()]
                _csv_lines.append(_dl_df.to_csv(encoding="utf-8-sig"))
                st.download_button(
                    "📥 CSV",
                    "\n".join(_csv_lines),
                    f"AI추천주_{_mode}_top{len(_dl_df)}_{date_str}.csv",
                    "text/csv",
                    use_container_width=True,
                    key=f"rec_csv_{_mode}",
                )

    # ── AI급등주와 비교 ──
    _surge_results = st.session_state.get("screening_results", {})
    if _surge_results:
        st.subheader("🔄 AI급등주 vs AI추천주 비교")
        for _mode in _rec_results:
            _rec_pred = _rec_results.get(_mode)
            if _rec_pred is None or _rec_pred.empty:
                continue
            _rec_pass = set(_rec_pred[_rec_pred["AI_Pass"]].index)

            # AI급등주 결과는 screened DataFrame에서 가져오기
            _screened = st.session_state.get("screened")
            if _screened is not None:
                _mode_map = {
                    "surge": "AI_30%급등주",
                    "surge_1y": "AI_30%급등주_1Y",
                    "surge_6m": "AI_30%급등주_6M",
                    "surge_3m": "AI_30%급등주_3M",
                }
                _surge_cat = _mode_map.get(_mode)
                if _surge_cat and f"Pass_{_surge_cat}" in _screened.columns:
                    _surge_pass = set(_screened[_screened[f"Pass_{_surge_cat}"]].index)
                    _overlap = _rec_pass & _surge_pass
                    _rec_only = _rec_pass - _surge_pass
                    _surge_only = _surge_pass - _rec_pass

                    _period = _surge_labels[_surge_modes.index(_mode)]
                    st.caption(
                        f"**{_period}**: "
                        f"겹침 {len(_overlap)}개 | "
                        f"추천주만 {len(_rec_only)}개 | "
                        f"급등주만 {len(_surge_only)}개"
                    )


# ══════════════════════════════════════════════════════
# Section 2: AI추천주 수익률검사
# ══════════════════════════════════════════════════════

st.markdown("---")
st.header("📊 Section 2: AI추천주 수익률검사")
st.caption("AI추천주 모델의 과거 실제 수익률을 검증합니다 (주별 롤링)")

# ── 설정 패널 ──
with st.expander("수익률검사 설정", expanded=True):
    _bc1, _bc2, _bc3, _bc4, _bc5 = st.columns(5)
    with _bc1:
        _bt_train_window = st.selectbox(
            "학습 윈도우",
            [2, 1, 3, 4],
            format_func=lambda x: f"{x}년",
            key="rec_bt_train_window",
            help="학습에 사용할 과거 데이터 기간",
        )
    with _bc2:
        _bt_top_k = st.number_input(
            "Top K 종목",
            min_value=5,
            max_value=50,
            value=20,
            key="rec_bt_top_k",
        )
    with _bc3:
        _bt_horizon_weeks = st.selectbox(
            "수익목표기간 (horizon)",
            [3, 4, 6, 8, 10, 12, 16, 20, 24],
            index=8,
            format_func=lambda w: f"{w}주 ({w * 5}영업일)",
            key="rec_bt_horizon_weeks",
            help="라벨링 horizon: 이 기간 내 목표수익 달성 여부로 학습",
        )
    with _bc4:
        _bt_target = st.number_input(
            "학습목표수익률 (target_pct) %",
            min_value=5,
            max_value=100,
            value=50,
            key="rec_bt_target",
            help="이 수익률 이상 달성 시 '급등' 라벨 부여",
        )
    with _bc5:
        from core.ml_pattern import compute_recommended_stop as _crs

        _bt_stop_rec = _crs(_bt_horizon_weeks, _bt_target)
        # horizon / target 변경 시 stop 자동갱신
        _prev_bt_hw = st.session_state.get("_rec_bt_prev_hw", _bt_horizon_weeks)
        _prev_bt_tp = st.session_state.get("_rec_bt_prev_tp", _bt_target)
        if _bt_horizon_weeks != _prev_bt_hw or _bt_target != _prev_bt_tp:
            st.session_state["rec_bt_stop"] = _bt_stop_rec
        st.session_state["_rec_bt_prev_hw"] = _bt_horizon_weeks
        st.session_state["_rec_bt_prev_tp"] = _bt_target
        _bt_stop_kw = {
            "label": f"학습손절기준 (stop_pct) % — 추천: {_bt_stop_rec}%",
            "min_value": 5,
            "max_value": 30,
            "key": "rec_bt_stop",
        }
        if "rec_bt_stop" not in st.session_state:
            _bt_stop_kw["value"] = _bt_stop_rec
        _bt_stop = st.number_input(**_bt_stop_kw)

    _bc6, _bc7 = st.columns(2)
    with _bc6:
        _bt_boost = st.slider(
            "모델강화 (Boost Rounds)",
            300,
            800,
            400,
            step=50,
            key="rec_bt_boost",
        )
    with _bc7:
        st.caption("300: 빠른실행 | 400: 균형(권장) | 800: 최고정확도")

    _bc8, _bc9, _bc10, _bc11 = st.columns(4)
    with _bc8:
        _bt_min_profit = st.number_input(
            "A: 최소수익률 (%)",
            min_value=0,
            max_value=50,
            value=30,
            key="rec_bt_min_profit",
            help="트레일링 발동 조건",
        )
    with _bc9:
        _bt_trailing = st.number_input(
            "B: 고점대비 하락 (%)",
            min_value=3,
            max_value=30,
            value=5,
            key="rec_bt_trailing",
        )
    with _bc10:
        _bt_stop_loss = st.number_input(
            "손절 C (%)",
            min_value=5,
            max_value=30,
            value=10,
            key="rec_bt_stop_loss",
        )
    with _bc11:
        _bt_buy_type = st.selectbox(
            "매수가 기준",
            ["시가", "종가"],
            key="rec_bt_buy_type",
        )

    _buy_label = "시가" if _bt_buy_type == "시가" else "종가"
    st.caption(
        f"학습윈도우 {_bt_train_window}년 | "
        f"수익목표기간 {_bt_horizon_weeks}주({_bt_horizon_weeks * 5}일) | "
        f"목표수익률 {_bt_target}% | 손절기준 {_bt_stop}% | "
        f"Boost {_bt_boost} | Top {_bt_top_k}종목 | 매수가: {_buy_label} | "
        f"매도: A={_bt_min_profit}%→B={_bt_trailing}%하락, 손절 C={_bt_stop_loss}%"
    )

# ── 현재 수익률검사 목표 파라미터 표시 ──
st.info(
    f"학습 목표: **목표수익률 {_bt_target}%** | "
    f"**목표기간 {_bt_horizon_weeks}주 ({_bt_horizon_weeks * 5}영업일)** | "
    f"**손절 {_bt_stop}%**"
)

# ── 캐시된 결과 확인 ──
_cached_bt = load_recommend_backtest()
if _cached_bt:
    _cfg = _cached_bt.get("config", {})
    st.success(
        f"저장된 결과: {_cfg.get('start_date', '?')} ~ {_cfg.get('end_date', '?')} | "
        f"Top {_cfg.get('top_k', '?')} | "
        f"{_cached_bt['summary'].get('total_weeks', 0)}주"
    )

# ── 실행 버튼 ──
_run_col, _load_col = st.columns(2)
with _run_col:
    _bt_run_btn = st.button(
        "AI추천주 수익률검사 실행",
        type="primary",
        use_container_width=True,
        key="rec_bt_run",
    )
with _load_col:
    _bt_load_btn = st.button(
        "저장된 결과 보기",
        use_container_width=True,
        disabled=_cached_bt is None,
        key="rec_bt_load",
    )

if _bt_run_btn:
    _run_start = _time.time()
    with st.status(
        f"AI추천주 수익률검사 실행 중... (보유기간: {_bt_horizon_weeks}주)",
        expanded=True,
    ) as bt_status:
        bt_progress = st.progress(0)
        bt_log_lines = []
        bt_log_container = st.empty()

        def _bt_progress_cb(msg, pct=None):
            if pct is not None:
                bt_progress.progress(min(pct, 100))
            bt_log_lines.append(msg)
            bt_log_container.text("\n".join(bt_log_lines[-50:]))

        try:
            _buy_param = "open" if _bt_buy_type == "시가" else "close"
            bt_result = run_recommend_backtest(
                universe_df=universe_df,
                date_str=date_str,
                progress_callback=_bt_progress_cb,
                train_window_years=_bt_train_window,
                top_k=_bt_top_k,
                trailing_stop_pct=_bt_trailing / 100.0,
                stop_loss_pct=_bt_stop_loss / 100.0,
                min_profit_pct=_bt_min_profit / 100.0,
                resume=False,
                num_boost_round=_bt_boost,
                buy_price_type=_buy_param,
                label_target_pct=_bt_target / 100.0,
                label_stop_pct=_bt_stop / 100.0,
                label_horizon_weeks=_bt_horizon_weeks,
                target_hold_periods=[_bt_horizon_weeks],
                asset_type="stock",
                min_avg_tv=_rec_min_tv * 100_000_000,
            )

            bt_progress.progress(100)
            n_weeks = len(bt_result.get("daily_results", []))
            _elapsed = (_time.time() - _run_start) / 60

            sbp = bt_result.get("summary", {}).get("summary_by_period", {})
            _goal_sp = sbp.get(_bt_horizon_weeks, {})
            _cum = _goal_sp.get("realistic_cum_return", 1.0)

            st.session_state["recommend_bt_result"] = bt_result
            bt_status.update(
                label=(
                    f"✅ AI추천주 수익률검사 완료! ({n_weeks}주, "
                    f"{_elapsed:.1f}분, "
                    f"{_bt_horizon_weeks}주누적 {(_cum - 1) * 100:+.1f}%)"
                ),
                state="complete",
            )
        except Exception as e:
            _elapsed = (_time.time() - _run_start) / 60
            bt_status.update(label=f"❌ 오류 ({_elapsed:.1f}분): {e}", state="error")
            import traceback

            st.code(traceback.format_exc())

elif _bt_load_btn and _cached_bt:
    st.session_state["recommend_bt_result"] = _cached_bt

# ── 결과 렌더링 ──
_bt_result = st.session_state.get("recommend_bt_result")
if _bt_result is None and _cached_bt:
    _bt_result = _cached_bt

if _bt_result:
    _render_recommend_bt(_bt_result)


# ══════════════════════════════════════════════════════
# Section 3: AI급등주 vs AI추천주 비교
# ══════════════════════════════════════════════════════

st.markdown("---")
st.header("🔄 Section 3: AI급등주 vs AI추천주 비교")

# AI급등주 백테스트 결과 로드
try:
    from core.backtest_engine import load_backtest_result

    _surge_bt = load_backtest_result()
except Exception:
    _surge_bt = None

_rec_bt = st.session_state.get("recommend_bt_result") or _cached_bt

if _surge_bt and _rec_bt:
    _s_summary = _surge_bt.get("summary", {})
    _r_summary = _rec_bt.get("summary", {})
    _s_config = _surge_bt.get("config", {})
    _r_config = _rec_bt.get("config", {})

    _s_goal_w = _s_config.get("label_horizon_weeks", 4)
    _r_goal_w = _r_config.get("label_horizon_weeks", 4)

    _s_sbp = _s_summary.get("summary_by_period", {}).get(_s_goal_w, {})
    _r_sbp = _r_summary.get("summary_by_period", {}).get(_r_goal_w, {})

    # 평균 주간수익률 (avg_return은 이미 주 단위 평균)
    _s_weekly_avg = _s_sbp.get("avg_return", 0)
    _r_weekly_avg = _r_sbp.get("avg_return", 0)

    compare_data = [
        {
            "지표": "평균 주간수익률",
            "AI급등주": f"{_s_weekly_avg * 100:+.2f}%",
            "AI추천주": f"{_r_weekly_avg * 100:+.2f}%",
        },
        {
            "지표": "분할투자 수익률",
            "AI급등주": f"{(_s_sbp.get('realistic_cum_return', 1) - 1) * 100:+.1f}%",
            "AI추천주": f"{(_r_sbp.get('realistic_cum_return', 1) - 1) * 100:+.1f}%",
        },
        {
            "지표": "일괄투자 수익률",
            "AI급등주": f"{(_s_sbp.get('lumpsum_cum_return', 1) - 1) * 100:+.1f}%",
            "AI추천주": f"{(_r_sbp.get('lumpsum_cum_return', 1) - 1) * 100:+.1f}%",
        },
        {
            "지표": "승률",
            "AI급등주": f"{_s_sbp.get('win_rate', 0) * 100:.1f}%",
            "AI추천주": f"{_r_sbp.get('win_rate', 0) * 100:.1f}%",
        },
        {
            "지표": "샤프비율",
            "AI급등주": f"{_s_sbp.get('sharpe_ratio', 0):.2f}",
            "AI추천주": f"{_r_sbp.get('sharpe_ratio', 0):.2f}",
        },
        {
            "지표": "최대낙폭",
            "AI급등주": f"{_s_sbp.get('max_drawdown', 0) * 100:.1f}%",
            "AI추천주": f"{_r_sbp.get('max_drawdown', 0) * 100:.1f}%",
        },
        {
            "지표": "거래주",
            "AI급등주": _s_sbp.get("n_weeks", 0),
            "AI추천주": _r_sbp.get("n_weeks", 0),
        },
        {
            "지표": "수익목표기간",
            "AI급등주": f"{_s_goal_w}주",
            "AI추천주": f"{_r_goal_w}주",
        },
        {
            "지표": "목표수익률",
            "AI급등주": f"{_s_config.get('label_target_pct', 0.2) * 100:.0f}%",
            "AI추천주": f"{_r_config.get('label_target_pct', 0.2) * 100:.0f}%",
        },
    ]

    st.dataframe(
        pd.DataFrame(compare_data),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Phase A: 동일 모델이므로 설정이 같으면 유사한 결과. "
        "Phase B에서 24주 안정형 모델로 진화 시 차이 발생 예정."
    )
elif not _surge_bt:
    st.info("AI급등주 백테스트 결과가 없습니다. AI수익률검사 페이지에서 실행하세요.")
elif not _rec_bt:
    st.info("AI추천주 백테스트 결과가 없습니다. 위 Section 2에서 실행하세요.")
