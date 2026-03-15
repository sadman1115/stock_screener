"""Page 2: 스크리닝 — 12개 전략 카테고리 동시 실행."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data_dart as dart
from core import data_fetcher as fetcher
from core import derived_metrics as dm
from core import indicators
from core.config import (
    CACHE_DIR,
    RESULT_MCAP_DEFAULT_BILLION,
    RESULT_MCAP_MIN_BILLION,
    get_categories_config,
    get_default_config,
)
from core.report_generator import generate_excel_report
from core.screener import run_all_categories
from core.styles import apply_mobile_styles

st.set_page_config(page_title="스크리닝", page_icon="🎯", layout="wide")
apply_mobile_styles()

st.title("🎯 스크리닝")

# ── 데이터 확인 ──
_REQUIRED_FILES = [
    "all_tickers.pkl",
    "all_ohlcv.pkl",
    "all_fundamentals.pkl",
    "all_market_cap.pkl",
    "all_ohlcv_20d.pkl",
]

date_str = st.session_state.get("date_str")

# session_state에 date_str 없으면 캐시에서 최신 날짜 자동 탐색
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
            st.session_state["data_collected"] = True
            st.toast(f"📦 캐시 자동 감지: {_cd[:4]}-{_cd[4:6]}-{_cd[6:]}")
            break

    if not date_str:
        st.warning(
            "⚠️ 수집된 데이터가 없습니다. **📊 데이터 수집** 페이지에서 데이터를 수집하세요."
        )
        st.stop()

cache_dir = CACHE_DIR / date_str
missing = [f for f in _REQUIRED_FILES if not (cache_dir / f).exists()]
if missing:
    st.warning(
        f"⚠️ 필수 데이터 없음: {', '.join(missing)}\n\n**📊 데이터 수집** 페이지에서 수집하세요."
    )
    st.stop()

st.caption(f"📅 기준일: **{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}**")

# ══════════════════════════════════════════════════════
# 카테고리 정의: (이름, 아이콘, 설명)
# ══════════════════════════════════════════════════════
RULE_PRESETS = [
    ("자산괴리주", "💎", "자산>시총120% + 영업흑자 + 수급"),
    ("고배당_가치주", "💰", "DIV4%↑ ROE5%↑ PBR≤0.6"),
    ("실적_턴어라운드", "📈", "ROE5%↑ ROA2%↑ PBR≤0.55"),
    ("외인기관_쌍끌이", "🌍", "외인+기관 AND 7일↑ 순매수"),
    ("기술적_바닥반전", "⚡", "52주저점근접 MACD + 반등2개↑"),
    ("초저PBR_안전망", "🔥", "PBR≤0.25 + 3중 안전장치"),
    ("수급전환_모멘텀", "📊", "8일↑ 집중순매수 PBR≤0.60"),
    ("저PER_가치주", "💲", "0<PER≤5 + 영업흑자 + ROE3%↑"),
]

AI_PRESETS = {
    25: [
        ("AI_25%급등주", "🤖", "25% 목표 2년 전체 학습"),
        ("AI_25%급등주_1Y", "🤖", "25% 목표 최근 1년"),
        ("AI_25%급등주_6M", "🤖", "25% 목표 최근 6개월"),
        ("AI_25%급등주_3M", "🤖", "25% 목표 최근 3개월"),
    ],
    30: [
        ("AI_30%급등주", "🤖", "30% 목표 2년 전체 학습"),
        ("AI_30%급등주_1Y", "🤖", "30% 목표 최근 1년"),
        ("AI_30%급등주_6M", "🤖", "30% 목표 최근 6개월"),
        ("AI_30%급등주_3M", "🤖", "30% 목표 최근 3개월"),
    ],
}


ETF_PRESETS = {
    10: [
        ("AI_10%추천ETF", "📦", "10% 목표 2년 전체 ETF학습"),
        ("AI_10%추천ETF_1Y", "📦", "10% 목표 최근 1년 ETF"),
        ("AI_10%추천ETF_6M", "📦", "10% 목표 최근 6개월 ETF"),
        ("AI_10%추천ETF_3M", "📦", "10% 목표 최근 3개월 ETF"),
    ],
    15: [
        ("AI_15%추천ETF", "📦", "15% 목표 2년 전체 ETF학습"),
        ("AI_15%추천ETF_1Y", "📦", "15% 목표 최근 1년 ETF"),
        ("AI_15%추천ETF_6M", "📦", "15% 목표 최근 6개월 ETF"),
        ("AI_15%추천ETF_3M", "📦", "15% 목표 최근 3개월 ETF"),
    ],
}


def _get_preset_list():
    tp_list = st.session_state.get("label_target_pct_list", [25, 30])
    etf_tp_list = st.session_state.get("etf_target_pct_list", [10, 15])
    presets = list(RULE_PRESETS)
    for tp in sorted(tp_list):
        presets += AI_PRESETS.get(tp, [])
    for tp in sorted(etf_tp_list):
        presets += ETF_PRESETS.get(tp, [])
    return presets


# AI 기간별 카테고리 모드 매핑 (모든 목표수익률 포함)
AI_PERIOD_MODES = {}
for _tp_key, _ai_list in AI_PRESETS.items():
    _surge_modes = ["surge", "surge_1y", "surge_6m", "surge_3m"]
    for (_ai_name, _, _), _sm in zip(_ai_list, _surge_modes, strict=False):
        AI_PERIOD_MODES[_ai_name] = _sm

# ETF 기간별 카테고리 모드 매핑
ETF_PERIOD_MODES = {}
for _tp_key, _etf_list in ETF_PRESETS.items():
    _etf_modes = ["etf_surge", "etf_surge_1y", "etf_surge_6m", "etf_surge_3m"]
    for (_etf_name, _, _), _em in zip(_etf_list, _etf_modes, strict=False):
        ETF_PERIOD_MODES[_etf_name] = _em

# ══════════════════════════════════════════════════════
# 시가총액 설정
# ══════════════════════════════════════════════════════
st.markdown("---")

min_mcap_billion = st.slider(
    "시가총액 최소 (억)",
    RESULT_MCAP_MIN_BILLION,
    5000,
    value=st.session_state.get("filter1_min_mcap", RESULT_MCAP_DEFAULT_BILLION),
    step=100,
    help=f"최소 {RESULT_MCAP_MIN_BILLION}억. 기본 {RESULT_MCAP_DEFAULT_BILLION}억",
)
min_mcap = min_mcap_billion * 100_000_000
st.session_state["filter1_min_mcap"] = min_mcap_billion
st.session_state["result_min_mcap_billion"] = min_mcap_billion

_init_presets = _get_preset_list()
st.caption(
    f"{len(_init_presets)}개 전략 동시 실행: "
    + " · ".join([f"{icon}{name}" for name, icon, _ in _init_presets])
)

# ══════════════════════════════════════════════════════
# 스크리닝 캐시 로드
# ══════════════════════════════════════════════════════
screening_cache_path = cache_dir / "screening_result.pkl"
ohlcv_cache_path = cache_dir / "screening_ohlcv.pkl"
ohlcv_1y_cache_path = cache_dir / "screening_ohlcv_1y.pkl"
universe_cache_path = cache_dir / "screening_universe.pkl"

if screening_cache_path.exists() and "screened" not in st.session_state:
    try:
        combined_cache = pd.read_pickle(screening_cache_path)
        pr = {}
        for name, _icon, _ in _get_preset_list():
            key = f"Pass_{name}"
            pr[name] = (
                int(combined_cache[key].sum()) if key in combined_cache.columns else 0
            )
        st.session_state["screened"] = combined_cache
        st.session_state["preset_results"] = pr
        if ohlcv_cache_path.exists():
            with open(ohlcv_cache_path, "rb") as f:
                st.session_state["candidates_ohlcv_100d"] = pickle.load(f)
        if ohlcv_1y_cache_path.exists():
            with open(ohlcv_1y_cache_path, "rb") as f:
                st.session_state["candidates_ohlcv_1y"] = pickle.load(f)
        if universe_cache_path.exists():
            st.session_state["universe"] = pd.read_pickle(universe_cache_path)
        st.toast("📦 캐시된 스크리닝 결과를 로드했습니다.")
    except Exception:
        pass

# ══════════════════════════════════════════════════════
# 스크리닝 실행
# ══════════════════════════════════════════════════════
st.markdown("---")

TIMING_BADGE = {
    "STRONG_BUY": "🔴 즉시매수",
    "BUY_NOW": "🟢 매수",
    "WATCH": "🟡 관심",
}

TIMING_RANK = {"STRONG_BUY": 3, "BUY_NOW": 2, "WATCH": 1, "": 0}

has_cache = screening_cache_path.exists()
bc1, bc2 = st.columns([3, 1])
with bc1:
    btn_label = (
        "▶ 스크리닝 재실행 (12개 전략)" if has_cache else "▶ 스크리닝 실행 (12개 전략)"
    )
    run_screening = st.button(btn_label, type="primary", use_container_width=True)
with bc2:
    clear_screening = st.button(
        "🗑️ 캐시 삭제", use_container_width=True, disabled=not has_cache
    )

if clear_screening:
    for _p in [
        screening_cache_path,
        ohlcv_cache_path,
        ohlcv_1y_cache_path,
        universe_cache_path,
    ]:
        if _p.exists():
            _p.unlink()
    for key in [
        "screened",
        "preset_results",
        "candidates_ohlcv_100d",
        "candidates_ohlcv_1y",
        "universe",
    ]:
        st.session_state.pop(key, None)
    st.success("✅ 스크리닝 캐시 삭제 완료")
    st.rerun()

if run_screening:
    # 썸네일 캐시 초기화
    for key in list(st.session_state.keys()):
        if key.startswith("thumbnails_"):
            del st.session_state[key]

    status = st.status("스크리닝 중...", expanded=True)
    progress = st.progress(0)

    try:
        # Step 1: 원시 데이터 로드 (캐시에서 직접 — API 호출 없음)
        status.write("Step 1/5: 원시 데이터 로드...")

        def _load_pkl(filename):
            with open(cache_dir / filename, "rb") as f:
                return pickle.load(f)

        tickers_df = _load_pkl("all_tickers.pkl")
        ohlcv_df = _load_pkl("all_ohlcv.pkl")
        fund_df = _load_pkl("all_fundamentals.pkl")
        cap_df = _load_pkl("all_market_cap.pkl")
        ohlcv_20d = _load_pkl("all_ohlcv_20d.pkl")
        progress.progress(10)

        # 업종분류 (캐시 있으면 로드, 없으면 빈 DataFrame — API 호출 없음)
        sector_file = cache_dir / "all_sectors.pkl"
        sector_df = (
            _load_pkl("all_sectors.pkl") if sector_file.exists() else pd.DataFrame()
        )

        # Universe 구축
        universe = tickers_df.set_index("Ticker")
        if not ohlcv_df.empty:
            ohlcv_cols = ["시가", "고가", "저가", "종가", "거래량", "거래대금"]
            avail = [c for c in ohlcv_cols if c in ohlcv_df.columns]
            universe = universe.join(ohlcv_df[avail], how="left")
            universe.rename(columns={"종가": "Close"}, inplace=True)
        if not fund_df.empty:
            universe = universe.join(fund_df, how="left", rsuffix="_fund")
        if not cap_df.empty:
            universe = universe.join(cap_df[["시가총액"]], how="left")
            universe.rename(columns={"시가총액": "MarketCap"}, inplace=True)
        if not sector_df.empty and "업종명" in sector_df.columns:
            universe = universe.join(sector_df[["업종명"]], how="left")
            universe.rename(columns={"업종명": "Sector"}, inplace=True)

        if "PBR" in universe.columns and "PER" in universe.columns:
            universe["ROE"] = universe.apply(
                lambda r: dm.calc_roe_from_pbr_per(
                    r.get("PBR", np.nan), r.get("PER", np.nan)
                ),
                axis=1,
            )

        avg_tv = fetcher.calc_avg_trading_value(ohlcv_20d)
        universe["AvgTradingValue20D"] = avg_tv
        suspended = fetcher.detect_suspended(ohlcv_20d)
        universe["Suspended"] = universe.index.isin(suspended)
        status.write(f"  ✅ 전종목: {len(universe)}개")
        progress.progress(20)

        # Step 2: 내부 사전필터 (시가총액 + PBR 스코프 + 거래정지 제외)
        status.write("Step 2/5: 사전필터 적용...")
        pbr_col = universe.get("PBR", pd.Series(np.nan, index=universe.index))
        mcap_col = universe.get("MarketCap", pd.Series(0, index=universe.index)).fillna(
            0
        )

        scope_mask = (
            (~universe["Suspended"])
            & (pbr_col > 0)
            & (pbr_col <= 1.00)  # 8개 카테고리 중 최대 PBR = 1.0 (고배당)
        )
        if min_mcap > 0:
            scope_mask = scope_mask & (mcap_col >= min_mcap)

        candidates_idx = universe[scope_mask].index.tolist()
        status.write(f"  ✅ 사전필터 통과: {len(candidates_idx)}종목")
        progress.progress(30)

        # Step 3: 후보군 상세 데이터
        status.write(f"Step 3/5: 후보군 상세 ({len(candidates_idx)}종목)...")
        detail_status = st.empty()

        detail_cache_file = cache_dir / "candidates_detail.pkl"
        detail = None

        # (A) 금일자 캐시 확인 (가장 빠른 경로)
        if detail_cache_file.exists():
            try:
                status.write("  📦 캐시된 상세 데이터 로드...")
                with open(detail_cache_file, "rb") as f:
                    detail = pickle.load(f)
                cached_set = set(detail.get("ohlcv_100d", {}).keys())
                missing_tickers = [t for t in candidates_idx if t not in cached_set]
                if missing_tickers:
                    status.write(f"  추가 {len(missing_tickers)}종목 수집 필요...")

                    def _extra_cb(msg):
                        detail_status.caption(f"  📡 {msg}")

                    extra = fetcher.fetch_candidates_detail(
                        missing_tickers, date_str, progress_callback=_extra_cb
                    )
                    detail_status.empty()
                    for key in ["ohlcv_100d", "ohlcv_1y", "investor"]:
                        detail[key].update(extra.get(key, {}))
                status.write(
                    f"  ✅ 상세 데이터: {len(detail.get('ohlcv_100d', {}))}종목"
                )
            except Exception:
                detail = None

        # (B) 이전 날짜 캐시에서 증분 업데이트
        if detail is None:
            prev_detail, prev_date = fetcher._find_nearest_cache(
                "candidates_detail", date_str, max_age_days=7
            )
            if prev_detail is not None and isinstance(prev_detail, dict):
                status.write(f"  📦 {prev_date} 상세 데이터 증분 업데이트...")
                prev_1y = prev_detail.get("ohlcv_1y", {})
                prev_inv = prev_detail.get("investor", {})

                today_ts = pd.Timestamp(date_str)
                n_updated = 0
                new_100d, new_1y, new_inv = {}, {}, {}

                for t in candidates_idx:
                    if (
                        t in prev_1y
                        and isinstance(prev_1y[t], pd.DataFrame)
                        and not prev_1y[t].empty
                    ):
                        old_1y = prev_1y[t]
                        if not isinstance(old_1y.index, pd.DatetimeIndex):
                            old_1y.index = pd.to_datetime(old_1y.index)

                        if t in ohlcv_df.index:
                            # ohlcv_df에서 금일 행 추출 → 1행 DataFrame
                            row_data = ohlcv_df.loc[t]
                            today_row = pd.DataFrame([row_data], index=[today_ts])
                            # per-ticker OHLCV와 동일한 컬럼만 유지
                            common_cols = [
                                c for c in old_1y.columns if c in today_row.columns
                            ]
                            if common_cols:
                                today_row = today_row[common_cols]
                                for col in old_1y.columns:
                                    if col not in today_row.columns:
                                        today_row[col] = np.nan
                                today_row = today_row[old_1y.columns]
                                updated_1y = pd.concat([old_1y, today_row])
                                updated_1y = updated_1y[
                                    ~updated_1y.index.duplicated(keep="last")
                                ]
                            else:
                                updated_1y = old_1y
                        else:
                            updated_1y = old_1y

                        # 1년 윈도우 트리밍 (~260영업일)
                        if len(updated_1y) > 260:
                            updated_1y = updated_1y.iloc[-260:]

                        new_1y[t] = updated_1y
                        new_100d[t] = (
                            updated_1y.iloc[-100:]
                            if len(updated_1y) >= 100
                            else updated_1y
                        )
                        new_inv[t] = prev_inv.get(t)
                        n_updated += 1

                # 신규 종목 (이전 캐시에 없는 종목)
                missing_tickers = [t for t in candidates_idx if t not in new_1y]
                if missing_tickers:
                    status.write(f"  신규 {len(missing_tickers)}종목 API 수집...")

                    def _new_cb(msg):
                        detail_status.caption(f"  📡 {msg}")

                    extra = fetcher.fetch_candidates_detail(
                        missing_tickers, date_str, progress_callback=_new_cb
                    )
                    detail_status.empty()
                    new_100d.update(extra.get("ohlcv_100d", {}))
                    new_1y.update(extra.get("ohlcv_1y", {}))
                    new_inv.update(extra.get("investor", {}))

                detail = {
                    "ohlcv_100d": new_100d,
                    "ohlcv_1y": new_1y,
                    "investor": new_inv,
                }
                fetcher._save_cache("candidates_detail", date_str, detail)
                status.write(
                    f"  ✅ 증분 업데이트: {n_updated}종목 재사용, "
                    f"{len(missing_tickers)}종목 신규 수집"
                )

        # (C) 이전 캐시도 없으면 전체 수집 (콜드 스타트)
        if detail is None:

            def detail_callback(msg):
                detail_status.caption(f"  📡 {msg}")
                if "/" in msg:
                    try:
                        parts = msg.split(":")[0] if ":" in msg else msg
                        nums = parts.split("/")
                        if len(nums) >= 2:
                            current = int("".join(filter(str.isdigit, nums[0])))
                            total = int(
                                "".join(filter(str.isdigit, nums[1].split()[0]))
                            )
                            pct = 30 + int(30 * current / total)
                            progress.progress(min(pct, 60))
                    except Exception:
                        pass

            detail = fetcher.fetch_candidates_detail(
                candidates_idx, date_str, progress_callback=detail_callback
            )
            detail_status.empty()

        progress.progress(60)

        # Step 4: 지표 계산
        status.write("Step 4/5: 기술지표/파생지표 계산...")
        indicator_rows = {}
        for t in candidates_idx:
            ohlcv_100d = detail["ohlcv_100d"].get(t)
            ind = indicators.compute_all_indicators(ohlcv_100d)
            ohlcv_1y = detail["ohlcv_1y"].get(t)
            hw = dm.calc_52w_high_low(ohlcv_1y)
            ind.update(hw)
            close = universe.at[t, "Close"] if "Close" in universe.columns else np.nan
            ind["Near52wLow"] = dm.calc_near_52w_low(close, hw["Low52"])
            ind["DrawdownFrom52wHigh"] = dm.calc_drawdown_from_52w_high(
                close, hw["High52"]
            )
            inv_df = detail["investor"].get(t)
            ind.update(dm.calc_flow_metrics(inv_df))
            ind["Low52_Breach_20d"] = dm.check_52w_low_breach(ohlcv_100d, hw["Low52"])
            indicator_rows[t] = ind

        ind_df = pd.DataFrame.from_dict(indicator_rows, orient="index")
        ind_df.index.name = "Ticker"
        for col in ind_df.columns:
            if col not in universe.columns:
                universe[col] = np.nan
            universe.loc[ind_df.index, col] = ind_df[col]
        progress.progress(70)

        # DART 재무 병합
        dart_df = dart._load_cache("financial_all", date_str)
        _dart_v2 = (
            dart_df is not None and not dart_df.empty and "revenue" in dart_df.columns
        )
        if not _dart_v2:
            status.write(
                "  ⚠ DART 재무 데이터 없음 또는 구버전 → "
                "📊 데이터 수집 페이지에서 DART 재수집 필요 "
                "(성장우량주A/B 카테고리)"
            )
        if dart_df is not None and not dart_df.empty:
            status.write("  DART 재무 데이터 병합...")
            mcap = universe["MarketCap"].fillna(0)
            for t in dart_df.index:
                if t not in universe.index:
                    continue
                fd = dart_df.loc[t]
                if not fd.get("available", False):
                    continue
                mc = mcap.get(t, 0)
                if mc > 0:
                    universe.at[t, "CashRatio"] = dm.calc_cash_ratio(
                        fd.get("cash", 0), fd.get("st_financial", 0), mc
                    )
                    universe.at[t, "RealEstateRatio"] = dm.calc_realestate_ratio(
                        fd.get("land", 0),
                        fd.get("building", 0),
                        fd.get("invest_property", 0),
                        mc,
                    )

            for t in dart_df.index:
                if t not in universe.index:
                    continue
                fd = dart_df.loc[t]
                if fd.get("available", False) and fd.get("total_assets", 0) > 0:
                    universe.at[t, "ROA"] = dm.calc_roa(
                        fd.get("net_income", 0), fd.get("total_assets", 0)
                    )
                # operating_profit 병합
                if (
                    fd.get("available", False)
                    and fd.get("operating_profit") is not None
                ):
                    universe.at[t, "operating_profit"] = fd.get("operating_profit", 0)

            # v2+v3: 우량주 등급용 파생지표
            for t in dart_df.index:
                if t not in universe.index:
                    continue
                fd = dart_df.loc[t]
                if not fd.get("available", False):
                    continue
                rev = fd.get("revenue", 0)
                rev_prev = fd.get("revenue_prev", 0)
                op = fd.get("operating_profit", 0)
                universe.at[t, "Revenue"] = rev
                universe.at[t, "RevenueGrowth"] = dm.calc_revenue_growth(rev, rev_prev)
                universe.at[t, "OperatingMargin"] = dm.calc_operating_margin(op, rev)
                universe.at[t, "OpCashFlow"] = fd.get("op_cash_flow", 0)
                universe.at[t, "net_income"] = fd.get("net_income", 0)
                liab = fd.get("total_liabilities", 0)
                equity = fd.get("total_equity", 0)
                universe.at[t, "DebtRatio"] = dm.calc_debt_ratio(liab, equity)

                # v3: 다년도 + 안전핀 지표
                rev_prev2 = fd.get("revenue_prev2", 0)
                op_prev = fd.get("op_prev", 0)
                interest = fd.get("interest_expense", 0)
                depreciation = fd.get("depreciation", 0)
                short_b = fd.get("short_borrowing", 0)
                long_b = fd.get("long_borrowing", 0)
                cash = fd.get("cash", 0)
                st_fin = fd.get("st_financial", 0)
                ni = fd.get("net_income", 0)
                ni_prev = fd.get("net_income_prev", 0)

                universe.at[t, "RevCAGR3Y"] = dm.calc_revenue_cagr_3y(rev, rev_prev2)
                op_m_prev = dm.calc_operating_margin(op_prev, rev_prev)
                universe.at[t, "OpMarginPrev"] = op_m_prev
                universe.at[t, "OpCashFlowPrev"] = fd.get("op_cash_flow_prev", 0)
                universe.at[t, "ICR"] = dm.calc_interest_coverage(op, interest)
                ebitda = dm.calc_ebitda(op, depreciation)
                universe.at[t, "NetDebtEBITDA"] = dm.calc_net_debt_to_ebitda(
                    short_b, long_b, cash, st_fin, ebitda
                )
                eps_g = dm.calc_eps_growth(ni, ni_prev, equity)
                universe.at[t, "EPSGrowth"] = eps_g
                per_val = universe.at[t, "PER"] if "PER" in universe.columns else np.nan
                universe.at[t, "PEG"] = dm.calc_peg_ratio(per_val, eps_g)
        progress.progress(80)

        # Step 5: 전략 카테고리 동시 스크리닝
        # 카테고리 config 로드 (선택된 목표수익률에 맞춰 필터)
        _tp_list_run = st.session_state.get("label_target_pct_list", [25, 30])
        categories = get_categories_config(target_pct_list=_tp_list_run)
        status.write(f"Step 5/5: {len(categories)}개 전략 카테고리 동시 스크리닝...")

        ohlcv_dict = detail["ohlcv_100d"]

        # AI 모델 로드 (목표수익률별 × 기간별)
        from core.ml_pattern import load_model as _load_ai_model

        ml_models = {}
        for _tp_run in _tp_list_run:
            _tp_f_run = _tp_run / 100.0
            for _ai_mode in ("surge", "surge_1y", "surge_6m", "surge_3m"):
                _key = f"{_ai_mode}_t{_tp_run}"
                _m = st.session_state.get(f"ai_model_{_key}") or _load_ai_model(
                    mode=_ai_mode, target_pct=_tp_f_run
                )
                if _m:
                    ml_models[_key] = _m

        combined = run_all_categories(
            universe, categories, ohlcv_dict, ml_model=ml_models or None
        )

        # 카테고리별 통과 종목 수 집계
        _cur_presets = _get_preset_list()
        preset_results = {}
        for i, (name, icon, _desc) in enumerate(_cur_presets):
            pass_key = f"Pass_{name}"
            n_pass = (
                int(combined[pass_key].sum()) if pass_key in combined.columns else 0
            )
            preset_results[name] = n_pass
            status.write(f"  {icon} {name}: {n_pass}종목")
            pct = 80 + int(18 * (i + 1) / len(_cur_presets))
            progress.progress(min(pct, 98))

        # C등급 경고 (Excel 전용)
        _qd = combined.apply(dm.compute_quality_detail, axis=1)
        combined["QualityGrade"] = _qd.apply(lambda d: d["grade"])
        combined["QualityWarnings"] = _qd.apply(lambda d: " ".join(d["warnings"]))
        n_c = int((combined["QualityGrade"] == "C").sum())
        if n_c > 0:
            status.write(f"  ⚠ C등급 경고: {n_c}종목")

        progress.progress(100)
        status.update(label="✅ 스크리닝 완료!", state="complete", expanded=False)

        # 결과 저장 (session + 디스크 캐시)
        st.session_state["universe"] = universe
        st.session_state["screened"] = combined
        st.session_state["preset_results"] = preset_results
        st.session_state["candidates_ohlcv_100d"] = ohlcv_dict
        st.session_state["candidates_ohlcv_1y"] = detail.get("ohlcv_1y", {})

        combined.to_pickle(screening_cache_path)
        universe.to_pickle(universe_cache_path)
        with open(ohlcv_cache_path, "wb") as f:
            pickle.dump(ohlcv_dict, f)
        with open(ohlcv_1y_cache_path, "wb") as f:
            pickle.dump(detail.get("ohlcv_1y", {}), f)

        st.rerun()

    except TimeoutError as e:
        status.update(label=f"⏱️ 타임아웃: {e}", state="error")
        st.warning(
            "💡 주말/공휴일에는 KRX API가 느려 개별종목 데이터 수집이 불가합니다.\n"
            "평일 장 마감 후 다시 실행하세요."
        )
    except Exception as e:
        status.update(label=f"❌ 스크리닝 실패: {e}", state="error")
        import traceback

        st.error(traceback.format_exc())

# ══════════════════════════════════════════════════════
# 결과
# ══════════════════════════════════════════════════════
if "screened" in st.session_state and "preset_results" in st.session_state:
    df = st.session_state["screened"]
    preset_results = st.session_state["preset_results"]

    total_any = int(df["Pass_any"].sum()) if "Pass_any" in df.columns else 0

    st.markdown("---")

    # ── 버튼 스타일링: 숨긴 마커 div → 인접 형제 button 색상 제어 ──
    def _cnt_class(n):
        """종목 수 → CSS 클래스. 0=gray, 1-3=red, 4-10=orange, 11+=blue."""
        if n == 0:
            return "cnt-zero"
        if n <= 3:
            return "cnt-low"
        if n <= 10:
            return "cnt-mid"
        return "cnt-high"

    st.markdown(
        """<style>
    .cnt-marker{display:none}
    div:has(.cnt-zero)+div button:not([kind="primary"]) p{
        color:#999!important;font-weight:bold!important}
    div:has(.cnt-low)+div button:not([kind="primary"]) p{
        color:#d63031!important;font-weight:bold!important}
    div:has(.cnt-mid)+div button:not([kind="primary"]) p{
        color:#e17055!important;font-weight:bold!important}
    div:has(.cnt-high)+div button:not([kind="primary"]) p{
        color:#0984e3!important;font-weight:bold!important}
    button[kind="primary"] p{font-weight:bold!important}
    .ai-green{display:none}
    div:has(.ai-green)+div button:not([kind="primary"]) p{
        color:#27ae60!important;font-weight:bold!important}
    div:has(.ai-green)+div button:not([kind="primary"]){
        border-color:#27ae60!important}
    div:has(.ai-green)+div button[kind="primary"]{
        background-color:#27ae60!important;border-color:#27ae60!important}
    .etf-red{display:none}
    div:has(.etf-red)+div button:not([kind="primary"]) p{
        color:#e74c3c!important;font-weight:bold!important}
    div:has(.etf-red)+div button:not([kind="primary"]){
        border-color:#e74c3c!important}
    div:has(.etf-red)+div button[kind="primary"]{
        background-color:#e74c3c!important;border-color:#e74c3c!important}
    .ai-purple{display:none}
    div:has(.ai-purple)+div button:not([kind="primary"]) p{
        color:#8e44ad!important;font-weight:bold!important}
    div:has(.ai-purple)+div button:not([kind="primary"]){
        border-color:#8e44ad!important}
    div:has(.ai-purple)+div button[kind="primary"]{
        background-color:#8e44ad!important;border-color:#8e44ad!important}
    </style>""",
        unsafe_allow_html=True,
    )

    # ── Best Timing 계산 (전체 df에 대해 1회) ──
    PRESET_LIST = _get_preset_list()

    def _best_timing_rank(row):
        best = 0
        for name, _, _ in PRESET_LIST:
            t = str(row.get(f"Timing_{name}", ""))
            best = max(best, TIMING_RANK.get(t, 0))
        return best

    df["_TimingRank"] = df.apply(_best_timing_rank, axis=1)
    pass_any_df = df[df["Pass_any"]] if "Pass_any" in df.columns else df.head(0)
    n_strong = int((pass_any_df["_TimingRank"] == 3).sum())
    n_buy = int((pass_any_df["_TimingRank"] == 2).sum())
    n_watch = int((pass_any_df["_TimingRank"] == 1).sum())

    # ── 헤더: 전체 요약 (클릭 시 전체보기) ──
    _hdr1, _hdr2 = st.columns([5, 1])
    with _hdr1:
        st.markdown(
            f"### 📊 결과 — **{total_any}**개 통과 &nbsp;&nbsp; "
            f"🔴 {n_strong} &nbsp; 🟢 {n_buy} &nbsp; 🟡 {n_watch}"
        )
    with _hdr2:
        if st.button(
            "📋 전체보기",
            key="v_전체",
            type="primary"
            if st.session_state.get("selected_view") == "전체"
            else "secondary",
            use_container_width=True,
        ):
            st.session_state["selected_view"] = "전체"
            st.rerun()

    # ── 보기 선택 (session_state 기반 통합 UI) ──
    if "selected_view" not in st.session_state:
        st.session_state["selected_view"] = "전체"

    # 선택된 뷰가 0개 카테고리면 전체로 폴백
    _cur_view = st.session_state["selected_view"]
    if _cur_view not in ("전체", "즉시매수", "매수", "관심"):
        if preset_results.get(_cur_view, 0) == 0:
            st.session_state["selected_view"] = "전체"

    # ── 카테고리 카드 그리드 (Row 1: 4열) ──
    cols_r1 = st.columns(4)
    for i, (name, icon, desc) in enumerate(RULE_PRESETS[:4]):
        with cols_r1[i]:
            n = preset_results.get(name, 0)
            sel = st.session_state["selected_view"] == name
            if st.button(
                f"{icon} {name} ({n})",
                key=f"v_{name}",
                type="primary" if sel else "secondary",
                use_container_width=True,
                disabled=(n == 0),
            ):
                st.session_state["selected_view"] = name
                st.rerun()
            st.caption(desc)

    st.markdown("")  # 행 간 여백

    # ── Row 2: 4열 (저PER + AI_급등주) ──
    cols_r2 = st.columns(4)
    for i, (name, icon, desc) in enumerate(RULE_PRESETS[4:8]):
        with cols_r2[i]:
            n = preset_results.get(name, 0)
            sel = st.session_state["selected_view"] == name
            if st.button(
                f"{icon} {name} ({n})",
                key=f"v_{name}",
                type="primary" if sel else "secondary",
                use_container_width=True,
                disabled=(n == 0),
            ):
                st.session_state["selected_view"] = name
                st.rerun()
            st.caption(desc)

    st.markdown("")  # 행 간 여백

    # ── Row 3+: AI 기간별 급등주 (초록색 버튼, 목표수익률별 1행) ──
    from core.ml_pattern import load_model as _load_ai

    _tp_list = st.session_state.get("label_target_pct_list", [25, 30])
    _any_surge = any(
        st.session_state.get(f"ai_model_{m}_t{tp}")
        or _load_ai(mode=m, target_pct=tp / 100.0)
        for tp in _tp_list
        for m in ("surge", "surge_1y", "surge_6m", "surge_3m")
    )
    _surge_label = (
        "🔄 AI 급등주학습 (업데이트)" if _any_surge else "🚀 AI 급등주학습 (최초)"
    )

    # AI 학습 고급 옵션 모달
    @st.dialog("⚙️ AI 학습 고급 옵션")
    def _ai_settings_dialog():
        st.markdown("**급등주 멀티기간 학습 옵션**")
        _cur_val = st.session_state.get("use_full_retrain_surge", False)
        use_retrain = st.checkbox(
            "📚 전체 재학습 (데이터 캐싱 활용)",
            value=_cur_val,
            help="캐시된 2년 데이터 + 최근 7일 신규 데이터로 전체 재학습\n"
            "• 속도: 25분 → 6분\n"
            "• 4개 기간(2Y/1Y/6M/3M) 모델 동시 학습\n"
            "• 기본 모드(체크 해제): 캐시된 모델 사용 (빠르지만 재학습 없음)",
        )
        if use_retrain:
            st.info(
                "✅ 전체 재학습 모드 활성화\n"
                "• 2년 전~7일 전: 캐시 로드 (20분 절약)\n"
                "• 최근 7일: 신규 수집 (1분)\n"
                "• 4개 기간별 모델 학습 (4분)\n"
                "• 예상 소요 시간: 약 6분"
            )
        else:
            st.info(
                "기본 모드: 캐시된 모델 사용\n"
                "• 빠른 실행 (재학습 없음)\n"
                "• 이전 학습 결과 활용"
            )

        st.markdown("---")
        _gpu_val = st.session_state.get("use_gpu_surge", False)
        use_gpu = st.checkbox(
            "🖥️ GPU 사용 최대화",
            value=_gpu_val,
            help="LightGBM GPU 학습 모드 활성화\n"
            "• NVIDIA GPU + CUDA 드라이버 필요\n"
            "• lightgbm --install-option=--gpu 빌드 필요\n"
            "• 대규모 데이터에서 학습 속도 2~5배 향상\n"
            "• GPU 미지원 환경에서는 자동으로 CPU 폴백",
        )
        if use_gpu:
            st.info(
                "🖥️ GPU 학습 모드 활성화\n"
                "• device_type=gpu로 LightGBM 실행\n"
                "• 단정밀도(FP32) 사용으로 속도 극대화\n"
                "• GPU 미설치 시 에러 → CPU로 자동 전환 안 됨"
            )

        st.markdown("---")
        st.markdown("**라벨링 설정 (Triple-Barrier)**")
        _col_a, _col_b = st.columns(2)
        with _col_a:
            _cur_hw = st.session_state.get("label_horizon_days", 20) // 5
            _hw_options = [3, 4, 6, 8, 10, 12]
            _hw_idx = _hw_options.index(_cur_hw) if _cur_hw in _hw_options else 0
            horizon_weeks = st.selectbox(
                "급등판정기간 (horizon_days)",
                _hw_options,
                index=_hw_idx,
                format_func=lambda w: f"{w}주 ({w * 5}영업일)",
                help="이 기간 내에 목표수익률 도달 여부로 라벨링",
            )
        with _col_b:
            target_pct_selected = st.multiselect(
                "학습 목표수익률",
                options=[25, 30],
                default=st.session_state.get("label_target_pct_list", [25, 30]),
                format_func=lambda x: f"{x}%",
                help="25%, 30%, 또는 둘 다 선택 가능",
            )

        from core.ml_pattern import compute_recommended_stop

        _rep_target = max(target_pct_selected) if target_pct_selected else 30
        _rec = compute_recommended_stop(horizon_weeks, _rep_target)
        # horizon / target 변경 시 stop_pct를 추천값으로 자동 변경
        _prev_hw = st.session_state.get("_dlg_prev_hw", horizon_weeks)
        _prev_tp = st.session_state.get("_dlg_prev_tp", _rep_target)
        if horizon_weeks != _prev_hw or _rep_target != _prev_tp:
            st.session_state["_dlg_stop_input"] = _rec
        st.session_state["_dlg_prev_hw"] = horizon_weeks
        st.session_state["_dlg_prev_tp"] = _rep_target
        # key가 session_state에 이미 있으면 value 생략 (Streamlit 충돌 방지)
        _stop_kw = {
            "label": f"라벨손절기준 (stop_pct) % — 추천: {_rec}%",
            "min_value": 8,
            "max_value": 20,
            "key": "_dlg_stop_input",
            "help": "이 비율 이상 하락 시 'Hard Negative'로 강한 부정 라벨 (1.5배 가중)",
        }
        if "_dlg_stop_input" not in st.session_state:
            _stop_kw["value"] = int(st.session_state.get("label_stop_pct", 0.10) * 100)
        stop_pct_ui = st.number_input(**_stop_kw)

        st.markdown("---")
        with st.expander("📊 AI 모델 하이퍼파라미터 (현재 설정)", expanded=False):
            st.markdown(
                "| 설정 | 값 | 설명 |\n"
                "|------|-----|------|\n"
                "| **나뭇잎 수** (num_leaves) | 63 | "
                "트리 한 그루의 복잡도. 클수록 세밀하지만 과적합 위험 |\n"
                "| **최소 잎 데이터** (min_data_in_leaf) | 50 | "
                "판단 근거 최소 샘플 수. 높을수록 안정적 |\n"
                "| **학습 속도** (learning_rate) | 0.03 | "
                "한 번에 배우는 양. 작을수록 정밀하지만 느림 |\n"
                "| **피처 사용률** (feature_fraction) | 80% | "
                "매 트리마다 80%의 지표만 랜덤 사용 → 과적합 방지 |\n"
                "| **데이터 사용률** (bagging_fraction) | 80% | "
                "매 5회마다 80% 데이터 랜덤 추출 → 일반화 향상 |\n"
                "| **학습 라운드** (num_boost_round) | 1000 | "
                "트리를 최대 1000그루까지 쌓음 |\n"
                "| **조기 종료** (early_stopping) | 50 | "
                "50라운드 동안 개선 없으면 자동 중단 → 과적합 방지 |\n"
                "| **검증 방식** | Walk-Forward 4단계 | "
                "과거→미래 순서로 4번 검증. 미래 데이터 누수 차단 |\n"
                "| **Hard Negative 가중치** | 1.5배 | "
                "크게 하락한 종목을 1.5배 더 강하게 학습 |"
            )
            st.caption(
                "기간별 조정: "
                "2Y/1Y → 기본값 그대로 | "
                "6M → 나뭇잎 47, 최소잎 40 | "
                "3M → 나뭇잎 31, 최소잎 30 (데이터가 적어 단순화)"
            )

        st.markdown("---")
        st.markdown("**ETF 학습 설정**")
        _ecol_a, _ecol_b = st.columns(2)
        with _ecol_a:
            etf_target_selected = st.multiselect(
                "ETF 목표수익률",
                options=[8, 10, 15],
                default=st.session_state.get("etf_target_pct_list", [10, 15]),
                format_func=lambda x: f"{x}%",
                help="ETF는 주식보다 변동성이 낮아 10~15% 권장",
            )
        with _ecol_b:
            _etf_hw_cur = st.session_state.get("etf_horizon_days", 20) // 5
            _etf_hw_opts = [3, 4, 5, 6, 8]
            _etf_hw_idx = (
                _etf_hw_opts.index(_etf_hw_cur) if _etf_hw_cur in _etf_hw_opts else 1
            )
            etf_horizon_weeks = st.selectbox(
                "ETF 급등판정기간",
                _etf_hw_opts,
                index=_etf_hw_idx,
                format_func=lambda w: f"{w}주 ({w * 5}영업일)",
                help="ETF는 주식보다 느리게 움직여 4~6주 권장",
            )
        _etf_stop_kw = {
            "label": "ETF 손절기준 (%)",
            "min_value": 3,
            "max_value": 15,
            "key": "_dlg_etf_stop_input",
            "help": "ETF 하락 손절 기준 (기본 5%)",
        }
        if "_dlg_etf_stop_input" not in st.session_state:
            _etf_stop_kw["value"] = int(
                st.session_state.get("etf_stop_pct", 0.05) * 100
            )
        etf_stop_pct_ui = st.number_input(**_etf_stop_kw)

        if not target_pct_selected:
            st.warning("최소 1개 목표수익률을 선택하세요")
        elif st.button("적용", type="primary", use_container_width=True):
            st.session_state["use_full_retrain_surge"] = use_retrain
            st.session_state["use_gpu_surge"] = use_gpu
            st.session_state["label_horizon_days"] = horizon_weeks * 5
            st.session_state["label_target_pct_list"] = sorted(target_pct_selected)
            # 하위 호환: 백테스트 페이지용 단일 값
            st.session_state["label_target_pct"] = target_pct_selected[0] / 100.0
            st.session_state["label_stop_pct"] = stop_pct_ui / 100.0
            # ETF 설정
            st.session_state["etf_target_pct_list"] = sorted(etf_target_selected)
            st.session_state["etf_horizon_days"] = etf_horizon_weeks * 5
            st.session_state["etf_stop_pct"] = etf_stop_pct_ui / 100.0
            st.rerun()

    # 목표수익률별 AI 버튼 행
    for _tp_btn in sorted(_tp_list):
        _ai_presets = AI_PRESETS.get(_tp_btn, [])
        if not _ai_presets:
            continue
        cols_ai = st.columns(4)
        for i, (name, icon, desc) in enumerate(_ai_presets):
            with cols_ai[i]:
                n = preset_results.get(name, 0)
                sel = st.session_state["selected_view"] == name
                st.markdown('<div class="ai-green"></div>', unsafe_allow_html=True)
                if st.button(
                    f"{icon} {name} ({n})",
                    key=f"v_{name}",
                    type="primary" if sel else "secondary",
                    use_container_width=True,
                    disabled=(n == 0),
                ):
                    st.session_state["selected_view"] = name
                    st.rerun()
                st.caption(desc)

    # 현재 학습 목표 파라미터 표시
    _disp_hw = st.session_state.get("label_horizon_days", 20)
    _disp_tp = st.session_state.get("label_target_pct_list", [25, 30])
    _disp_sp = st.session_state.get("label_stop_pct", 0.10)
    st.info(
        f"학습 목표: **목표수익률 {'/'.join(f'{t}%' for t in _disp_tp)}** | "
        f"**목표기간 {_disp_hw // 5}주 ({_disp_hw}영업일)** | "
        f"**손절 {_disp_sp * 100:.0f}%**"
    )

    # AI 급등주학습 + 고급옵션 아이콘
    _sub1, _sub2, _sub3 = st.columns([2, 1, 5])
    with _sub1:
        st.markdown('<div class="ai-green"></div>', unsafe_allow_html=True)
        train_surge = st.button(
            _surge_label, key="train_surge", use_container_width=True
        )
    with _sub2:
        st.markdown('<div class="ai-green"></div>', unsafe_allow_html=True)
        if st.button("⚙️", key="ai_settings_btn", use_container_width=True):
            _ai_settings_dialog()

    # ── ETF 추천 프리셋 버튼 (빨간색) ──
    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
    _etf_tp_list = st.session_state.get("etf_target_pct_list", [10, 15])
    _any_etf = any(
        st.session_state.get(f"ai_model_{m}_t{tp}")
        or _load_ai(mode=m, target_pct=tp / 100.0)
        for tp in _etf_tp_list
        for m in ("etf_surge", "etf_surge_1y", "etf_surge_6m", "etf_surge_3m")
    )
    for _tp_btn in sorted(_etf_tp_list):
        _etf_presets = ETF_PRESETS.get(_tp_btn, [])
        if not _etf_presets:
            continue
        cols_etf = st.columns(4)
        for i, (name, icon, desc) in enumerate(_etf_presets):
            with cols_etf[i]:
                n = preset_results.get(name, 0)
                sel = st.session_state["selected_view"] == name
                st.markdown('<div class="etf-red"></div>', unsafe_allow_html=True)
                if st.button(
                    f"{icon} {name} ({n})",
                    key=f"v_{name}",
                    type="primary" if sel else "secondary",
                    use_container_width=True,
                    disabled=(n == 0),
                ):
                    st.session_state["selected_view"] = name
                    st.rerun()
                st.caption(desc)

    # AI ETF학습 버튼 + 통합학습 버튼
    _etf_label = "🔄 AI ETF학습 (업데이트)" if _any_etf else "📦 AI ETF학습 (최초)"
    _esub1, _esub2, _esub3 = st.columns([2, 2, 4])
    with _esub1:
        st.markdown('<div class="etf-red"></div>', unsafe_allow_html=True)
        train_etf = st.button(_etf_label, key="train_etf", use_container_width=True)
    with _esub2:
        st.markdown('<div class="ai-purple"></div>', unsafe_allow_html=True)
        _all_label = (
            "🔄 AI 통합학습 (업데이트)"
            if (_any_surge or _any_etf)
            else "🚀 AI 통합학습 (최초)"
        )
        train_all = st.button(_all_label, key="train_all", use_container_width=True)

    # ── 급등주 멀티기간 학습 실행 (목표수익률별 순차) ──
    if train_surge or train_all:
        from core.ml_pattern import get_learned_patterns
        from core.ml_pattern import predict_current_surge as _predict_surge
        from core.ml_pattern import save_model as _save_ai
        from core.ml_pattern import train_multi_period_surge as _train_multi
        from core.ml_pattern import train_multi_period_with_cache as _train_multi_cached

        _use_cache = st.session_state.get("use_full_retrain_surge", False)
        _use_gpu = st.session_state.get("use_gpu_surge", False)
        _gpu_tag = " +GPU" if _use_gpu else ""
        _tp_desc = "/".join(f"{t}%" for t in _tp_list)
        _status_label = (
            f"🚀 AI 급등주 학습 중 ({_tp_desc}, 캐싱{_gpu_tag})..."
            if _use_cache
            else f"🚀 AI 급등주 학습 중 ({_tp_desc}{_gpu_tag})..."
        )

        with st.status(_status_label, expanded=True) as ai_status:
            ai_progress = st.progress(0)
            _universe = st.session_state.get("universe")
            if _universe is None:
                ai_status.update(
                    label="❌ universe 데이터 없음 (스크리닝 먼저 실행)", state="error"
                )
            else:
                try:

                    def _ai_cb_surge(msg, pct=None):
                        st.write(msg)
                        if pct is not None:
                            ai_progress.progress(min(int(pct), 100))

                    _horizon = st.session_state.get("label_horizon_days", 20)
                    _stop = st.session_state.get("label_stop_pct", 0.10)
                    if _use_gpu:
                        st.write("🖥️ GPU 학습 모드 활성화")

                    _all_trained = {}  # {(tp_int, mode): model}
                    _n_tp = len(_tp_list)

                    for _ti, _tp_int in enumerate(_tp_list):
                        _tp_f = _tp_int / 100.0
                        st.write(f"📊 [{_ti + 1}/{_n_tp}] 목표수익률 {_tp_int}% 학습")
                        st.write(
                            f"라벨링: {_horizon}일 / 목표 {_tp_int}% / "
                            f"손절 {_stop * 100:.0f}%"
                        )
                        if _use_cache:
                            _models = _train_multi_cached(
                                _universe,
                                date_str,
                                _ai_cb_surge,
                                use_gpu=_use_gpu,
                                horizon_days=_horizon,
                                target_pct=_tp_f,
                                stop_pct=_stop,
                            )
                        else:
                            _models = _train_multi(
                                _universe,
                                date_str,
                                _ai_cb_surge,
                                use_gpu=_use_gpu,
                                horizon_days=_horizon,
                                target_pct=_tp_f,
                                stop_pct=_stop,
                            )

                        for _mode, _model in _models.items():
                            _key = f"{_mode}_t{_tp_int}"
                            _save_ai(_model, date_str, mode=_mode, target_pct=_tp_f)
                            st.session_state[f"ai_model_{_key}"] = _model
                            _all_trained[(_tp_int, _mode)] = _model
                            _period = _model.get("period", _mode)
                            _auc = _model.get("validation_auc", 0)
                            st.write(f"  ✅ {_tp_int}% {_period}: AUC={_auc:.3f}")

                        # 2Y 모델 패턴 요약
                        if "surge" in _models:
                            patterns = get_learned_patterns(_models["surge"])
                            st.code(patterns["description"])

                    # 모든 모델 예측 적용
                    _screened = st.session_state.get("screened")
                    _ohlcv_dict = st.session_state.get("candidates_ohlcv_100d")
                    if _screened is not None and _ohlcv_dict is not None:
                        st.write("AI 기간별 급등주 예측 적용 중...")
                        from core.config import get_categories_config as _get_cfg

                        _cfg = _get_cfg(target_pct_list=_tp_list)
                        _kospi_close = None
                        try:
                            from core.data_fetcher import fetch_kospi_index_3year

                            _kospi_df = fetch_kospi_index_3year(date_str)
                            if _kospi_df is not None and len(_kospi_df) > 0:
                                _kospi_close = _kospi_df["종가"].astype(float)
                        except Exception:
                            pass

                        _mcap = _screened.get(
                            "MarketCap", pd.Series(0, index=_screened.index)
                        )
                        pr = st.session_state.get("preset_results", {})

                        for (_tp_int, _mode), _model in _all_trained.items():
                            ai_cat = f"AI_{_tp_int}%급등주"
                            _suffix_map = {
                                "surge_1y": "_1Y",
                                "surge_6m": "_6M",
                                "surge_3m": "_3M",
                            }
                            if _mode in _suffix_map:
                                ai_cat += _suffix_map[_mode]

                            _pf = _cfg.get(ai_cat, {}).get("ml_pre_filter", {})
                            _min_m = _pf.get("min_mcap", 100_000_000_000)
                            _eligible = _mcap >= _min_m
                            _ai_tickers = _screened[_eligible].index.tolist()
                            _mcap_lbl = f"{_min_m / 100_000_000:,.0f}억"
                            st.write(
                                f"  {ai_cat}: {len(_ai_tickers)}종목 "
                                f"(시총≥{_mcap_lbl})"
                            )

                            _ai_result = _predict_surge(
                                _model,
                                _ohlcv_dict,
                                _ai_tickers,
                                kospi_close=_kospi_close,
                            )

                            _screened[f"Pass_{ai_cat}"] = False
                            _screened[f"Score_{ai_cat}"] = 0.0
                            _screened[f"Timing_{ai_cat}"] = ""
                            _screened[f"Reason_{ai_cat}"] = ""
                            for t in _ai_tickers:
                                if (
                                    t in _ai_result.index
                                    and _ai_result.at[t, "AI_Pass"]
                                ):
                                    _screened.at[t, f"Pass_{ai_cat}"] = True
                                    _screened.at[t, f"Score_{ai_cat}"] = float(
                                        _ai_result.at[t, "AI_Score"]
                                    )
                                    _screened.at[t, f"Timing_{ai_cat}"] = str(
                                        _ai_result.at[t, "AI_Timing"]
                                    )
                                    _screened.at[t, f"Reason_{ai_cat}"] = str(
                                        _ai_result.at[t, "AI_Reason"]
                                    )

                            n_ai = int(_screened[f"Pass_{ai_cat}"].sum())
                            pr[ai_cat] = n_ai
                            _period = _model.get("period", _mode)
                            st.write(f"  ✅ {ai_cat} ({_period}): **{n_ai}종목**")

                        pass_cols = [
                            c
                            for c in _screened.columns
                            if c.startswith("Pass_") and c != "Pass_any"
                        ]
                        _screened["Pass_any"] = _screened[pass_cols].any(axis=1)

                        st.session_state["screened"] = _screened
                        st.session_state["preset_results"] = pr
                        _screened.to_pickle(screening_cache_path)

                    _total_models = len(_all_trained)
                    ai_progress.progress(100)
                    ai_status.update(
                        label=f"✅ 멀티기간 학습 완료! ({_total_models}개 모델)",
                        state="complete",
                    )
                    if not train_all:
                        st.rerun()
                except Exception as e:
                    ai_status.update(
                        label=f"❌ 급등주 멀티기간 학습 실패: {e}", state="error"
                    )
                    import traceback

                    st.error(traceback.format_exc())
                    if not train_all:
                        pass  # 개별 학습은 에러 후 중단
                    # 통합학습은 에러 나도 ETF로 계속

    # ── ETF 멀티기간 학습 실행 ──
    if train_etf or train_all:
        from core.data_fetcher import fetch_etf_market_cap as _fetch_etf_mcap
        from core.data_fetcher import fetch_etf_ohlcv_3year as _fetch_etf_ohlcv
        from core.data_fetcher import fetch_etf_ohlcv_all as _fetch_etf_all
        from core.data_fetcher import fetch_etf_tickers as _fetch_etf_tickers
        from core.ml_pattern import predict_current_etf_surge as _predict_etf
        from core.ml_pattern import save_model as _save_ai_etf
        from core.ml_pattern import train_multi_period_etf_surge as _train_etf_multi

        _etf_tp_desc = "/".join(f"{t}%" for t in _etf_tp_list)

        with st.status(
            f"📦 AI ETF 학습 중 ({_etf_tp_desc})...", expanded=True
        ) as etf_status:
            etf_progress = st.progress(0)
            try:

                def _etf_cb(msg, pct=None):
                    st.write(msg)
                    if pct is not None:
                        etf_progress.progress(min(int(pct), 100))

                # Step 1: ETF 티커 수집
                _etf_cb("ETF 티커 목록 수집 중...", 1)
                _etf_tickers_df = _fetch_etf_tickers(date_str)
                _etf_tickers = _etf_tickers_df["Ticker"].tolist()
                _etf_names = dict(
                    zip(
                        _etf_tickers_df["Ticker"], _etf_tickers_df["Name"], strict=False
                    )
                )
                st.session_state["etf_names"] = _etf_names
                _etf_cb(f"ETF 총 {len(_etf_tickers)}종목", 3)

                # Step 2: 거래대금 기준 필터 (≥5천만)
                _etf_ohlcv_today = _fetch_etf_all(date_str)
                if _etf_ohlcv_today is not None and not _etf_ohlcv_today.empty:
                    # ETF 시가총액 조인
                    try:
                        _etf_mcap_df = _fetch_etf_mcap(date_str)
                        if (
                            _etf_mcap_df is not None
                            and "시가총액" in _etf_mcap_df.columns
                        ):
                            _etf_ohlcv_today = _etf_ohlcv_today.join(
                                _etf_mcap_df[["시가총액"]], how="left"
                            )
                    except Exception:
                        pass
                    st.session_state["etf_ohlcv_today"] = _etf_ohlcv_today
                    _tv = _etf_ohlcv_today.get("거래대금", pd.Series(dtype=float))
                    _active_set = set(_tv[_tv >= 50_000_000].index.tolist())
                    _etf_tickers = [t for t in _etf_tickers if t in _active_set]
                _etf_cb(f"활성 ETF: {len(_etf_tickers)}종목 (거래대금≥5천만)", 5)

                # KOSPI 지수
                _kospi_close_etf = None
                try:
                    from core.data_fetcher import fetch_kospi_index_3year

                    _kospi_df_etf = fetch_kospi_index_3year(date_str)
                    if _kospi_df_etf is not None and len(_kospi_df_etf) > 0:
                        _kospi_close_etf = _kospi_df_etf["종가"].astype(float)
                except Exception:
                    pass

                _etf_horizon = st.session_state.get("etf_horizon_days", 20)
                _etf_stop = st.session_state.get("etf_stop_pct", 0.05)

                _all_etf_trained = {}
                for _ti, _tp_int in enumerate(_etf_tp_list):
                    _tp_f = _tp_int / 100.0
                    st.write(
                        f"📦 [{_ti + 1}/{len(_etf_tp_list)}] "
                        f"ETF 목표수익률 {_tp_int}% 학습"
                    )
                    st.write(
                        f"라벨링: {_etf_horizon}일 / 목표 {_tp_int}% / "
                        f"손절 {_etf_stop * 100:.0f}%"
                    )
                    _models = _train_etf_multi(
                        _etf_tickers,
                        date_str,
                        _etf_cb,
                        use_gpu=False,
                        horizon_days=_etf_horizon,
                        target_pct=_tp_f,
                        stop_pct=_etf_stop,
                    )
                    for _mode, _model in _models.items():
                        _key = f"{_mode}_t{_tp_int}"
                        _save_ai_etf(_model, date_str, mode=_mode, target_pct=_tp_f)
                        st.session_state[f"ai_model_{_key}"] = _model
                        _all_etf_trained[(_tp_int, _mode)] = _model
                        _period = _model.get("period", _mode)
                        _auc = _model.get("validation_auc", 0)
                        st.write(f"  ✅ ETF {_tp_int}% {_period}: AUC={_auc:.3f}")

                # Step 3: ETF OHLCV 수집 + 예측 적용
                _etf_target = _etf_tickers[:300]
                _etf_total = len(_etf_target)
                _etf_cb(f"ETF OHLCV 수집 중... (0/{_etf_total})", 85)
                _etf_ohlcv_dict = {}
                for _ei, t in enumerate(_etf_target):
                    if (_ei + 1) % 20 == 0 or _ei == _etf_total - 1:
                        _pct = 85 + int(5 * (_ei + 1) / _etf_total)
                        _etf_cb(f"ETF OHLCV 수집 중... ({_ei + 1}/{_etf_total})", _pct)
                    try:
                        _ohlcv = _fetch_etf_ohlcv(t, date_str)
                        if _ohlcv is not None and len(_ohlcv) >= 60:
                            _etf_ohlcv_dict[t] = _ohlcv
                    except Exception:
                        continue
                st.session_state["etf_ohlcv_dict"] = _etf_ohlcv_dict
                _etf_cb(f"ETF OHLCV: {len(_etf_ohlcv_dict)}종목", 90)

                pr = st.session_state.get("preset_results", {})
                etf_screened = {}

                _suffix_map = {
                    "etf_surge_1y": "_1Y",
                    "etf_surge_6m": "_6M",
                    "etf_surge_3m": "_3M",
                }
                for (_tp_int, _mode), _model in _all_etf_trained.items():
                    etf_cat = f"AI_{_tp_int}%추천ETF"
                    if _mode in _suffix_map:
                        etf_cat += _suffix_map[_mode]

                    _ai_result = _predict_etf(
                        _model,
                        _etf_ohlcv_dict,
                        list(_etf_ohlcv_dict.keys()),
                        kospi_close=_kospi_close_etf,
                    )

                    etf_screened[etf_cat] = _ai_result
                    n_pass = (
                        int(_ai_result["AI_Pass"].sum()) if not _ai_result.empty else 0
                    )
                    pr[etf_cat] = n_pass
                    st.write(f"  ✅ {etf_cat}: **{n_pass}종목**")

                st.session_state["etf_screened"] = etf_screened
                st.session_state["preset_results"] = pr

                _total_etf = len(_all_etf_trained)
                etf_progress.progress(100)
                etf_status.update(
                    label=f"✅ ETF 학습 완료! ({_total_etf}개 모델)",
                    state="complete",
                )
                st.rerun()
            except Exception as e:
                _err_msg = str(e)
                if "비거래일" in _err_msg or "캐시에도" in _err_msg:
                    etf_status.update(
                        label="❌ ETF 학습 실패: KRX API 비거래일 오류",
                        state="error",
                    )
                    st.error(
                        f"**KRX API 비거래일(주말/공휴일) 오류**\n\n"
                        f"{_err_msg}\n\n"
                        f"💡 **해결 방법**: 평일 장 마감 후(16:00 이후) "
                        f"데이터 수집 페이지에서 먼저 데이터를 수집한 뒤 "
                        f"ETF 학습을 실행하세요."
                    )
                else:
                    etf_status.update(label=f"❌ ETF 학습 실패: {e}", state="error")
                    import traceback

                    st.error(traceback.format_exc())

    st.markdown("")  # 행 간 여백

    # ── 타이밍 필터 버튼 ──
    cols_tf = st.columns(3)
    _special_views = [
        ("즉시매수", f"🔴 즉시매수 ({n_strong})", n_strong),
        ("매수", f"🟢 매수 ({n_buy})", n_buy),
        ("관심", f"🟡 관심 ({n_watch})", n_watch),
    ]
    for i, (vname, vlabel, vn) in enumerate(_special_views):
        with cols_tf[i]:
            sel = st.session_state["selected_view"] == vname
            if st.button(
                vlabel,
                key=f"v_{vname}",
                type="primary" if sel else "secondary",
                use_container_width=True,
                disabled=(vn == 0),
            ):
                st.session_state["selected_view"] = vname
                st.rerun()

    # ── 선택된 보기에 따른 필터링 ──
    view = st.session_state["selected_view"]

    if view in ("전체", "즉시매수", "매수", "관심"):
        if view == "전체":
            filtered = df[df["Pass_any"]].copy()
        elif view == "즉시매수":
            filtered = df[(df["Pass_any"]) & (df["_TimingRank"] == 3)].copy()
        elif view == "매수":
            filtered = df[(df["Pass_any"]) & (df["_TimingRank"] == 2)].copy()
        else:
            filtered = df[(df["Pass_any"]) & (df["_TimingRank"] == 1)].copy()
        score_cols = [f"Score_{name}" for name, _, _ in _get_preset_list()]
        avail_score = [c for c in score_cols if c in filtered.columns]
        if avail_score:
            filtered["_BestScore"] = filtered[avail_score].max(axis=1)
        else:
            filtered["_BestScore"] = 0
    elif view in ETF_PERIOD_MODES:
        # ── ETF 뷰: etf_screened에서 데이터 로드 ──
        _etf_data = st.session_state.get("etf_screened", {}).get(view)
        if _etf_data is not None and not _etf_data.empty:
            filtered = _etf_data[_etf_data["AI_Pass"]].copy()
            filtered["_BestScore"] = filtered["AI_Score"]
            _etf_nm = st.session_state.get("etf_names", {})
            filtered["Name"] = filtered.index.map(lambda t: _etf_nm.get(t, t))
            filtered["Market"] = "ETF"
            filtered["_TimingRank"] = (
                filtered["AI_Timing"]
                .map({"즉시매수": 3, "매수": 2, "관심": 1})
                .fillna(0)
                .astype(int)
            )
        else:
            filtered = pd.DataFrame()
    else:
        pass_key = f"Pass_{view}"
        score_key = f"Score_{view}"
        if pass_key in df.columns:
            filtered = df[df[pass_key]].copy()
            filtered["_BestScore"] = (
                filtered[score_key] if score_key in filtered.columns else 0
            )
        else:
            filtered = df[df["Pass_any"]].copy()
            filtered["_BestScore"] = 0

    # ── ETF 뷰인지 판별 ──
    _is_etf_view = view in ETF_PERIOD_MODES

    # ── 정렬 ──
    sort_options = {
        "_BestScore": "점수 (높은순)",
        "_TimingRank": "타이밍 (즉시매수 우선)",
    }
    if not _is_etf_view:
        sort_options["PBR"] = "PBR (낮은순)"
    if "MarketCap" in filtered.columns:
        sort_options["MarketCap"] = "시가총액 (높은순)"
    if "DIV" in filtered.columns:
        sort_options["DIV"] = "배당률 (높은순)"
    sort_by = st.selectbox(
        "정렬", list(sort_options.keys()), format_func=lambda x: sort_options[x]
    )
    ascending = sort_by == "PBR" if not _is_etf_view else False
    if not filtered.empty:
        filtered = filtered.sort_values(sort_by, ascending=ascending)

    # ── 메타 정보 배너 ──
    _meta_parts_p2 = []
    _meta_parts_p2.append(f"시총 {min_mcap_billion:,}억 이상")
    _p2_hw = st.session_state.get("label_horizon_days", 20)
    _p2_tp = st.session_state.get("label_target_pct_list", [25, 30])
    _p2_sp = st.session_state.get("label_stop_pct", 0.10)
    _p2_tp_str = "/".join(f"{t}%" for t in _p2_tp)
    _p2_hw_w = _p2_hw // 5
    _meta_parts_p2.append(f"AI급등주 목표 {_p2_tp_str}/{_p2_hw_w}주")
    _meta_parts_p2.append(f"손절 {_p2_sp * 100:.0f}%")
    _p2_date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    _meta_parts_p2.append(f"{_p2_date_fmt} 종가 기준")
    st.caption(f"📋 {' | '.join(_meta_parts_p2)}")

    st.caption(f"**{len(filtered)}{'ETF' if _is_etf_view else '종목'}** 표시")

    if _is_etf_view:
        # ── ETF 전용 테이블 표시 ──
        if not filtered.empty:
            filtered["매수타이밍"] = filtered["_TimingRank"].map(
                {3: "🔴 즉시매수", 2: "🟢 매수", 1: "🟡 관심", 0: "-"}
            )
            filtered["Naver"] = filtered.index.map(
                lambda t: f"https://finance.naver.com/item/main.naver?code={t}"
            )

            # 시총(억), 거래대금(억), 현재가 조인
            _etf_today = st.session_state.get("etf_ohlcv_today")
            _etf_ohlcv_d = st.session_state.get("etf_ohlcv_dict", {})
            if _etf_today is not None and not _etf_today.empty:
                if "시가총액" in _etf_today.columns:
                    filtered["시총(억)"] = filtered.index.map(
                        lambda t: round(_etf_today.at[t, "시가총액"] / 1e8)
                        if t in _etf_today.index
                        and pd.notna(_etf_today.at[t, "시가총액"])
                        else 0
                    )
                if "거래대금" in _etf_today.columns:
                    filtered["거래대금(억)"] = filtered.index.map(
                        lambda t: round(_etf_today.at[t, "거래대금"] / 1e8, 1)
                        if t in _etf_today.index
                        and pd.notna(_etf_today.at[t, "거래대금"])
                        else 0
                    )
                if "종가" in _etf_today.columns:
                    filtered["현재가"] = filtered.index.map(
                        lambda t: int(_etf_today.at[t, "종가"])
                        if t in _etf_today.index and pd.notna(_etf_today.at[t, "종가"])
                        else 0
                    )
            elif _etf_ohlcv_d:
                # 당일 데이터 없으면 3년 OHLCV 마지막 종가
                filtered["현재가"] = filtered.index.map(
                    lambda t: int(_etf_ohlcv_d[t]["종가"].astype(float).iloc[-1])
                    if t in _etf_ohlcv_d and len(_etf_ohlcv_d[t]) > 0
                    else 0
                )

            _etf_display = [
                "Name",
                "Naver",
                "Market",
                "시총(억)",
                "현재가",
                "거래대금(억)",
                "_BestScore",
                "매수타이밍",
                "AI_Timing",
                "AI_Reason",
            ]
            _etf_display = [c for c in _etf_display if c in filtered.columns]
            _etf_rename = {
                "Name": "ETF명",
                "Naver": "Naver",
                "Market": "시장",
                "시총(억)": "시총(억)",
                "현재가": "현재가",
                "거래대금(억)": "거래대금(억)",
                "_BestScore": "점수",
                "매수타이밍": "타이밍",
                "AI_Timing": "AI판정",
                "AI_Reason": "근거",
            }
            st.dataframe(
                filtered[_etf_display].rename(
                    columns={c: _etf_rename.get(c, c) for c in _etf_display}
                ),
                column_config={
                    "Naver": st.column_config.LinkColumn(
                        "Naver", display_text="📊 Naver"
                    ),
                },
                use_container_width=True,
                height=min(500, 35 * len(filtered) + 38),
            )
    else:
        # ── 통과 전략 태그 + 타이밍 뱃지 ──
        def _strategy_tags(row):
            tags = []
            for name, icon, _ in _get_preset_list():
                if name in ETF_PERIOD_MODES:
                    continue  # ETF 프리셋은 주식 전략 태그에서 제외
                if row.get(f"Pass_{name}", False):
                    timing = row.get(f"Timing_{name}", "")
                    badge = TIMING_BADGE.get(timing, "")
                    label = f"{icon}{name}"
                    if badge:
                        label += f" {badge}"
                    tags.append(label)
            return " · ".join(tags) if tags else "-"

        if not filtered.empty:
            filtered["전략"] = filtered.apply(_strategy_tags, axis=1)
            filtered["매수타이밍"] = filtered["_TimingRank"].map(
                {3: "🔴 즉시매수", 2: "🟢 매수", 1: "🟡 관심", 0: "-"}
            )

        # ── 테이블 ──
        # Daum 종목 링크 컬럼 생성
        if not filtered.empty:
            filtered["Naver"] = filtered.index.map(
                lambda t: f"https://finance.naver.com/item/main.naver?code={t}"
            )

        # 시가총액(조) 컬럼 추가: 1조=1.0, 1000억=0.1
        if "MarketCap" in filtered.columns:
            filtered = filtered.copy()
            filtered["MarketCap_T"] = (
                filtered["MarketCap"].fillna(0).astype(float) / 1e12
            ).round(2)

        display_cols = [
            "Name",
            "Naver",
            "Market",
            "MarketCap_T",
            "Close",
            "PBR",
            "PER",
            "EPS",
        ]
        if "ROE" in filtered.columns:
            display_cols.append("ROE")
        if "DIV" in filtered.columns:
            display_cols.append("DIV")
        if "CashRatio" in filtered.columns:
            display_cols.append("CashRatio")
        display_cols += ["_BestScore", "매수타이밍", "전략"]

        display_cols = [c for c in display_cols if c in filtered.columns]

        col_rename = {
            "Name": "종목명",
            "Naver": "Naver",
            "Market": "시장",
            "MarketCap_T": "시총(조)",
            "Close": "현재가",
            "PBR": "PBR",
            "PER": "PER",
            "EPS": "EPS",
            "ROE": "ROE",
            "DIV": "배당률",
            "_BestScore": "점수",
            "CashRatio": "현금비율",
            "매수타이밍": "타이밍",
            "전략": "통과전략",
            "MarketCap": "시가총액",
        }

        if display_cols and not filtered.empty:
            st.dataframe(
                filtered[display_cols].rename(
                    columns={c: col_rename.get(c, c) for c in display_cols}
                ),
                column_config={
                    "Naver": st.column_config.LinkColumn(
                        "Naver", display_text="📊 Naver"
                    ),
                },
                use_container_width=True,
                height=min(500, 35 * len(filtered) + 38),
            )

    # 분석 차트
    chart_tabs = st.tabs(["전략별 분포", "업종 분포", "점수 분포"])

    with chart_tabs[0]:
        if not filtered.empty:
            preset_counts = {}
            for name, icon, _ in _get_preset_list():
                key = f"Pass_{name}"
                if key in filtered.columns:
                    preset_counts[f"{icon}{name}"] = int(filtered[key].sum())
            if preset_counts:
                st.bar_chart(pd.Series(preset_counts))

    with chart_tabs[1]:
        if "Sector" in filtered.columns and not filtered.empty:
            st.bar_chart(filtered["Sector"].value_counts().head(15))

    with chart_tabs[2]:
        if "_BestScore" in filtered.columns and not filtered.empty:
            score_data = filtered["_BestScore"].dropna()
            if not score_data.empty:
                hist_vals = np.histogram(score_data, bins=10)
                chart_df = pd.DataFrame(
                    {
                        "점수구간": [
                            f"{int(hist_vals[1][i])}-{int(hist_vals[1][i+1])}"
                            for i in range(len(hist_vals[0]))
                        ],
                        "종목수": hist_vals[0],
                    }
                ).set_index("점수구간")
                st.bar_chart(chart_df)

    # 다운로드
    st.markdown("---")
    dc1, dc2 = st.columns(2)
    with dc1:
        cfg_default = get_default_config()
        # AI급등주 모델 설정 병합
        cfg_default["AI급등주_시총필터(억)"] = min_mcap_billion
        cfg_default["AI급등주_목표수익률(%)"] = _p2_tp
        cfg_default["AI급등주_목표기간(주)"] = _p2_hw_w
        cfg_default["AI급등주_손절(%)"] = round(_p2_sp * 100)
        cfg_default["종가기준일"] = date_str
        if st.button("📥 Excel 리포트", use_container_width=True):
            with st.spinner("Excel 생성 중..."):
                filepath = generate_excel_report(df, cfg_default, date_str)
                with open(filepath, "rb") as f:
                    st.download_button(
                        "💾 다운로드",
                        data=f.read(),
                        file_name=filepath.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
    with dc2:
        _csv_meta_lines = [
            "#META:model=AI급등주",
            f"#META:min_mcap={min_mcap_billion}억",
            f"#META:target={_p2_tp_str}",
            f"#META:horizon={_p2_hw_w}주",
            f"#META:stop={_p2_sp * 100:.0f}%",
            f"#META:date={date_str}",
        ]
        csv_data = (
            "\n".join(_csv_meta_lines) + "\n" + filtered.to_csv(encoding="utf-8-sig")
        )
        st.download_button(
            "📥 CSV 다운로드",
            data=csv_data,
            file_name=f"candidates_{date_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )
