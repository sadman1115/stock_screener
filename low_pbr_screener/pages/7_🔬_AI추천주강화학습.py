"""Page 7: AI추천주 강화학습 (개발중) — AI추천주 백테스트 기반 패턴 분석 + 강화 모델 학습."""

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (  # noqa: E402
    CACHE_DIR,
    RESULT_MCAP_DEFAULT_BILLION,
    RESULT_MCAP_MIN_BILLION,
)
from core.recommend_model import (
    SURGE_FEATURE_NAMES_KR,  # noqa: E402
    load_recommend_backtest,  # noqa: E402
)
from core.recommend_reinforcement import (  # noqa: E402
    analyze_recommend_patterns,
    generate_recommend_rf_rules,
    list_recommend_rf_rule_versions,
    load_recommend_rf_backtest,
    load_recommend_rf_result,
    load_recommend_rf_rules,
    reset_recommend_rf_cache,
    rollback_recommend_rf_rules,
    run_recommend_rf_backtest,
    save_recommend_rf_backtest,
    save_recommend_rf_result,
    save_recommend_rf_rules,
    train_recommend_rf_model,
)
from core.styles import apply_mobile_styles  # noqa: E402

# ──────────────────────────────────────────────
# 렌더링 함수 (먼저 정의)
# ──────────────────────────────────────────────


def _render_pattern_analysis(pat_analysis):
    """패턴 분석 결과 렌더링."""
    stats = pat_analysis.get("stats", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("총 예측 건수", f"{stats.get('total', 0):,}")
    col2.metric(
        "고수익 종목 (+15%↑)",
        f"{stats.get('high_return_count', 0)}건",
        f"평균 {stats.get('high_return_avg', 0) * 100:+.1f}%",
    )
    col3.metric(
        "손실 종목 (-5%↓)",
        f"{stats.get('loss_count', 0)}건",
        f"평균 {stats.get('loss_avg', 0) * 100:.1f}%",
    )

    # 피처-수익률 상관관계 Top 10
    st.subheader("📊 피처-수익률 상관관계 Top 10")
    corr = pat_analysis.get("feature_return_corr", {})
    sorted_corr = sorted(corr.items(), key=lambda x: -abs(x[1]))[:10]

    if sorted_corr:
        corr_rows = []
        for feat, c in sorted_corr:
            kr_name = SURGE_FEATURE_NAMES_KR.get(feat, feat)
            direction = "양(+)" if c > 0 else "음(-)"
            corr_rows.append(
                {
                    "피처": kr_name,
                    "영문": feat,
                    "상관계수": f"{c:+.4f}",
                    "방향": direction,
                    "강도": "●" * min(5, max(1, int(abs(c) * 50))),
                }
            )
        st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

    # 고수익 패턴 vs 저수익 패턴
    col_high, col_loss = st.columns(2)

    with col_high:
        st.subheader("🟢 고수익 패턴 프로파일")
        high_profile = pat_analysis.get("high_return_profile", {})
        optimal = pat_analysis.get("optimal_ranges", {})
        if high_profile:
            diff = pat_analysis.get("profile_diff", {})
            top_diff = sorted(diff.items(), key=lambda x: -abs(x[1]))[:8]
            rows = []
            for feat, d in top_diff:
                kr = SURGE_FEATURE_NAMES_KR.get(feat, feat)
                r = optimal.get(feat, (0, 0))
                rows.append(
                    {
                        "피처": kr,
                        "고수익 평균": f"{high_profile.get(feat, 0):.4f}",
                        "최적 범위": f"[{r[0]:.3f}, {r[1]:.3f}]",
                        "차이": f"{d:+.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_loss:
        st.subheader("🔴 저수익 패턴 프로파일")
        loss_profile = pat_analysis.get("loss_profile", {})
        if loss_profile:
            diff = pat_analysis.get("profile_diff", {})
            bottom_diff = sorted(diff.items(), key=lambda x: x[1])[:8]
            rows = []
            for feat, d in bottom_diff:
                kr = SURGE_FEATURE_NAMES_KR.get(feat, feat)
                rows.append(
                    {
                        "피처": kr,
                        "손실 평균": f"{loss_profile.get(feat, 0):.4f}",
                        "차이": f"{d:+.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _compare_results(reinforced_top20, existing_results, existing_name):
    """강화학습 vs 기존 AI추천주 비교 테이블."""
    rf_tickers = {s["ticker"] for s in reinforced_top20}

    existing_df = (
        existing_results if isinstance(existing_results, pd.DataFrame) else None
    )
    if existing_df is None:
        st.info(f"기존 {existing_name} 결과를 DataFrame으로 변환할 수 없습니다.")
        return

    score_col = f"Score_{existing_name}"
    if score_col in existing_df.columns:
        existing_top = existing_df.nlargest(20, score_col)
    else:
        existing_top = existing_df.head(20)

    ex_tickers = set(existing_top.index.tolist())
    both = rf_tickers & ex_tickers
    only_rf = rf_tickers - ex_tickers
    only_ex = ex_tickers - rf_tickers

    c1, c2, c3 = st.columns(3)
    c1.metric("공통 종목", f"{len(both)}개")
    c2.metric("강화학습만", f"{len(only_rf)}개")
    c3.metric(f"{existing_name}만", f"{len(only_ex)}개")

    rows = []
    for s in reinforced_top20:
        ticker = s["ticker"]
        in_existing = "✅" if ticker in ex_tickers else ""
        ex_score = ""
        if ticker in existing_df.index and score_col in existing_df.columns:
            ex_score = f"{existing_df.at[ticker, score_col]:.1f}"
        rows.append(
            {
                "종목코드": ticker,
                "종목명": s.get("name", ""),
                "강화학습 점수": f"{s.get('score', 0):.1f}",
                f"{existing_name} 점수": ex_score,
                "기존포함": in_existing,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# 메인 UI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI추천주 강화학습 (개발중)", page_icon="🔬", layout="wide"
)
apply_mobile_styles()

st.title("🔬 AI추천주 강화학습 (개발중)")
st.caption(
    "수익률검사 결과에서 고수익 패턴을 분석하고, 개선된 모델로 Top 20을 도출합니다"
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
        st.warning("⚠️ 수집된 데이터가 없습니다.")
        st.stop()

cache_dir = CACHE_DIR / date_str
missing = [f for f in _REQUIRED_FILES if not (cache_dir / f).exists()]
if missing:
    st.warning(f"⚠️ 필수 데이터 없음: {', '.join(missing)}")
    st.stop()

st.caption(f"📅 기준일: **{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}**")


@st.cache_data(ttl=3600)
def _load_universe(ds):
    import pickle

    import numpy as np

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
    """universe_df에서 종목별 재무정보 dict 생성."""
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
            "영업이익률": "",
        }
    return info


universe_df = _load_universe(date_str)


# ══════════════════════════════════════════════════════
# Section 0: 일괄 실행 / 재실행
# ══════════════════════════════════════════════════════
_bt_for_batch = load_recommend_backtest()

_existing_rules = load_recommend_rf_rules()
_has_rules = _existing_rules is not None
_cur_ver = _existing_rules.get("version", 0) if _existing_rules else 0

# 백테스트 파라미터는 항상 AI급등주 base에서 (동일 조건 유지)
_base_cfg = (_bt_for_batch or {}).get("config", {})

if _bt_for_batch is not None:
    # ── 학습 목표 파라미터 표시 (AI추천주 백테스트 설정 기반) ──
    _disp_tp_bt = _base_cfg.get("label_target_pct", 0.50)
    _disp_hw_bt = _base_cfg.get("label_horizon_weeks", 24)
    _disp_sp_bt = _base_cfg.get("label_stop_pct", 0.08)
    st.info(
        f"학습 목표 (AI추천주 기반): "
        f"**목표수익률 {_disp_tp_bt * 100:.0f}%** | "
        f"**목표기간 {_disp_hw_bt}주 ({_disp_hw_bt * 5}영업일)** | "
        f"**손절 {_disp_sp_bt * 100:.0f}%**"
    )
    if _has_rules:
        # 규칙이 이미 있으면 → 학습 + 백테스트만 (2단계)
        st.info(
            f"현재 규칙 v{_cur_ver} 로 강화학습 + 수익률검사를 실행합니다. "
            f"결과 확인 후 아래 '모델 진화' 버튼으로 다음 버전을 생성하세요."
        )
        _btn_label = f"⚡ 실행 (v{_cur_ver} 규칙으로 강화학습 → 수익률검사)"
    else:
        # 규칙이 없으면 → 전체 4단계
        _btn_label = "⚡ 일괄 실행 (패턴분석 → 규칙생성 → 강화학습 → 수익률검사)"

    batch_btn = st.button(
        _btn_label,
        type="primary",
        use_container_width=True,
        key="batch_run",
    )
    if batch_btn:
        _batch_progress = st.progress(0)
        _batch_status = st.empty()
        _batch_failed = False

        if _has_rules:
            # ── 규칙 있음: Step 1~2 스킵, 기존 규칙 사용 ──
            _batch_rules = _existing_rules
            _batch_progress.progress(5)
            _batch_status.text(f"⏳ v{_cur_ver} 규칙 사용")
        else:
            # ── 규칙 없음: Step 1/4 패턴 분석 ──
            _batch_status.text("⏳ [1/4] 패턴 분석 중... (AI급등주 결과 기반)")
            _batch_progress.progress(2)
            try:
                _batch_analysis = analyze_recommend_patterns(_bt_for_batch)
                st.session_state["recommend_rf_pattern_analysis"] = _batch_analysis
            except Exception as e:
                st.error(f"[1/4] 패턴 분석 실패: {e}")
                _batch_failed = True

            # ── Step 2/4: 학습규칙 생성 ──
            if not _batch_failed:
                _batch_status.text("⏳ [2/4] 학습규칙 v1 생성 중...")
                _batch_progress.progress(5)
                try:
                    _batch_rules = generate_recommend_rf_rules(
                        _batch_analysis, _bt_for_batch
                    )
                    save_recommend_rf_rules(_batch_rules)
                    st.session_state["recommend_rf_rules"] = _batch_rules
                except Exception as e:
                    st.error(f"[2/4] 규칙 생성 실패: {e}")
                    _batch_failed = True

        # ── Step 3: 강화학습 실행 ──
        if not _batch_failed:

            def _batch_cb_train(msg, pct=None):
                if pct is not None:
                    mapped = 5 + int(pct * 0.35)
                    _batch_progress.progress(min(mapped, 40))
                _batch_status.text(f"⏳ 강화학습 중: {msg}")

            try:
                _batch_rf_res = train_recommend_rf_model(
                    universe_df=universe_df,
                    date_str=date_str,
                    rules=_batch_rules,
                    progress_callback=_batch_cb_train,
                )
                save_recommend_rf_result(_batch_rf_res)
                st.session_state["recommend_rf_result"] = _batch_rf_res
            except Exception as e:
                st.error(f"강화학습 실패: {e}")
                import traceback

                st.code(traceback.format_exc())
                _batch_failed = True

        # ── Step 4: AI추천주 강화학습 수익률검사 ──
        if not _batch_failed:

            def _batch_cb_bt(msg, pct=None):
                if pct is not None:
                    mapped = 40 + int(pct * 0.58)
                    _batch_progress.progress(min(mapped, 98))
                _batch_status.text(f"⏳ 수익률검사 중: {msg}")

            try:
                _batch_bt_result = run_recommend_rf_backtest(
                    universe_df=universe_df,
                    date_str=date_str,
                    rules=_batch_rules,
                    progress_callback=_batch_cb_bt,
                    train_window_years=int(_base_cfg.get("train_window", "2Y")[0]),
                    top_k=_base_cfg.get("top_k", 20),
                    trailing_stop_pct=_base_cfg.get("trailing_stop", 0.05),
                    stop_loss_pct=_base_cfg.get("stop_loss", 0.10),
                    min_profit_pct=_base_cfg.get("min_profit", 0.30),
                    label_horizon_weeks=_base_cfg.get("label_horizon_weeks", 4),
                    target_hold_periods=_base_cfg.get("hold_periods", [4]),
                    buy_price_type=_base_cfg.get("buy_price_type", "close"),
                    num_boost_round=_base_cfg.get("num_boost_round", 400),
                )
                save_recommend_rf_backtest(_batch_bt_result)
                st.session_state["recommend_rf_bt_result"] = _batch_bt_result
            except Exception as e:
                st.error(f"수익률검사 실패: {e}")
                import traceback

                st.code(traceback.format_exc())
                _batch_failed = True

        if not _batch_failed:
            _batch_progress.progress(100)
            _batch_status.empty()
            st.success(
                f"v{_cur_ver if _has_rules else 1} 실행 완료! "
                f"아래 비교 결과를 확인한 후 '모델 진화' 버튼으로 다음 버전을 생성하세요."
            )
            st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════
# Section 1: 패턴 분석
# ══════════════════════════════════════════════════════
st.header("🔍 패턴 분석")

bt_result = load_recommend_backtest()
if bt_result is None:
    st.warning(
        "⚠️ 수익률검사 결과가 없습니다. "
        "먼저 **📊 AI수익률검사** 페이지에서 백테스트를 실행하세요."
    )
    st.stop()

bt_cfg = bt_result.get("config", {})
bt_summary = bt_result.get("summary", {})
st.info(
    f"백테스트 결과: {bt_cfg.get('start_date', '?')} ~ {bt_cfg.get('end_date', '?')} | "
    f"누적수익률(1M실투자): {(bt_summary.get('summary_by_period', {}).get(1, {}).get('realistic_cum_return', 1) - 1) * 100:+.1f}% | "
    f"승률: {bt_summary.get('win_rate', 0) * 100:.1f}%"
)

analyze_btn = st.button("🔍 패턴 분석 실행", type="primary", use_container_width=True)

if analyze_btn:
    with st.spinner("패턴 분석 중..."):
        try:
            analysis = analyze_recommend_patterns(bt_result)
            st.session_state["recommend_rf_pattern_analysis"] = analysis
            st.success("패턴 분석 완료!")
        except Exception as e:
            st.error(f"분석 오류: {e}")

analysis = st.session_state.get("recommend_rf_pattern_analysis")
if analysis:
    _render_pattern_analysis(analysis)


# ══════════════════════════════════════════════════════
# Section 2: 학습규칙 생성 및 저장
# ══════════════════════════════════════════════════════
st.divider()
st.header("💾 학습규칙 생성")

if analysis:
    gen_rules_btn = st.button(
        "💾 학습규칙 자동 생성 및 저장", type="primary", use_container_width=True
    )
    if gen_rules_btn:
        with st.spinner("규칙 생성 중..."):
            try:
                rules = generate_recommend_rf_rules(analysis, bt_result)
                save_path = save_recommend_rf_rules(rules)
                st.session_state["recommend_rf_rules"] = rules
                st.success(f"학습규칙 저장 완료: {save_path.name}")
            except Exception as e:
                st.error(f"규칙 생성 오류: {e}")
else:
    st.info("패턴 분석을 먼저 실행하세요.")

# 저장된 규칙 로드
saved_rules = st.session_state.get("recommend_rf_rules") or load_recommend_rf_rules()
if saved_rules:
    st.session_state["recommend_rf_rules"] = saved_rules
    with st.expander("📋 현재 학습규칙 상세", expanded=False):
        st.write(f"**버전**: {saved_rules.get('version', '?')}")
        st.write(f"**생성일시**: {saved_rules.get('created_at', '?')}")
        st.write(f"**백테스트 기간**: {saved_rules.get('backtest_period', '?')}")

        bt_stats = saved_rules.get("backtest_stats", {})
        st.write(
            f"**백테스트 성능**: 수익률 {bt_stats.get('avg_return', 0) * 100:+.1f}% | "
            f"승률 {bt_stats.get('win_rate', 0) * 100:.0f}% | "
            f"샤프 {bt_stats.get('sharpe_ratio', 0):.2f}"
        )

        boosts = saved_rules.get("pattern_boosts", [])
        if boosts:
            st.subheader("고수익 패턴 부스트")
            for i, b in enumerate(boosts, 1):
                st.write(
                    f"  {i}. **{b['name']}** → "
                    f"가중치 {b['weight_multiplier']}x | "
                    f"매칭 {b.get('match_count', '?')}건 | "
                    f"평균수익률 {b.get('avg_return', 0) * 100:+.1f}%"
                )

        lgb_p = saved_rules.get("lgb_params", {})
        if lgb_p:
            st.subheader("LightGBM 파라미터")
            st.json(lgb_p)

        excluded = saved_rules.get("excluded_features", [])
        if excluded:
            kr_excluded = [SURGE_FEATURE_NAMES_KR.get(f, f) for f in excluded]
            st.write(f"**제외 피처**: {', '.join(kr_excluded)}")

        st.write(
            f"**Triple-Barrier**: 목표 {saved_rules.get('target_pct', 0.2) * 100:.0f}% / "
            f"손절 {saved_rules.get('stop_pct', 0.08) * 100:.0f}%"
        )


# ══════════════════════════════════════════════════════
# Section 3: 강화학습 실행
# ══════════════════════════════════════════════════════
st.divider()
st.header("🚀 AI추천주 강화학습 실행")

if saved_rules:
    st.info(
        f"현재 규칙: v{saved_rules.get('version', '?')} "
        f"({saved_rules.get('created_at', '?')} 생성)"
    )

    reinforce_btn = st.button(
        "🚀 강화학습 실행", type="primary", use_container_width=True
    )

    if reinforce_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _rcb(msg, pct=None):
            if pct is not None:
                progress_bar.progress(min(pct, 100))
            status_text.text(msg)

        try:
            rf_res = train_recommend_rf_model(
                universe_df=universe_df,
                date_str=date_str,
                rules=saved_rules,
                progress_callback=_rcb,
            )
            save_recommend_rf_result(rf_res)
            st.session_state["recommend_rf_result"] = rf_res
            progress_bar.progress(100)
            st.success("강화학습 완료!")
            st.rerun()
        except Exception as e:
            st.error(f"강화학습 오류: {e}")
            import traceback

            st.code(traceback.format_exc())
else:
    st.warning("학습규칙이 없습니다. 패턴 분석 → 규칙 생성을 먼저 수행하세요.")

# 결과 표시
rf_result = st.session_state.get("recommend_rf_result") or load_recommend_rf_result()

if rf_result:
    st.divider()
    st.header("🏆 AI추천주 강화학습 Top 20")

    # ── 결과 시총 필터 ──
    _result_mcap_b = st.slider(
        "결과 최소 시총 (억)",
        RESULT_MCAP_MIN_BILLION,
        5000,
        value=st.session_state.get(
            "result_min_mcap_billion", RESULT_MCAP_DEFAULT_BILLION
        ),
        step=100,
        key="rec_rf_result_min_mcap",
        help=f"Top K 결과에서 시총 이 값 이상만 표시 (최소 {RESULT_MCAP_MIN_BILLION}억)",
    )
    st.session_state["result_min_mcap_billion"] = _result_mcap_b

    model_info = rf_result.get("model_info", {})
    st.caption(
        f"Train AUC: {model_info.get('train_auc', 0):.3f} | "
        f"학습샘플: {model_info.get('n_samples', 0):,} | "
        f"양성: {model_info.get('n_positive', 0):,} | "
        f"피처: {model_info.get('n_features_used', 0)}개"
    )

    # ── 메타 정보 배너 ──
    _meta_parts = []
    _meta_mcap = _base_cfg.get("min_market_cap", 100_000_000_000)
    _meta_mcap_eok = int(_meta_mcap / 1e8) if _meta_mcap else 1000
    _meta_parts.append(f"시총 {_result_mcap_b:,}억 이상")
    _meta_tp = _base_cfg.get("label_target_pct", 0.50)
    _meta_hw = _base_cfg.get("label_horizon_weeks", 24)
    _meta_sp = _base_cfg.get("label_stop_pct", 0.15)
    _meta_parts.append(f"목표 {_meta_tp * 100:.0f}%/{_meta_hw}주")
    _meta_parts.append(f"손절 {_meta_sp * 100:.0f}%")
    _meta_date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    _meta_parts.append(f"{_meta_date_fmt} 종가 기준")
    st.caption(f"📋 {' | '.join(_meta_parts)}")

    top20 = rf_result.get("top20", [])
    if top20:
        df_top = pd.DataFrame(top20)

        # 재무정보 조인
        for _col in ["Market", "MarketCap", "PBR", "PER", "ROE", "DIV"]:
            if _col in universe_df.columns:
                df_top[_col] = df_top["ticker"].map(
                    lambda t, c=_col: universe_df.at[t, c]
                    if t in universe_df.index
                    else None
                )

        # 시총 필터 적용
        _mcap_threshold = _result_mcap_b * 1e8
        if "MarketCap" in df_top.columns:
            df_top = df_top[df_top["MarketCap"].fillna(0) >= _mcap_threshold]
        elif "시총_억" in df_top.columns:
            df_top = df_top[df_top["시총_억"].fillna(0) >= _result_mcap_b]

        df_top = df_top.head(20)
        df_top.index = range(1, len(df_top) + 1)
        df_top.index.name = "순위"

        df_top["시총(조)"] = (df_top.get("MarketCap", 0).fillna(0) / 1e12).round(2)
        df_top["시장"] = df_top.get("Market", "")
        df_top["Naver"] = df_top["ticker"].map(
            lambda t: f"https://finance.naver.com/item/main.naver?code={t}"
        )

        # 현재가 조인
        _ohlcv_dict = st.session_state.get("candidates_ohlcv_100d", {})
        df_top["현재가"] = df_top["ticker"].map(
            lambda t: int(_ohlcv_dict[t]["종가"].astype(float).iloc[-1])
            if t in _ohlcv_dict and len(_ohlcv_dict[t]) > 0
            else 0
        )

        display_cols = {
            "ticker": "종목코드",
            "name": "종목명",
            "Naver": "Naver",
            "시장": "시장",
            "시총(조)": "시총(조)",
            "현재가": "현재가",
            "PBR": "PBR",
            "PER": "PER",
            "ROE": "ROE",
            "DIV": "배당률",
            "predicted_prob": "예측확률",
            "score": "점수",
        }
        df_display = df_top.rename(columns=display_cols)
        show_cols = [c for c in display_cols.values() if c in df_display.columns]

        st.dataframe(
            df_display[show_cols],
            column_config={
                "Naver": st.column_config.LinkColumn("Naver", display_text="📊 Naver"),
            },
            use_container_width=True,
        )

        # ── Excel / CSV 다운로드 ──
        _dl_df = df_display[show_cols].copy()
        _dl_meta = {
            "모델": "AI추천주강화학습",
            "시총필터": f"{_result_mcap_b}억",
            "목표수익률": f"{_meta_tp * 100:.0f}%",
            "수익목표기간": f"{_meta_hw}주",
            "손절기준": f"{_meta_sp * 100:.0f}%",
            "종가기준일": _meta_date_fmt,
            "Top_K": str(len(_dl_df)),
            "생성일시": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _dl_c1, _dl_c2 = st.columns(2)
        with _dl_c1:
            _xl_buf = io.BytesIO()
            with pd.ExcelWriter(_xl_buf, engine="openpyxl") as xw:
                pd.DataFrame(list(_dl_meta.items()), columns=["항목", "값"]).to_excel(
                    xw, sheet_name="META", index=False
                )
                _dl_df.to_excel(xw, sheet_name="TOP_K", index=True)
            st.download_button(
                "📥 Excel",
                _xl_buf.getvalue(),
                f"AI추천주강화학습_top{len(_dl_df)}_{date_str}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with _dl_c2:
            _csv_lines = [f"#META:{k}={v}" for k, v in _dl_meta.items()]
            _csv_lines.append(_dl_df.to_csv(encoding="utf-8-sig"))
            st.download_button(
                "📥 CSV",
                "\n".join(_csv_lines),
                f"AI추천주강화학습_top{len(_dl_df)}_{date_str}.csv",
                "text/csv",
                use_container_width=True,
            )

        # 기존 AI추천주 결과와 비교
        existing_ai = st.session_state.get("screening_results", {})
        ai_surge_key = None
        for key in [
            "AI_25%급등주",
            "AI_30%급등주",
            "AI_25%급등주_1Y",
            "AI_30%급등주_1Y",
        ]:
            if key in existing_ai:
                ai_surge_key = key
                break

        if ai_surge_key and ai_surge_key in existing_ai:
            st.subheader("🔄 기존 AI추천주 vs 강화학습 비교")
            _compare_results(top20, existing_ai[ai_surge_key], ai_surge_key)
    else:
        st.info("예측 결과가 없습니다.")


# ══════════════════════════════════════════════════════
# Section 4: AI추천주 강화학습 수익률검사
# ══════════════════════════════════════════════════════
st.divider()
st.header("📊 AI추천주 강화학습 수익률검사")

if saved_rules:
    _rules_ver = saved_rules.get("version", "?")
    _rules_target = saved_rules.get("target_pct", 0.30)
    _rules_stop = saved_rules.get("stop_pct", 0.10)
    st.info(
        f"규칙 v{_rules_ver} 적용 "
        f"(규칙 기본: 목표 {_rules_target * 100:.0f}% / 손절 {_rules_stop * 100:.0f}%). "
        f"손절기준은 horizon·target 연동으로 자동 조정됩니다."
    )

    # ── 기존 AI추천주 백테스트 config에서 기본값 읽기 ──
    _base_cfg = (_bt_for_batch or {}).get("config", {})
    _def_tw = int(str(_base_cfg.get("train_window", "2Y"))[0]) if _base_cfg else 2
    _def_topk = _base_cfg.get("top_k", 20)
    _def_horizon = _base_cfg.get("label_horizon_weeks", 4)
    _def_boost = _base_cfg.get("num_boost_round", 400)
    _def_min_profit = int(_base_cfg.get("min_profit", 0.30) * 100)
    _def_trailing = int(_base_cfg.get("trailing_stop", 0.05) * 100)
    _def_stop_loss = int(_base_cfg.get("stop_loss", 0.10) * 100)
    _def_buy_type = _base_cfg.get("buy_price_type", "close")

    with st.expander("설정", expanded=True):
        if _base_cfg:
            st.caption("기본값: 기존 AI추천주 백테스트 설정에서 자동 로드됨")

        # ── Row 1: 핵심 파라미터 (5열) ──
        _c1, _c2, _c3, _c4, _c5 = st.columns(5)
        _tw_options = [2, 1, 3, 4]
        _tw_idx = _tw_options.index(_def_tw) if _def_tw in _tw_options else 0
        with _c1:
            rf_train_window = st.selectbox(
                "학습 윈도우",
                _tw_options,
                index=_tw_idx,
                format_func=lambda x: f"{x}년",
                help="학습에 사용할 과거 데이터 기간",
                key="rf_train_window",
            )
        with _c2:
            rf_top_k = st.number_input(
                "Top K 종목",
                min_value=5,
                max_value=50,
                value=min(max(_def_topk, 5), 50),
                key="rf_top_k",
            )
        _hz_options = [3, 4, 6, 8, 10, 12]
        _hz_idx = _hz_options.index(_def_horizon) if _def_horizon in _hz_options else 1
        with _c3:
            rf_horizon = st.selectbox(
                "수익목표기간 (horizon)",
                _hz_options,
                index=_hz_idx,
                format_func=lambda w: f"{w}주 ({w * 5}영업일)",
                help="라벨링 horizon: 이 기간 내 목표수익 달성 여부로 학습",
                key="rf_horizon",
            )
        with _c4:
            st.number_input(
                "학습목표수익률 (target_pct) %",
                value=int(_rules_target * 100),
                disabled=True,
                help="규칙에서 자동 설정 (기존 AI추천주와 동일)",
                key="rf_label_target_display",
            )
        with _c5:
            from core.ml_pattern import compute_recommended_stop

            _rf_stop_rec = compute_recommended_stop(
                rf_horizon, int(_rules_target * 100)
            )
            # horizon 변경 시 stop 자동갱신
            _rf_prev_hw = st.session_state.get("_rf_prev_hw", rf_horizon)
            if rf_horizon != _rf_prev_hw:
                st.session_state["rf_label_stop_input"] = _rf_stop_rec
            st.session_state["_rf_prev_hw"] = rf_horizon
            _rf_stop_kw = {
                "label": f"학습손절기준 (stop_pct) % — 추천: {_rf_stop_rec}%",
                "min_value": 5,
                "max_value": 30,
                "key": "rf_label_stop_input",
                "help": f"규칙 기본값: {int(_rules_stop * 100)}% | horizon·target 연동 추천: {_rf_stop_rec}%",
            }
            if "rf_label_stop_input" not in st.session_state:
                _rf_stop_kw["value"] = _rf_stop_rec
            rf_label_stop = st.number_input(**_rf_stop_kw)

        # ── Row 2: Boost Rounds ──
        _c6, _c7 = st.columns(2)
        with _c6:
            rf_boost = st.slider(
                "모델강화 (Boost Rounds)",
                300,
                800,
                min(max(_def_boost, 300), 800),
                step=50,
                help="높을수록 정확도 향상, 실행시간 증가",
                key="rf_boost",
            )
        with _c7:
            st.caption("300: 빠른실행 | 400: 균형(권장) | 800: 최고정확도")

        # ── Row 3: 매도 조건 (4열) ──
        _c8, _c9, _c10, _c11 = st.columns(4)
        with _c8:
            rf_min_profit = st.number_input(
                "A: 최소수익률 (%)",
                min_value=0,
                max_value=50,
                value=min(max(_def_min_profit, 0), 50),
                help="트레일링 발동 조건: 매수가 대비 최소 A% 이상 상승 경험",
                key="rf_min_profit",
            )
        with _c9:
            rf_trailing = st.number_input(
                "B: 고점대비 하락 (%)",
                min_value=3,
                max_value=30,
                value=min(max(_def_trailing, 3), 30),
                help="트레일링 스탑: 고점 대비 B% 하락 시 매도",
                key="rf_trailing",
            )
        with _c10:
            rf_stop = st.number_input(
                "손절 C (%)",
                min_value=5,
                max_value=30,
                value=min(max(_def_stop_loss, 5), 30),
                help="매수가 대비 해당% 이상 하락 시 손절",
                key="rf_stop",
            )
        with _c11:
            rf_buy_type = st.selectbox(
                "매수가 기준",
                ["시가", "종가"],
                index=0 if _def_buy_type == "open" else 1,
                help="지정일 다음 거래일의 시가 또는 종가로 매수",
                key="rf_buy_type",
            )

        _rf_buy_label = "시가" if rf_buy_type == "시가" else "종가"
        st.caption(
            f"학습윈도우 {rf_train_window}년 | "
            f"수익목표기간 {rf_horizon}주({rf_horizon * 5}일) | "
            f"목표수익률 {_rules_target * 100:.0f}% | "
            f"손절기준 {rf_label_stop}% | "
            f"Boost {rf_boost} | Top {rf_top_k}종목 | "
            f"매수가: {_rf_buy_label} | "
            f"매도: A={rf_min_profit}%이상→고점대비 B={rf_trailing}%하락, "
            f"손절 C={rf_stop}% | 보유기간: {rf_horizon}주"
        )

    rf_bt_btn = st.button(
        "📊 AI추천주 강화학습 수익률검사 실행",
        type="primary",
        use_container_width=True,
        key="run_rf_bt",
    )

    if rf_bt_btn:
        rf_bt_progress = st.progress(0)
        rf_bt_status = st.empty()

        def _rf_bt_cb(msg, pct=None):
            if pct is not None:
                rf_bt_progress.progress(min(pct, 100))
            rf_bt_status.text(msg)

        try:
            import time as _time

            _rf_start = _time.time()
            _rf_buy_param = "open" if rf_buy_type == "시가" else "close"
            rf_bt_result = run_recommend_rf_backtest(
                universe_df=universe_df,
                date_str=date_str,
                rules=saved_rules,
                progress_callback=_rf_bt_cb,
                train_window_years=rf_train_window,
                top_k=rf_top_k,
                trailing_stop_pct=rf_trailing / 100.0,
                stop_loss_pct=rf_stop / 100.0,
                min_profit_pct=rf_min_profit / 100.0,
                label_horizon_weeks=rf_horizon,
                target_hold_periods=[rf_horizon],
                buy_price_type=_rf_buy_param,
                num_boost_round=rf_boost,
                label_stop_pct=rf_label_stop / 100.0,
            )
            save_recommend_rf_backtest(rf_bt_result)
            st.session_state["recommend_rf_bt_result"] = rf_bt_result
            rf_bt_progress.progress(100)
            _elapsed = (_time.time() - _rf_start) / 60
            _sbp = rf_bt_result.get("summary", {}).get("summary_by_period", {})
            _cum = _sbp.get(rf_horizon, {}).get("realistic_cum_return", 1.0)
            st.success(
                f"AI추천주 강화학습 수익률검사 완료! "
                f"({len(rf_bt_result.get('daily_results', []))}주, "
                f"{_elapsed:.1f}분, "
                f"누적수익률 {(_cum - 1) * 100:+.1f}%)"
            )
            st.rerun()
        except Exception as e:
            st.error(f"강화학습 백테스트 오류: {e}")
            import traceback

            st.code(traceback.format_exc())
else:
    st.warning("학습규칙이 없습니다. 먼저 패턴 분석 → 규칙 생성을 수행하세요.")


# ══════════════════════════════════════════════════════
# Section 5: AI추천주 기존 vs 강화학습 성과 비교
# ══════════════════════════════════════════════════════
rf_bt_result = (
    st.session_state.get("recommend_rf_bt_result") or load_recommend_rf_backtest()
)
base_bt_result = bt_result  # Section 1에서 이미 로드됨

if rf_bt_result and base_bt_result:
    st.divider()
    st.header("⚖️ AI추천주 기존 vs 강화학습 성과 비교")

    # 공통 보유기간 결정
    rf_cfg = rf_bt_result.get("config", {})
    base_cfg = base_bt_result.get("config", {})
    rf_hw = rf_cfg.get("label_horizon_weeks", 4)
    base_hw = base_cfg.get("label_horizon_weeks", 4)

    rf_sbp = rf_bt_result.get("summary", {}).get("summary_by_period", {})
    base_sbp = base_bt_result.get("summary", {}).get("summary_by_period", {})

    # 강화학습 결과 기준 보유기간
    rf_period_data = rf_sbp.get(rf_hw, {})
    base_period_data = base_sbp.get(base_hw, {})

    # ── 핵심 지표 비교 테이블 ──
    def _fmt_pct(v):
        return f"{v * 100:+.1f}%" if v else "N/A"

    def _fmt_f(v, fmt=".2f"):
        return f"{v:{fmt}}" if v else "N/A"

    rf_cum = rf_period_data.get("realistic_cum_return", 1.0)
    base_cum = base_period_data.get("realistic_cum_return", 1.0)
    rf_avg = rf_period_data.get("avg_return", 0)
    base_avg = base_period_data.get("avg_return", 0)
    rf_wr = rf_period_data.get("win_rate", 0)
    base_wr = base_period_data.get("win_rate", 0)
    rf_sharpe = rf_period_data.get("sharpe_ratio", 0)
    base_sharpe = base_period_data.get("sharpe_ratio", 0)
    rf_mdd = rf_period_data.get("max_drawdown", 0)
    base_mdd = base_period_data.get("max_drawdown", 0)
    rf_lump = rf_period_data.get("lumpsum_cum_return", 1.0)
    base_lump = base_period_data.get("lumpsum_cum_return", 1.0)

    m0, m1, m2, m3, m4 = st.columns(5)
    m0.metric(
        "평균 주간수익률",
        f"{rf_avg * 100:+.2f}%",
        f"{(rf_avg - base_avg) * 100:+.2f}%p vs 기존({base_avg * 100:+.2f}%)",
    )
    m1.metric(
        "누적수익률(분할매수)",
        _fmt_pct(rf_cum - 1),
        f"{(rf_cum - base_cum) * 100:+.1f}%p vs 기존({_fmt_pct(base_cum - 1)})",
    )
    m2.metric(
        "누적수익률(일괄매수)",
        _fmt_pct(rf_lump - 1),
        f"{(rf_lump - base_lump) * 100:+.1f}%p vs 기존({_fmt_pct(base_lump - 1)})",
    )
    m3.metric(
        "승률",
        f"{rf_wr * 100:.1f}%",
        f"{(rf_wr - base_wr) * 100:+.1f}%p",
    )
    m4.metric(
        "샤프비율",
        _fmt_f(rf_sharpe),
        f"{rf_sharpe - base_sharpe:+.2f}",
    )

    # ── 상세 비교 테이블 ──
    with st.expander("상세 비교 테이블", expanded=False):
        comp_rows = [
            {
                "지표": "평균 주간수익률",
                "기존 AI추천주": f"{base_avg * 100:+.2f}%",
                "강화학습": f"{rf_avg * 100:+.2f}%",
                "차이": f"{(rf_avg - base_avg) * 100:+.2f}%p",
            },
            {
                "지표": "누적수익률(분할매수)",
                "기존 AI추천주": _fmt_pct(base_cum - 1),
                "강화학습": _fmt_pct(rf_cum - 1),
                "차이": f"{(rf_cum - base_cum) * 100:+.1f}%p",
            },
            {
                "지표": "누적수익률(일괄매수)",
                "기존 AI추천주": _fmt_pct(base_lump - 1),
                "강화학습": _fmt_pct(rf_lump - 1),
                "차이": f"{(rf_lump - base_lump) * 100:+.1f}%p",
            },
            {
                "지표": "승률",
                "기존 AI추천주": f"{base_wr * 100:.1f}%",
                "강화학습": f"{rf_wr * 100:.1f}%",
                "차이": f"{(rf_wr - base_wr) * 100:+.1f}%p",
            },
            {
                "지표": "샤프비율",
                "기존 AI추천주": _fmt_f(base_sharpe),
                "강화학습": _fmt_f(rf_sharpe),
                "차이": f"{rf_sharpe - base_sharpe:+.2f}",
            },
            {
                "지표": "최대낙폭",
                "기존 AI추천주": _fmt_pct(base_mdd),
                "강화학습": _fmt_pct(rf_mdd),
                "차이": f"{(rf_mdd - base_mdd) * 100:+.1f}%p",
            },
            {
                "지표": "총 주수",
                "기존 AI추천주": str(base_period_data.get("n_weeks", 0)),
                "강화학습": str(rf_period_data.get("n_weeks", 0)),
                "차이": "",
            },
        ]
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ── 누적수익률 오버레이 차트 (분할매수 & 일괄매수) ──
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 분할매수 weekly values
        base_wv = base_period_data.get("realistic_weekly_values", [])
        rf_wv = rf_period_data.get("realistic_weekly_values", [])
        # 일괄매수 weekly values
        base_lump_wv = base_period_data.get("lumpsum_weekly_values", [])
        rf_lump_wv = rf_period_data.get("lumpsum_weekly_values", [])

        if base_wv and rf_wv:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["분할매수 (슬롯 순환)", "일괄매수"],
                horizontal_spacing=0.08,
            )

            # ── 분할매수 차트 ──
            base_dates = [w["date"] for w in base_wv]
            base_vals = [
                (w["total_value"] / base_wv[0]["total_value"] - 1) * 100
                for w in base_wv
            ]
            rf_dates = [w["date"] for w in rf_wv]
            rf_vals = [
                (w["total_value"] / rf_wv[0]["total_value"] - 1) * 100 for w in rf_wv
            ]

            fig.add_trace(
                go.Scatter(
                    x=base_dates,
                    y=base_vals,
                    mode="lines",
                    name="기존 AI추천주",
                    line={"color": "#2196F3", "width": 2},
                    legendgroup="base",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=rf_dates,
                    y=rf_vals,
                    mode="lines",
                    name="강화학습",
                    line={"color": "#E91E63", "width": 2},
                    legendgroup="rf",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # ── 일괄매수 차트 ──
            if base_lump_wv and rf_lump_wv:
                bl_dates = [w["date"] for w in base_lump_wv]
                bl_vals = [
                    (w["total_value"] / base_lump_wv[0]["total_value"] - 1) * 100
                    for w in base_lump_wv
                ]
                rl_dates = [w["date"] for w in rf_lump_wv]
                rl_vals = [
                    (w["total_value"] / rf_lump_wv[0]["total_value"] - 1) * 100
                    for w in rf_lump_wv
                ]

                fig.add_trace(
                    go.Scatter(
                        x=bl_dates,
                        y=bl_vals,
                        mode="lines",
                        name="기존 AI추천주",
                        line={"color": "#2196F3", "width": 2},
                        legendgroup="base",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter(
                        x=rl_dates,
                        y=rl_vals,
                        mode="lines",
                        name="강화학습",
                        line={"color": "#E91E63", "width": 2},
                        legendgroup="rf",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            fig.update_layout(
                title="누적수익률 비교 — 분할매수 vs 일괄매수",
                hovermode="x unified",
                height=420,
            )
            fig.update_yaxes(title_text="수익률(%)", row=1, col=1)
            fig.update_yaxes(title_text="수익률(%)", row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Plotly가 설치되지 않아 차트를 표시할 수 없습니다.")

    # ── 강화학습 종목별 거래 내역 ──
    _rf_daily = rf_bt_result.get("daily_results", [])
    if _rf_daily:
        with st.expander("📋 강화학습 종목별 거래 내역", expanded=False):
            _fi = _build_fund_info(universe_df)
            _rf_rows = []
            for d in _rf_daily:
                for s in d["stocks"]:
                    rbp = s.get("returns_by_period", {}).get(rf_hw)
                    if rbp is None:
                        continue
                    _t = s["ticker"]
                    _f = _fi.get(_t, {})
                    _rf_rows.append(
                        {
                            "지정일": d["date"],
                            "매수일": d["buy_date"],
                            "종목코드": _t,
                            "종목명": s.get("name", _t),
                            "시총(조)": _f.get("시총(조)", ""),
                            "PBR": _f.get("PBR", ""),
                            "PER": _f.get("PER", ""),
                            "EPS": _f.get("EPS", ""),
                            "ROE": _f.get("ROE", ""),
                            "배당률": _f.get("배당률", ""),
                            "영업이익률": _f.get("영업이익률", ""),
                            "예측확률": f"{s['predicted_prob']:.2%}",
                            "매수가": f"{s['buy_price']:,.0f}",
                            "매도가": f"{rbp['sell_price']:,.0f}",
                            f"{rf_hw}주수익률": f"{rbp['return_pct'] * 100:+.2f}%",
                            "매도사유": rbp["exit_reason"],
                            "보유일(거래일)": rbp["hold_days"],
                        }
                    )
            if _rf_rows:
                st.dataframe(
                    pd.DataFrame(_rf_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, len(_rf_rows) * 35 + 38),
                )

    # ── 모델 적용 버튼 (진화 루프) ──
    st.divider()
    st.subheader("🔄 모델 진화")

    if rf_avg > base_avg:
        st.success(
            f"강화학습 주간평균 {rf_avg * 100:+.2f}% — 기존 대비 "
            f"**+{(rf_avg - base_avg) * 100:.2f}%p** 우수 | "
            f"분할매수 {(rf_cum - base_cum) * 100:+.1f}%p / "
            f"일괄매수 {(rf_lump - base_lump) * 100:+.1f}%p"
        )
    elif rf_avg < base_avg:
        st.warning(
            f"강화학습 주간평균 {rf_avg * 100:+.2f}% — 기존 대비 "
            f"**{(rf_avg - base_avg) * 100:.2f}%p** 낮음 | "
            f"분할매수 {(rf_cum - base_cum) * 100:+.1f}%p / "
            f"일괄매수 {(rf_lump - base_lump) * 100:+.1f}%p"
        )
    else:
        st.info("강화학습 모델과 기존 모델의 수익률이 동일합니다.")

    _cur_ver = saved_rules.get("version", 0) if saved_rules else 0
    apply_btn = st.button(
        f"✅ 강화학습 결과로 규칙 업데이트 (v{_cur_ver} → v{_cur_ver + 1})",
        use_container_width=True,
        key="apply_rf_rules",
        help="강화학습 백테스트 결과를 기반으로 패턴 재분석 → 새 규칙 생성",
    )

    if apply_btn:
        with st.spinner("규칙 업데이트 중... (패턴 재분석 → 새 규칙 생성)"):
            try:
                # 1. 강화학습 백테스트 결과로 패턴 재분석
                new_analysis = analyze_recommend_patterns(rf_bt_result)
                # 2. 새 규칙 생성 (version +1)
                new_rules = generate_recommend_rf_rules(new_analysis, rf_bt_result)
                new_rules["version"] = _cur_ver + 1
                # 3. 저장
                save_path = save_recommend_rf_rules(new_rules)
                # 4. 세션 업데이트
                st.session_state["recommend_rf_rules"] = new_rules
                st.session_state["recommend_rf_pattern_analysis"] = new_analysis
                st.success(
                    f"규칙 v{_cur_ver + 1} 생성 완료! ({save_path.name}) "
                    f"→ 'AI추천주 강화학습 수익률검사'를 다시 실행하여 개선 효과를 확인하세요."
                )
                st.rerun()
            except Exception as e:
                st.error(f"규칙 업데이트 오류: {e}")
                import traceback

                st.code(traceback.format_exc())

    # ── 진화 이력 + 버전 관리 ──
    with st.expander("📈 진화 이력 / 버전 관리", expanded=False):
        _rule_versions = list_recommend_rf_rule_versions()

        if _rule_versions:
            st.caption("**규칙 버전 목록** (최신 순)")
            _ver_rows = []
            for rv in _rule_versions:
                _ver_rows.append(
                    {
                        "버전": f"v{rv['version']}",
                        "파일": rv["filename"],
                        "생성일시": rv["created_at"],
                        "승률": f"{rv['win_rate'] * 100:.0f}%",
                    }
                )
            st.dataframe(
                pd.DataFrame(_ver_rows),
                use_container_width=True,
                hide_index=True,
            )

            # 이전 버전 원복
            st.markdown("---")
            st.caption("**⏪ 이전 버전 원복**")
            _rollback_options = [
                f"v{rv['version']} — {rv['created_at']} ({rv['filename']})"
                for rv in _rule_versions
            ]
            _selected_rollback = st.selectbox(
                "복원할 버전 선택",
                _rollback_options,
                key="rollback_select",
            )
            if st.button(
                "⏪ 선택한 버전으로 원복",
                key="rollback_btn",
                use_container_width=True,
            ):
                _sel_idx = _rollback_options.index(_selected_rollback)
                _sel_file = _rule_versions[_sel_idx]["filename"]
                try:
                    restored = rollback_recommend_rf_rules(_sel_file)
                    st.session_state["recommend_rf_rules"] = restored
                    st.session_state.pop("recommend_rf_result", None)
                    st.session_state.pop("recommend_rf_bt_result", None)
                    st.success(
                        f"v{restored.get('version', '?')} 으로 원복 완료! "
                        f"강화학습 / 수익률검사를 다시 실행하세요."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"원복 실패: {e}")
        else:
            st.info("아직 이력이 없습니다.")

        # 전체 초기화
        st.markdown("---")
        st.caption("**🗑️ 전체 초기화** — 모든 강화학습 캐시 삭제, v1부터 재시작")
        _rc1, _rc2 = st.columns([3, 1])
        with _rc1:
            _confirm_reset = st.checkbox(
                "초기화에 동의합니다 (되돌릴 수 없음)", key="confirm_reset"
            )
        with _rc2:
            if st.button(
                "🗑️ 초기화",
                key="reset_btn",
                disabled=not _confirm_reset,
                type="primary",
            ):
                n_deleted = reset_recommend_rf_cache()
                st.session_state.pop("recommend_rf_rules", None)
                st.session_state.pop("recommend_rf_result", None)
                st.session_state.pop("recommend_rf_bt_result", None)
                st.session_state.pop("recommend_rf_pattern_analysis", None)
                st.success(
                    f"초기화 완료! {n_deleted}개 파일 삭제. v1부터 재시작합니다."
                )
                st.rerun()
