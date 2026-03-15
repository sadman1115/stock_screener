"""Page 8: 지수예측 — KOSPI/KOSDAQ 3년(36개월) 미래 지수 AI 예측."""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.index_predictor import (  # noqa: E402
    INDEX_CODES,
    INDEX_PRED_DIR,
    KB_SECTOR_MONTHLY,
    N_PRED_MONTHS,
    REGIME_EVENTS,
    compute_index_features,
    compute_index_labels,
    fetch_external_features,
    fetch_index_fundamentals_long,
    fetch_index_ohlcv_long,
    fetch_market_investor_long,
    generate_monthly_strategy,
    interpolate_weekly_predictions,
    load_index_model,
    load_index_prediction,
    predict_index_future,
    save_index_model,
    save_index_prediction,
    train_index_model,
)

_N_YEARS = N_PRED_MONTHS // 12  # 3
_N_WEEKS = N_PRED_MONTHS * 52 // 12  # 156
from core.styles import apply_mobile_styles  # noqa: E402

st.set_page_config(page_title="지수예측", page_icon="📈", layout="wide")
apply_mobile_styles()

st.title("📈 지수예측")
st.caption(
    f"KOSPI/KOSDAQ 지수를 AI로 학습하고 향후 {_N_YEARS}년"
    f"({N_PRED_MONTHS}개월)을 예측합니다."
)

# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

st.sidebar.header("📈 지수예측 설정")

index_choice = st.sidebar.radio("예측 대상", ["KOSPI", "KOSDAQ"], key="idx_pred_target")
index_code = INDEX_CODES[index_choice]

# KOSPI: 1983년~ (Stooq+pykrx), KOSDAQ: 2000년~ (pykrx)
_current_year = datetime.now().year
if index_choice == "KOSPI":
    _data_start_year = 1983  # Stooq 1983~ + pykrx 1997~
    _max_years = _current_year - _data_start_year
    _default_years = _max_years
else:
    _data_start_year = 2000
    _max_years = _current_year - _data_start_year
    _default_years = min(_max_years, 15)

train_years = st.sidebar.slider(
    "학습 기간 (년)",
    8,
    _max_years,
    _default_years,
    key=f"idx_pred_train_years_{index_choice}",
    help=(
        f"{index_choice}: {_data_start_year}년~ (최대 {_max_years}년)"
        + (" | 1983~1996: Stooq, 1997~: pykrx/KRX" if index_choice == "KOSPI" else "")
    ),
)

# 모델 상태 표시
model = st.session_state.get(f"idx_pred_model_{index_choice.lower()}")
if model is None:
    model = load_index_model(index_choice)
    if model:
        st.session_state[f"idx_pred_model_{index_choice.lower()}"] = model

if model:
    st.sidebar.success(f"모델: {model.get('train_date', '?')} 학습")
    v = model.get("validation", {})
    st.sidebar.caption(
        f"MAE: {v.get('avg_mae', 0):.4f} | "
        f"방향: {v.get('avg_direction_accuracy', 0):.1%} | "
        f"CI: {v.get('avg_coverage', 0):.1%}"
    )
else:
    st.sidebar.warning("학습된 모델 없음")

# 학습기준일 — 학습 데이터의 끝 날짜 (이 날짜까지만 학습)
_train_end_date = st.sidebar.date_input(
    "학습 기준일",
    value=date.today(),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
    key=f"idx_pred_train_end_date_{index_choice}",
    help=(
        "학습 데이터의 마지막 날짜를 설정합니다. "
        "이 날짜까지의 데이터만으로 모델을 학습합니다. "
        "과거 날짜를 선택하면 해당 시점까지만 알려진 데이터로 학습합니다."
    ),
)
_train_is_past = _train_end_date < date.today()
if _train_is_past:
    st.sidebar.caption(
        f"📅 학습기준일: {_train_end_date} (과거) → " f"해당일까지 데이터로만 학습"
    )

do_train = st.sidebar.button("🔄 모델 학습", key="idx_pred_train_btn")

# 예측 기준일 — 이 날짜의 종가를 현재가로 사용, 다음 날부터 예측
st.sidebar.markdown("---")
_ref_date = st.sidebar.date_input(
    "예측 기준일",
    value=date.today(),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
    key=f"idx_pred_ref_date_{index_choice}",
    help=(
        "이 날짜의 종가를 기준으로 예측합니다. "
        "과거 날짜를 선택하면 해당일 종가 기준으로 예측하고, "
        "이후 실제 지수와 비교할 수 있습니다."
    ),
)
_ref_is_past = _ref_date < date.today()
if _ref_is_past:
    st.sidebar.caption(f"📅 기준일: {_ref_date} (과거) → " f"예측 vs 실제 비교 가능")

do_predict = st.sidebar.button("🔮 예측 실행", key="idx_pred_predict_btn")

# ── KRX 로그인 (pykrx 데이터 수집 필수) ──
st.sidebar.markdown("---")
st.sidebar.subheader("KRX 로그인")
_krx_id = st.sidebar.text_input(
    "KRX 회원 ID",
    value=st.session_state.get("krx_mbr_id", ""),
    key="idx_krx_id_input",
)
_krx_pw = st.sidebar.text_input(
    "KRX 비밀번호",
    value=st.session_state.get("krx_pw", ""),
    type="password",
    key="idx_krx_pw_input",
)
if _krx_id and _krx_pw:
    st.session_state["krx_mbr_id"] = _krx_id
    st.session_state["krx_pw"] = _krx_pw
    # 즉시 로그인 시도
    try:
        from core import krx_login

        _login_ok, _login_msg = krx_login.ensure_login(_krx_id, _krx_pw)
        if _login_ok:
            st.sidebar.success(f"KRX 로그인 완료: {_login_msg}")
        else:
            st.sidebar.error(f"KRX 로그인 실패: {_login_msg}")
    except Exception as _login_err:
        st.sidebar.warning(f"KRX 로그인 모듈 오류: {_login_err}")
else:
    st.sidebar.caption(
        "pykrx(KRX API)는 로그인이 필요합니다. "
        "[open.krx.co.kr](https://open.krx.co.kr) 에서 회원가입하세요."
    )

# 기준 날짜
date_str = st.session_state.get("date_str")
if not date_str:
    date_str = datetime.now().strftime("%Y%m%d")

other_index = "KOSDAQ" if index_choice == "KOSPI" else "KOSPI"
other_code = INDEX_CODES[other_index]

# ──────────────────────────────────────────────
# 학습 워크플로
# ──────────────────────────────────────────────

if do_train:
    status = st.status(
        f"{index_choice} 모델 학습 중... (학습기준일: {_train_end_date})",
        expanded=True,
    )
    progress = st.progress(0)

    def _cb(msg, pct=None):
        status.write(msg)
        if pct is not None:
            progress.progress(min(int(pct), 100))

    try:
        # 학습기준일까지 수집
        _train_end = _train_end_date.strftime("%Y%m%d")

        # KRX 로그인
        try:
            from core import krx_login

            _mid = st.session_state.get("krx_mbr_id", "")
            _mpw = st.session_state.get("krx_pw", "")
            if _mid and _mpw:
                _lok, _lmsg = krx_login.ensure_login(_mid, _mpw)
                if _lok:
                    _cb(f"KRX 로그인 성공: {_lmsg}", 2)
                else:
                    st.warning(
                        f"⚠️ KRX 로그인 실패: {_lmsg}\n\n"
                        "pykrx 데이터 수집이 실패할 수 있습니다."
                    )
            else:
                st.warning(
                    "⚠️ KRX 로그인 정보가 없습니다. "
                    "사이드바에서 KRX ID/PW를 입력하세요."
                )
        except Exception as _le:
            st.warning(f"KRX 로그인 모듈 오류: {_le}")

        # Step 1: 데이터 수집
        _cb(f"Step 1/4: {index_choice} 지수 OHLCV 수집 ({train_years}년)...", 5)
        ohlcv = fetch_index_ohlcv_long(
            index_code, _train_end, years=train_years, progress_callback=_cb
        )
        if ohlcv is None or ohlcv.empty:
            st.error("지수 OHLCV 데이터 수집 실패")
            st.stop()
        _cb(f"  OHLCV: {len(ohlcv)}일 (~{ohlcv.index[-1].strftime('%Y-%m-%d')})", 10)

        _cb("Step 1/4: 지수 펀더멘털 수집...", 15)
        fundamentals = fetch_index_fundamentals_long(
            index_code, _train_end, years=train_years, progress_callback=_cb
        )
        _cb(f"  펀더멘털: {len(fundamentals) if fundamentals is not None else 0}일", 20)

        _cb("Step 1/4: 투자자 매매 데이터 수집...", 25)
        investor = fetch_market_investor_long(
            index_choice, _train_end, years=train_years, progress_callback=_cb
        )
        _cb(f"  투자자: {len(investor) if investor is not None else 0}일", 30)

        _cb("Step 1/4: 외부 시장 데이터 수집...", 35)
        external = fetch_external_features(
            _train_end, years=train_years, progress_callback=_cb
        )
        ext_msg = f"{len(external)}일" if external is not None else "없음 (FDR 미설치)"
        _cb(f"  외부데이터: {ext_msg}", 38)

        # 상대지수
        _cb(f"Step 1/4: {other_index} 지수 수집 (상대강도)...", 40)
        other_ohlcv = fetch_index_ohlcv_long(other_code, _train_end, years=train_years)

        # Step 2: 피처 계산
        _cb("Step 2/4: 월간 피처 계산 (35개)...", 45)
        features = compute_index_features(
            ohlcv, fundamentals, investor, external, other_ohlcv
        )
        if features.empty or len(features) < 48:
            st.error(f"피처 부족: {len(features)}개월 (최소 48 필요)")
            st.stop()
        _cb(f"  피처: {features.shape[0]}개월 × {features.shape[1]}열", 48)

        # Step 3: 라벨 생성
        from core.index_predictor import _monthly_resample

        monthly_close = _monthly_resample(ohlcv, "종가")
        labels = compute_index_labels(monthly_close)
        _cb(f"  라벨: {labels.shape[0]}개월 × 12구간", 50)

        # Step 4: 모델 학습
        model_data = train_index_model(features, labels, index_choice, _cb)

        # 저장
        _cb("모델 저장...", 98)
        save_index_model(model_data, _train_end, index_choice)
        st.session_state[f"idx_pred_model_{index_choice.lower()}"] = model_data

        progress.progress(100)
        status.update(label=f"✅ {index_choice} 모델 학습 완료!", state="complete")

    except Exception as e:
        status.update(label=f"❌ 학습 실패: {e}", state="error")
        import traceback

        st.error(traceback.format_exc())

# ──────────────────────────────────────────────
# 예측 워크플로
# ──────────────────────────────────────────────

if do_predict:
    model = st.session_state.get(f"idx_pred_model_{index_choice.lower()}")
    if model is None:
        model = load_index_model(index_choice)

    if model is None:
        st.error("학습된 모델이 없습니다. 먼저 '모델 학습'을 실행하세요.")
    else:
        _ref_dt_str = _ref_date.strftime("%Y%m%d")
        _ref_ts = pd.Timestamp(_ref_date)
        with st.spinner(f"{index_choice} 예측 중... (기준일: {_ref_date})"):
            try:
                # 항상 오늘까지 수집 (전체 실제 데이터 확보)
                _pred_end = datetime.now().strftime("%Y%m%d")

                # KRX 로그인 (pykrx 증분 수집 필수)
                try:
                    from core import krx_login

                    _mid = st.session_state.get("krx_mbr_id", "")
                    _mpw = st.session_state.get("krx_pw", "")
                    if _mid and _mpw:
                        _lok, _lmsg = krx_login.ensure_login(_mid, _mpw)
                        if not _lok:
                            st.warning(f"⚠️ KRX 로그인 실패: {_lmsg}")
                    else:
                        st.warning(
                            "⚠️ KRX 로그인 정보 없음 — "
                            "사이드바에서 ID/PW를 입력하세요."
                        )
                except Exception as _le:
                    st.warning(f"KRX 로그인 모듈 오류: {_le}")

                # 오늘까지 전체 수집 (기준일 이후 실제 데이터도 포함)
                ohlcv_full = fetch_index_ohlcv_long(
                    index_code, _pred_end, years=train_years
                )
                fund_full = fetch_index_fundamentals_long(
                    index_code, _pred_end, years=train_years
                )
                inv_full = fetch_market_investor_long(
                    index_choice, _pred_end, years=train_years
                )
                ext_full = fetch_external_features(_pred_end, years=train_years)
                other_full = fetch_index_ohlcv_long(
                    other_code, _pred_end, years=train_years
                )

                # ── 기준일까지 truncate (리와인드) ──
                # 문자열 슬라이싱으로 인덱스 타입 문제 회피
                _ref_slice = _ref_date.strftime("%Y-%m-%d")

                def _safe_truncate(df, cutoff_str):
                    """DatetimeIndex 보장 후 기준일까지 슬라이싱."""
                    if df is None or df.empty:
                        return df
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            return df
                    return df.loc[:cutoff_str]

                ohlcv_full = _safe_truncate(ohlcv_full, None)  # 전체 유지
                if ohlcv_full is not None and not isinstance(
                    ohlcv_full.index, pd.DatetimeIndex
                ):
                    try:
                        ohlcv_full.index = pd.to_datetime(ohlcv_full.index)
                    except Exception:
                        pass
                ohlcv = (
                    ohlcv_full.loc[:_ref_slice]
                    if ohlcv_full is not None and not ohlcv_full.empty
                    else pd.DataFrame()
                )
                fundamentals = _safe_truncate(fund_full, _ref_slice)
                investor = _safe_truncate(inv_full, _ref_slice)
                external = _safe_truncate(ext_full, _ref_slice)
                other_ohlcv = _safe_truncate(other_full, _ref_slice)

                if ohlcv.empty:
                    st.error(f"기준일({_ref_date}) 이전 OHLCV 데이터가 없습니다.")
                else:
                    # 데이터 최종일이 기준일과 너무 차이나면 경고
                    _ohlcv_last_dt = ohlcv.index[-1]
                    _gap = (_ref_ts - _ohlcv_last_dt).days
                    if _gap > 30:
                        st.error(
                            f"⚠️ OHLCV 데이터가 "
                            f"{_ohlcv_last_dt.strftime('%Y-%m-%d')}까지만 "
                            f"존재합니다 (기준일 {_ref_date}와 {_gap}일 차이).\n\n"
                            f"**pykrx 수집이 실패**했을 수 있습니다. "
                            f"KRX 로그인 정보를 확인하거나, "
                            f"캐시를 삭제 후 다시 시도하세요.\n\n"
                            f"캐시 경로: `{INDEX_PRED_DIR}`"
                        )
                    else:
                        features = compute_index_features(
                            ohlcv, fundamentals, investor, external, other_ohlcv
                        )

                        if features.empty:
                            st.error("피처 계산 실패")
                        else:
                            latest_feats = features.iloc[-1]
                            # 현재가: 기준일(또는 직전 거래일) 종가
                            current_price = float(ohlcv["종가"].iloc[-1])
                            _actual_ref = ohlcv.index[-1].strftime("%Y-%m-%d")

                            pred = predict_index_future(
                                model, latest_feats, current_price
                            )
                            # 주별 보간 + KB 전략 오버레이
                            pred = interpolate_weekly_predictions(
                                pred,
                                current_price=current_price,
                                apply_kb_overlay=True,
                            )
                            # 기준일 정보 저장
                            pred["ref_date"] = _actual_ref
                            save_index_prediction(pred, _ref_dt_str, index_choice)
                            st.session_state[
                                f"idx_pred_result_{index_choice.lower()}"
                            ] = pred
                            # 전체 OHLCV 저장 (차트에서 실제 vs 예측 비교용)
                            st.session_state[
                                f"idx_pred_ohlcv_{index_choice.lower()}"
                            ] = ohlcv_full

                            # 월별전략 자동 생성
                            _strategy = generate_monthly_strategy(
                                pred,
                                index_name=index_choice,
                                recommend_results=st.session_state.get(
                                    "recommend_screening_results"
                                ),
                            )
                            st.session_state[
                                f"idx_monthly_strategy_{index_choice.lower()}"
                            ] = _strategy

                            _msg = (
                                f"✅ {index_choice} 예측 완료 — "
                                f"기준가 {current_price:,.0f} "
                                f"({_actual_ref} 종가), "
                                f"주별 {_N_WEEKS}주 + KB전략 반영"
                            )
                            if _ref_is_past:
                                _msg += (
                                    "\n\n📊 과거 기준일 예측 — "
                                    "차트에서 실제 지수와 비교하세요."
                                )
                            st.success(_msg)

            except Exception as e:
                st.error(f"예측 실패: {e}")

# ──────────────────────────────────────────────
# 탭 레이아웃
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 예측 차트", "📋 월별전략", "📋 모델 정보", "📈 역사적 패턴"]
)

# ── Tab 1: 예측 차트 ──
with tab1:
    pred_result = st.session_state.get(f"idx_pred_result_{index_choice.lower()}")
    if pred_result is None:
        pred_result = load_index_prediction(index_choice)
        if pred_result:
            st.session_state[f"idx_pred_result_{index_choice.lower()}"] = pred_result

    ohlcv_data = st.session_state.get(f"idx_pred_ohlcv_{index_choice.lower()}")

    # 차트용 OHLCV는 항상 *오늘 날짜*까지 시도 (date_str이 과거일 수 있음)
    _today_str = datetime.now().strftime("%Y%m%d")

    # OHLCV 데이터 갱신: 없거나 오래된 경우 최신 데이터 시도
    _ohlcv_stale = False
    if ohlcv_data is not None and not ohlcv_data.empty:
        _last_ohlcv_date = ohlcv_data.index[-1]
        _ohlcv_stale = (pd.Timestamp.now() - _last_ohlcv_date).days > 2

    if pred_result and (ohlcv_data is None or ohlcv_data.empty or _ohlcv_stale):
        # KRX 로그인 시도 (pykrx 증분 수집을 위해)
        try:
            from core import krx_login

            _mid = st.session_state.get("krx_mbr_id", "")
            _mpw = st.session_state.get("krx_pw", "")
            if _mid and _mpw:
                _lok, _lmsg = krx_login.ensure_login(_mid, _mpw)
                if not _lok:
                    st.warning(f"차트 데이터 수집: KRX 로그인 실패 — {_lmsg}")
        except Exception:
            pass

        # 기준일이 과거이면 해당 시점까지 포함하도록 넉넉히 수집
        _ref_str = pred_result.get("ref_date") or pred_result.get("prediction_date")
        if _ref_str:
            _ref_ts_chart = pd.Timestamp(_ref_str)
            _years_needed = max(3, (pd.Timestamp.now() - _ref_ts_chart).days // 365 + 3)
        else:
            _years_needed = 3
        try:
            _fresh_ohlcv = fetch_index_ohlcv_long(
                index_code, _today_str, years=_years_needed
            )
            if _fresh_ohlcv is not None and not _fresh_ohlcv.empty:
                ohlcv_data = _fresh_ohlcv
                st.session_state[f"idx_pred_ohlcv_{index_choice.lower()}"] = ohlcv_data
        except Exception:
            pass

    if pred_result and ohlcv_data is not None and not ohlcv_data.empty:
        import plotly.graph_objects as go

        # 기준일(ref_date) 기반으로 분리
        _ref_date_str = pred_result.get("ref_date")
        if _ref_date_str:
            pred_date = pd.Timestamp(_ref_date_str)
        else:
            pred_date = pd.Timestamp(pred_result.get("prediction_date", date_str))

        # 현재가 = 예측 기준일 종가 (pred_result에 저장된 값)
        current_price = pred_result["current_price"]

        # 기준일 이전 데이터 (차트 과거 영역)
        recent_start = pred_date - pd.DateOffset(years=2)
        hist_ohlcv = ohlcv_data[
            (ohlcv_data.index >= recent_start) & (ohlcv_data.index <= pred_date)
        ]

        # hist_ohlcv 가 비어있으면 전체 ohlcv 중 기준일 이전 데이터 탐색 (범위 확장)
        if hist_ohlcv.empty:
            hist_ohlcv = ohlcv_data[ohlcv_data.index <= pred_date]
        if hist_ohlcv.empty:
            st.warning(
                f"기준일({pred_date.strftime('%Y-%m-%d')}) 이전 OHLCV가 "
                f"없습니다. 예측 기준일을 확인하거나 데이터를 다시 수집하세요."
            )

        # 기준일 이후 실제 데이터 (예측 vs 실제 비교용)
        actual_ohlcv = ohlcv_data[ohlcv_data.index > pred_date]
        _has_actual = len(actual_ohlcv) > 0

        # OHLCV 최신 날짜 vs 오늘 — 오래되면 경고
        _ohlcv_last = ohlcv_data.index[-1]
        _days_stale = (pd.Timestamp.now() - _ohlcv_last).days
        if _days_stale > 3:
            st.warning(
                f"OHLCV 데이터가 "
                f"{_ohlcv_last.strftime('%Y-%m-%d')}까지만 "
                f"있습니다 ({_days_stale}일 전). "
                f"**예측 실행**을 다시 클릭하면 최신 데이터로 "
                f"갱신됩니다."
            )

        # 주별 예측 사용 (있으면), 없으면 월별에서 실시간 보간
        weekly_preds = pred_result.get("weekly_predictions")
        if not weekly_preds:
            pred_result = interpolate_weekly_predictions(
                pred_result, current_price=current_price, apply_kb_overlay=True
            )
            weekly_preds = pred_result["weekly_predictions"]

        # 주별 데이터 추출
        pred_dates_w = []
        pred_medians_w = []
        pred_lowers_w = []
        pred_uppers_w = []
        for wp in weekly_preds:
            pred_dates_w.append(pd.Timestamp(wp["date"]))
            pred_medians_w.append(current_price * (1 + wp["return_median"]))
            pred_lowers_w.append(current_price * (1 + wp["return_lower"]))
            pred_uppers_w.append(current_price * (1 + wp["return_upper"]))

        # 월별 앵커 포인트 (마커 표시용)
        monthly_preds = pred_result["predictions"]
        monthly_dates = []
        monthly_medians = []
        monthly_returns = []
        for p in monthly_preds:
            mdate = pred_date + pd.DateOffset(months=p["month"])
            monthly_dates.append(mdate)
            monthly_medians.append(current_price * (1 + p["return_median"]))
            monthly_returns.append(p["return_median"])

        import numpy as np

        # ── 차트 제목 (차트 밖 st.markdown으로 — 겹침 방지) ──
        kb_applied = pred_result.get("kb_overlay_applied", False)
        _kb_suffix = " + KB전략" if kb_applied else ""
        _ref_suffix = (
            f" [기준일: {pred_date.strftime('%Y-%m-%d')}]" if _has_actual else ""
        )
        st.markdown(
            f"**{index_choice} 향후 {_N_YEARS}년 주별 예측"
            f"{_kb_suffix}{_ref_suffix}**"
        )

        fig = go.Figure()

        # ── 3저호황 과거 그래프 오버레이 매핑 ──
        # KB 그림 39 기준: 1986-08(금리인하 종료) = 예측기준일(pred_date)
        # 1:1 일 단위 매핑, 스케일은 실제 기준가 기준
        from core.index_predictor import _stooq_index_ohlcv

        _show_3low = st.checkbox(
            "3저호황(1984~1990) 패턴 오버레이 (KB 그림39 기준)",
            value=True,
            key="idx_show_3low_overlay",
        )
        if _show_3low and index_choice == "KOSPI":
            # 매핑: 1986-08-01(금리인하 종료) ↔ pred_date (1:1 일 단위)
            _3LOW_ANCHOR_HIST = pd.Timestamp("1986-08-01")
            _3LOW_ANCHOR_NOW = pd.Timestamp(pred_date)
            _3low_data = _stooq_index_ohlcv("1001", "19840101", "19900701")
            if _3low_data is not None and not _3low_data.empty:
                # KB 그림39 스케일: 1984-07 KOSPI → 2023-07 KOSPI(≈2,600)
                # 시간 시프트와 스케일은 독립: 시간은 1986-08→pred_date,
                # 스케일은 원래 KB 앵커(1984-07/2023-07) 기준 유지
                _kb_1984 = _3low_data.loc[_3low_data.index >= "1984-07-01", "종가"]
                if not _kb_1984.empty:
                    _3low_anchor_price = float(_kb_1984.iloc[0])
                    _kb_anchor_now = 2600.0  # 2023-07 KOSPI
                    _3low_scale = _kb_anchor_now / _3low_anchor_price
                    # 일별 데이터를 현재 타임라인으로 매핑
                    _day_offsets = (_3low_data.index - _3LOW_ANCHOR_HIST).days
                    _3low_mapped_dates = [
                        _3LOW_ANCHOR_NOW + pd.DateOffset(days=int(d))
                        for d in _day_offsets
                    ]
                    _3low_scaled_vals = (_3low_data["종가"] * _3low_scale).values
                    _3low_orig_vals = _3low_data["종가"].values
                    # 주별 리샘플로 포인트 수 줄임 (가독성)
                    _3low_weekly = _3low_data["종가"].resample("W").last().dropna()
                    _day_off_w = (_3low_weekly.index - _3LOW_ANCHOR_HIST).days
                    _3low_w_dates = [
                        _3LOW_ANCHOR_NOW + pd.DateOffset(days=int(d))
                        for d in _day_off_w
                    ]
                    _3low_w_scaled = (_3low_weekly * _3low_scale).values
                    _3low_w_orig = _3low_weekly.values
                    # 원본 날짜 (호버용)
                    _3low_w_labels = [d.strftime("%Y-%m") for d in _3low_weekly.index]
                    fig.add_trace(
                        go.Scatter(
                            x=_3low_w_dates,
                            y=_3low_w_scaled,
                            mode="lines",
                            name="3저호황 패턴 (1984~90, KB 그림39)",
                            line={
                                "color": "#FF9800",
                                "width": 2.5,
                            },
                            opacity=0.8,
                            hovertemplate=(
                                "3저호황 매핑: %{y:,.0f}<br>"
                                "원본: %{customdata}<extra></extra>"
                            ),
                            customdata=[
                                f"{d} KOSPI {v:.0f}"
                                for d, v in zip(
                                    _3low_w_labels,
                                    _3low_w_orig,
                                    strict=False,
                                )
                            ],
                        )
                    )

        # 기준일 이전 실제 지수 (파란선)
        _hist_anchor = hist_ohlcv.index[-1] if not hist_ohlcv.empty else pred_date
        if not hist_ohlcv.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist_ohlcv.index,
                    y=hist_ohlcv["종가"],
                    mode="lines",
                    name=f"실제 {index_choice} (기준일 이전)",
                    line={"color": "#1976D2", "width": 2},
                )
            )

        # 기준일 이후 실제 지수 오버레이 (녹색 — 예측 vs 실제 비교)
        if _has_actual:
            _actual_x = [_hist_anchor] + list(actual_ohlcv.index)
            _actual_y = [current_price] + list(actual_ohlcv["종가"])
            fig.add_trace(
                go.Scatter(
                    x=_actual_x,
                    y=_actual_y,
                    mode="lines",
                    name="실제 지수 (기준일 이후)",
                    line={
                        "color": "#4CAF50",
                        "width": 2.5,
                        "dash": "solid",
                    },
                )
            )

        # 현재가 → 예측 연결점 (주별)
        connect_dates = [_hist_anchor] + pred_dates_w
        connect_medians = [current_price] + pred_medians_w
        connect_lowers = [current_price] + pred_lowers_w
        connect_uppers = [current_price] + pred_uppers_w

        # 90% 신뢰구간 상한선 (점선)
        fig.add_trace(
            go.Scatter(
                x=connect_dates,
                y=connect_uppers,
                mode="lines",
                name="상한 (90%)",
                line={"color": "rgba(229,57,53,0.3)", "width": 1, "dash": "dot"},
                hoverinfo="skip",
            )
        )

        # 90% 신뢰구간 하한선 (더 진한 점선으로 가시성 확보)
        fig.add_trace(
            go.Scatter(
                x=connect_dates,
                y=connect_lowers,
                mode="lines",
                name="하한 (10%)",
                line={
                    "color": "rgba(229,57,53,0.5)",
                    "width": 1.5,
                    "dash": "dash",
                },
                fill="tonexty",
                fillcolor="rgba(229,57,53,0.06)",
                hoverinfo="skip",
            )
        )

        # 예측 (빨간 실선 — 첫 3개월 일별 + 이후 주간)
        _w_hover = [f"현재: {current_price:,.0f}"]
        for wp in weekly_preds:
            _is_daily = wp.get("is_daily", False)
            if _is_daily:
                _label = f"+{wp['week']:.1f}주"
                _tag = "일별"
            else:
                _label = f"+{wp['week']}주"
                _tag = "주간"
            _w_hover.append(
                f"<b>{_label} ({wp['date']}) [{_tag}]</b><br>"
                f"예측: {wp['price_median']:,.0f}<br>"
                f"수익률: {wp['return_median']:+.1%}<br>"
                f"범위: {wp['price_lower']:,.0f} ~ {wp['price_upper']:,.0f}"
            )
        fig.add_trace(
            go.Scatter(
                x=connect_dates,
                y=connect_medians,
                mode="lines",
                name="예측 (일별+주간)",
                line={"color": "#E53935", "width": 2, "dash": "solid"},
                hovertext=_w_hover,
                hoverinfo="text",
            )
        )

        # ── 변곡점 지수 표시 (방향 전환 포인트마다 지수값 annotation) ──
        _med_arr = np.array(connect_medians)
        _inflection_indices = []
        for _ii in range(1, len(_med_arr) - 1):
            _d1 = _med_arr[_ii] - _med_arr[_ii - 1]
            _d2 = _med_arr[_ii + 1] - _med_arr[_ii]
            # 방향 전환 (부호 변경) 또는 6개월 간격
            if (_d1 > 0 and _d2 < 0) or (_d1 < 0 and _d2 > 0):
                _inflection_indices.append(_ii)
        # 최소 6개월 간격으로 중요 변곡점만 선별
        _filtered_inflections = []
        _last_idx = -26  # 첫 변곡점은 무조건 포함
        for _ii in _inflection_indices:
            if _ii - _last_idx >= 13:  # 최소 13주(~3개월) 간격
                _filtered_inflections.append(_ii)
                _last_idx = _ii
        # 마지막 포인트 항상 포함
        if len(connect_dates) > 1:
            _filtered_inflections.append(len(connect_dates) - 1)
        # 시작점도 포함
        _filtered_inflections = [0] + _filtered_inflections
        _filtered_inflections = sorted(set(_filtered_inflections))

        for _fi in _filtered_inflections:
            _val = connect_medians[_fi]
            _is_top = True
            if 0 < _fi < len(_med_arr) - 1:
                _is_top = _med_arr[_fi] >= max(
                    _med_arr[max(0, _fi - 4)], _med_arr[min(len(_med_arr) - 1, _fi + 4)]
                )
            fig.add_annotation(
                x=str(connect_dates[_fi]),
                y=_val,
                text=f"{_val:,.0f}",
                showarrow=True,
                arrowhead=0,
                ax=0,
                ay=-25 if _is_top else 25,
                font={"size": 9, "color": "#333"},
                bgcolor="rgba(255,255,255,0.8)",
                borderpad=2,
            )

        # 월별 앵커 마커 (마커만 — 텍스트는 변곡점에서만 표시)
        fig.add_trace(
            go.Scatter(
                x=monthly_dates,
                y=monthly_medians,
                mode="markers",
                name="월별 앵커",
                marker={
                    "size": 6,
                    "color": "#E53935",
                    "symbol": "circle",
                    "line": {"width": 1, "color": "white"},
                },
                hovertext=[
                    f"{md.strftime('%Y-%m')}: {mv:,.0f} ({mr:+.1%})"
                    for md, mv, mr in zip(
                        monthly_dates, monthly_medians, monthly_returns, strict=False
                    )
                ],
                hoverinfo="text",
            )
        )

        # KB 전략 분기 영역 표시
        if kb_applied:
            _q_colors = {
                "Q1": "rgba(76,175,80,0.06)",
                "Q2": "rgba(244,67,54,0.06)",
                "Q3": "rgba(76,175,80,0.04)",
                "Q4": "rgba(158,158,158,0.04)",
            }
            _q_labels = {
                "Q1": "Q1 알파↑",
                "Q2": "Q2 베타↓",
                "Q3": "Q3 회복↑",
                "Q4": "Q4 중립",
            }
            for _yr in range(pred_date.year, pred_date.year + _N_YEARS + 1):
                for _qi, (_qm_start, _qm_end) in enumerate(
                    [(1, 3), (4, 6), (7, 9), (10, 12)], 1
                ):
                    _q_start = pd.Timestamp(f"{_yr}-{_qm_start:02d}-01")
                    _q_end = pd.Timestamp(f"{_yr}-{_qm_end:02d}-28") + pd.DateOffset(
                        days=3
                    )
                    _q_end = _q_end.replace(day=1) - pd.DateOffset(days=1)
                    if _q_end < pred_date or _q_start > pred_dates_w[-1]:
                        continue
                    _q_start = max(_q_start, pred_date)
                    _q_end = min(_q_end, pred_dates_w[-1])
                    _qname = f"Q{_qi}"
                    fig.add_shape(
                        type="rect",
                        x0=str(_q_start),
                        x1=str(_q_end),
                        y0=0,
                        y1=1,
                        yref="paper",
                        fillcolor=_q_colors.get(_qname, "rgba(0,0,0,0)"),
                        layer="below",
                        line_width=0,
                    )
                    _mid_q = _q_start + (_q_end - _q_start) / 2
                    fig.add_annotation(
                        x=str(_mid_q),
                        y=0.02,
                        yref="paper",
                        text=_q_labels.get(_qname, _qname),
                        showarrow=False,
                        font={"size": 9, "color": "gray"},
                    )

        # 기준일 시점 표시
        _vline_x = _hist_anchor
        if hasattr(_vline_x, "to_pydatetime"):
            _vline_x = _vline_x.to_pydatetime()
        _vline_x_str = str(_vline_x)
        _vline_label = (
            f"기준일 ({pred_date.strftime('%m/%d')})" if _has_actual else "현재"
        )
        fig.add_shape(
            type="line",
            x0=_vline_x_str,
            x1=_vline_x_str,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": "gray", "width": 1, "dash": "dot"},
        )
        fig.add_annotation(
            x=_vline_x_str,
            y=1.02,
            yref="paper",
            text=_vline_label,
            showarrow=False,
            font={"size": 11, "color": "gray"},
        )

        # 초기 표시 범위 (1년 과거 ~ 예측 끝)
        _range_start = (pred_date - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        _range_end = (pred_dates_w[-1] + pd.DateOffset(weeks=4)).strftime("%Y-%m-%d")

        fig.update_layout(
            xaxis_title="날짜",
            yaxis_title="지수",
            height=700,
            margin={"l": 50, "r": 20, "t": 30, "b": 60},
            legend={"orientation": "h", "y": 1.08, "x": 0},
            hovermode="x",
            xaxis={
                "range": [_range_start, _range_end],
                "rangeslider": {"visible": True, "thickness": 0.05},
                "rangeselector": {
                    "buttons": [
                        {
                            "count": 1,
                            "label": "1년",
                            "step": "year",
                            "stepmode": "backward",
                        },
                        {
                            "count": 2,
                            "label": "2년",
                            "step": "year",
                            "stepmode": "backward",
                        },
                        {
                            "count": 3,
                            "label": "3년",
                            "step": "year",
                            "stepmode": "backward",
                        },
                        {"label": "전체", "step": "all"},
                    ],
                    "y": 1.18,
                    "x": 1.0,
                    "xanchor": "right",
                },
                "dtick": "M3",
                "tickformat": "%Y-%m",
                "tickangle": -45,
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        if kb_applied:
            st.caption(
                "KB증권 '1월 전략' 분기별 알파/베타 전략 반영: "
                "Q1 알파 드리븐(상향), Q2 베타 컨트롤(하향), Q3 회복(상향), Q4 중립. "
                "| 주황선: 3저호황(1984~90) 역사적 시나리오"
            )
            st.caption(
                "주황선은 1986-08 금리인하 종료 시점을 기준일에 매핑한 참고 패턴입니다. "
                "3저호황 당시 KOSPI는 3년간 +525% 상승했으나, "
                "AI 예측(빨간선) = AI모델 50% + KB전략 25% + 상승장패턴 25% 블렌딩. "
                "주황선과의 차이는 패턴 가중평균(3저호황 40%+유동성 25%+IMF 20%+코로나 15%) 반영 때문."
            )
        else:
            st.caption("하단 슬라이더를 드래그하거나 상단 버튼으로 기간을 조절하세요.")

        # ── 예측 vs 실제 비교 (기준일이 과거일 때) ──
        if _has_actual:
            # 기준일 직후 첫 번째 실제 거래일 기준으로 비교
            _cmp_actual = float(actual_ohlcv["종가"].iloc[0])
            _cmp_date = actual_ohlcv.index[0]
            _cmp_date_str = _cmp_date.strftime("%m/%d")
            _cmp_days = (_cmp_date - pred_date).days
            _cmp_actual_ret = _cmp_actual / current_price - 1

            # 해당 날짜의 예측값 보간 (주별 예측에서)
            _cmp_frac_week = max(0.1, _cmp_days / 7.0)
            if _cmp_frac_week <= 1.0 and len(weekly_preds) >= 1:
                # 1주 미만: 선형 보간
                _cmp_pred_ret = weekly_preds[0]["return_median"] * _cmp_frac_week
                _cmp_pred_lo = weekly_preds[0]["return_lower"] * _cmp_frac_week
                _cmp_pred_hi = weekly_preds[0]["return_upper"] * _cmp_frac_week
            else:
                _cmp_wi = max(0, int(_cmp_frac_week) - 1)
                _cmp_wi = min(_cmp_wi, len(weekly_preds) - 1)
                _cmp_pred_ret = weekly_preds[_cmp_wi]["return_median"]
                _cmp_pred_lo = weekly_preds[_cmp_wi]["return_lower"]
                _cmp_pred_hi = weekly_preds[_cmp_wi]["return_upper"]
            _cmp_pred_price = current_price * (1 + _cmp_pred_ret)
            _cmp_error = _cmp_actual - _cmp_pred_price
            _cmp_error_pct = _cmp_error / current_price
            _cmp_in_ci = _cmp_pred_lo <= _cmp_actual_ret <= _cmp_pred_hi

            st.subheader("📊 예측 vs 실제 비교")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric(
                    f"기준가 ({pred_date.strftime('%m/%d')})",
                    f"{current_price:,.0f}",
                )
            with c2:
                st.metric(
                    f"예측 ({_cmp_date_str})",
                    f"{_cmp_pred_price:,.0f}",
                    f"{_cmp_pred_ret:+.1%}",
                )
            with c3:
                st.metric(
                    f"실제 ({_cmp_date_str})",
                    f"{_cmp_actual:,.0f}",
                    f"{_cmp_actual_ret:+.1%}",
                )
            with c4:
                st.metric(
                    "오차",
                    f"{_cmp_error:+,.0f}",
                    f"{_cmp_error_pct:+.1%}",
                    delta_color="off",
                )
            with c5:
                st.metric(
                    "CI 내 적중",
                    "O" if _cmp_in_ci else "X",
                    "90% 신뢰구간",
                    delta_color="off",
                )

        # 전쟁패턴 정보 표시
        _war_start_str = pred_result.get("iran_war_start")
        if _war_start_str:
            _war_el = pred_result.get("war_months_elapsed", 0)
            st.info(
                f"⚔️ **이란전쟁 패턴 반영** | 발발: {_war_start_str} | "
                f"경과: {_war_el}개월 | "
                f"과거 중동전쟁(걸프전·이라크전·이란긴장) 실제 KOSPI 데이터 기반 오버레이 적용 | "
                f"첫 3개월 일별(2일 간격) 세밀 예측"
            )

        # 예측 요약 — 1달/6개월/1년/2년/3년
        st.subheader("예측 요약")
        _horizons = [
            ("1달후", 1),
            ("6개월후", 6),
            ("1년후", 12),
            ("2년후", 24),
            ("3년후", min(36, len(monthly_preds))),
        ]
        _sum_cols = st.columns(1 + len(_horizons))
        with _sum_cols[0]:
            st.metric(
                "기준 지수",
                f"{current_price:,.0f}",
                f"{pred_date.strftime('%Y-%m-%d')}",
                delta_color="off",
            )
        for _hi, (_hlabel, _hmonth) in enumerate(_horizons):
            _hmi = min(_hmonth - 1, len(monthly_preds) - 1)
            _hp = monthly_preds[_hmi]
            _hret = _hp["return_median"]
            _hprice = current_price * (1 + _hret)
            _htarget = (pred_date + pd.DateOffset(months=_hmonth)).strftime("%Y-%m")
            with _sum_cols[_hi + 1]:
                st.metric(
                    _hlabel,
                    f"{_hprice:,.0f}",
                    f"{_hret:+.1%} ({_htarget})",
                )

        # 탭: 월별 / 주별 예측 상세
        _detail_tabs = ["월별 예측 상세", "일별/주별 예측 상세"]
        detail_tab1, detail_tab2 = st.tabs(_detail_tabs)

        with detail_tab1:
            table_data = []
            for p in monthly_preds:
                _pdate = pred_date + pd.DateOffset(months=p["month"])
                _med = current_price * (1 + p["return_median"])
                _low = current_price * (1 + p["return_lower"])
                _up = current_price * (1 + p["return_upper"])
                _chg = _med - current_price
                row = {
                    "개월": f"+{p['month']}M",
                    "예측일": _pdate.strftime("%Y-%m"),
                    "하한 (10%)": f"{_low:,.0f}",
                    "중앙값": f"{_med:,.0f}",
                    "상한 (90%)": f"{_up:,.0f}",
                    "등락폭": f"{_chg:+,.0f}",
                    "예측수익률": f"{p['return_median']:+.1%}",
                    "CI 폭": f"{_up - _low:,.0f}",
                }
                # 실제 데이터가 있으면 비교 컬럼 추가
                if _has_actual:
                    _target = _pdate
                    _near = actual_ohlcv.index[
                        actual_ohlcv.index.searchsorted(_target, side="right").clip(
                            0, len(actual_ohlcv) - 1
                        )
                    ]
                    if len(actual_ohlcv) > 0 and abs((_near - _target).days) <= 15:
                        _av = float(actual_ohlcv.loc[_near, "종가"])
                        _ar = _av / current_price - 1
                        row["실제"] = f"{_av:,.0f}"
                        row["실제수익률"] = f"{_ar:+.1%}"
                        _err = _av - _med
                        row["오차"] = f"{_err:+,.0f}"
                    else:
                        row["실제"] = "-"
                        row["실제수익률"] = "-"
                        row["오차"] = "-"
                table_data.append(row)
            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True,
            )

        with detail_tab2:
            # 전쟁패턴 정보 표시
            _war_info = pred_result.get("iran_war_start")
            if _war_info:
                _war_elapsed = pred_result.get("war_months_elapsed", 0)
                st.caption(
                    f"⚔️ 이란전쟁 발발: {_war_info} | "
                    f"경과: {_war_elapsed}개월 | "
                    f"전쟁패턴 오버레이 적용 중 "
                    f"(걸프전 35% + 이라크전 40% + 이란긴장 25%)"
                )

            # 일별(첫 3개월) + 주간(이후) 구분 표시
            _daily_preds = [wp for wp in weekly_preds if wp.get("is_daily", False)]
            _weekly_only = [wp for wp in weekly_preds if not wp.get("is_daily", False)]
            if _daily_preds:
                st.markdown(
                    f"**첫 3개월 일별 예측** ({len(_daily_preds)}개 포인트, "
                    f"2일 간격)"
                )

            w_table = []
            for wp in weekly_preds:
                _chg_w = wp["price_median"] - current_price
                _is_daily = wp.get("is_daily", False)
                if _is_daily:
                    _period_label = f"+{wp['week']:.1f}W"
                    _type_label = "일별"
                else:
                    _period_label = f"+{wp['week']}W"
                    _type_label = "주간"
                row_w = {
                    "구분": _type_label,
                    "기간": _period_label,
                    "날짜": wp["date"],
                    "하한": f"{wp['price_lower']:,.0f}",
                    "중앙값": f"{wp['price_median']:,.0f}",
                    "상한": f"{wp['price_upper']:,.0f}",
                    "등락폭": f"{_chg_w:+,.0f}",
                    "수익률": f"{wp['return_median']:+.1%}",
                }
                if _has_actual:
                    _wt = pd.Timestamp(wp["date"])
                    _idx = actual_ohlcv.index.searchsorted(_wt, side="right")
                    _idx = min(_idx, len(actual_ohlcv) - 1)
                    if (
                        len(actual_ohlcv) > 0
                        and _idx >= 0
                        and abs((actual_ohlcv.index[_idx] - _wt).days) <= 5
                    ):
                        _wav = float(actual_ohlcv["종가"].iloc[_idx])
                        row_w["실제"] = f"{_wav:,.0f}"
                        row_w["오차"] = f"{_wav - wp['price_median']:+,.0f}"
                    else:
                        row_w["실제"] = "-"
                        row_w["오차"] = "-"
                w_table.append(row_w)
            st.dataframe(
                pd.DataFrame(w_table),
                use_container_width=True,
                hide_index=True,
            )

    elif pred_result:
        st.info(
            "예측 결과는 있지만 차트 데이터가 없습니다. '예측 실행'을 다시 실행하세요."
        )
    else:
        st.info(
            "예측 결과가 없습니다.\n\n"
            "1. 사이드바에서 **🔄 모델 학습**을 실행하세요.\n"
            "2. 학습 완료 후 **🔮 예측 실행**을 클릭하세요."
        )

# ── Tab 2: 월별전략 (AI 50% + KB 25% + 상승장패턴 25%) ──
with tab2:
    _strat_key = f"idx_monthly_strategy_{index_choice.lower()}"
    _strategies = st.session_state.get(_strat_key)

    if not _strategies:
        _pred_for_strat = st.session_state.get(
            f"idx_pred_result_{index_choice.lower()}"
        )
        if _pred_for_strat is None:
            _pred_for_strat = load_index_prediction(index_choice)
        if _pred_for_strat and _pred_for_strat.get("predictions"):
            _strategies = generate_monthly_strategy(
                _pred_for_strat,
                index_name=index_choice,
                recommend_results=st.session_state.get("recommend_screening_results"),
            )
            st.session_state[_strat_key] = _strategies

    if _strategies:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        st.subheader(
            f"{index_choice} 36개월 월별 투자전략 " "(AI 50% + KB 25% + 상승장패턴 25%)"
        )

        # ── A. 요약 대시보드 (확장: 5열 + 투자유형 태그) ──
        cur = _strategies[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("분기 전략", f"Q{cur['quarter']} {cur['style']}")
        c2.metric("포지션/비중", f"{cur['position']} / {cur['equity_weight']:.0%}")
        c3.metric("매수우선순위", cur.get("buy_priority", "")[:20])
        c4.metric(
            "예측수익률",
            f"{cur['predicted_return']:+.1%}",
            delta=cur["direction"],
        )
        c5.metric("KB 확신도", f"{cur.get('kb_conviction', 0.5):.0%}")

        # 투자유형 태그
        _type_tags = " ".join(
            f"{it.get('icon', '')} **{it['label']}**"
            for it in cur.get("invest_types", [])
        )
        if _type_tags:
            st.markdown(f"**추천 투자유형**: {_type_tags}")

        # 매수조건 & 리스크
        _conds = cur.get("specific_conditions", [])
        if _conds:
            st.info("**매수 조건**: " + " | ".join(_conds[:3]))
        _risks = cur.get("risk_notes", [])
        if _risks:
            st.warning("**리스크**: " + " | ".join(_risks[:3]))

        st.divider()

        # ── B. 비중/수익률 차트 ──
        st.subheader("월별 비중 & 누적예측수익률")

        dates = [s["date_label"] for s in _strategies]
        weights = [s["equity_weight"] for s in _strategies]
        returns = [s["predicted_return"] for s in _strategies]
        cum_ret = np.cumprod([1 + r for r in returns]).tolist()
        cum_ret_pct = [(v - 1) * 100 for v in cum_ret]

        bar_colors = []
        for s in _strategies:
            if s["direction"] in ("상승", "소폭상승", "강세상승"):
                bar_colors.append("#2ca02c")
            elif s["direction"] in ("하락", "소폭하락", "강세하락"):
                bar_colors.append("#d62728")
            else:
                bar_colors.append("#7f7f7f")

        fig_wt = make_subplots(specs=[[{"secondary_y": True}]])
        fig_wt.add_trace(
            go.Bar(
                x=dates,
                y=[w * 100 for w in weights],
                name="주식비중(%)",
                marker_color=bar_colors,
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig_wt.add_trace(
            go.Scatter(
                x=dates,
                y=cum_ret_pct,
                name="누적수익률(%)",
                mode="lines+markers",
                line={"color": "#1f77b4", "width": 2},
                marker={"size": 4},
            ),
            secondary_y=True,
        )
        for idx, d in enumerate(dates):
            m = int(d.split("-")[1])
            if m in (1, 4, 7, 10) and idx > 0:
                fig_wt.add_vline(
                    x=idx - 0.5, line_dash="dot", line_color="gray", opacity=0.4
                )
        fig_wt.update_layout(
            height=350,
            margin={"l": 50, "r": 50, "t": 30, "b": 50},
            legend={"orientation": "h", "y": 1.12},
            xaxis_tickangle=-45,
            xaxis_type="category",
        )
        fig_wt.update_yaxes(title_text="주식비중(%)", secondary_y=False)
        fig_wt.update_yaxes(title_text="누적수익률(%)", secondary_y=True)
        st.plotly_chart(fig_wt, use_container_width=True)

        st.divider()

        # ── C. 36개월 전략 테이블 (확장 컬럼) ──
        st.subheader("상세 전략 테이블")
        from collections import defaultdict

        year_groups: dict[str, list] = defaultdict(list)
        for s in _strategies:
            yr = s["date_label"][:4]
            year_groups[yr].append(s)

        for yr in sorted(year_groups.keys()):
            with st.expander(f"{yr}년", expanded=(yr == sorted(year_groups.keys())[0])):
                rows = []
                for s in year_groups[yr]:
                    _types_str = ", ".join(
                        f"{it.get('icon', '')}{it['label']}"
                        for it in s.get("invest_types", [])
                    )
                    rows.append(
                        {
                            "월": s["date_label"],
                            "방향": s["direction"],
                            "포지션": s["position"],
                            "비중": f"{s['equity_weight']:.0%}",
                            "매수우선순위": s.get("buy_priority", ""),
                            "투자유형": _types_str,
                            "추천업종": ", ".join(s["top_sectors"]),
                            "회피업종": ", ".join(s.get("avoid_sectors", [])),
                            "수익률": f"{s['predicted_return']:+.1%}",
                            "HR": f"{s['hit_ratio']:.0f}%",
                            "확신도": f"{s.get('kb_conviction', 0.5):.0%}",
                            "전략메모": s["strategy_note"],
                        }
                    )
                df_year = pd.DataFrame(rows)
                st.dataframe(
                    df_year,
                    use_container_width=True,
                    hide_index=True,
                    height=min(len(rows) * 40 + 40, 500),
                )

        st.divider()

        # ── D. 업종 히트맵 (18개 업종) ──
        st.subheader("업종별 월간 어닝스 모멘텀 히트맵 (18개 업종)")

        sectors = list(KB_SECTOR_MONTHLY.keys())
        months = list(range(1, 13))
        z_data = [[KB_SECTOR_MONTHLY[sec].get(m, 0) for m in months] for sec in sectors]

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=[f"{m}월" for m in months],
                y=sectors,
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{v:+d}" for v in row] for row in z_data],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar_title="%p",
            )
        )
        fig_hm.update_layout(
            height=max(350, len(sectors) * 22),
            margin={"l": 100, "r": 20, "t": 30, "b": 40},
            xaxis_title="월",
            yaxis_title="업종",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.divider()

        # ── E. 투자유형 분포 차트 (신규) ──
        st.subheader("투자유형 분포 (36개월)")
        _type_counts: dict[str, int] = {}
        for s in _strategies:
            for it in s.get("invest_types", []):
                _k = it["label"]
                _type_counts[_k] = _type_counts.get(_k, 0) + 1
        _sorted_types = sorted(_type_counts.items(), key=lambda x: x[1], reverse=True)
        if _sorted_types:
            fig_types = go.Figure(
                go.Bar(
                    x=[t[1] for t in _sorted_types],
                    y=[t[0] for t in _sorted_types],
                    orientation="h",
                    marker_color="#FF6F00",
                )
            )
            fig_types.update_layout(
                height=max(250, len(_sorted_types) * 30),
                margin={"l": 120, "r": 20, "t": 10, "b": 20},
                xaxis_title="추천 월수 (36개월 중)",
            )
            st.plotly_chart(fig_types, use_container_width=True)

        st.divider()

        # ── F. KB 확신도 타임라인 (신규) ──
        st.subheader("KB 전략 확신도 타임라인")
        _conv_vals = [s.get("kb_conviction", 0.5) * 100 for s in _strategies]
        fig_conv = go.Figure()
        fig_conv.add_trace(
            go.Scatter(
                x=dates,
                y=_conv_vals,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(255,152,0,0.15)",
                line={"color": "#FF9800", "width": 2},
                marker={"size": 4},
            )
        )
        fig_conv.add_hline(y=50, line_dash="dot", line_color="gray")
        fig_conv.update_layout(
            height=250,
            yaxis_title="확신도 (%)",
            yaxis_range=[0, 100],
            margin={"l": 50, "r": 20, "t": 10, "b": 50},
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # ── G. 추천종목 (KB Research Coverage 기반) ──
        st.divider()
        st.subheader("KB 추천종목 (월별)")
        st.caption(
            "KB증권 Research Coverage (PDF pp.56-80) + Table 19 Model Portfolio 기반. "
            "분기별 활성 업종에서 Buy 의견 종목을 자동 선정합니다."
        )

        # 6개월치 탭으로 표시
        _pick_months = min(6, len(_strategies))
        _pick_tab_labels = [_strategies[j]["date_label"] for j in range(_pick_months)]
        _pick_tabs = st.tabs(_pick_tab_labels)
        for _pti, _pt in enumerate(_pick_tabs):
            with _pt:
                _s = _strategies[_pti]
                _picks = _s.get("stock_picks", [])
                if _picks:
                    _pick_rows = []
                    for _pk in _picks:
                        _pick_rows.append(
                            {
                                "종목명": _pk.get("name", ""),
                                "업종": _pk.get("sector", ""),
                                "의견": _pk.get("opinion", ""),
                                "사유": _pk.get("reason", ""),
                                "업종비중(bp)": (
                                    f"{_pk['weight_bp']:+d}"
                                    if _pk.get("weight_bp")
                                    else "-"
                                ),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(_pick_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                    # 투자유형/전략 요약
                    _types_str = ", ".join(
                        f"{it.get('icon', '')}{it['label']}"
                        for it in _s.get("invest_types", [])
                    )
                    if _types_str:
                        st.caption(f"추천 투자유형: {_types_str}")
                    if _s.get("buy_priority"):
                        st.caption(f"매수우선순위: {_s['buy_priority']}")
                else:
                    st.info(f"{_s['date_label']} — 해당 분기 추천종목 없음")
    else:
        st.info(
            "월별전략을 생성하려면 먼저 예측을 실행하세요.\n\n"
            "1. 사이드바에서 **모델 학습**을 실행하세요.\n"
            "2. 학습 완료 후 **예측 실행**을 클릭하세요."
        )

# ── Tab 3: 모델 정보 ──
with tab3:
    model = st.session_state.get(f"idx_pred_model_{index_choice.lower()}")
    if model is None:
        model = load_index_model(index_choice)

    if model:
        # 기본 정보
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("학습일", model.get("train_date", "?"))
        with col2:
            st.metric("학습 샘플", f"{model.get('n_train_samples', 0)}개월")
        with col3:
            st.metric("피처 수", f"{len(model.get('feature_names', []))}개")
        with col4:
            v = model.get("validation", {})
            st.metric("방향 정확도", f"{v.get('avg_direction_accuracy', 0):.1%}")

        # 검증 결과
        v = model.get("validation", {})
        st.subheader("Walk-Forward 검증 결과")
        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            st.metric("평균 MAE", f"{v.get('avg_mae', 0):.4f}")
        with vcol2:
            st.metric("평균 방향 정확도", f"{v.get('avg_direction_accuracy', 0):.1%}")
        with vcol3:
            st.metric("90% CI 커버리지", f"{v.get('avg_coverage', 0):.1%}")

        folds = v.get("folds", [])
        if folds:
            fold_df = pd.DataFrame(
                [
                    {
                        "Fold": f"Fold {i + 1}",
                        "MAE": f"{f['mae']:.4f}",
                        "방향 정확도": f"{f['direction_accuracy']:.1%}",
                        "CI 커버리지": f"{f['coverage']:.1%}",
                    }
                    for i, f in enumerate(folds)
                ]
            )
            st.dataframe(fold_df, use_container_width=True, hide_index=True)

        # 피처 중요도
        importance = model.get("importance", {})
        if importance:
            st.subheader("피처 중요도 (Top 15)")

            # 전체 구간 중요도 합산
            combined_imp = {}
            for _period_key, imp_dict in importance.items():
                for feat, val in imp_dict.items():
                    combined_imp[feat] = combined_imp.get(feat, 0) + val

            sorted_imp = sorted(combined_imp.items(), key=lambda x: x[1], reverse=True)[
                :15
            ]
            imp_names = [x[0] for x in sorted_imp]
            imp_vals = [x[1] for x in sorted_imp]

            import plotly.graph_objects as go

            fig_imp = go.Figure(
                go.Bar(
                    x=imp_vals[::-1],
                    y=imp_names[::-1],
                    orientation="h",
                    marker_color="#1976D2",
                )
            )
            fig_imp.update_layout(
                title="피처 중요도 (gain 합산)",
                height=400,
                margin={"l": 150, "r": 20, "t": 40, "b": 20},
                xaxis_title="Importance (gain)",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("학습된 모델이 없습니다. 사이드바에서 '모델 학습'을 실행하세요.")

# ── Tab 4: 역사적 패턴 ──
with tab4:
    st.subheader(f"{index_choice} 장기 차트 + 역사적 리짐")

    # 장기 OHLCV 로드 (캐시에서 또는 3년분)
    long_ohlcv = st.session_state.get(f"idx_pred_ohlcv_{index_choice.lower()}")

    if long_ohlcv is not None and not long_ohlcv.empty:
        import plotly.graph_objects as go

        fig_hist = go.Figure()

        # 지수 선
        fig_hist.add_trace(
            go.Scatter(
                x=long_ohlcv.index,
                y=long_ohlcv["종가"],
                mode="lines",
                name=index_choice,
                line={"color": "#1976D2", "width": 1.5},
            )
        )

        # 리짐 밴드 (Plotly datetime 축 호환을 위해 문자열 변환)
        colors = {"bull": "rgba(76,175,80,0.15)", "bear": "rgba(244,67,54,0.15)"}
        for event in REGIME_EVENTS:
            start_dt = pd.Timestamp(event["start"])
            end_dt = pd.Timestamp(event["end"])
            # 데이터 범위 내인지 확인
            if end_dt < long_ohlcv.index[0] or start_dt > long_ohlcv.index[-1]:
                continue

            _x0_str = str(start_dt)
            _x1_str = str(end_dt)
            fig_hist.add_shape(
                type="rect",
                x0=_x0_str,
                x1=_x1_str,
                y0=0,
                y1=1,
                yref="paper",
                fillcolor=colors.get(event["type"], "rgba(0,0,0,0.05)"),
                layer="below",
                line_width=0,
            )
            fig_hist.add_annotation(
                x=_x0_str,
                y=1.0,
                yref="paper",
                text=event["label"],
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font={"size": 10},
            )

        fig_hist.update_layout(
            title=f"{index_choice} 장기 차트 (리짐 하이라이트)",
            xaxis_title="날짜",
            yaxis_title="지수",
            height=500,
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
            yaxis_type="log",
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # 리짐별 통계 테이블
        st.subheader("역사적 리짐 요약")
        regime_stats = []
        for event in REGIME_EVENTS:
            start_dt = pd.Timestamp(event["start"])
            end_dt = pd.Timestamp(event["end"])
            mask = (long_ohlcv.index >= start_dt) & (long_ohlcv.index <= end_dt)
            period_data = long_ohlcv[mask]
            if period_data.empty:
                continue

            start_price = period_data["종가"].iloc[0]
            end_price = period_data["종가"].iloc[-1]
            total_ret = end_price / start_price - 1
            duration_months = (end_dt - start_dt).days / 30

            regime_stats.append(
                {
                    "기간": f"{event['start']} ~ {event['end']}",
                    "이벤트": event["label"],
                    "유형": "상승" if event["type"] == "bull" else "하락",
                    "시작 지수": f"{start_price:,.0f}",
                    "종료 지수": f"{end_price:,.0f}",
                    "수익률": f"{total_ret:+.1%}",
                    "기간(월)": f"{duration_months:.0f}",
                }
            )

        if regime_stats:
            st.dataframe(
                pd.DataFrame(regime_stats),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "리짐 레이블은 사후 분석 기반이며, 모델 학습 피처로는 사용되지 않습니다. "
            "대신 vol_regime, per_deviation, foreign_net 등 피처가 리짐 특성을 간접 반영합니다."
        )

        # ── 중동전쟁 패턴 비교 차트 ──
        st.subheader("⚔️ 중동전쟁 패턴 비교 (KOSPI 실제 데이터)")
        st.caption(
            "과거 미국 중동전쟁 발발 후 18개월간 KOSPI 누적수익률. "
            "이란전쟁(2026-02-28) 예측에 가중평균으로 반영됩니다."
        )
        try:
            from core.index_predictor import _load_war_patterns

            _war_pats = _load_war_patterns()
            if _war_pats:
                fig_war = go.Figure()
                _war_colors = {
                    "걸프전(1990-91)": "#FF9800",
                    "이라크전(2003)": "#4CAF50",
                    "이란긴장(2020)": "#2196F3",
                }
                for _wname, _wpat in _war_pats.items():
                    _cum = [0.0] + _wpat["monthly_cum"]
                    _months = list(range(len(_cum)))
                    fig_war.add_trace(
                        go.Scatter(
                            x=_months,
                            y=[c * 100 for c in _cum],
                            mode="lines+markers",
                            name=f"{_wname} (w={_wpat['weight']:.0%})",
                            line={
                                "color": _war_colors.get(_wname, "#999"),
                                "width": 2,
                            },
                            marker={"size": 4},
                        )
                    )
                # 가중평균
                from core.index_predictor import _war_pattern_return

                _wavg = [0.0] + [_war_pattern_return(m) * 100 for m in range(1, 19)]
                fig_war.add_trace(
                    go.Scatter(
                        x=list(range(19)),
                        y=_wavg,
                        mode="lines",
                        name="가중평균 (예측 반영)",
                        line={"color": "#E53935", "width": 3, "dash": "dash"},
                    )
                )
                fig_war.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_war.update_layout(
                    xaxis_title="전쟁 발발 후 경과 (개월)",
                    yaxis_title="누적수익률 (%)",
                    height=400,
                    margin={"l": 40, "r": 20, "t": 30, "b": 40},
                    legend={"yanchor": "top", "y": 0.99, "x": 0.01},
                )
                st.plotly_chart(fig_war, use_container_width=True)
            else:
                st.warning("전쟁 패턴 데이터를 가져올 수 없습니다 (Stooq 연결 실패).")
        except Exception as e:
            st.warning(f"전쟁 패턴 차트 오류: {e}")

    else:
        st.info(
            "장기 데이터가 없습니다. '모델 학습' 또는 '예측 실행'을 먼저 실행하여 "
            "데이터를 수집하세요."
        )
