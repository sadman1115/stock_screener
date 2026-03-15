"""Page 1: 데이터 수집 — 기준일 설정, 전종목 원시 데이터 다운로드.

필터링 없이 날짜 기반으로 전종목 데이터만 수집한다.
1차/2차 필터링은 조건설정 페이지에서 수행한다.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CACHE_DIR, ensure_dirs
from core.styles import apply_mobile_styles

# NOTE: data_fetcher / data_dart 는 지연 임포트 (pykrx→matplotlib 초기화 ~2초 회피)
# → 버튼 클릭 시에만 import

ensure_dirs()


# ══════════════════════════════════════════════════════
# 한국 증시 공휴일 (주말 외 휴장일)
# ══════════════════════════════════════════════════════
_KR_MARKET_HOLIDAYS = {
    # ── 2025 ──
    date(2025, 1, 1),  # 신정
    date(2025, 1, 28),
    date(2025, 1, 29),
    date(2025, 1, 30),  # 설날
    date(2025, 3, 3),  # 삼일절 대체 (3/1 토)
    date(2025, 5, 5),
    date(2025, 5, 6),  # 어린이날 + 석가탄신일 대체
    date(2025, 6, 6),  # 현충일
    date(2025, 8, 15),  # 광복절
    date(2025, 10, 3),  # 개천절
    date(2025, 10, 6),
    date(2025, 10, 7),
    date(2025, 10, 8),  # 추석 + 대체
    date(2025, 10, 9),  # 한글날
    date(2025, 12, 25),  # 크리스마스
    # ── 2026 ──
    date(2026, 1, 1),  # 신정
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 2, 18),  # 설날
    date(2026, 3, 2),  # 삼일절 대체 (3/1 일)
    date(2026, 5, 5),  # 어린이날
    date(2026, 5, 25),  # 석가탄신일 대체 (5/24 일)
    date(2026, 8, 17),  # 광복절 대체 (8/15 토)
    date(2026, 9, 24),
    date(2026, 9, 25),
    date(2026, 9, 28),  # 추석 + 대체 (9/26 토)
    date(2026, 10, 5),  # 개천절 대체 (10/3 토)
    date(2026, 10, 9),  # 한글날
    date(2026, 12, 25),  # 크리스마스
    # ── 2027 ──
    date(2027, 1, 1),  # 신정
    date(2027, 2, 8),
    date(2027, 2, 9),  # 설날 + 대체 (2/6토, 2/7일)
    date(2027, 3, 1),  # 삼일절
    date(2027, 5, 5),  # 어린이날
    date(2027, 5, 13),  # 석가탄신일
    date(2027, 6, 6),  # 현충일 (일→대체 없음)
    date(2027, 8, 16),  # 광복절 대체 (8/15 일)
    date(2027, 9, 14),
    date(2027, 9, 15),
    date(2027, 9, 16),  # 추석
    date(2027, 10, 4),  # 개천절 대체 (10/3 일)
    date(2027, 10, 11),  # 한글날 대체 (10/9 토)
    date(2027, 12, 27),  # 크리스마스 대체 (12/25 토)
}


def _get_last_trading_day(today=None):
    """주말/공휴일이면 직전 개장일 반환."""
    if today is None:
        today = date.today()
    d = today
    while d.weekday() >= 5 or d in _KR_MARKET_HOLIDAYS:
        d -= timedelta(days=1)
    return d


def _get_cache_meta_path(date_str: str) -> Path:
    """캐시 메타데이터 파일 경로."""
    return CACHE_DIR / date_str / "_meta.json"


def _save_cache_meta(date_str: str, has_dart: bool):
    """캐시 메타데이터 저장."""
    meta = {
        "date": date_str,
        "collected_at": datetime.now().isoformat(),
        "has_dart": has_dart,
    }
    path = _get_cache_meta_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _load_cache_meta(date_str: str) -> dict | None:
    """캐시 메타데이터 로드. 없으면 None."""
    path = _get_cache_meta_path(date_str)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _get_quarter(date_str: str) -> str:
    """날짜에서 분기 문자열 반환."""
    year = int(date_str[:4])
    month = int(date_str[4:6])
    quarter = (month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def _find_dart_pkl(date_str: str) -> Path | None:
    """DART financial_all.pkl 경로 반환. 없으면 None."""
    quarter = _get_quarter(date_str)
    dart_path = CACHE_DIR / "dart" / quarter / "financial_all.pkl"
    if dart_path.exists():
        return dart_path
    year = int(date_str[:4])
    month = int(date_str[4:6])
    q = (month - 1) // 3 + 1
    if q == 1:
        prev_q = f"{year - 1}Q4"
    else:
        prev_q = f"{year}Q{q - 1}"
    dart_path = CACHE_DIR / "dart" / prev_q / "financial_all.pkl"
    if dart_path.exists():
        return dart_path
    return None


def _check_dart_version(date_str: str) -> str:
    """DART 데이터 버전 확인. 'v3'/'v2'/'v1'/'' 반환."""
    dart_path = _find_dart_pkl(date_str)
    if dart_path is None:
        return ""
    try:
        import pickle

        with open(dart_path, "rb") as f:
            df = pickle.load(f)
        if "revenue_prev2" in df.columns:
            return "v3"
        elif "revenue" in df.columns:
            return "v2"
        return "v1"
    except Exception:
        return ""


def _check_cache_exists(date_str: str) -> dict:
    """해당 날짜의 캐시 상태 확인."""
    cache_dir = CACHE_DIR / date_str
    dart_exists = _find_dart_pkl(date_str) is not None

    return {
        "tickers": (cache_dir / "all_tickers.pkl").exists(),
        "ohlcv": (cache_dir / "all_ohlcv.pkl").exists(),
        "fundamentals": (cache_dir / "all_fundamentals.pkl").exists(),
        "market_cap": (cache_dir / "all_market_cap.pkl").exists(),
        "sectors": (cache_dir / "all_sectors.pkl").exists(),
        "ohlcv_20d": (cache_dir / "all_ohlcv_20d.pkl").exists(),
        "etf_tickers": (cache_dir / "etf_tickers.pkl").exists(),
        "dart": dart_exists,
    }


st.set_page_config(page_title="데이터 수집", page_icon="📊", layout="wide")

# 모바일 반응형 스타일 적용
apply_mobile_styles()

st.title("📊 데이터 수집")
st.caption("전종목 원시 데이터를 다운로드합니다. 필터링은 조건설정에서 수행합니다.")
st.markdown("---")

# ── 기준일/API 키 입력 ──
col1, col2 = st.columns(2)
with col1:
    default_date = _get_last_trading_day()
    base_date = st.date_input("기준일", value=default_date)
    date_str = base_date.strftime("%Y%m%d")
    if base_date != default_date and default_date != date.today():
        pass  # 사용자가 수동 변경한 경우 그대로
    elif base_date == default_date and default_date != date.today():
        st.caption(f"📅 오늘({date.today():%m/%d})은 휴장일 → 직전 개장일 자동 선택")

    # 데이터 날짜 상태 확인 (로컬 판별 — KRX API 호출 없음)
    _today = date.today()
    _today_str = _today.strftime("%Y%m%d")
    if date_str == _today_str:
        _now = datetime.now()
        if _now.hour >= 16:
            st.caption("🟢 **금일 종가 (확정)** 기준 데이터")
        elif _now.hour > 15 or (_now.hour == 15 and _now.minute >= 30):
            st.caption("🟡 **금일 종가 (집계중)** 기준 데이터")
            st.caption("💡 장 마감 직후에는 데이터 확정까지 약 30분 소요")
        else:
            st.caption("🔴 **금일 장중 (미확정)** 기준 데이터")
    else:
        disp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        st.caption(f"🔵 **{disp} 종가** 기준 데이터")

with col2:
    dart_api_key = st.text_input(
        "DART API 키 (선택)",
        value=os.environ.get("DART_API_KEY", ""),
        type="password",
        help="https://opendart.fss.or.kr/ 에서 발급",
    )

# ── KRX 로그인 (2025-12-27~ 회원제 전환으로 필수) ──
with st.expander("KRX 로그인 설정 (data.krx.co.kr)", expanded=False):
    st.caption(
        "2025-12-27부터 KRX 정보데이터시스템이 회원제로 전환되어 "
        "로그인 없이는 데이터 조회가 불가합니다. "
        "[가입하기](https://data.krx.co.kr)"
    )
    _krx_col1, _krx_col2 = st.columns(2)
    with _krx_col1:
        _krx_id = st.text_input(
            "KRX 회원 ID",
            value=st.session_state.get("krx_mbr_id", ""),
            key="krx_id_input",
            help="data.krx.co.kr 로그인 ID (네이버/카카오 소셜 로그인 가입 가능)",
        )
    with _krx_col2:
        _krx_pw = st.text_input(
            "KRX 비밀번호",
            value=st.session_state.get("krx_pw", ""),
            type="password",
            key="krx_pw_input",
        )
    if _krx_id:
        st.session_state["krx_mbr_id"] = _krx_id
    if _krx_pw:
        st.session_state["krx_pw"] = _krx_pw

    # 로그인 상태 표시
    try:
        from core.krx_login import is_logged_in as _krx_is_logged_in

        if _krx_is_logged_in():
            st.caption("🟢 KRX 로그인: 인증됨")
        elif _krx_id and _krx_pw:
            st.caption("🟡 KRX 로그인: 미연결 (데이터 수집 시 자동 로그인)")
        else:
            st.caption("🔴 KRX 로그인: ID/PW 미입력")
    except Exception:
        st.caption("⬜ KRX 로그인: 모듈 미로드")

    _use_openapi = st.checkbox(
        "KRX Open API 우선 사용 (KRX 로그인 장애 시)",
        value=st.session_state.get("use_openapi_first", False),
        help=(
            "체크하면 pykrx(로그인 필요) 대신 KRX Open API를 1순위로 사용합니다. "
            "KRX data.krx.co.kr 로그인이 안 될 때 활성화하세요."
        ),
    )
    st.session_state["use_openapi_first"] = _use_openapi

st.markdown("---")

# ── 캐시 상태 (컴팩트) ──
cache_status = _check_cache_exists(date_str)
cache_meta = _load_cache_meta(date_str)
quarter = _get_quarter(date_str)

# 캐시 보유 날짜 목록
cached_dates = sorted(
    [
        d.name
        for d in CACHE_DIR.iterdir()
        if d.is_dir()
        and d.name.isdigit()
        and len(d.name) == 8
        and (d / "all_fundamentals.pkl").exists()
    ],
    reverse=True,
)

# Fallback 마커 확인
from core.data_fetcher import get_fallback_markers  # noqa: E402

_fb_markers = get_fallback_markers(date_str)
_fb_cache_names = {"all_ohlcv": "ohlcv", "all_market_cap": "market_cap"}

# 일별 캐시 아이콘 한 줄
daily_labels = ["티커", "OHLCV", "시총", "펀더멘털", "20일"]
daily_keys = ["tickers", "ohlcv", "market_cap", "fundamentals", "ohlcv_20d"]


def _cache_icon(label, key):
    if not cache_status[key]:
        return f"⬜{label}"
    # fallback 마커가 있으면 경고 아이콘
    for fb_name, fb_key in _fb_cache_names.items():
        if fb_key == key and fb_name in _fb_markers:
            return f"⚠️{label}"
    return f"✅{label}"


daily_icons = " ".join(
    _cache_icon(lbl, k) for lbl, k in zip(daily_labels, daily_keys, strict=False)
)
_dart_ver = _check_dart_version(date_str)
if _dart_ver == "":
    dart_label = "⬜DART"
elif _dart_ver in ("v1", "v2"):
    dart_label = f"⚠️DART({_dart_ver}·재수집필요)"
else:
    dart_label = "✅DART(v3)"
quarterly_icons = f"{'✅' if cache_status['sectors'] else '⬜'}업종 {dart_label}"

# 캐시 상태 + 날짜 정보를 한 블록으로
meta_text = ""
if cache_meta:
    collected_at = cache_meta.get("collected_at", "")
    if isinstance(collected_at, str) and "T" in collected_at:
        try:
            dt = datetime.fromisoformat(collected_at)
            meta_text = f" · 수집 {dt.strftime('%m/%d %H:%M')}"
        except Exception:
            pass

dates_text = ""
if cached_dates:
    formatted = [f"{d[4:6]}/{d[6:]}" for d in cached_dates[:5]]
    dates_text = f" · 보유: {', '.join(formatted)}"

st.caption(
    f"📦 **일별**({date_str[4:6]}/{date_str[6:]}) {daily_icons} "
    f"| **분기**({quarter}) {quarterly_icons}"
    f"{meta_text}{dates_text}"
)

# Fallback 경고 표시
if _fb_markers:
    _fb_items = []
    for _fn, _sd in _fb_markers.items():
        _label = {
            "all_ohlcv": "OHLCV(종가)",
            "all_market_cap": "시가총액",
            "etf_ohlcv_all": "ETF OHLCV",
        }.get(_fn, _fn)
        _fb_items.append(f"{_label} → {_sd[:4]}-{_sd[4:6]}-{_sd[6:]}일 데이터")
    st.warning(
        "⚠️ KRX API 장애로 일부 데이터가 다른 날짜 기준입니다:\n"
        + "\n".join(f"- {i}" for i in _fb_items)
        + "\n\n💡 평일 장 마감 후 **캐시 초기화 → 재수집**하면 정확한 데이터로 갱신됩니다."
    )

# ── 수집 버튼 ──
col1, col2 = st.columns(2)

with col1:
    btn_daily = st.button(
        "📥 일별 데이터 수집", type="primary", use_container_width=True
    )

with col2:
    btn_dart = st.button(
        "📊 DART 재무 수집", use_container_width=True, disabled=not dart_api_key
    )

# ── 캐시 초기화 버튼 (분리) ──
st.markdown("")
col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    btn_clear_daily = st.button(
        "🗑️ 일별 캐시 초기화",
        use_container_width=True,
        help="해당 날짜의 일별 데이터만 삭제",
    )
with col_c2:
    btn_clear_dart = st.button(
        "🗑️ DART 캐시 초기화",
        use_container_width=True,
        help="현재 분기의 DART 재무 데이터 삭제",
    )
with col_c3:
    btn_clear_all = st.button(
        "🗑️ 전체 초기화", use_container_width=True, help="일별 + DART 모두 삭제"
    )


def collect_daily_data(date_str: str):
    """일별 데이터 수집 (매일 변경되는 데이터).

    최적화:
    - Step 1: 티커 먼저 수집 (Fundamental 보완에 필요)
    - Step 2: OHLCV + Fundamental + 시가총액 병렬 수집
    - Step 3: 20일 OHLCV — 증분 업데이트 + 병렬 수집
    """
    import concurrent.futures

    from core import data_fetcher as fetcher

    # Open API 우선 모드 설정
    _openapi_first = st.session_state.get("use_openapi_first", False)
    fetcher.set_force_openapi(_openapi_first)

    # KRX 로그인 (pykrx 세션 주입)
    # Open API 우선 모드에서도 pykrx 로그인 시도 (Open API 실패 시 fallback용)
    if _openapi_first:
        st.info(
            "KRX Open API 우선 모드: Open API를 먼저 시도하고, "
            "데이터가 없으면 pykrx로 전환합니다."
        )
    try:
        from core import krx_login

        _mid = st.session_state.get("krx_mbr_id", "")
        _mpw = st.session_state.get("krx_pw", "")
        if _mid and _mpw:
            ok, msg = krx_login.ensure_login(_mid, _mpw)
            if ok:
                st.toast("🟢 KRX 로그인 성공")
            elif not _openapi_first:
                st.warning(f"KRX 로그인 실패: {msg}")
        else:
            if not _openapi_first:
                st.warning(
                    "KRX 로그인 정보가 없습니다. "
                    "위 'KRX 로그인 설정' 에서 ID/PW를 입력하세요."
                )
    except Exception as e:
        if not _openapi_first:
            st.warning(f"KRX 로그인 모듈 오류: {e}")

    # KRX Open API 세션 캐시/인증 플래그 초기화
    try:
        from core import krx_openapi

        krx_openapi.clear_session_cache()
        krx_openapi.reset_auth_flag()
    except Exception:
        pass

    # Open API 우선 모드: fallback 마커가 있는 캐시 삭제 (불완전 데이터 재수집)
    if _openapi_first:
        _fb = fetcher.get_fallback_markers(date_str)
        if _fb:
            for cache_name in _fb:
                fetcher.clear_fallback_cache(cache_name, date_str)
            st.toast(f"🔄 불완전 캐시 {len(_fb)}건 삭제 → Open API로 재수집합니다")

    # 캐시 전부 있으면 즉시 완료 (rerun 불필요)
    cs = _check_cache_exists(date_str)
    daily_keys = ["tickers", "ohlcv", "fundamentals", "market_cap", "ohlcv_20d"]
    if all(cs[k] for k in daily_keys):
        st.session_state["date_str"] = date_str
        st.session_state["data_collected"] = True
        st.toast("✅ 이미 수집된 데이터입니다. 재수집하려면 캐시를 초기화하세요.")
        return False  # rerun 불필요

    status = st.status("일별 데이터 수집 중...", expanded=True)
    progress = st.progress(0)

    try:
        # ── Step 1/3: 전종목 티커 (Fundamental 보완에 선행 필요) ──
        status.write("Step 1/3: 전종목 티커 수집...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fetcher.fetch_all_tickers, date_str)
            try:
                tickers_df = fut.result(timeout=60)
            except concurrent.futures.TimeoutError as e:
                raise TimeoutError("티커: KRX API 응답 없음 (60초 초과)") from e
        status.write(f"  ✅ 티커: {len(tickers_df)}종목")
        progress.progress(15)

        # ── Step 2/3: OHLCV + Fundamental + 시가총액 병렬 수집 ──
        status.write("Step 2/3: 전종목 기본 데이터 병렬 수집...")
        bulk_tasks = {}
        if not cs["ohlcv"]:
            bulk_tasks["OHLCV"] = ("fetch_all_ohlcv", 60)
        if not cs["fundamentals"]:
            bulk_tasks["Fundamental"] = ("fetch_all_fundamentals", 180)
        if not cs["market_cap"]:
            bulk_tasks["시가총액"] = ("fetch_all_market_cap", 60)
        if not cs["sectors"]:
            bulk_tasks["업종분류"] = ("fetch_sector_classifications", 60)
        if not cs.get("etf_tickers", False):
            bulk_tasks["ETF 티커"] = ("fetch_etf_tickers", 60)

        if bulk_tasks:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(bulk_tasks)
            ) as executor:
                futures = {}
                for label, (fn_name, _timeout) in bulk_tasks.items():
                    fn = getattr(fetcher, fn_name)
                    futures[label] = (executor.submit(fn, date_str), _timeout)

                for label, (future, _timeout) in futures.items():
                    try:
                        result = future.result(timeout=_timeout)
                        status.write(f"  ✅ {label}: {len(result)}종목")
                    except concurrent.futures.TimeoutError as e:
                        raise TimeoutError(
                            f"{label}: KRX API 응답 없음 ({_timeout}초 초과). "
                            "주말/공휴일에는 KRX 서버가 느릴 수 있습니다."
                        ) from e
        else:
            status.write("  ✅ 모든 기본 데이터 캐시됨")
        progress.progress(60)

        # ── Step 3/3: 20일 OHLCV (증분 + 병렬) ──
        status.write("Step 3/3: 20일 OHLCV 수집 (증분 업데이트)...")
        detail_status = st.empty()

        def ohlcv_callback(msg):
            detail_status.caption(f"  📡 {msg}")

        ohlcv_20d = fetcher.fetch_all_ohlcv_ndays(
            date_str, 20, progress_callback=ohlcv_callback
        )
        detail_status.empty()
        progress.progress(100)
        status.write(f"  ✅ {len(ohlcv_20d)}일분 데이터")

        # 메타데이터 저장
        _save_cache_meta(date_str, cs["dart"])

        # Fallback 경고 확인
        fb_warnings = fetcher.get_fallback_warnings()
        if fb_warnings:
            # KRX Open API로 대체된 항목이 있는지 확인
            has_openapi = any("KRX Open API" in w for w in fb_warnings)
            has_cache_fb = any("캐시 데이터" in w for w in fb_warnings)
            is_forced = _openapi_first
            if has_openapi and not has_cache_fb:
                if is_forced:
                    status.update(
                        label="✅ 일별 데이터 수집 완료 (KRX Open API 우선 모드)",
                        state="complete",
                        expanded=False,
                    )
                    st.info(
                        "KRX Open API 우선 모드로 수집 완료:\n"
                        + "\n".join(f"- {w}" for w in fb_warnings)
                    )
                else:
                    status.update(
                        label="🔄 일별 데이터 수집 완료 (KRX Open API 사용)",
                        state="complete",
                        expanded=True,
                    )
                    st.info(
                        "🔄 pykrx 장애로 KRX Open API를 통해 수집했습니다:\n"
                        + "\n".join(f"- {w}" for w in fb_warnings)
                    )
            else:
                status.update(
                    label="⚠️ 일별 데이터 수집 완료 (일부 전일 데이터 사용)",
                    state="complete",
                    expanded=True,
                )
                st.warning(
                    "⚠️ KRX 응답 불능으로 일부 데이터가 이전 캐시를 사용합니다:\n"
                    + "\n".join(f"- {w}" for w in fb_warnings)
                )
        else:
            status.update(
                label="✅ 일별 데이터 수집 완료!", state="complete", expanded=False
            )

        # session_state에 날짜 저장
        st.session_state["date_str"] = date_str
        st.session_state["data_collected"] = True

        return True

    except TimeoutError as e:
        status.update(label=f"⏱️ 타임아웃: {e}", state="error")
        st.warning(
            "💡 주말/공휴일에는 KRX API가 느릴 수 있습니다. 평일 장 마감 후 재시도하세요."
        )
        return False
    except Exception as e:
        status.update(label=f"❌ 수집 실패: {e}", state="error")
        import traceback

        st.error(traceback.format_exc())
        return False


def collect_dart_data(date_str: str, api_key: str):
    """DART 재무제표 수집 (분기별 데이터)."""
    from core import data_dart as dart
    from core import data_fetcher as fetcher

    if not api_key:
        st.error("DART API 키가 필요합니다.")
        return False

    if not dart.init_dart(api_key):
        st.error("DART API 초기화 실패")
        return False

    status = st.status("DART 재무제표 수집 중...", expanded=True)
    progress = st.progress(0)
    detail_status = st.empty()

    try:
        # 먼저 전종목 티커 로드
        tickers_df = fetcher.fetch_all_tickers(date_str)
        all_tickers = tickers_df["Ticker"].tolist()

        status.write(f"전종목 DART 재무제표 수집 ({len(all_tickers)}종목)...")
        status.write("⚠️ 약 15~30분 소요될 수 있습니다. (병렬 수집)")

        year = int(date_str[:4])

        def dart_callback(msg):
            detail_status.caption(f"  📡 {msg}")
            # "DART 재무제표: 150/2773 (5%) - 005930" 형식 파싱
            if "/" in msg and ":" in msg:
                try:
                    after_colon = msg.split(":", 1)[1].strip()
                    nums = after_colon.split("/")
                    if len(nums) >= 2:
                        current = int("".join(filter(str.isdigit, nums[0])))
                        total = int("".join(filter(str.isdigit, nums[1].split()[0])))
                        pct = int(100 * current / total)
                        progress.progress(min(pct, 99))
                except Exception:
                    pass

        dart_df = dart.fetch_candidates_financial(
            all_tickers, year, date_str, progress_callback=dart_callback
        )
        detail_status.empty()
        progress.progress(100)

        # 메타데이터 업데이트
        _save_cache_meta(date_str, True)

        status.update(
            label=f"✅ DART 재무제표 수집 완료! ({len(dart_df)}종목)",
            state="complete",
            expanded=False,
        )
        return True

    except Exception as e:
        status.update(label=f"❌ DART 수집 실패: {e}", state="error")
        return False


# ── 버튼 동작 ──
if btn_daily:
    ok = collect_daily_data(date_str)
    if ok:
        st.rerun()

if btn_dart:
    ok = collect_dart_data(date_str, dart_api_key)
    if ok:
        st.rerun()

if btn_clear_daily:
    from core import data_fetcher as fetcher

    fetcher.clear_cache(date_str)
    st.success(f"✅ {date_str} 일별 캐시 초기화 완료")
    st.rerun()

if btn_clear_dart:
    import shutil

    quarter = _get_quarter(date_str)
    dart_dir = CACHE_DIR / "dart" / quarter
    if dart_dir.exists():
        shutil.rmtree(dart_dir, ignore_errors=True)
    st.success(f"✅ DART 캐시 초기화 완료 ({quarter})")
    st.rerun()

if btn_clear_all:
    import shutil

    from core import data_fetcher as fetcher

    fetcher.clear_cache(date_str)
    quarter = _get_quarter(date_str)
    dart_dir = CACHE_DIR / "dart" / quarter
    if dart_dir.exists():
        shutil.rmtree(dart_dir, ignore_errors=True)
    st.success(f"✅ 전체 캐시 초기화 완료 ({date_str} + {quarter})")
    st.rerun()

st.markdown("---")

# ── 수집된 데이터 요약 ──


@st.cache_data(show_spinner=False)
def _compute_data_summary(date_str, has_dart):
    """데이터 요약 통계 계산 (동일 날짜는 캐시 — 매 렌더 시 재로드 방지)."""
    import pickle as _pkl

    cache_base = CACHE_DIR / date_str

    with open(cache_base / "all_tickers.pkl", "rb") as f:
        tickers_df = _pkl.load(f)
    with open(cache_base / "all_fundamentals.pkl", "rb") as f:
        fund_df = _pkl.load(f)

    n_total = len(tickers_df)

    pbr_le05 = pbr_le10 = pbr_gt10 = 0
    if "PBR" in fund_df.columns:
        pbr = fund_df["PBR"]
        pbr_le05 = int(((pbr > 0) & (pbr <= 0.5)).sum())
        pbr_le10 = int(((pbr > 0.5) & (pbr <= 1.0)).sum())
        pbr_gt10 = int((pbr > 1.0).sum())

    n_dart = n_fcf_high = n_fcf_mid = n_fcf_low = 0
    if has_dart:
        # DART 데이터 직접 로드 (data_dart 모듈 임포트 회피)
        _year = int(date_str[:4])
        _month = int(date_str[4:6])
        _q = (_month - 1) // 3 + 1
        _dart_path = CACHE_DIR / "dart" / f"{_year}Q{_q}" / "financial_all.pkl"
        if not _dart_path.exists() and _q == 1:
            _dart_path = CACHE_DIR / "dart" / f"{_year - 1}Q4" / "financial_all.pkl"
        elif not _dart_path.exists():
            _dart_path = CACHE_DIR / "dart" / f"{_year}Q{_q - 1}" / "financial_all.pkl"
        dart_df = None
        if _dart_path.exists():
            with open(_dart_path, "rb") as f:
                dart_df = _pkl.load(f)
        if dart_df is not None and not dart_df.empty:
            avail = (
                dart_df["available"]
                if "available" in dart_df.columns
                else pd.Series(False, index=dart_df.index)
            )
            n_dart = int(avail.sum())
            avail_df = dart_df[avail]
            if all(
                c in avail_df.columns for c in ["cash", "st_financial", "total_assets"]
            ):
                cash_total = avail_df["cash"].fillna(0) + avail_df[
                    "st_financial"
                ].fillna(0)
                assets = avail_df["total_assets"].fillna(0)
                ratio = cash_total / assets.replace(0, float("nan"))
                n_fcf_high = int((ratio >= 0.20).sum())
                n_fcf_mid = int(((ratio >= 0.05) & (ratio < 0.20)).sum())
                n_fcf_low = int((ratio < 0.05).sum())

    return {
        "n_total": n_total,
        "n_dart": n_dart,
        "pbr_le05": pbr_le05,
        "pbr_le10": pbr_le10,
        "pbr_gt10": pbr_gt10,
        "n_fcf_high": n_fcf_high,
        "n_fcf_mid": n_fcf_mid,
        "n_fcf_low": n_fcf_low,
    }


if all(cache_status[k] for k in ["tickers", "ohlcv", "fundamentals"]):
    st.markdown(
        """<style>
    [data-testid="stMetric"] { padding: 2px 0 !important; }
    [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
    </style>""",
        unsafe_allow_html=True,
    )

    s = _compute_data_summary(date_str, cache_status["dart"])

    disp_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    st.markdown(f"##### 📊 수집 데이터 요약 ({disp_date})")
    c = st.columns([1, 1, 1, 1, 1, 0.3, 1, 1, 1])
    c[0].metric("전종목", f"{s['n_total']:,}")
    c[1].metric("DART", f"{s['n_dart']:,}")
    c[2].metric("PBR ≤0.5", f"{s['pbr_le05']:,}")
    c[3].metric("PBR ≤1.0", f"{s['pbr_le10']:,}")
    c[4].metric("PBR >1.0", f"{s['pbr_gt10']:,}")
    c[6].metric("현금풍부", f"{s['n_fcf_high']:,}")
    c[7].metric("현금보통", f"{s['n_fcf_mid']:,}")
    c[8].metric("현금부족", f"{s['n_fcf_low']:,}")

    st.session_state["date_str"] = date_str
    st.session_state["data_collected"] = True

else:
    st.info("📥 '일별 데이터 수집' 버튼을 클릭하여 데이터를 수집하세요.")
