"""pykrx 기반 데이터 수집 모듈.

전종목 일괄 API → 1차 필터 → 후보군 개별 API 순서로 수집한다.
모든 결과는 pickle 캐시로 저장하여 재실행 시 API 호출을 방지한다.

Fallback 체인:
    pykrx (1순위) → KRX Open API (2순위) → 캐시 fallback (3순위)
"""

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pykrx import stock

from .config import CACHE_DIR

# KRX API 차단 감지 플래그 — 첫 호출 실패 시 True로 설정되어 이후 KRX 호출 스킵
_krx_ohlcv_blocked = False
# FDR 연속 실패 카운터 — 3회 연속 실패 시 나머지 스킵
_fdr_consecutive_fails = 0
_FDR_MAX_FAILS = 3

# Open API 우선 모드 — True이면 pykrx를 건너뛰고 KRX Open API를 1순위로 사용
_force_openapi = False


def set_force_openapi(enabled: bool):
    """KRX Open API 우선 모드 설정.

    enabled=True: pykrx(로그인 필요)를 건너뛰고 Open API를 1순위로 사용.
    KRX 로그인 장애 시 유용하다.
    """
    global _force_openapi
    _force_openapi = enabled


def _cache_path(name: str, date: str) -> Path:
    d = CACHE_DIR / date
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.pkl"


def _load_cache(name: str, date: str):
    p = _cache_path(name, date)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(name: str, date: str, data):
    p = _cache_path(name, date)
    with open(p, "wb") as f:
        pickle.dump(data, f)


def _save_fallback_marker(name: str, date: str, src_date: str):
    """Fallback 캐시 사용 시 마커 파일 저장. src_date = 실제 데이터 원본 날짜."""
    import json

    marker = _cache_path(name, date).with_suffix(".fallback.json")
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"src_date": src_date, "saved_at": datetime.now().isoformat()}, f)


def _load_fallback_marker(name: str, date: str) -> str | None:
    """Fallback 마커가 있으면 원본 날짜 반환, 없으면 None."""
    import json

    marker = _cache_path(name, date).with_suffix(".fallback.json")
    if marker.exists():
        try:
            with open(marker, encoding="utf-8") as f:
                return json.load(f).get("src_date")
        except Exception:
            pass
    return None


def _clear_fallback_marker(name: str, date: str):
    """Fallback 마커 삭제 (정상 데이터로 갱신 시)."""
    marker = _cache_path(name, date).with_suffix(".fallback.json")
    if marker.exists():
        marker.unlink(missing_ok=True)


def clear_fallback_cache(name: str, date: str):
    """Fallback 마커가 있는 캐시를 삭제하여 재수집 유도."""
    _clear_fallback_marker(name, date)
    p = _cache_path(name, date)
    if p.exists():
        p.unlink(missing_ok=True)


def get_fallback_markers(date: str) -> dict[str, str]:
    """해당 날짜의 모든 fallback 마커 반환. {name: src_date}."""
    import json

    markers = {}
    cache_dir = CACHE_DIR / date
    if cache_dir.exists():
        for f in cache_dir.glob("*.fallback.json"):
            name = f.name.replace(".fallback.json", "")
            try:
                with open(f, encoding="utf-8") as fh:
                    markers[name] = json.load(fh).get("src_date", "?")
            except Exception:
                pass
    return markers


# ──────────────────────────────────────────────
# KRX Open API Fallback 헬퍼
# ──────────────────────────────────────────────


def _try_krx_openapi(func_name: str, date: str) -> pd.DataFrame | None:
    """KRX Open API fallback 시도. 실패 시 None 반환.

    - AUTH_KEY 미설정이면 즉시 None
    - 이전 인증 실패 시 즉시 None (재시도 방지)
    - API 오류 시 조용히 None (기존 캐시 fallback으로 전환)
    - _force_openapi 모드에서 실패하면 경고 로그 추가
    """
    try:
        from . import krx_openapi

        if not krx_openapi.is_configured():
            if _force_openapi:
                _add_fallback_warning(
                    f"{func_name}: KRX Open API 인증키 미설정 또는 인증 실패 → "
                    "pykrx fallback으로 전환"
                )
            return None
        fn = getattr(krx_openapi, func_name, None)
        if fn is None:
            return None
        result = fn(date)
        if isinstance(result, pd.DataFrame) and result.empty:
            return None
        return result
    except Exception as exc:
        if _force_openapi:
            _add_fallback_warning(
                f"{func_name}: KRX Open API 호출 실패({exc}) → pykrx fallback으로 전환"
            )
        return None


# ──────────────────────────────────────────────
# 범용 캐시 Fallback 인프라
# ──────────────────────────────────────────────


def _find_nearest_cache(name: str, date: str, max_age_days: int = 7) -> tuple:
    """가장 가까운 이전 날짜의 유효한 캐시 반환.

    Returns:
        (data, source_date) — 유효한 캐시를 찾은 경우
        (None, None)        — 캐시가 없거나 max_age_days 이내에 없는 경우
    """
    if not CACHE_DIR.exists():
        return None, None

    min_date_str = (
        datetime.strptime(date, "%Y%m%d") - timedelta(days=max_age_days)
    ).strftime("%Y%m%d")

    candidates = []
    for d in CACHE_DIR.iterdir():
        if (
            d.is_dir()
            and d.name.isdigit()
            and len(d.name) == 8
            and d.name != date
            and d.name >= min_date_str
            and d.name < date
        ):
            if (d / f"{name}.pkl").exists():
                candidates.append(d.name)

    if not candidates:
        return None, None

    candidates.sort(reverse=True)  # 최근 날짜 우선

    for src_date in candidates:
        data = _load_cache(name, src_date)
        if data is None:
            continue
        if isinstance(data, pd.DataFrame) and data.empty:
            continue
        if isinstance(data, dict) and len(data) == 0:
            continue
        return data, src_date

    return None, None


# Fallback 경고 수집기 — 데이터 수집 세션 중 사용
_fallback_warnings: list[str] = []


def get_fallback_warnings() -> list[str]:
    """현재 세션의 fallback 경고 목록 반환 후 초기화."""
    warnings = _fallback_warnings.copy()
    _fallback_warnings.clear()
    return warnings


def _add_fallback_warning(msg: str):
    """fallback 사용 시 경고 추가."""
    _fallback_warnings.append(msg)


def _call_with_timeout(func, args=(), kwargs=None, timeout=30):
    """pykrx API 호출을 별도 스레드에서 실행하여 타임아웃 적용."""
    import concurrent.futures

    if kwargs is None:
        kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as err:
            raise TimeoutError(
                f"KRX API 응답 없음 ({timeout}초 초과). "
                "주말/공휴일에는 KRX 서버가 느릴 수 있습니다."
            ) from err


def _is_krx_nontrading_error(e: Exception) -> bool:
    """pykrx 내부 KeyError가 비거래일(주말/공휴일)로 인한 것인지 판별."""
    if isinstance(e, KeyError):
        key_str = str(e)
        # pykrx Ticker 클래스가 빈 DataFrame에서 컬럼 접근 시 발생하는 KeyError 패턴
        if any(
            k in key_str for k in ("시장", "지수명", "종목", "종목명", "BPS", "PER")
        ):
            return True
    if isinstance(e, RuntimeError | ValueError):
        msg = str(e)
        if "빈 응답" in msg or "비거래일" in msg:
            return True
    return False


def _krx_api_failure_msg(func_name: str, e: Exception) -> str:
    """KRX API 실패 시 원인을 포함한 명확한 로그 메시지 생성."""
    if _is_krx_nontrading_error(e):
        reason = "비거래일(주말/공휴일) 데이터 미제공"
    elif isinstance(e, TimeoutError):
        reason = "API 응답 타임아웃"
    else:
        reason = f"API 오류({type(e).__name__}: {e})"
    return f"기본 KRX API({func_name})가 {reason}으로 실패"


def _retry(func, *args, retries=3, delay=2, timeout=30, **kwargs):
    for attempt in range(retries):
        try:
            result = _call_with_timeout(func, args=args, kwargs=kwargs, timeout=timeout)
            return result
        except TimeoutError:
            raise
        except Exception as e:
            if _is_krx_nontrading_error(e):
                # 비거래일 오류는 재시도 무의미 → 즉시 전파
                raise RuntimeError(_krx_api_failure_msg(func.__name__, e)) from e
            if attempt < retries - 1:
                wait = delay * (2**attempt)
                time.sleep(wait)
            else:
                raise e


def _get_business_days(end_date: str, n_days: int) -> list[str]:
    """end_date 이전 n_days 영업일 리스트 반환."""
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=n_days * 2)
    start_str = start_dt.strftime("%Y%m%d")
    try:
        days = _call_with_timeout(
            stock.get_previous_business_days,
            kwargs={"fromdate": start_str, "todate": end_date},
            timeout=15,
        )
        result = [d.strftime("%Y%m%d") for d in days]
        return result[-n_days:]
    except Exception:
        # 타임아웃/에러 시 — 단순 역산 fallback (API 호출 없음)
        days = []
        dt = end_dt
        max_iter = n_days * 3  # 무한루프 방지
        while len(days) < n_days and max_iter > 0:
            max_iter -= 1
            if dt.weekday() < 5:  # 평일만
                days.append(dt.strftime("%Y%m%d"))
            dt -= timedelta(days=1)
        return sorted(days)[-n_days:]


# ──────────────────────────────────────────────
# Step 1: 전종목 기본 데이터 (일괄 API)
# ──────────────────────────────────────────────


def _get_special_tickers(date: str) -> set:
    """ETF/ETN/ELW 티커 목록을 가져와서 set으로 반환."""
    special = set()
    for getter in [
        stock.get_etf_ticker_list,
        stock.get_etn_ticker_list,
        stock.get_elw_ticker_list,
    ]:
        try:
            tickers = _call_with_timeout(getter, args=(date,), timeout=15)
            special.update(tickers)
        except Exception:
            pass
    return special


def fetch_all_tickers(date: str, progress_callback=None) -> pd.DataFrame:
    """코스피+코스닥 전종목 티커/종목명/시장 반환 (ETF/ETN/ELW 제외)."""
    cached = _load_cache("all_tickers", date)
    if cached is not None and len(cached) > 0:
        return cached

    # Open API 우선 모드
    if _force_openapi:
        openapi_result = _try_krx_openapi("fetch_all_tickers", date)
        if openapi_result is not None:
            _save_cache("all_tickers", date, openapi_result)
            msg = (
                f"전종목 티커: KRX Open API 우선 모드로 수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

    try:
        # ETF/ETN/ELW 제외 목록
        special = _get_special_tickers(date)

        rows = []
        for mkt in ["KOSPI", "KOSDAQ"]:
            tickers = _retry(stock.get_market_ticker_list, date, market=mkt)
            for t in tickers:
                if t in special:
                    continue
                name = stock.get_market_ticker_name(t)
                rows.append({"Ticker": t, "Name": name, "Market": mkt})
            time.sleep(1)

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("KRX 빈 응답")

        _save_cache("all_tickers", date, df)
        if progress_callback:
            progress_callback(
                f"전종목 티커 수집 완료: {len(df)}종목 (ETF/ETN/ELW 제외)"
            )
        return df

    except Exception as e:
        api_msg = _krx_api_failure_msg("get_market_ticker_list", e)

        # Fallback 1: KRX Open API
        openapi_result = _try_krx_openapi("fetch_all_tickers", date)
        if openapi_result is not None:
            _save_cache("all_tickers", date, openapi_result)
            msg = (
                f"전종목 티커: {api_msg} → "
                f"KRX Open API로 대체 수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

        # Fallback 2: 캐시
        fallback, src_date = _find_nearest_cache("all_tickers", date)
        if fallback is not None:
            _save_cache("all_tickers", date, fallback)
            msg = (
                f"전종목 티커: {api_msg} → "
                f"{src_date}일자 캐시 데이터({len(fallback)}종목)로 대체 사용"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"⚠️ {msg}")
            return fallback
        raise RuntimeError(
            f"전종목 티커: {api_msg}. "
            "캐시에도 이전 데이터가 없습니다. "
            "평일 장 마감 후 최초 1회 데이터 수집이 필요합니다."
        ) from e


def fetch_all_ohlcv(date: str, progress_callback=None) -> pd.DataFrame:
    """전종목 당일 OHLCV+거래대금."""
    cached = _load_cache("all_ohlcv", date)
    if cached is not None and not cached.empty:
        return cached

    # Open API 우선 모드
    if _force_openapi:
        openapi_result = _try_krx_openapi("fetch_all_ohlcv", date)
        if openapi_result is not None:
            _save_cache("all_ohlcv", date, openapi_result)
            _clear_fallback_marker("all_ohlcv", date)
            msg = f"전종목 OHLCV: KRX Open API 우선 모드로 수집({len(openapi_result)}종목)"
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

    try:
        frames = []
        for mkt in ["KOSPI", "KOSDAQ"]:
            df = _retry(stock.get_market_ohlcv, date, market=mkt)
            if df is not None and not df.empty:
                df["Market"] = mkt
                frames.append(df)
            time.sleep(1)

        if not frames:
            raise RuntimeError("KRX 빈 응답")

        result = pd.concat(frames)
        result.index.name = "Ticker"
        _save_cache("all_ohlcv", date, result)
        _clear_fallback_marker("all_ohlcv", date)
        if progress_callback:
            progress_callback(f"전종목 OHLCV 수집 완료: {len(result)}종목")
        return result

    except Exception as e:
        api_msg = _krx_api_failure_msg("get_market_ohlcv", e)

        # Fallback 1: KRX Open API
        openapi_result = _try_krx_openapi("fetch_all_ohlcv", date)
        if openapi_result is not None:
            _save_cache("all_ohlcv", date, openapi_result)
            _clear_fallback_marker("all_ohlcv", date)
            msg = (
                f"전종목 OHLCV: {api_msg} → "
                f"KRX Open API로 대체 수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

        # Fallback 2: 캐시
        fallback, src_date = _find_nearest_cache("all_ohlcv", date)
        if fallback is not None:
            _save_cache("all_ohlcv", date, fallback)
            _save_fallback_marker("all_ohlcv", date, src_date)
            msg = (
                f"전종목 OHLCV: {api_msg} → "
                f"{src_date}일자 캐시 데이터({len(fallback)}종목)로 대체 사용 "
                f"(주의: 종가가 {src_date}일 기준)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"⚠️ {msg}")
            return fallback
        raise RuntimeError(
            f"전종목 OHLCV: {api_msg}. 캐시에도 이전 데이터가 없습니다."
        ) from e


def check_data_date(date: str) -> dict:
    """데이터 날짜 확인 - 금일 종가인지 전일 종가인지 판별."""
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")

    # 실제 거래일 확인
    try:
        actual_date = stock.get_nearest_business_day_in_a_week(date, prev=True)
    except Exception:
        actual_date = date

    is_today = actual_date == today
    is_requested_today = date == today

    # 장 마감 여부 (오후 3:30 이후)
    now = datetime.now()
    market_closed = now.hour > 15 or (now.hour == 15 and now.minute >= 30)
    # 데이터 확정 시간 (보통 4:00 이후)
    data_finalized = now.hour >= 16

    # 상태 판정
    if is_requested_today:
        if data_finalized:
            status = "금일 종가 (확정)"
        elif market_closed:
            status = "금일 종가 (집계중)"
        else:
            status = "금일 장중 (미확정)"
    elif actual_date == date:
        status = f"{date[:4]}-{date[4:6]}-{date[6:]} 종가"
    else:
        status = f"전일({actual_date}) 종가"

    return {
        "requested_date": date,
        "actual_date": actual_date,
        "is_today": is_today,
        "market_closed": market_closed,
        "data_finalized": data_finalized,
        "data_status": status,
    }


def _supplement_fundamentals(
    result: pd.DataFrame, date: str, progress_callback=None
) -> pd.DataFrame:
    """벌크 API 누락 종목을 개별 조회로 보완.

    개별 호출당 3초 타임아웃, 전체 30초 제한.
    """
    all_tickers_cache = _load_cache("all_tickers", date)
    if all_tickers_cache is None:
        return result

    all_tickers = set(all_tickers_cache["Ticker"].tolist())
    fetched_tickers = set(result.index.tolist())
    missing = all_tickers - fetched_tickers
    if not missing:
        return result

    if progress_callback:
        progress_callback(f"Fundamental 누락 {len(missing)}종목 개별 보완 중...")
    extra_rows = []
    deadline = time.time() + 30  # 전체 30초 제한
    for t in missing:
        if time.time() > deadline:
            break
        try:
            df_ind = _call_with_timeout(
                stock.get_market_fundamental_by_date, args=(date, date, t), timeout=3
            )
            if df_ind is not None and not df_ind.empty:
                row = df_ind.iloc[-1]
                row.name = t
                extra_rows.append(row)
        except Exception:
            continue
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        extra_df.index.name = "Ticker"
        result = pd.concat([result, extra_df])
        if progress_callback:
            progress_callback(f"Fundamental 보완 완료: +{len(extra_rows)}종목")
    return result


def _find_nearest_fundamentals(date: str) -> pd.DataFrame | None:
    """현재 날짜 캐시가 없으면 가장 가까운 날짜의 캐시를 찾아 반환."""
    cache_base = CACHE_DIR
    if not cache_base.exists():
        return None

    # 날짜 역순으로 검색 (최근 캐시 우선)
    candidates = []
    for d in cache_base.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            fund_file = d / "all_fundamentals.pkl"
            if fund_file.exists() and d.name != date:
                candidates.append(d.name)

    if not candidates:
        return None

    # 가장 가까운 날짜 선택
    candidates.sort(reverse=True)
    nearest = candidates[0]
    cached = _load_cache("all_fundamentals", nearest)
    return cached


def fetch_all_fundamentals(date: str, progress_callback=None) -> pd.DataFrame:
    """전종목 BPS/PER/PBR/EPS/DIV/DPS.

    벌크 API에서 누락된 종목(900xxx/950xxx 등)은 개별 조회로 보완.
    캐시가 있으면 누락분만 보완 (주말에도 동작).
    API 실패 시 가장 가까운 날짜 캐시로 fallback.
    """
    # Open API 우선 모드 (캐시 체크 전에 — 캐시가 있으면 어차피 사용)
    cached = _load_cache("all_fundamentals", date)
    if _force_openapi and (cached is None or cached.empty):
        openapi_result = _try_krx_openapi("fetch_all_fundamentals", date)
        if openapi_result is not None:
            _save_cache("all_fundamentals", date, openapi_result)
            msg = (
                f"전종목 Fundamental: KRX Open API 우선 모드로 "
                f"수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

    if cached is not None and not cached.empty:
        # 캐시 존재 → 누락 종목만 보완 시도 (실패해도 기존 캐시 반환)
        try:
            updated = _supplement_fundamentals(cached, date, progress_callback)
            if len(updated) > len(cached):
                _save_cache("all_fundamentals", date, updated)
                if progress_callback:
                    progress_callback(
                        f"전종목 Fundamental: {len(updated)}종목 (보완 완료)"
                    )
            return updated
        except Exception:
            return cached

    # 캐시 없음 → 벌크 API 호출 (타임아웃 60초로 증가)
    try:
        frames = []
        for mkt in ["KOSPI", "KOSDAQ"]:
            df = _retry(stock.get_market_fundamental, date, market=mkt, timeout=60)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(1)

        if not frames:
            raise RuntimeError("KRX 빈 응답")

        result = pd.concat(frames)
        result.index.name = "Ticker"

        # 누락 종목 개별 보완
        result = _supplement_fundamentals(result, date, progress_callback)

        _save_cache("all_fundamentals", date, result)
        if progress_callback:
            progress_callback(f"전종목 Fundamental 수집 완료: {len(result)}종목")
        return result

    except Exception as e:
        api_msg = _krx_api_failure_msg("get_market_fundamental", e)

        # Fallback 1: KRX Open API
        openapi_result = _try_krx_openapi("fetch_all_fundamentals", date)
        if openapi_result is not None:
            _save_cache("all_fundamentals", date, openapi_result)
            msg = (
                f"전종목 Fundamental: {api_msg} → "
                f"KRX Open API로 대체 수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

        # Fallback 2: 캐시
        fallback, src_date = _find_nearest_cache("all_fundamentals", date)
        if fallback is not None:
            _save_cache("all_fundamentals", date, fallback)
            msg = (
                f"전종목 Fundamental: {api_msg} → "
                f"{src_date}일자 캐시 데이터({len(fallback)}종목)로 대체 사용"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"⚠️ {msg}")
            return fallback
        raise


def fetch_all_market_cap(date: str, progress_callback=None) -> pd.DataFrame:
    """전종목 시가총액/거래량/거래대금/상장주식수."""
    cached = _load_cache("all_market_cap", date)
    if cached is not None and not cached.empty:
        return cached

    # Open API 우선 모드
    if _force_openapi:
        openapi_result = _try_krx_openapi("fetch_all_market_cap", date)
        if openapi_result is not None:
            _save_cache("all_market_cap", date, openapi_result)
            _clear_fallback_marker("all_market_cap", date)
            msg = (
                f"전종목 시가총액: KRX Open API 우선 모드로 "
                f"수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

    try:
        frames = []
        for mkt in ["KOSPI", "KOSDAQ"]:
            df = _retry(stock.get_market_cap, date, market=mkt)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(1)

        if not frames:
            raise RuntimeError("KRX 빈 응답")

        result = pd.concat(frames)
        result.index.name = "Ticker"
        _save_cache("all_market_cap", date, result)
        _clear_fallback_marker("all_market_cap", date)
        if progress_callback:
            progress_callback(f"전종목 시가총액 수집 완료: {len(result)}종목")
        return result

    except Exception as e:
        api_msg = _krx_api_failure_msg("get_market_cap", e)

        # Fallback 1: KRX Open API
        openapi_result = _try_krx_openapi("fetch_all_market_cap", date)
        if openapi_result is not None:
            _save_cache("all_market_cap", date, openapi_result)
            _clear_fallback_marker("all_market_cap", date)
            msg = (
                f"전종목 시가총액: {api_msg} → "
                f"KRX Open API로 대체 수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

        # Fallback 2: 캐시
        fallback, src_date = _find_nearest_cache("all_market_cap", date)
        if fallback is not None:
            _save_cache("all_market_cap", date, fallback)
            _save_fallback_marker("all_market_cap", date, src_date)
            msg = (
                f"전종목 시가총액: {api_msg} → "
                f"{src_date}일자 캐시 데이터({len(fallback)}종목)로 대체 사용"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"⚠️ {msg}")
            return fallback
        raise RuntimeError(
            f"전종목 시가총액: {api_msg}. 캐시에도 이전 데이터가 없습니다."
        ) from e


def fetch_sector_classifications(date: str, progress_callback=None) -> pd.DataFrame:
    """전종목 업종분류."""
    cached = _load_cache("all_sectors", date)
    if cached is not None and not cached.empty:
        return cached

    # Open API 우선 모드
    if _force_openapi:
        openapi_result = _try_krx_openapi("fetch_sector_classifications", date)
        if openapi_result is not None:
            _save_cache("all_sectors", date, openapi_result)
            msg = (
                f"업종분류: KRX Open API 우선 모드로 "
                f"수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

    frames = []
    for mkt in ["KOSPI", "KOSDAQ"]:
        try:
            df = _retry(stock.get_market_sector_classifications, date, mkt)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            pass
        time.sleep(1)

    if frames:
        result = pd.concat(frames)
        result.index.name = "Ticker"
        _save_cache("all_sectors", date, result)
        if progress_callback:
            progress_callback(f"업종분류 수집 완료: {len(result)}종목")
        return result

    # Fallback 1: KRX Open API
    openapi_result = _try_krx_openapi("fetch_sector_classifications", date)
    if openapi_result is not None:
        _save_cache("all_sectors", date, openapi_result)
        msg = f"업종분류: pykrx 실패 → KRX Open API로 대체 수집({len(openapi_result)}종목)"
        _add_fallback_warning(msg)
        if progress_callback:
            progress_callback(f"🔄 {msg}")
        return openapi_result

    # Fallback 2: 캐시 (업종은 거의 변하지 않으므로 30일까지 허용)
    fallback, src_date = _find_nearest_cache("all_sectors", date, max_age_days=30)
    if fallback is not None:
        _save_cache("all_sectors", date, fallback)
        msg = (
            f"업종분류: 기본 KRX API가 비거래일(주말/공휴일) 응답 불가로 실패 → "
            f"{src_date}일자 캐시 데이터({len(fallback)}종목)로 대체 사용"
        )
        _add_fallback_warning(msg)
        if progress_callback:
            progress_callback(f"⚠️ {msg}")
        return fallback

    if progress_callback:
        progress_callback("⚠️ 업종분류: KRX API 실패, 캐시 데이터도 없음")
    return pd.DataFrame()


# ──────────────────────────────────────────────
# Step 2: 전종목 20일분 OHLCV (거래대금/거래정지 판별)
# ──────────────────────────────────────────────


def _find_nearest_ohlcv_cache(date: str, n_days: int) -> dict | None:
    """가장 가까운 이전 날짜의 N일 OHLCV 캐시를 찾아 재활용."""
    if not CACHE_DIR.exists():
        return None
    candidates = []
    for d in CACHE_DIR.iterdir():
        if (
            d.is_dir()
            and d.name.isdigit()
            and len(d.name) == 8
            and d.name != date
            and d.name < date
        ):
            cache_file = d / f"all_ohlcv_{n_days}d.pkl"
            if cache_file.exists():
                candidates.append(d.name)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return _load_cache(f"all_ohlcv_{n_days}d", candidates[0])


def _fetch_single_day_ohlcv(d: str) -> pd.DataFrame:
    """단일 날짜 전종목 OHLCV 수집 (KOSPI+KOSDAQ).

    pykrx 실패 시 KRX Open API로 fallback.
    """
    frames = []
    try:
        for mkt in ["KOSPI", "KOSDAQ"]:
            df = _retry(stock.get_market_ohlcv, d, market=mkt)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(0.3)
    except Exception:
        pass
    if not frames:
        # KRX Open API fallback
        openapi_result = _try_krx_openapi("fetch_single_day_ohlcv", d)
        if openapi_result is not None and not openapi_result.empty:
            return openapi_result
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined.index.name = "Ticker"
    return combined


def fetch_all_ohlcv_ndays(
    date: str, n_days: int = 20, progress_callback=None
) -> dict[str, pd.DataFrame]:
    """전종목 N일분 일별 OHLCV. {date_str: df} 반환.

    최적화:
    - 이전 날짜 캐시에서 겹치는 날짜 데이터 재활용 (증분 업데이트)
    - 신규 날짜는 병렬 수집 (최대 3워커)
    """
    cached = _load_cache(f"all_ohlcv_{n_days}d", date)
    if cached is not None:
        return cached

    bdays = _get_business_days(date, n_days)

    # ── 증분 업데이트: 이전 캐시에서 겹치는 날짜 재활용 ──
    result = {}
    reused = 0
    prev_cache = _find_nearest_ohlcv_cache(date, n_days)
    if prev_cache:
        for d in bdays:
            if d in prev_cache:
                result[d] = prev_cache[d]
                reused += 1
    missing_days = [d for d in bdays if d not in result]

    if progress_callback:
        if reused > 0:
            progress_callback(
                f"전종목 {n_days}일 OHLCV: {reused}일 캐시 재활용, "
                f"{len(missing_days)}일 신규 수집"
            )
        elif missing_days:
            progress_callback(
                f"전종목 {n_days}일 OHLCV: {len(missing_days)}일 수집 시작..."
            )

    # ── 신규 날짜 병렬 수집 (최대 3워커) ──
    if missing_days:
        n_workers = min(3, len(missing_days))
        fetched = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_day = {
                executor.submit(_fetch_single_day_ohlcv, d): d for d in missing_days
            }
            for future in as_completed(future_to_day):
                d = future_to_day[future]
                try:
                    result[d] = future.result(timeout=60)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"⚠️ {d} 수집 실패: {e}")
                fetched += 1
                if progress_callback:
                    progress_callback(
                        f"전종목 {n_days}일 OHLCV: {reused + fetched}/{len(bdays)}"
                    )

    _save_cache(f"all_ohlcv_{n_days}d", date, result)
    return result


def calc_avg_trading_value(ohlcv_ndays: dict[str, pd.DataFrame]) -> pd.Series:
    """N일 평균 거래대금 계산."""
    trading_values = []
    for _d, df in ohlcv_ndays.items():
        if "거래대금" in df.columns:
            trading_values.append(df["거래대금"])
    if not trading_values:
        return pd.Series(dtype=float)
    combined = pd.concat(trading_values, axis=1)
    return combined.mean(axis=1)


def detect_suspended(ohlcv_ndays: dict[str, pd.DataFrame]) -> set[str]:
    """최근 N일 중 거래량=0인 날이 있는 종목 → 거래정지 의심."""
    suspended = set()
    for _d, df in ohlcv_ndays.items():
        if "거래량" in df.columns:
            zeros = df[df["거래량"] == 0].index.tolist()
            suspended.update(zeros)
    return suspended


# ──────────────────────────────────────────────
# Step 3: 후보군 개별 상세 데이터
# ──────────────────────────────────────────────


def _fdr_ohlcv_fallback(
    ticker: str, start_str: str, end_str: str
) -> pd.DataFrame | None:
    """FinanceDataReader가 설치된 경우에만 동작하는 개별 종목 OHLCV fallback.

    미설치 시 None 반환 (기존 동작과 동일).
    타임아웃 15초 적용 — ETF 대량 호출 시 지연 방지.
    """
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return None
    try:
        df = _call_with_timeout(
            fdr.DataReader,
            args=(ticker, start_str, end_str),
            timeout=15,
        )
        if df is None or df.empty:
            return None
        # FDR 컬럼명 → pykrx 컬럼명 변환
        rename_map = {
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가",
            "Volume": "거래량",
        }
        df = df.rename(columns=rename_map)
        # pykrx와 동일하게 DatetimeIndex 유지, 필요한 컬럼만
        cols = [
            c for c in ["시가", "고가", "저가", "종가", "거래량"] if c in df.columns
        ]
        return df[cols]
    except Exception:
        return None


def _fdr_index_ohlcv_fallback(
    index_code: str,
    start_str: str,
    end_str: str,
) -> pd.DataFrame | None:
    """FinanceDataReader로 지수 OHLCV 수집 (pykrx 실패 시 fallback).

    Args:
        index_code: "1001" (KOSPI) or "2001" (KOSDAQ)
        start_str: YYYYMMDD or YYYY-MM-DD
        end_str: YYYYMMDD or YYYY-MM-DD

    Returns:
        DataFrame (DatetimeIndex, 시가/고가/저가/종가/거래량) or None
    """
    _INDEX_YAHOO = {"1001": "^KS11", "2001": "^KQ11"}
    yahoo_code = _INDEX_YAHOO.get(index_code)
    if yahoo_code is None:
        return None

    try:
        import FinanceDataReader as fdr
    except ImportError:
        return None
    try:
        # FDR은 YYYY-MM-DD 형식 선호
        s = start_str.replace("-", "")
        e = end_str.replace("-", "")
        s_fmt = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        e_fmt = f"{e[:4]}-{e[4:6]}-{e[6:8]}"
        df = fdr.DataReader(yahoo_code, s_fmt, e_fmt)
        if df is None or df.empty:
            return None
        rename_map = {
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가",
            "Volume": "거래량",
        }
        df = df.rename(columns=rename_map)
        cols = [
            c for c in ["시가", "고가", "저가", "종가", "거래량"] if c in df.columns
        ]
        return df[cols]
    except Exception:
        return None


def _fdr_etf_tickers_fallback() -> pd.DataFrame | None:
    """FinanceDataReader로 한국 ETF 목록 조회 (pykrx 실패 시 fallback).

    FDR 미설치 시 None 반환.
    """
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return None
    try:
        etf_kr = fdr.StockListing("ETF/KR")
        if etf_kr is None or etf_kr.empty:
            return None
        code_col = next(
            (c for c in ("Code", "Symbol", "code", "symbol") if c in etf_kr.columns),
            None,
        )
        name_col = next(
            (c for c in ("Name", "name") if c in etf_kr.columns),
            None,
        )
        if code_col is None:
            return None
        result = pd.DataFrame(
            {
                "Ticker": etf_kr[code_col].astype(str),
                "Name": etf_kr[name_col] if name_col else etf_kr[code_col].astype(str),
                "Market": "ETF",
            }
        )
        return result if not result.empty else None
    except Exception:
        return None


def _try_previous_weekdays(
    func, date: str, max_tries: int = 3, timeout: int = 30, **kwargs
):
    """비거래일 에러 시 직전 평일 날짜들로 pykrx API 재시도.

    Returns:
        (result, adj_date) — 성공 시
        (None, None)       — 모든 시도 실패
    """
    dt = datetime.strptime(date, "%Y%m%d")
    tried = 0
    for offset in range(1, 8):
        adj_dt = dt - timedelta(days=offset)
        if adj_dt.weekday() >= 5:  # 주말 skip
            continue
        adj_date = adj_dt.strftime("%Y%m%d")
        tried += 1
        try:
            result = _call_with_timeout(
                func, args=(adj_date,), kwargs=kwargs, timeout=timeout
            )
            return result, adj_date
        except Exception:
            pass
        if tried >= max_tries:
            break
    return None, None


def fetch_ohlcv_history(ticker: str, end_date: str, days: int = 100) -> pd.DataFrame:
    """개별 종목 N일치 OHLCV."""
    cache_name = f"ohlcv_{ticker}_{days}d"
    cached = _load_cache(cache_name, end_date)
    if cached is not None:
        return cached

    inc = _incremental_ohlcv(cache_name, end_date, ticker, window_days=int(days * 1.8))
    if inc is not None:
        if len(inc) > days:
            inc = inc.iloc[-days:]
        return inc

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=int(days * 1.8))
    start_str = start_dt.strftime("%Y%m%d")

    try:
        df = _retry(stock.get_market_ohlcv, start_str, end_date, ticker)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        fdr_df = _fdr_ohlcv_fallback(ticker, start_str, end_date)
        if fdr_df is not None and not fdr_df.empty:
            df = fdr_df

    if len(df) > days:
        df = df.iloc[-days:]

    if not df.empty:
        _save_cache(cache_name, end_date, df)
    return df


def fetch_ohlcv_1year(ticker: str, end_date: str) -> pd.DataFrame:
    """개별 종목 1년치 OHLCV (52주 고/저 계산용)."""
    cache_name = f"ohlcv_{ticker}_1y"
    cached = _load_cache(cache_name, end_date)
    if cached is not None:
        return cached

    inc = _incremental_ohlcv(cache_name, end_date, ticker, window_days=370)
    if inc is not None:
        return inc

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=370)
    start_str = start_dt.strftime("%Y%m%d")

    try:
        df = _retry(stock.get_market_ohlcv, start_str, end_date, ticker)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        fdr_df = _fdr_ohlcv_fallback(ticker, start_str, end_date)
        if fdr_df is not None and not fdr_df.empty:
            df = fdr_df

    if not df.empty:
        _save_cache(cache_name, end_date, df)
    return df


def fetch_ohlcv_2year(ticker: str, end_date: str) -> pd.DataFrame:
    """개별 종목 2년치 OHLCV (장기 차트용)."""
    cache_name = f"ohlcv_{ticker}_2y"
    cached = _load_cache(cache_name, end_date)
    if cached is not None:
        return cached

    inc = _incremental_ohlcv(cache_name, end_date, ticker, window_days=740)
    if inc is not None:
        return inc

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=740)  # 약 2년
    start_str = start_dt.strftime("%Y%m%d")

    try:
        df = _retry(stock.get_market_ohlcv, start_str, end_date, ticker)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        fdr_df = _fdr_ohlcv_fallback(ticker, start_str, end_date)
        if fdr_df is not None and not fdr_df.empty:
            df = fdr_df

    if not df.empty:
        _save_cache(cache_name, end_date, df)
    return df


def _incremental_ohlcv(
    cache_name: str, end_date: str, ticker: str, window_days: int
) -> pd.DataFrame | None:
    """이전 날짜 캐시에서 증분만 API 호출하여 OHLCV 업데이트.

    Returns:
        업데이트된 DataFrame 또는 None (이전 캐시 없음).
    """
    prev_data, _prev_date = _find_nearest_cache(cache_name, end_date, max_age_days=7)
    if (
        prev_data is None
        or not isinstance(prev_data, pd.DataFrame)
        or len(prev_data) < 100
    ):
        return None

    if not isinstance(prev_data.index, pd.DatetimeIndex):
        prev_data.index = pd.to_datetime(prev_data.index)

    last_cached = prev_data.index[-1]
    next_day = (last_cached + timedelta(days=1)).strftime("%Y%m%d")

    if next_day > end_date:
        # 이미 end_date까지 데이터 보유 → 그대로 사용
        _save_cache(cache_name, end_date, prev_data)
        return prev_data

    try:
        incremental = _retry(stock.get_market_ohlcv, next_day, end_date, ticker)
        if incremental is not None and not incremental.empty:
            if not isinstance(incremental.index, pd.DatetimeIndex):
                incremental.index = pd.to_datetime(incremental.index)
            df = pd.concat([prev_data, incremental])
            df = df[~df.index.duplicated(keep="last")]
            # 윈도우 트리밍
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            start_dt = end_dt - timedelta(days=window_days)
            df = df[df.index >= pd.Timestamp(start_dt)]
            _save_cache(cache_name, end_date, df)
            return df
    except Exception:
        pass
    return None


def fetch_ohlcv_3year(ticker: str, end_date: str) -> pd.DataFrame:
    """개별 종목 3년치 OHLCV (AI 정밀학습 v7.0용)."""
    cache_name = f"ohlcv_{ticker}_3y"
    cached = _load_cache(cache_name, end_date)
    if cached is not None:
        return cached

    # 증분 수집 시도 (이전 날짜 캐시 + 신규 데이터만 API 호출)
    inc = _incremental_ohlcv(cache_name, end_date, ticker, window_days=1110)
    if inc is not None:
        return inc

    # 전체 수집 (콜드 스타트 또는 증분 실패)
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=1110)  # 약 3년
    start_str = start_dt.strftime("%Y%m%d")

    try:
        df = _retry(stock.get_market_ohlcv, start_str, end_date, ticker)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        fdr_df = _fdr_ohlcv_fallback(ticker, start_str, end_date)
        if fdr_df is not None and not fdr_df.empty:
            df = fdr_df

    if not df.empty:
        _save_cache(cache_name, end_date, df)
    return df


def fetch_kospi_index_3year(end_date: str) -> pd.DataFrame:
    """KOSPI 지수 3년 OHLCV (AI 학습용, 단일 API 호출).

    Returns:
        DataFrame with 시가/고가/저가/종가/거래량 columns, DatetimeIndex
    """
    cache_name = "kospi_index_3y"
    cached = _load_cache(cache_name, end_date)
    if cached is not None:
        return cached

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=1110)
    start_str = start_dt.strftime("%Y%m%d")

    try:
        df = _retry(
            stock.get_index_ohlcv_by_date,
            start_str,
            end_date,
            "1001",
            name_display=False,
        )
        if df is not None and not df.empty:
            _save_cache(cache_name, end_date, df)
            return df
        raise RuntimeError("KRX 빈 응답")
    except Exception as e:
        api_msg = _krx_api_failure_msg("get_index_ohlcv_by_date(KOSPI)", e)

        # Fallback 1: FinanceDataReader (Yahoo ^KS11)
        fdr_df = _fdr_index_ohlcv_fallback("1001", start_str, end_date)
        if fdr_df is not None and not fdr_df.empty:
            _save_cache(cache_name, end_date, fdr_df)
            msg = f"KOSPI 지수 3Y: {api_msg} → FinanceDataReader로 대체 수집({len(fdr_df)}일)"
            _add_fallback_warning(msg)
            return fdr_df

        # Fallback 2: 이전 캐시
        fallback, src_date = _find_nearest_cache(cache_name, end_date)
        if fallback is not None:
            _save_cache(cache_name, end_date, fallback)
            msg = f"KOSPI 지수 3Y: {api_msg} → {src_date}일자 캐시 데이터로 대체 사용"
            _add_fallback_warning(msg)
            return fallback
        raise RuntimeError(f"KOSPI 지수 3Y: {api_msg}. 캐시 데이터도 없습니다.") from e


def fetch_kosdaq_index_3year(end_date: str) -> pd.DataFrame:
    """KOSDAQ 지수 3년 OHLCV (AI 학습용, 단일 API 호출).

    Returns:
        DataFrame with 시가/고가/저가/종가/거래량 columns, DatetimeIndex
    """
    cache_name = "kosdaq_index_3y"
    cached = _load_cache(cache_name, end_date)
    if cached is not None:
        return cached

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=1110)
    start_str = start_dt.strftime("%Y%m%d")

    try:
        df = _retry(
            stock.get_index_ohlcv_by_date,
            start_str,
            end_date,
            "2001",
            name_display=False,
        )
        if df is not None and not df.empty:
            _save_cache(cache_name, end_date, df)
            return df
        raise RuntimeError("KRX 빈 응답")
    except Exception as e:
        api_msg = _krx_api_failure_msg("get_index_ohlcv_by_date(KOSDAQ)", e)

        # Fallback 1: FinanceDataReader (Yahoo ^KQ11)
        fdr_df = _fdr_index_ohlcv_fallback("2001", start_str, end_date)
        if fdr_df is not None and not fdr_df.empty:
            _save_cache(cache_name, end_date, fdr_df)
            msg = f"KOSDAQ 지수 3Y: {api_msg} → FinanceDataReader로 대체 수집({len(fdr_df)}일)"
            _add_fallback_warning(msg)
            return fdr_df

        # Fallback 2: 이전 캐시
        fallback, src_date = _find_nearest_cache(cache_name, end_date)
        if fallback is not None:
            _save_cache(cache_name, end_date, fallback)
            msg = f"KOSDAQ 지수 3Y: {api_msg} → {src_date}일자 캐시 데이터로 대체 사용"
            _add_fallback_warning(msg)
            return fallback
        raise RuntimeError(f"KOSDAQ 지수 3Y: {api_msg}. 캐시 데이터도 없습니다.") from e


# ──────────────────────────────────────────────
# ETF 전용 데이터 수집
# ──────────────────────────────────────────────


def fetch_etf_tickers(date: str, progress_callback=None) -> pd.DataFrame:
    """국내 ETF 전종목 티커/종목명 반환.

    Returns:
        DataFrame with columns: Ticker, Name, Market("ETF")
    """
    cached = _load_cache("etf_tickers", date)
    if cached is not None and not cached.empty:
        return cached

    # Open API 우선 모드
    if _force_openapi:
        openapi_result = _try_krx_openapi("fetch_etf_tickers", date)
        if openapi_result is not None:
            _save_cache("etf_tickers", date, openapi_result)
            msg = (
                f"ETF 티커: KRX Open API 우선 모드로 "
                f"수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

    try:
        tickers = _retry(stock.get_etf_ticker_list, date)
        rows = []
        for t in tickers:
            try:
                name = stock.get_etf_ticker_name(t)
                if not isinstance(name, str) or not name:
                    name = t
            except Exception:
                name = t
            rows.append({"Ticker": t, "Name": name, "Market": "ETF"})

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("KRX 빈 응답")

        _save_cache("etf_tickers", date, df)
        if progress_callback:
            progress_callback(f"ETF 티커 수집 완료: {len(df)}종목")
        return df

    except Exception as e:
        api_msg = _krx_api_failure_msg("get_etf_ticker_list", e)

        # Fallback 1: 직전 평일 날짜로 KRX API 재시도
        if _is_krx_nontrading_error(e) or "비거래일" in str(e):
            adj_tickers, adj_date = _try_previous_weekdays(
                stock.get_etf_ticker_list, date
            )
            if adj_tickers:
                rows = []
                for t in adj_tickers:
                    try:
                        name = stock.get_etf_ticker_name(t)
                        if not isinstance(name, str) or not name:
                            name = t
                    except Exception:
                        name = t
                    rows.append({"Ticker": t, "Name": name, "Market": "ETF"})
                df = pd.DataFrame(rows)
                if not df.empty:
                    _save_cache("etf_tickers", date, df)
                    msg = (
                        f"ETF 티커: {api_msg} → "
                        f"{adj_date}일자 KRX API로 재수집 성공({len(df)}종목)"
                    )
                    _add_fallback_warning(msg)
                    if progress_callback:
                        progress_callback(f"⚠️ {msg}")
                    return df

        # Fallback 2: KRX Open API
        openapi_result = _try_krx_openapi("fetch_etf_tickers", date)
        if openapi_result is not None:
            _save_cache("etf_tickers", date, openapi_result)
            msg = (
                f"ETF 티커: {api_msg} → "
                f"KRX Open API로 대체 수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

        # Fallback 3: FinanceDataReader (설치 시에만 동작)
        fdr_result = _fdr_etf_tickers_fallback()
        if fdr_result is not None:
            _save_cache("etf_tickers", date, fdr_result)
            msg = (
                f"ETF 티커: {api_msg} → "
                f"FinanceDataReader로 대체 수집({len(fdr_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"⚠️ {msg}")
            return fdr_result

        # Fallback 4: 이전 캐시
        fallback, src_date = _find_nearest_cache("etf_tickers", date)
        if fallback is not None:
            _save_cache("etf_tickers", date, fallback)
            msg = (
                f"ETF 티커: {api_msg} → "
                f"{src_date}일자 캐시 데이터({len(fallback)}종목)로 대체 사용"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"⚠️ {msg}")
            return fallback
        raise RuntimeError(
            f"ETF 티커: {api_msg}. "
            "캐시에도 이전 데이터가 없습니다. "
            "평일 장 마감 후 최초 1회 데이터 수집이 필요합니다."
        ) from e


def fetch_etf_ohlcv_all(date: str, progress_callback=None) -> pd.DataFrame:
    """전체 ETF 당일 OHLCV+NAV+거래대금 (유니버스 필터링용).

    Returns:
        DataFrame indexed by Ticker, with NAV/시가/고가/저가/종가/거래량/거래대금 columns
    """
    cached = _load_cache("etf_ohlcv_all", date)
    if cached is not None and not (isinstance(cached, pd.DataFrame) and cached.empty):
        return cached

    try:
        df = _retry(stock.get_etf_ohlcv_by_ticker, date)
        if df is None or df.empty:
            raise RuntimeError("KRX 빈 응답")

        df.index.name = "Ticker"
        _save_cache("etf_ohlcv_all", date, df)
        _clear_fallback_marker("etf_ohlcv_all", date)
        if progress_callback:
            progress_callback(f"ETF 전종목 OHLCV: {len(df)}종목")
        return df

    except Exception as e:
        api_msg = _krx_api_failure_msg("get_etf_ohlcv_by_ticker", e)

        # Fallback 1: 직전 평일 날짜로 KRX API 재시도
        if _is_krx_nontrading_error(e) or "비거래일" in str(e):
            adj_df, adj_date = _try_previous_weekdays(
                stock.get_etf_ohlcv_by_ticker, date
            )
            if adj_df is not None and not adj_df.empty:
                adj_df.index.name = "Ticker"
                _save_cache("etf_ohlcv_all", date, adj_df)
                msg = (
                    f"ETF OHLCV: {api_msg} → "
                    f"{adj_date}일자 KRX API로 재수집 성공({len(adj_df)}종목)"
                )
                _add_fallback_warning(msg)
                if progress_callback:
                    progress_callback(f"⚠️ {msg}")
                return adj_df

        # Fallback 2: KRX Open API
        openapi_result = _try_krx_openapi("fetch_etf_ohlcv_all", date)
        if openapi_result is not None:
            _save_cache("etf_ohlcv_all", date, openapi_result)
            _clear_fallback_marker("etf_ohlcv_all", date)
            msg = (
                f"ETF OHLCV: {api_msg} → "
                f"KRX Open API로 대체 수집({len(openapi_result)}종목)"
            )
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"🔄 {msg}")
            return openapi_result

        # Fallback 3: 이전 캐시
        fallback, src_date = _find_nearest_cache("etf_ohlcv_all", date)
        if fallback is not None:
            _save_cache("etf_ohlcv_all", date, fallback)
            _save_fallback_marker("etf_ohlcv_all", date, src_date)
            msg = f"ETF OHLCV: {api_msg} → {src_date}일자 캐시 데이터로 대체 사용"
            _add_fallback_warning(msg)
            if progress_callback:
                progress_callback(f"⚠️ {msg}")
            return fallback
        if progress_callback:
            progress_callback(f"⚠️ ETF OHLCV: {api_msg}. 캐시 데이터도 없음")
        return pd.DataFrame()


def fetch_etf_market_cap(date: str) -> pd.DataFrame:
    """전체 ETF 시가총액(AUM 대용) 조회.

    pykrx의 get_market_cap_by_ticker()는 ETF 티커에 대해서도
    시가총액(상장주식수 × 종가)을 반환한다.
    """
    cached = _load_cache("etf_market_cap", date)
    if cached is not None and not (isinstance(cached, pd.DataFrame) and cached.empty):
        return cached

    try:
        df = _retry(stock.get_market_cap_by_ticker, date)
        if df is None or df.empty:
            raise RuntimeError("KRX 빈 응답")

        df.index.name = "Ticker"
        _save_cache("etf_market_cap", date, df)
        return df

    except Exception as e:
        api_msg = _krx_api_failure_msg("get_market_cap_by_ticker", e)
        fallback, src_date = _find_nearest_cache("etf_market_cap", date)
        if fallback is not None:
            msg = f"ETF 시가총액: {api_msg} → {src_date}일자 캐시 데이터로 대체 사용"
            _add_fallback_warning(msg)
            return fallback
        return pd.DataFrame()


def fetch_etf_ohlcv_3year(ticker: str, end_date: str) -> pd.DataFrame:
    """개별 ETF 3년치 OHLCV (AI ETF 학습용).

    pykrx의 stock.get_market_ohlcv()는 ETF 티커도 동일하게 처리.
    컬럼: 시가/고가/저가/종가/거래량/거래대금 — compute_ml_features_surge()와 호환.
    캐시 사용 + KRX/FDR 연속 실패 시 fast-fail.
    """
    global _krx_ohlcv_blocked, _fdr_consecutive_fails

    # 캐시 확인 (7일 이내)
    cache_name = f"etf_ohlcv_3y_{ticker}"
    cached = _load_cache(cache_name, end_date)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached
    # 이전 날짜 캐시도 확인
    prev_cached, _src = _find_nearest_cache(cache_name, end_date, max_age_days=7)
    if (
        prev_cached is not None
        and isinstance(prev_cached, pd.DataFrame)
        and not prev_cached.empty
    ):
        return prev_cached

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=1110)
    start_str = start_dt.strftime("%Y%m%d")

    # KRX가 이미 차단된 것으로 확인된 경우 → 바로 FDR
    if not _krx_ohlcv_blocked:
        try:
            df = _retry(stock.get_market_ohlcv, start_str, end_date, ticker)
            if df is not None and not df.empty:
                _save_cache(cache_name, end_date, df)
                return df
            raise RuntimeError("KRX 빈 응답")
        except Exception:
            _krx_ohlcv_blocked = True

    # FDR도 연속 실패 시 나머지 스킵
    if _fdr_consecutive_fails >= _FDR_MAX_FAILS:
        return pd.DataFrame()

    # Fallback: FinanceDataReader
    fdr_df = _fdr_ohlcv_fallback(ticker, start_str, end_date)
    if fdr_df is not None and not fdr_df.empty:
        _fdr_consecutive_fails = 0
        _save_cache(cache_name, end_date, fdr_df)
        return fdr_df

    _fdr_consecutive_fails += 1
    return pd.DataFrame()


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 데이터를 주봉으로 리샘플링."""
    if df is None or df.empty:
        return df

    # 인덱스가 DatetimeIndex인지 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    weekly = (
        df.resample("W")
        .agg(
            {
                "시가": "first",
                "고가": "max",
                "저가": "min",
                "종가": "last",
                "거래량": "sum",
            }
        )
        .dropna()
    )

    if "거래대금" in df.columns:
        weekly["거래대금"] = df["거래대금"].resample("W").sum()

    return weekly


def fetch_investor_trading(ticker: str, end_date: str, days: int = 22) -> pd.DataFrame:
    """개별 종목 투자자별 순매수 거래대금 (일별)."""
    cache_name = f"investor_{ticker}_{days}d"
    cached = _load_cache(cache_name, end_date)
    if cached is not None:
        return cached

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=int(days * 1.8))
    start_str = start_dt.strftime("%Y%m%d")

    try:
        df = _retry(stock.get_market_trading_value_by_date, start_str, end_date, ticker)
        if df is not None and not df.empty:
            _save_cache(cache_name, end_date, df)
            return df
        return pd.DataFrame()
    except Exception as e:
        api_msg = _krx_api_failure_msg("get_market_trading_value_by_date", e)
        # 이전 캐시 찾기 (투자자 데이터는 최대 7일 허용)
        fallback, src_date = _find_nearest_cache(cache_name, end_date)
        if fallback is not None:
            _save_cache(cache_name, end_date, fallback)
            _add_fallback_warning(
                f"투자자매매 {ticker}: {api_msg} → {src_date}일자 캐시로 대체 사용"
            )
            return fallback
        return pd.DataFrame()


def _detail_worker(args: tuple) -> tuple:
    """병렬 워커: 단일 종목의 상세 데이터 수집 (최적화).

    1y OHLCV에서 100d를 추출하여 API 호출 절감 (3회→2회).
    KRX 세션 인증 안정성을 위해 요청 간 0.3초 딜레이.
    """
    ticker, end_date = args
    # 1y OHLCV 수집 (100d 데이터를 포함)
    ohlcv_1y = fetch_ohlcv_1year(ticker, end_date)
    time.sleep(0.3)
    # 100d는 1y 마지막 100행에서 추출 (별도 API 호출 불필요)
    if ohlcv_1y is not None and len(ohlcv_1y) > 100:
        ohlcv_100d = ohlcv_1y.iloc[-100:]
    else:
        ohlcv_100d = ohlcv_1y
    investor = fetch_investor_trading(ticker, end_date, 22)
    time.sleep(0.3)
    return (ticker, ohlcv_100d, ohlcv_1y, investor)


def fetch_candidates_detail(
    tickers: list[str], end_date: str, progress_callback=None
) -> dict:
    """후보군 전체의 상세 데이터를 병렬 수집 (5워커).

    최적화:
    - 1y OHLCV에서 100d 추출 (API 호출 3→2회/종목, 33% 절감)
    - 5워커 병렬 수집 (KRX 세션 인증 안정성 확보)
    연속 5회 타임아웃 시 KRX API 불능으로 판단하고 조기 중단한다.
    """
    result = {
        "ohlcv_100d": {},
        "ohlcv_1y": {},
        "investor": {},
    }

    total = len(tickers)
    completed = 0
    consecutive_timeouts = 0
    MAX_CONSECUTIVE_TIMEOUTS = 5
    n_workers = min(5, total)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_ticker = {}
        for t in tickers:
            f = executor.submit(_detail_worker, (t, end_date))
            future_to_ticker[f] = t

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                t, ohlcv_100d, ohlcv_1y, investor = future.result(timeout=90)
                result["ohlcv_100d"][t] = ohlcv_100d
                result["ohlcv_1y"][t] = ohlcv_1y
                result["investor"][t] = investor
                consecutive_timeouts = 0

            except TimeoutError:
                consecutive_timeouts += 1
                if progress_callback:
                    progress_callback(
                        f"⏱️ {ticker} 타임아웃 "
                        f"({consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS})"
                    )
                if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    if progress_callback:
                        progress_callback(
                            f"🛑 KRX API 연속 타임아웃 → 수집 중단 "
                            f"({completed}/{total})"
                        )
                    for f in future_to_ticker:
                        f.cancel()
                    break

            except Exception as e:
                if progress_callback:
                    progress_callback(f"⚠️ {ticker} 수집 실패: {e}")

            completed += 1
            if progress_callback:
                progress_callback(f"후보군 상세 수집: {completed}/{total} ({ticker})")

            # 100건마다 중간 저장
            if completed % 100 == 0:
                _save_cache("candidates_detail_partial", end_date, result)

    _save_cache("candidates_detail", end_date, result)
    return result


def _rmtree_robust(path: Path, retries: int = 3, delay: float = 1.0):
    """Windows 파일 잠금 문제를 처리하는 강건한 디렉터리 삭제."""
    import gc
    import shutil

    def onerror(func, path, exc_info):
        """삭제 실패 시 권한 변경 후 재시도."""
        import os
        import stat

        # 읽기 전용 속성 제거 후 재시도
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    for attempt in range(retries):
        try:
            # 가비지 컬렉션으로 파일 핸들 해제 유도
            gc.collect()
            shutil.rmtree(path, onerror=onerror)
            return True
        except OSError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                # 최종 실패 시 개별 파일 삭제 시도
                try:
                    for item in path.rglob("*"):
                        try:
                            if item.is_file():
                                item.unlink()
                        except Exception:
                            pass
                    # 빈 디렉터리들 삭제
                    for item in sorted(path.rglob("*"), reverse=True):
                        try:
                            if item.is_dir():
                                item.rmdir()
                        except Exception:
                            pass
                    path.rmdir()
                    return True
                except Exception:
                    # 삭제 실패해도 계속 진행
                    return False
    return False


def clear_cache(date: str = None):
    """캐시 초기화."""
    if date:
        d = CACHE_DIR / date
        if d.exists():
            _rmtree_robust(d)
    else:
        for item in CACHE_DIR.iterdir():
            if item.is_dir() and item.name != "presets":
                _rmtree_robust(item)


def cleanup_old_caches(max_age_days: int = 7) -> dict:
    """date-stamped 캐시 디렉토리 중 max_age_days 이상 오래된 것 삭제.

    Returns:
        {"deleted": [(날짜, 바이트)], "kept": [날짜], "freed_bytes": int}
    """
    cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y%m%d")
    deleted, kept = [], []
    freed = 0

    for d in sorted(CACHE_DIR.iterdir()):
        if not d.is_dir() or not d.name.isdigit() or len(d.name) != 8:
            continue
        if d.name < cutoff:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            if _rmtree_robust(d):
                deleted.append((d.name, size))
                freed += size
        else:
            kept.append(d.name)

    return {"deleted": deleted, "kept": kept, "freed_bytes": freed}
