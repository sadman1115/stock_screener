"""KRX Open API 래퍼 모듈 — pykrx 실패 시 2차 데이터 소스.

KRX 공식 REST API (http://data-dbg.krx.co.kr/svc/apis/)를 사용하여
전종목 OHLCV, 시가총액, 기본정보, ETF 데이터를 수집한다.

인증: auth_key 쿼리 파라미터 (openapi.krx.co.kr 에서 무료 발급)
제한: 일일 10,000건 / 비상업적 이용
데이터: 익영업일 08:00 갱신 (실시간 아님)

사용 전 필수 절차:
1. openapi.krx.co.kr 회원가입
2. 마이페이지 → 인증키 신청 (관리자 승인 필요)
3. 서비스별 이용 신청 (각 API 엔드포인트 개별 활성화)
"""

import logging
import os
from typing import Any

import pandas as pd
import requests

from .config import KRX_OPENAPI_AUTH_KEY

logger = logging.getLogger(__name__)

_BASE_URL = "http://data-dbg.krx.co.kr/svc/apis"
_AUTH_KEY: str = os.environ.get("KRX_OPENAPI_AUTH_KEY", "") or KRX_OPENAPI_AUTH_KEY

# ── 세션 내 메모리 캐시 (같은 날짜를 여러 함수가 공유) ──
_session_cache: dict[str, Any] = {}

# ── 인증 실패 플래그 — 한번 401 받으면 이후 호출 스킵 ──
_auth_failed = False


def is_configured() -> bool:
    """AUTH_KEY가 설정되어 있고 인증 실패 상태가 아닌지 확인."""
    return bool(_AUTH_KEY) and not _auth_failed


def reset_auth_flag():
    """인증 실패 플래그 초기화 (새 세션 시작 시)."""
    global _auth_failed
    _auth_failed = False


def clear_session_cache():
    """세션 캐시 초기화."""
    _session_cache.clear()


# ══════════════════════════════════════════════
# 내부 HTTP 클라이언트
# ══════════════════════════════════════════════


def _get(endpoint: str, params: dict, timeout: int = 30) -> list[dict]:
    """KRX Open API GET 요청. OutBlock_1 리스트 반환.

    Raises:
        RuntimeError: 인증 실패, 빈 응답, HTTP 에러
    """
    global _auth_failed

    if not _AUTH_KEY:
        raise RuntimeError("KRX Open API AUTH_KEY 미설정")
    if _auth_failed:
        raise RuntimeError("KRX Open API 인증 실패 상태 (이전 401 오류)")

    url = f"{_BASE_URL}/{endpoint}"
    headers = {"Authorization": _AUTH_KEY}

    resp = requests.get(url, params=params, headers=headers, timeout=timeout)

    # 401 → 인증 실패 플래그 설정 (이후 호출 스킵)
    if resp.status_code == 401:
        _auth_failed = True
        data = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        msg = data.get("respMsg", "Unauthorized")
        raise RuntimeError(f"KRX Open API 인증 실패: {msg}")

    resp.raise_for_status()
    data = resp.json()

    # 에러 응답 체크
    if data.get("respCode") and data["respCode"] != "200":
        raise RuntimeError(f"KRX Open API 오류: {data.get('respMsg', data)}")

    block = data.get("OutBlock_1", [])
    if not block:
        raise RuntimeError(f"KRX Open API 빈 응답: {endpoint}")

    return block


# ══════════════════════════════════════════════
# 숫자 변환 유틸리티
# ══════════════════════════════════════════════


def _to_int(v) -> int:
    """문자열(콤마 포함 가능)을 정수로 변환."""
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).replace(",", "").strip()
    if not s or s == "-" or s == "":
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _to_float(v) -> float:
    """문자열을 실수로 변환."""
    if isinstance(v, int | float):
        return float(v)
    s = str(v).replace(",", "").strip()
    if not s or s == "-" or s == "":
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _get_field(row: dict, *candidates, default="") -> str:
    """여러 가능한 필드명 중 첫 번째로 존재하는 값 반환."""
    for key in candidates:
        if key in row:
            return row[key]
    return default


# ══════════════════════════════════════════════
# 캐시된 벌크 데이터 조회
# ══════════════════════════════════════════════


def _fetch_daily_trade(date: str) -> dict[str, list[dict]]:
    """KOSPI/KOSDAQ 전종목 일별매매정보 조회 (세션 캐시).

    Returns: {"KOSPI": [...], "KOSDAQ": [...]}
    """
    cache_key = f"daily_trade_{date}"
    if cache_key in _session_cache:
        return _session_cache[cache_key]

    result: dict[str, list[dict]] = {}
    endpoints = [
        ("sto/stk_bydd_trd", "KOSPI"),
        ("sto/ksq_bydd_trd", "KOSDAQ"),
    ]
    errors = []
    for ep, market in endpoints:
        try:
            rows = _get(ep, {"basDd": date})
            result[market] = rows
        except Exception as e:
            logger.warning("KRX Open API %s 실패: %s", ep, e)
            result[market] = []
            errors.append(str(e))

    if not result.get("KOSPI") and not result.get("KOSDAQ"):
        raise RuntimeError(
            f"KRX Open API 전종목 일별매매정보 양쪽 시장 모두 실패: {'; '.join(errors)}"
        )

    _session_cache[cache_key] = result
    return result


def _fetch_etf_daily_trade(date: str) -> list[dict]:
    """ETF 전종목 일별매매정보 조회 (세션 캐시)."""
    cache_key = f"etf_daily_trade_{date}"
    if cache_key in _session_cache:
        return _session_cache[cache_key]

    rows = _get("etp/etf_bydd_trd", {"basDd": date})
    _session_cache[cache_key] = rows
    return rows


# ══════════════════════════════════════════════
# 외부 인터페이스 — pykrx 호환 DataFrame 반환
# ══════════════════════════════════════════════


def fetch_all_ohlcv(date: str) -> pd.DataFrame:
    """전종목 당일 OHLCV (pykrx get_market_ohlcv 호환).

    Returns:
        DataFrame indexed by Ticker with columns:
        시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률, Market
    """
    data = _fetch_daily_trade(date)
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        rows = data.get(market, [])
        if not rows:
            continue
        records = []
        for r in rows:
            ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
            if not ticker:
                continue
            records.append(
                {
                    "Ticker": ticker,
                    "시가": _to_int(_get_field(r, "TDD_OPNPRC", "시가")),
                    "고가": _to_int(_get_field(r, "TDD_HGPRC", "고가")),
                    "저가": _to_int(_get_field(r, "TDD_LWPRC", "저가")),
                    "종가": _to_int(_get_field(r, "TDD_CLSPRC", "종가")),
                    "거래량": _to_int(_get_field(r, "ACC_TRDVOL", "거래량")),
                    "거래대금": _to_int(_get_field(r, "ACC_TRDVAL", "거래대금")),
                    "등락률": _to_float(
                        _get_field(r, "FLUC_RT", "등락률", default="0")
                    ),
                    "Market": market,
                }
            )
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        raise RuntimeError("KRX Open API OHLCV 데이터 없음")

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index("Ticker")
    result.index.name = "Ticker"
    return result


def fetch_all_market_cap(date: str) -> pd.DataFrame:
    """전종목 시가총액 (pykrx get_market_cap 호환).

    Returns:
        DataFrame indexed by Ticker with columns:
        종가, 거래량, 거래대금, 시가총액, 상장주식수
    """
    data = _fetch_daily_trade(date)
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        rows = data.get(market, [])
        if not rows:
            continue
        records = []
        for r in rows:
            ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
            if not ticker:
                continue
            records.append(
                {
                    "Ticker": ticker,
                    "종가": _to_int(_get_field(r, "TDD_CLSPRC", "종가")),
                    "거래량": _to_int(_get_field(r, "ACC_TRDVOL", "거래량")),
                    "거래대금": _to_int(_get_field(r, "ACC_TRDVAL", "거래대금")),
                    "시가총액": _to_int(_get_field(r, "MKTCAP", "시가총액")),
                    "상장주식수": _to_int(_get_field(r, "LIST_SHRS", "상장주식수")),
                }
            )
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        raise RuntimeError("KRX Open API 시가총액 데이터 없음")

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index("Ticker")
    result.index.name = "Ticker"
    return result


def fetch_all_tickers(date: str) -> pd.DataFrame:
    """전종목 티커/종목명/시장 (pykrx 호환).

    Returns:
        DataFrame with columns: Ticker, Name, Market
    """
    data = _fetch_daily_trade(date)
    rows_out = []
    for market in ["KOSPI", "KOSDAQ"]:
        rows = data.get(market, [])
        for r in rows:
            ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
            name = _get_field(r, "ISU_ABBRV", "ISU_NM", "종목명", default=ticker)
            if ticker:
                rows_out.append({"Ticker": ticker, "Name": name, "Market": market})

    if not rows_out:
        raise RuntimeError("KRX Open API 티커 데이터 없음")

    return pd.DataFrame(rows_out)


def fetch_all_fundamentals(date: str) -> pd.DataFrame:
    """전종목 BPS/PER/PBR/EPS/DIV/DPS (pykrx get_market_fundamental 호환).

    KRX Open API의 PER/PBR 엔드포인트 사용.
    해당 엔드포인트가 없으면 빈 DataFrame 반환 (캐시 fallback으로 전환).

    Returns:
        DataFrame indexed by Ticker with columns: BPS, PER, PBR, EPS, DIV, DPS
    """
    frames = []
    endpoints = [
        ("sto/stk_bydd_trd_per_pbr_dvd", "KOSPI"),
        ("sto/ksq_bydd_trd_per_pbr_dvd", "KOSDAQ"),
    ]
    for ep, _market in endpoints:
        try:
            rows = _get(ep, {"basDd": date})
        except Exception:
            continue
        records = []
        for r in rows:
            ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
            if not ticker:
                continue
            records.append(
                {
                    "Ticker": ticker,
                    "BPS": _to_int(_get_field(r, "BPS", default="0")),
                    "PER": _to_float(_get_field(r, "PER", default="0")),
                    "PBR": _to_float(_get_field(r, "PBR", default="0")),
                    "EPS": _to_int(_get_field(r, "EPS", default="0")),
                    "DIV": _to_float(
                        _get_field(r, "DVD_YLD", "DIV", "배당수익률", default="0")
                    ),
                    "DPS": _to_int(_get_field(r, "DPS", default="0")),
                }
            )
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        raise RuntimeError("KRX Open API Fundamental 데이터 없음")

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index("Ticker")
    result.index.name = "Ticker"
    return result


def fetch_sector_classifications(date: str) -> pd.DataFrame:
    """전종목 업종분류 (pykrx get_market_sector_classifications 호환).

    일별매매정보 응답의 SECT_TP_NM 필드 또는 종목기본정보의 업종 필드 사용.

    Returns:
        DataFrame indexed by Ticker with columns: 종목명, 업종명
    """
    data = _fetch_daily_trade(date)
    records = []
    for market in ["KOSPI", "KOSDAQ"]:
        rows = data.get(market, [])
        for r in rows:
            ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
            name = _get_field(r, "ISU_ABBRV", "ISU_NM", "종목명", default=ticker)
            sector = _get_field(r, "SECT_TP_NM", "IDX_IND_NM", "업종명", default="")
            if ticker and sector:
                records.append(
                    {
                        "Ticker": ticker,
                        "종목명": name,
                        "업종명": sector,
                    }
                )

    if not records:
        raise RuntimeError("KRX Open API 업종분류 데이터 없음")

    result = pd.DataFrame(records)
    result = result.set_index("Ticker")
    result.index.name = "Ticker"
    return result


def fetch_etf_tickers(date: str) -> pd.DataFrame:
    """ETF 전종목 티커/종목명 (pykrx 호환).

    Returns:
        DataFrame with columns: Ticker, Name, Market("ETF")
    """
    rows = _fetch_etf_daily_trade(date)
    records = []
    for r in rows:
        ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
        name = _get_field(r, "ISU_ABBRV", "ISU_NM", "종목명", default=ticker)
        if ticker:
            records.append({"Ticker": ticker, "Name": name, "Market": "ETF"})

    if not records:
        raise RuntimeError("KRX Open API ETF 티커 데이터 없음")

    return pd.DataFrame(records)


def fetch_etf_ohlcv_all(date: str) -> pd.DataFrame:
    """ETF 전종목 당일 OHLCV (pykrx get_etf_ohlcv_by_ticker 호환).

    Returns:
        DataFrame indexed by Ticker with columns:
        NAV, 시가, 고가, 저가, 종가, 거래량, 거래대금
    """
    rows = _fetch_etf_daily_trade(date)
    records = []
    for r in rows:
        ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
        if not ticker:
            continue
        records.append(
            {
                "Ticker": ticker,
                "NAV": _to_float(_get_field(r, "NAV", "NETASST_TOTAMT", default="0")),
                "시가": _to_int(_get_field(r, "TDD_OPNPRC", "시가")),
                "고가": _to_int(_get_field(r, "TDD_HGPRC", "고가")),
                "저가": _to_int(_get_field(r, "TDD_LWPRC", "저가")),
                "종가": _to_int(_get_field(r, "TDD_CLSPRC", "종가")),
                "거래량": _to_int(_get_field(r, "ACC_TRDVOL", "거래량")),
                "거래대금": _to_int(_get_field(r, "ACC_TRDVAL", "거래대금")),
            }
        )

    if not records:
        raise RuntimeError("KRX Open API ETF OHLCV 데이터 없음")

    result = pd.DataFrame(records)
    result = result.set_index("Ticker")
    result.index.name = "Ticker"
    return result


def fetch_single_day_ohlcv(date: str) -> pd.DataFrame:
    """특정 일자 전종목 OHLCV (fetch_all_ohlcv_ndays 내부용).

    fetch_all_ohlcv와 동일하지만 Market 컬럼 없이 반환.
    """
    data = _fetch_daily_trade(date)
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        rows = data.get(market, [])
        if not rows:
            continue
        records = []
        for r in rows:
            ticker = _get_field(r, "ISU_SRT_CD", "ISU_CD", "종목코드")
            if not ticker:
                continue
            records.append(
                {
                    "Ticker": ticker,
                    "시가": _to_int(_get_field(r, "TDD_OPNPRC", "시가")),
                    "고가": _to_int(_get_field(r, "TDD_HGPRC", "고가")),
                    "저가": _to_int(_get_field(r, "TDD_LWPRC", "저가")),
                    "종가": _to_int(_get_field(r, "TDD_CLSPRC", "종가")),
                    "거래량": _to_int(_get_field(r, "ACC_TRDVOL", "거래량")),
                    "거래대금": _to_int(_get_field(r, "ACC_TRDVAL", "거래대금")),
                }
            )
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index("Ticker")
    result.index.name = "Ticker"
    return result
