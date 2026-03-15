"""OpenDartReader 기반 재무제표 수집 모듈.

DART API 키가 없으면 재무데이터 없이 진행하되 경고를 반환한다.

2단계 하이브리드 수집:
  Phase 1: fnlttMultiAcnt.json 배치(100종목/호출)로 데이터 존재 여부 스캔 → ~30 API 호출
  Phase 2: 데이터 있는 종목만 fnlttSinglAcntAll.json 개별 호출 → ~60% API 절감
"""

import time
import pickle
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as _requests
import pandas as pd

from .config import CACHE_DIR

_dart = None
_dart_available = False
_api_key = None

# 병렬 수집 설정
_MAX_WORKERS = 10
_WORKER_DELAY = 0.0  # API 자체 지연으로 충분, 추가 sleep 불필요


def init_dart(api_key: str) -> bool:
    """DART API 초기화. 성공 시 True."""
    global _dart, _dart_available, _api_key
    if not api_key:
        _dart_available = False
        return False
    try:
        import OpenDartReader
        _dart = OpenDartReader(api_key)
        _dart_available = True
        _api_key = api_key
        return True
    except Exception:
        _dart_available = False
        return False


def is_available() -> bool:
    return _dart_available


def _get_quarter(date: str) -> str:
    """날짜에서 분기 문자열 반환 (예: '2025Q1')."""
    year = int(date[:4])
    month = int(date[4:6])
    quarter = (month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def _dart_cache_dir() -> Path:
    """DART 전용 캐시 디렉터리 (분기별)."""
    d = CACHE_DIR / "dart"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(name: str, date: str) -> Path:
    """DART 캐시 경로 - 분기별 폴더 사용."""
    quarter = _get_quarter(date)
    d = _dart_cache_dir() / quarter
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.pkl"


def _load_cache(name: str, date: str):
    """캐시 로드 - 현재 분기 우선, 이전 분기 fallback."""
    # 현재 분기에서 찾기
    p = _cache_path(name, date)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)

    # 이전 분기에서 찾기 (분기 초에 유용)
    year = int(date[:4])
    month = int(date[4:6])
    quarter = (month - 1) // 3 + 1

    if quarter == 1:
        prev_quarter = f"{year - 1}Q4"
    else:
        prev_quarter = f"{year}Q{quarter - 1}"

    prev_path = _dart_cache_dir() / prev_quarter / f"{name}.pkl"
    if prev_path.exists():
        with open(prev_path, "rb") as f:
            return pickle.load(f)

    return None


def _save_cache(name: str, date: str, data):
    p = _cache_path(name, date)
    with open(p, "wb") as f:
        pickle.dump(data, f)


def _extract_amount(fs_df: pd.DataFrame, account_names: list[str],
                     amount_col: str = "thstrm_amount") -> float:
    """재무제표 DataFrame에서 특정 계정과목의 금액을 추출.

    Args:
        amount_col: 추출할 금액 컬럼.
            "thstrm_amount" = 당기금액 (기본),
            "frmtrm_amount" = 전기금액.
    """
    if fs_df is None or fs_df.empty:
        return 0.0

    # amount_col 우선, fallback으로 _dt 변형
    if amount_col == "thstrm_amount":
        amt_cols = ["thstrm_amount", "thstrm_dt_amount"]
    elif amount_col == "frmtrm_amount":
        amt_cols = ["frmtrm_amount", "frmtrm_dt_amount"]
    elif amount_col == "bfefrmtrm_amount":
        amt_cols = ["bfefrmtrm_amount", "bfefrmtrm_dt_amount"]
    else:
        amt_cols = [amount_col]

    for name in account_names:
        # account_nm 컬럼에서 검색
        col_candidates = ["account_nm", "sj_nm"]
        for col in col_candidates:
            if col not in fs_df.columns:
                continue
            matches = fs_df[fs_df[col].str.contains(name, na=False, regex=False)]
            if not matches.empty:
                for ac in amt_cols:
                    if ac in matches.columns:
                        val = matches[ac].iloc[0]
                        if pd.notna(val):
                            try:
                                return float(str(val).replace(",", ""))
                            except (ValueError, TypeError):
                                continue
    return 0.0


def _empty_result() -> dict:
    """빈 재무 결과 딕셔너리 생성."""
    return {
        "cash": 0.0, "st_financial": 0.0,
        "land": 0.0, "building": 0.0, "invest_property": 0.0,
        "total_assets": 0.0, "operating_profit": 0.0, "net_income": 0.0,
        "revenue": 0.0, "revenue_prev": 0.0, "op_cash_flow": 0.0,
        "total_liabilities": 0.0, "total_equity": 0.0,
        "revenue_prev2": 0.0, "op_prev": 0.0, "op_cash_flow_prev": 0.0,
        "net_income_prev": 0.0, "interest_expense": 0.0,
        "short_borrowing": 0.0, "long_borrowing": 0.0,
        "depreciation": 0.0, "available": False,
    }


def _parse_fs(dart_inst, ticker: str, year: int) -> pd.DataFrame | None:
    """DART API에서 재무제표 DataFrame을 가져온다.

    최신 데이터를 우선 시도하는 전략:
      1) 전년(year-1) 3분기보고서 — 가장 최근 확정 데이터
      2) 전년(year-1) 사업보고서 — 제출 완료 시 가장 완전
      3) 전전년(year-2) 사업보고서 — 확실한 fallback

    예: year=2026 → 2025-Q3 → 2025-연간 → 2024-연간
    """
    # (year-1) 3분기보고서(11014): 전년 11월 제출 완료, 가장 최신
    try:
        fs = dart_inst.finstate_all(ticker, year - 1, reprt_code="11014")
        if fs is not None and not fs.empty:
            return fs
    except Exception:
        pass

    # (year-1) 사업보고서(11011): 당해 3월 제출, 일부 조기 제출
    try:
        fs = dart_inst.finstate_all(ticker, year - 1, reprt_code="11011")
        if fs is not None and not fs.empty:
            return fs
    except Exception:
        pass

    # (year-2) 사업보고서(11011): 확정 fallback
    try:
        fs = dart_inst.finstate_all(ticker, year - 2, reprt_code="11011")
        if fs is not None and not fs.empty:
            return fs
    except Exception:
        pass

    return None


def _resolve_corp_codes(dart_inst, tickers: list[str]) -> dict[str, str]:
    """ticker → corp_code 매핑 (API 호출 없음, corp_codes DataFrame 참조)."""
    mapping = {}
    for t in tickers:
        cc = dart_inst.find_corp_code(t)
        if cc:
            mapping[t] = cc
    return mapping


def _batch_scan(api_key: str, ticker_to_corp: dict[str, str],
                year: int, progress_callback=None) -> dict[str, tuple]:
    """Phase 1: fnlttMultiAcnt.json 배치로 데이터 존재 여부 + 보고서코드 확인.

    Returns:
        {ticker: (bsns_year, reprt_code)} — 데이터가 있는 종목과 사용할 보고서 정보
    """
    code_to_ticker = {cc: t for t, cc in ticker_to_corp.items()}
    valid_codes = list(code_to_ticker.keys())

    ticker_report = {}  # ticker -> (bsns_year, reprt_code)
    found_codes = set()

    # 보고서 우선순위: Q3 → Annual year-1 → Annual year-2
    report_tries = [
        (year - 1, "11014"),
        (year - 1, "11011"),
        (year - 2, "11011"),
    ]

    batch_size = 100
    api_calls = 0

    for bsns_year, reprt_code in report_tries:
        remaining = [cc for cc in valid_codes if cc not in found_codes]
        if not remaining:
            break

        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]

            try:
                r = _requests.get(
                    "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json",
                    params={
                        "crtfc_key": api_key,
                        "corp_code": ",".join(batch),
                        "bsns_year": bsns_year,
                        "reprt_code": reprt_code,
                    },
                    timeout=15,
                )
                api_calls += 1
                jo = r.json()
                if "list" in jo:
                    df = pd.DataFrame(jo["list"])
                    for cc in df["corp_code"].unique():
                        if cc not in found_codes and cc in code_to_ticker:
                            ticker = code_to_ticker[cc]
                            ticker_report[ticker] = (bsns_year, reprt_code)
                            found_codes.add(cc)
            except Exception:
                pass

            if progress_callback:
                progress_callback(
                    f"DART 스캔: {len(found_codes)}/{len(valid_codes)} 발견 "
                    f"(API {api_calls}회)")

    return ticker_report


def _call_finstate_all(api_key: str, corp_code: str, bsns_year: int,
                       reprt_code: str, fs_div: str = "CFS") -> pd.DataFrame:
    """fnlttSinglAcntAll.json 직접 호출 (print 없음, timeout 포함)."""
    r = _requests.get(
        "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
        params={
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": bsns_year,
            "reprt_code": reprt_code,
            "fs_div": fs_div,
        },
        timeout=10,
    )
    jo = r.json()
    if "list" not in jo:
        return pd.DataFrame()
    return pd.DataFrame(jo["list"])


def _fetch_worker_v2(args: tuple) -> tuple[str, dict, bool]:
    """Phase 2 워커: 확인된 보고서코드로 finstate_all 직접 호출."""
    ticker, corp_code, bsns_year, reprt_code, date, api_key = args

    cache_name = f"financial_{ticker}"
    cached = _load_cache(cache_name, date)
    if cached is not None and "revenue_prev2" in cached:
        return (ticker, cached, False)

    result = _empty_result()
    try:
        # CFS(연결) 우선
        fs = _call_finstate_all(api_key, corp_code, bsns_year,
                                reprt_code, fs_div="CFS")
        if fs is None or fs.empty:
            # OFS(별도) fallback
            fs = _call_finstate_all(api_key, corp_code, bsns_year,
                                    reprt_code, fs_div="OFS")

        if fs is not None and not fs.empty:
            result = _extract_all_fields(fs)
        _save_cache(cache_name, date, result)
    except Exception:
        _save_cache(cache_name, date, result)

    return (ticker, result, True)


def _extract_all_fields(fs: pd.DataFrame) -> dict:
    """재무제표 DataFrame에서 모든 필드를 추출."""
    # 연결재무제표 우선, 없으면 별도
    if "fs_div" in fs.columns:
        cfs = fs[fs["fs_div"] == "CFS"]
        if cfs.empty:
            cfs = fs[fs["fs_div"] == "OFS"]
        if not cfs.empty:
            fs = cfs

    result = _empty_result()

    result["cash"] = _extract_amount(
        fs, ["현금및현금성자산", "현금및현금등가물", "현금과현금성자산"])
    result["st_financial"] = _extract_amount(
        fs, ["단기금융상품", "단기금융자산", "단기투자자산"])
    result["land"] = _extract_amount(fs, ["토지"])
    result["building"] = _extract_amount(fs, ["건물", "건축물"])
    result["invest_property"] = _extract_amount(
        fs, ["투자부동산", "투자용부동산"])
    result["total_assets"] = _extract_amount(
        fs, ["자산총계", "자산 총계"])
    result["operating_profit"] = _extract_amount(
        fs, ["영업이익", "영업손익"])
    result["net_income"] = _extract_amount(
        fs, ["당기순이익", "당기순손익", "분기순이익"])

    _rev_names = ["매출액", "영업수익", "수익(매출액)", "매출"]
    result["revenue"] = _extract_amount(fs, _rev_names)
    result["revenue_prev"] = _extract_amount(
        fs, _rev_names, amount_col="frmtrm_amount")
    _ocf_names = ["영업활동현금흐름", "영업활동으로인한현금흐름",
                  "영업활동으로 인한 현금흐름"]
    result["op_cash_flow"] = _extract_amount(fs, _ocf_names)
    result["total_liabilities"] = _extract_amount(
        fs, ["부채총계", "부채 총계"])
    result["total_equity"] = _extract_amount(
        fs, ["자본총계", "자본 총계"])

    result["revenue_prev2"] = _extract_amount(
        fs, _rev_names, amount_col="bfefrmtrm_amount")
    _op_names = ["영업이익", "영업손익"]
    result["op_prev"] = _extract_amount(
        fs, _op_names, amount_col="frmtrm_amount")
    result["op_cash_flow_prev"] = _extract_amount(
        fs, _ocf_names, amount_col="frmtrm_amount")
    _ni_names = ["당기순이익", "당기순손익", "분기순이익"]
    result["net_income_prev"] = _extract_amount(
        fs, _ni_names, amount_col="frmtrm_amount")
    result["interest_expense"] = _extract_amount(
        fs, ["이자비용", "금융비용", "금융원가"])
    result["short_borrowing"] = _extract_amount(
        fs, ["단기차입금", "단기차입금 및 유동성장기부채"])
    result["long_borrowing"] = _extract_amount(
        fs, ["장기차입금", "사채", "장기차입금및사채"])
    result["depreciation"] = _extract_amount(
        fs, ["감가상각비", "감가상각비와무형자산상각비",
             "유무형자산상각비"])
    result["available"] = True
    return result


def fetch_financial_data(ticker: str, year: int, date: str,
                         progress_callback=None,
                         dart_instance=None) -> dict:
    """단일 종목의 재무제표에서 자산/현금/영업이익 데이터 추출.

    Args:
        dart_instance: 사용할 OpenDartReader 인스턴스. None이면 글로벌 _dart 사용.
    """
    cache_name = f"financial_{ticker}"
    cached = _load_cache(cache_name, date)
    if cached is not None:
        if "revenue_prev2" in cached:
            return cached

    result = _empty_result()
    dart_inst = dart_instance or _dart

    if not _dart_available or dart_inst is None:
        return result

    try:
        fs = _parse_fs(dart_inst, ticker, year)
        if fs is None or fs.empty:
            _save_cache(cache_name, date, result)
            return result
        result = _extract_all_fields(fs)
    except Exception:
        pass

    _save_cache(cache_name, date, result)
    return result


def _create_dart_instances(n: int) -> list:
    """워커용 OpenDartReader 인스턴스를 순차 생성."""
    import OpenDartReader
    instances = []
    for _ in range(n):
        instances.append(OpenDartReader(_api_key))
    return instances


def _fetch_worker(args: tuple) -> tuple[str, dict, bool]:
    """병렬 워커: 단일 종목 fetch. (ticker, result, api_called) 반환."""
    ticker, year, date, dart_inst = args

    cache_name = f"financial_{ticker}"
    cached = _load_cache(cache_name, date)
    if cached is not None and "revenue_prev2" in cached:
        return (ticker, cached, False)

    result = _empty_result()
    try:
        fs = _parse_fs(dart_inst, ticker, year)
        if fs is not None and not fs.empty:
            result = _extract_all_fields(fs)
        _save_cache(cache_name, date, result)
    except Exception:
        _save_cache(cache_name, date, result)

    return (ticker, result, True)


def fetch_candidates_financial(tickers: list[str], year: int, date: str,
                               progress_callback=None) -> pd.DataFrame:
    """후보군 전체의 재무 데이터를 DataFrame으로 반환.

    2단계 하이브리드 수집:
      Phase 1: fnlttMultiAcnt 배치 스캔으로 데이터 존재 여부 확인 (~30 API 호출)
      Phase 2: 데이터 있는 종목만 fnlttSinglAcntAll 개별 호출 (병렬 10워커)
    """
    # 이미 전체 캐시가 있으면 바로 반환
    cached_all = _load_cache("financial_all", date)
    if (cached_all is not None and not cached_all.empty
            and "revenue_prev2" in cached_all.columns):
        if progress_callback:
            progress_callback(f"DART 재무제표: 캐시 로드 완료 ({len(cached_all)}종목)")
        return cached_all

    total = len(tickers)

    # ── 0단계: 개별 캐시 분류 ──
    cached_results = {}
    uncached_tickers = []
    for t in tickers:
        cached = _load_cache(f"financial_{t}", date)
        if cached is not None and "revenue_prev2" in cached:
            cached_results[t] = cached
        else:
            uncached_tickers.append(t)

    if progress_callback:
        progress_callback(
            f"DART 재무제표: 캐시 {len(cached_results)}/{total}, "
            f"미캐시 {len(uncached_tickers)}건")

    # 모두 캐시 히트면 바로 반환
    if not uncached_tickers:
        rows = []
        for t in tickers:
            d = cached_results[t].copy()
            d["Ticker"] = t
            rows.append(d)
        df = pd.DataFrame(rows).set_index("Ticker")
        _save_cache("financial_all", date, df)
        if progress_callback:
            progress_callback(f"DART 재무제표: {total}/{total} (전체 캐시)")
        return df

    # ── Phase 1: 배치 스캔 (데이터 존재 여부 확인) ──
    if progress_callback:
        progress_callback("DART Phase 1: 배치 스캔 시작...")

    ticker_to_corp = _resolve_corp_codes(_dart, uncached_tickers)
    ticker_report = _batch_scan(_api_key, ticker_to_corp, year,
                                progress_callback)

    # 데이터 없는 종목: 빈 결과로 캐시 즉시 저장
    has_data = [t for t in uncached_tickers if t in ticker_report]
    no_data = [t for t in uncached_tickers if t not in ticker_report]

    for t in no_data:
        empty = _empty_result()
        _save_cache(f"financial_{t}", date, empty)
        cached_results[t] = empty

    if progress_callback:
        progress_callback(
            f"DART 스캔 완료: {len(has_data)}종목 상세 수집 필요, "
            f"{len(no_data)}종목 데이터 없음")

    # ── Phase 2: 상세 수집 (데이터 있는 종목만, 직접 API 호출) ──
    api_results = {}
    completed_count = len(cached_results)

    if has_data:
        work_items = []
        for t in has_data:
            bsns_year, reprt_code = ticker_report[t]
            corp_code = ticker_to_corp[t]
            work_items.append(
                (t, corp_code, bsns_year, reprt_code, date, _api_key))

        n_workers = min(_MAX_WORKERS, len(work_items))
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_ticker = {}
            for item in work_items:
                f = executor.submit(_fetch_worker_v2, item)
                future_to_ticker[f] = item[0]

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    t_result, data, api_called = future.result(timeout=30)
                    api_results[t_result] = data
                except Exception:
                    api_results[ticker] = _empty_result()

                with lock:
                    completed_count += 1
                    pct = int(100 * completed_count / total)
                    if progress_callback:
                        progress_callback(
                            f"DART 재무제표: {completed_count}/{total} "
                            f"({pct}%) - {ticker}")

                # 100건마다 중간 저장
                if completed_count % 100 == 0:
                    with lock:
                        _partial_save(tickers, cached_results,
                                      api_results, date)

    # ── 결과 병합 및 저장 ──
    rows = []
    for t in tickers:
        if t in cached_results:
            d = cached_results[t].copy()
        elif t in api_results:
            d = api_results[t].copy()
        else:
            d = _empty_result()
        d["Ticker"] = t
        rows.append(d)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("Ticker")
    _save_cache("financial_all", date, df)

    if progress_callback:
        progress_callback(f"DART 재무제표: {total}/{total} 완료!")
    return df


def _partial_save(tickers, cached_results, api_results, date):
    """중간 저장 (진행 중 결과)."""
    rows = []
    for t in tickers:
        if t in cached_results:
            d = cached_results[t].copy()
        elif t in api_results:
            d = api_results[t].copy()
        else:
            continue
        d["Ticker"] = t
        rows.append(d)
    if rows:
        _save_cache("financial_partial", date,
                    pd.DataFrame(rows).set_index("Ticker"))
