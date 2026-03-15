"""KRX 정보데이터시스템 로그인 + pykrx 세션 주입.

2025-12-27부터 KRX data.krx.co.kr 이 회원제로 전환되어
로그인 세션 없이는 데이터 조회가 불가하다.

이 모듈은:
1. KRX에 로그인하여 인증된 세션(JSESSIONID) 확보
2. pykrx의 내부 HTTP 클라이언트를 monkey-patch하여 인증 세션 사용
3. 로그인 상태 관리 (자동 재로그인 등)

참고: GitHub pykrx#276, #278
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# ── 공유 세션 (pykrx에 주입됨) ──
_session: requests.Session | None = None
_logged_in: bool = False
_patched: bool = False

# ── KRX 로그인 URL ──
_LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
_LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
_LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _ensure_session() -> requests.Session:
    """세션 객체 생성 (lazy)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": _UA})
    return _session


def _patch_pykrx():
    """pykrx webio를 monkey-patch하여 공유 세션 사용."""
    global _patched
    if _patched:
        return

    try:
        from pykrx.website.comm import webio
    except ImportError:
        logger.warning("pykrx를 찾을 수 없어 monkey-patch를 건너뜁니다.")
        return

    session = _ensure_session()

    def _session_post_read(self, **params):
        return session.post(self.url, headers=self.headers, data=params, timeout=30)

    def _session_get_read(self, **params):
        return session.get(self.url, headers=self.headers, params=params, timeout=30)

    webio.Post.read = _session_post_read
    webio.Get.read = _session_get_read
    _patched = True
    logger.info("pykrx webio monkey-patch 완료")


def login(mbr_id: str | None = None, pw: str | None = None) -> tuple[bool, str]:
    """KRX 정보데이터시스템 로그인.

    Args:
        mbr_id: KRX 회원 ID (None이면 환경변수/config에서 읽음)
        pw: KRX 비밀번호 (None이면 환경변수/config에서 읽음)

    Returns:
        (성공여부, 메시지)
    """
    global _logged_in

    # 인증 정보 확인
    if not mbr_id:
        mbr_id = os.environ.get("KRX_MBR_ID", "")
    if not pw:
        pw = os.environ.get("KRX_PW", "")
    if not mbr_id or not pw:
        try:
            from .config import KRX_MBR_ID, KRX_PW

            if not mbr_id:
                mbr_id = KRX_MBR_ID
            if not pw:
                pw = KRX_PW
        except (ImportError, AttributeError):
            pass

    if not mbr_id or not pw:
        _logged_in = False
        return False, "KRX 로그인 ID/PW가 설정되지 않았습니다."

    session = _ensure_session()

    try:
        # Step 1: 로그인 페이지 → 초기 JSESSIONID 발급
        session.get(_LOGIN_PAGE, headers={"User-Agent": _UA}, timeout=15)

        # Step 2: 로그인 iframe 세션 초기화
        session.get(
            _LOGIN_JSP,
            headers={"User-Agent": _UA, "Referer": _LOGIN_PAGE},
            timeout=15,
        )

        # Step 3: 로그인 POST
        payload = {
            "mbrNm": "",
            "telNo": "",
            "di": "",
            "certType": "",
            "mbrId": mbr_id,
            "pw": pw,
        }
        headers = {"User-Agent": _UA, "Referer": _LOGIN_PAGE}

        resp = session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
        data = resp.json()
        error_code = data.get("_error_code", "")

        # CD011: 중복 로그인 → skipDup=Y 재시도
        if error_code == "CD011":
            payload["skipDup"] = "Y"
            resp = session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
            data = resp.json()
            error_code = data.get("_error_code", "")

        if error_code == "CD001":
            _logged_in = True
            # pykrx monkey-patch (로그인 성공 후)
            _patch_pykrx()
            logger.info("KRX 로그인 성공: %s", mbr_id)
            return True, "KRX 로그인 성공"
        else:
            _logged_in = False
            msg = data.get("_error_message", f"오류 코드: {error_code}")
            logger.warning("KRX 로그인 실패: %s — %s", error_code, msg)
            return False, f"KRX 로그인 실패: {msg}"

    except Exception as e:
        _logged_in = False
        logger.error("KRX 로그인 중 오류: %s", e)
        return False, f"KRX 로그인 오류: {e}"


def is_logged_in() -> bool:
    """현재 로그인 상태 확인."""
    return _logged_in


def ensure_login(mbr_id: str | None = None, pw: str | None = None) -> tuple[bool, str]:
    """로그인되어 있지 않으면 로그인 시도. 이미 로그인이면 스킵."""
    if _logged_in:
        return True, "이미 로그인됨"
    return login(mbr_id, pw)


def reset():
    """세션/로그인 상태 초기화."""
    global _session, _logged_in, _patched
    _session = None
    _logged_in = False
    _patched = False
