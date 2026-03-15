"""급등후보주 발굴 프로그램 — Streamlit 메인 진입점.

실행: streamlit run low_pbr_screener/app.py
"""

import streamlit as st
from core.config import ensure_dirs

st.set_page_config(
    page_title="급등후보주 발굴",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_dirs()

# ══════════════════════════════════════════════════════
# 모바일 반응형 CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
/* 모바일 반응형 기본 설정 */
@media (max-width: 768px) {
    /* 사이드바 폭 조절 */
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 250px !important;
    }

    /* 메인 콘텐츠 패딩 줄이기 */
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }

    /* 메트릭 카드 크기 조정 */
    [data-testid="stMetric"] {
        padding: 0.5rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }

    /* 컬럼 간격 줄이기 */
    [data-testid="column"] {
        padding: 0.25rem !important;
    }

    /* 테이블 스크롤 가능하게 */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }

    /* 버튼 크기 조정 */
    .stButton > button {
        width: 100% !important;
        padding: 0.5rem !important;
        font-size: 0.9rem !important;
    }

    /* 제목 크기 조정 */
    h1 {
        font-size: 1.5rem !important;
    }

    h2 {
        font-size: 1.3rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
    }
}

/* 태블릿 반응형 */
@media (min-width: 769px) and (max-width: 1024px) {
    .main .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
}

/* 공통 스타일 개선 */
/* 카드 스타일 */
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 1rem;
    color: white;
    margin-bottom: 1rem;
}

/* 성공/경고/에러 메시지 스타일 */
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 8px !important;
}

/* 프로그레스 바 스타일 */
.stProgress > div > div {
    border-radius: 10px !important;
}

/* 익스팬더 스타일 */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* 데이터프레임 헤더 스타일 */
[data-testid="stDataFrame"] th {
    background-color: #f0f2f6 !important;
    font-weight: 600 !important;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1rem !important;
}

/* 슬라이더 스타일 */
.stSlider > div > div {
    padding-top: 0.5rem !important;
}

/* 셀렉트박스 스타일 */
.stSelectbox > div > div {
    border-radius: 8px !important;
}

/* 인풋 필드 스타일 */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 8px !important;
}

/* 체크박스 레이블 */
.stCheckbox > label {
    font-weight: 500 !important;
}

/* 라디오 버튼 레이블 */
.stRadio > label {
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📈 급등후보주 스크리너")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **사용 순서**
    1. 📊 데이터 수집
    2. 🎯 스크리닝
    3. 📋 종목 상세
    """
)

# 모바일 안내
st.sidebar.markdown("---")
st.sidebar.caption("📱 모바일에서도 사용 가능합니다")

st.title("📈 급등후보주 발굴 프로그램")
st.markdown("---")

# 프로그램 개요를 모바일 친화적으로 재구성
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### 프로그램 개요

        코스피/코스닥 전종목을 대상으로 **8개 전략 카테고리**별 급등 후보주를 발굴합니다.

        **스크리닝 전략**
        - 💎 자산괴리 · 💰 고배당 · 📈 턴어라운드 · 🌍 쌍끌이
        - ⚡ 바닥반전 · 🔥 초저PBR · 📊 수급전환 · 🛡️ 종합밸런스
        """
    )

with col2:
    st.info("👈 **왼쪽 사이드바**에서\n페이지를 선택하세요")

st.markdown("---")

# 단계별 안내 (모바일에서 테이블 대신 카드 형식)
st.subheader("📌 사용 단계")

step_cols = st.columns(3)

with step_cols[0]:
    st.markdown("""
    **1단계**
    #### 📊 데이터 수집
    - 기준일 설정
    - 전종목 데이터 수집
    - DART 재무제표 (선택)
    """)

with step_cols[1]:
    st.markdown("""
    **2단계**
    #### 🎯 스크리닝
    - 조건 설정 + 실행
    - 통과 종목 확인
    - Excel 다운로드
    """)

with step_cols[2]:
    st.markdown("""
    **3단계**
    #### 📋 종목 상세
    - 썸네일 차트 그리드
    - 찐 신호 확인
    - 종목별 상세 분석
    """)

st.markdown("---")

# 주의사항
with st.expander("⚠️ 주의사항 및 면책조항"):
    st.warning("""
    **투자 주의사항**
    - 본 프로그램은 투자 참고용이며, 투자 손실에 대한 책임은 사용자에게 있습니다.
    - 스크리닝 결과는 매수 추천이 아닙니다. 반드시 개별 분석 후 투자하세요.
    - 거래량 급감, 관리종목 지정, 상장폐지 위험 등을 반드시 확인하세요.
    - DART 재무데이터가 없는 종목은 자산/영업이익 Gate가 스킵됩니다.
    """)

# 버전 정보
st.sidebar.markdown("---")
st.sidebar.caption("v2.0.0 | 2026-02-07")
