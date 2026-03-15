"""공유 CSS 스타일 모듈.

모든 페이지에서 일관된 모바일 반응형 스타일을 적용하기 위한 모듈.
"""

import streamlit as st


def apply_mobile_styles():
    """모바일 반응형 CSS를 적용."""
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

    /* 익스팬더 내부 간격 */
    .streamlit-expanderContent {
        padding: 0.5rem !important;
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

/* 하이라이트 박스 */
.highlight-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border-left: 4px solid #667eea;
}

/* 성공 하이라이트 */
.success-highlight {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left-color: #28a745;
}

/* 경고 하이라이트 */
.warning-highlight {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border-left-color: #ffc107;
}

/* 위험 하이라이트 */
.danger-highlight {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-left-color: #dc3545;
}
</style>
""", unsafe_allow_html=True)


def show_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """메트릭 카드를 표시 (모바일 친화적)."""
    delta_html = ""
    if delta:
        color = "#28a745" if delta_color == "normal" else "#dc3545"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem;">{delta}</div>'

    st.markdown(f"""
    <div style="background: #f8f9fa; border-radius: 8px; padding: 0.75rem; text-align: center;">
        <div style="font-size: 0.8rem; color: #6c757d;">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 600;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def show_status_badge(status: str, variant: str = "info"):
    """상태 배지를 표시."""
    colors = {
        "info": ("#17a2b8", "#fff"),
        "success": ("#28a745", "#fff"),
        "warning": ("#ffc107", "#000"),
        "danger": ("#dc3545", "#fff"),
    }
    bg, fg = colors.get(variant, colors["info"])

    st.markdown(f"""
    <span style="
        background-color: {bg};
        color: {fg};
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    ">{status}</span>
    """, unsafe_allow_html=True)
