# -*- coding: utf-8 -*-
"""
무료 심리상담 AI 웹 앱 (Free AI Psychological Counseling Web App)
--------------------------------------------------------------
- 사용자는 API Key, 토큰, 회원가입 등을 절대 입력하지 않습니다.
- huggingface_hub의 InferenceClient를 사용하여 무료 서버리스 인퍼런스(Inference Providers)로
  여러 LLM(Llama-3.1-8B, Qwen2.5-72B, Gemma-2-9B)을 선택해 호출합니다.
- 인증은 앱 내부에 보관된 토큰(st.secrets 또는 환경변수)으로 '조용히' 처리되며,
  사용자 화면에는 절대 노출되지 않습니다.
"""

import os
import time
import traceback

import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, BadRequestError


# =========================================================
# 0. 페이지 기본 설정 (가장 먼저 호출되어야 함)
# =========================================================
st.set_page_config(
    page_title="마음챙김 AI 상담실 🌿",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =========================================================
# 1. 아늑하고 차분한 커스텀 CSS (세이지 그린 & 밀크티 톤)
# =========================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* 전체 배경 - 눈이 편안한 따뜻한 밀크티/웜그레이 톤 */
        .stApp {
            background-color: #F7F4F0;
        }

        /* 메인 타이틀 영역 */
        .main-title {
            text-align: center;
            padding: 1.5rem 0 0.8rem 0;
        }
        .main-title h1 {
            color: #4A3E3D;
            font-weight: 700;
            margin-bottom: 0.4rem;
            font-size: 2.2rem;
            letter-spacing: -0.03em;
        }
        .main-title p {
            color: #8C7B75;
            font-size: 1.05rem;
            margin-top: 0;
        }

        /* 카드 느낌의 박스 (현재 설정 상태 등) */
        .soft-card {
            background-color: #FFFFFF;
            border: 1px solid #EBE4DC;
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(140, 123, 117, 0.04);
            color: #5C4E4B;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        /* AI 답변 박스 - 편안한 초록빛이 감도는 세이지 틴트 톤 */
        .answer-box {
            background-color: #F2F5F3;
            border: 1px solid #E1E6E2;
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            line-height: 1.8;
            color: #2F3E36;
            font-size: 1.05rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
        }

        /* 버튼 스타일 - 마음의 안정을 주는 다운된 세이지 그린 */
        div.stButton > button {
            background-color: #72847A;
            color: #FFFFFF;
            border: none;
            border-radius: 14px;
            padding: 0.7rem 1.2rem;
            font-weight: 600;
            font-size: 1.05rem;
            width: 100%;
            transition: all 0.25s ease-in-out;
            box-shadow: 0 2px 6px rgba(114, 132, 122, 0.15);
        }
        div.stButton > button:hover {
            background-color: #5F7066;
            color: #FFFFFF;
            box-shadow: 0 4px 10px rgba(114, 132, 122, 0.25);
        }

        /* 초기화 버튼 (secondary 형태) */
        div.stButton > button[data-testid="baseButton-secondary"] {
            background-color: transparent;
            color: #8C7B75;
            border: 1px solid #DCD1C4;
        }
        div.stButton > button[data-testid="baseButton-secondary"]:hover {
            background-color: #EBE4DC;
            color: #5C4E4B;
        }

        /* 사이드바 */
        section[data-testid="stSidebar"] {
            background-color: #EDE6DE;
        }

        /* 안내 캡션 */
        .small-note {
            color: #9C8E87;
            font-size: 0.85rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# =========================================================
# 2. 모델 / 상담 이론 메타데이터 정의 (UX 텍스트 비AI화)
# =========================================================
MODEL_OPTIONS = {
    "깊고 명쾌한 대화 · 이성적 분석가": {
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "desc": "복잡한 생각의 타래를 차근차근 짚어가며, 현명한 관점을 찾도록 이성적으로 돕습니다.",
    },
    "넓고 유연한 대화 · 공감형 조언자": {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "desc": "마음의 전체적인 맥락을 영리하게 파악하고, 여러 갈래의 따뜻한 시선을 열어줍니다.",
    },
    "다정하고 포근한 대화 · 이야기 친구": {
        "id": "google/gemma-2-9b-it",
        "desc": "가까운 친구처럼 부드러운 말투로 곁에 머물며, 부담 없이 이야기를 들어줍니다.",
    },
}

THEORY_OPTIONS = {
    "생각의 틀을 함께 짚어보는 대화 (인지행동치료)": "CBT",
    "있는 그대로 마음을 비춰주는 대화 (인간중심치료)": "PCT",
}

THEORY_DESCRIPTIONS = {
    "CBT": "나도 모르게 스스로를 괴롭히던 생각의 오해나 치우침을 살펴보고, 마음의 균형을 찾도록 이끕니다.",
    "PCT": "어떤 판단도 하지 않고, 당신이 느끼는 서러움이나 아픔을 온전히 존중하며 곁을 지킵니다.",
}


def build_system_prompt(theory_key: str) -> str:
    """선택된 상담 이론에 따라 시스템 프롬프트를 생성합니다."""
    if theory_key == "CBT":
        return (
            "너는 따뜻하고 깊이 있는 심리 상담사야. 아론 벡의 인지행동치료(CBT) 관점에서 대화해줘. "
            "내담자의 사연에서 스스로를 힘들게 만드는 생각의 함정(흑백논리, 파국화, 과잉일반화 등)을 세심히 살피되, "
            "기계적인 진단처럼 말하지 말고 대화하듯 풀어줘. "
            "1단계 [마음 알아채기 및 공감], 2단계 [생각의 타래 짚어보기], 3단계 [작은 변화를 위한 조심스러운 제안] "
            "순서로 답변해줘. 각 단계는 명확한 구절로 나누어 한국어로 다정하면서도 신뢰감 있게 작성해줘. "
            "의학적 진단은 피하고, 깊은 치유가 필요할 땐 조율된 어조로 전문가 상담을 권해줘."
        )
    else:  # PCT
        return (
            "너는 칼 로저스의 인간중심치료를 바탕으로 소통하는 따뜻한 상담사야. "
            "무조건적인 존중과 깊은 경청이 핵심이야. 섣부른 해결책이나 충고는 내려놓고, 오직 내담자의 마음에만 집중해줘. "
            "1단계 [온전한 경청과 마음 비추기], 2단계 [따스한 격려와 지지], 3단계 [내면의 힘 깨우기] "
            "순서로 답변해줘. 각 단계는 부드럽고 다정한 한국어로 작성하고, 결코 내담자를 평가하거나 판단하지 말아줘. "
            "더 깊은 위로와 치유가 필요할 땐 전문가의 손길을 조심스럽게 권유해줘."
        )


# =========================================================
# 3. 인증 토큰 처리
# =========================================================
def get_hidden_token() -> str | None:
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


_HIDDEN_TOKEN = get_hidden_token()
_TOKEN_IS_SET = bool(_HIDDEN_TOKEN)

PROVIDER_FALLBACKS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ["featherless-ai", "together", "novita", "fireworks-ai"],
    "Qwen/Qwen2.5-72B-Instruct": ["novita", "together", "fireworks-ai", "featherless-ai", "nebius"],
    "google/gemma-2-9b-it": ["featherless-ai", "nebius", "together"],
}


@st.cache_resource(show_spinner=False)
def get_client(_token: str | None, provider: str = "auto") -> InferenceClient:
    return InferenceClient(provider=provider, token=_token)


# =========================================================
# 4. 사이드바 UI (UX 텍스트 전면 개편)
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ 상담실 문 열기")
    st.caption("편안한 대화를 위해 마음에 맞는 방식을 골라주세요.")

    st.markdown("#### ✉️ 어떤 결의 상담사와 이야기하고 싶나요?")
    selected_model_label = st.selectbox(
        "목소리 톤 고르기",
        options=list(MODEL_OPTIONS.keys()),
        index=2,
        label_visibility="collapsed"
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]["id"]
    st.caption(f"✨ {MODEL_OPTIONS[selected_model_label]['desc']}")

    st.markdown("---")

    st.markdown("#### 📚 어떤 대화 방식으로 마음을 풀고 싶나요?")
    selected_theory_label = st.selectbox(
        "상담 접근방식 고르기",
        options=list(THEORY_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    selected_theory_key = THEORY_OPTIONS[selected_theory_label]
    st.caption(f"✨ {THEORY_DESCRIPTIONS[selected_theory_key]}")

    st.markdown("---")
    st.markdown(
        '<p class="small-note">🔒 이곳에서의 대화는 어디에도 남지 않으니 안심하세요.<br>'
        "본 공간은 마음을 돌보는 보조 도구입니다. 극심한 마음의 고통이나 "
        "위기 상황에는 꼭 전문기관(자살예방상담전화 1393 등)의 따뜻한 도움을 받아보시기를 권합니다.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("🛠️ 시스템 연결 확인 (운영자 확인용)"):
        if _TOKEN_IS_SET:
            st.success("상담실 연결 통로가 정상적으로 열려 있습니다.")
        else:
            st.error(
                "연결 토큰(HF_TOKEN)을 찾지 못했습니다.\n\n"
                "Streamlit Cloud 배포 시: Settings -> Secrets에 HF_TOKEN을 등록해주세요.\n"
                "로컬 실행 시: .streamlit/secrets.toml 파일에 토큰을 기입해주세요."
            )


# =========================================================
# 5. 메인 화면 UI
# =========================================================
st.markdown(
    """
    <div class="main-title">
        <h1>🌿 마음챙김 AI 상담실</h1>
        <p>복잡한 가입이나 절차 없이 — 오롯이 당신만을 위한 대화 공간입니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 간결하고 정돈된 카드 형태의 요약문
st.markdown(
    f"""
    <div class="soft-card">
        • <b>상담사 성향</b> : {selected_model_label.split(' · ')[0]}<br>
        • <b>대화의 흐름</b> : {selected_theory_label.split(' (')[0]}
    </div>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_area(
    label="지금 마음 한구석을 무겁게 만드는 고민이 있다면, 편하게 털어놓아 보세요. ✍️",
    placeholder=(
        "예) 요즘 작은 실수에도 온종일 마음이 쓰여요. 주변 사람들이 나를 안 좋게 "
        "평가할 것만 같아 불안하고, 밤이 되면 자꾸 서글퍼집니다..."
    ),
    height=220,
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    submit_clicked = st.button("내 마음 전하기 💬", use_container_width=True)
with col_clear:
    clear_clicked = st.button("처음부터 다시 쓰기", use_container_width=True)

if clear_clicked:
    st.rerun()


# =========================================================
# 6. 모델 호출 및 결과 출력 (UX 텍스트 비AI화)
# =========================================================
def call_counseling_model(model_id: str, system_prompt: str, user_text: str):
    if not _TOKEN_IS_SET:
        raise RuntimeError("HF_TOKEN_NOT_CONFIGURED")

    candidate_providers = ["auto"] + PROVIDER_FALLBACKS.get(model_id, [])
    last_error: Exception | None = None

    for provider in candidate_providers:
        try:
            client = get_client(_HIDDEN_TOKEN, provider)
            stream = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=900,
                temperature=0.7,
                top_p=0.9,
                stream=True,
            )

            got_any_chunk = False
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError, TypeError):
                    delta = None
                if delta:
                    got_any_chunk = True
                    yield delta

            if got_any_chunk:
                return

        except BadRequestError as e:
            last_error = e
            continue
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                raise
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        raise RuntimeError("PROVIDER_NOT_AVAILABLE")


if submit_clicked:
    cleaned_text = (user_input or "").strip()

    if not cleaned_text:
        st.warning("⚠️ 한두 구절이라도 마음을 적어주셔야 조심스레 답변을 건넬 수 있어요.")
    else:
        system_prompt = build_system_prompt(selected_theory_key)

        st.markdown("#### 💌 당신을 향한 따뜻한 답신")
        answer_placeholder = st.empty()
        answer_placeholder.markdown(
        '<div class="answer-box">이야기를 조용히 읽으며 생각을 정리하고 있어요...</div>',
        unsafe_allow_html=True,
    )

        full_response = ""
        error_occurred = False

        try:
        # 스피너 메시지도 다정하게 변경
        with st.spinner("마음을 가만히 들여다보는 중입니다..."):
            first_chunk_received = False
            for delta in call_counseling_model(
                model_id=selected_model_id,
                system_prompt=system_prompt,
                user_text=cleaned_text,
            ):
                full_response += delta
                first_chunk_received = True
                answer_placeholder.markdown(
                    f'<div class="answer-box">{full_response}▌</div>',
                    unsafe_allow_html=True,
                )

            if not first_chunk_received:
                error_occurred = True

        except RuntimeError as e:
        error_occurred = True
        if "HF_TOKEN_NOT_CONFIGURED" in str(e):
            friendly = "⚠️ 마음을 전달하는 연결 통로에 설정이 누락되었습니다. 상담실 관리자에게 확인을 부탁해주세요."
        elif "PROVIDER_NOT_AVAILABLE" in str(e):
            friendly = "🛠️ 선택하신 대화 통로에 잠시 정비가 필요합니다. 왼쪽 메뉴에서 다른 상담사를 선택해 이야기를 다시 건네어 보세요."
        else:
            friendly = "😥 통신에 작은 문제가 생겨 마음이 온전히 닿지 못했습니다. 잠시 후 편안할 때 다시 한 번 적어주세요."
        
        answer_placeholder.markdown(f'<div class="answer-box">{friendly}</div>', unsafe_allow_html=True)

    except HfHubHTTPError as e:
        error_occurred = True
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            friendly = "🔒 대화 공간의 권한에 작은 엉킴이 생겼습니다. 잠시 후 다시 시도해보거나 다른 상담사 스타일을 골라보세요."
        elif status == 429:
            friendly = "⏳ 지금 이 공간을 찾는 분들이 많아 상담실이 조금 붐비고 있습니다. 약 1분 정도 숨을 고른 뒤 다시 이야기를 건네어 주세요."
        elif status == 503:
            friendly = "🛠️ 상담사가 대화를 맞이할 준비(서버 로딩)를 하고 있습니다. 20~30초만 가만히 기다려 주신 뒤 다시 대화를 걸어주세요."
        else:
            friendly = "😥 이야기 통로에 예상치 못한 안개가 꼈습니다. 잠시 후 맑아지면 다시 찾아와주세요."

        answer_placeholder.markdown(f'<div class="answer-box">{friendly}</div>', unsafe_allow_html=True)

    except Exception as e:
        error_occurred = True
        answer_placeholder.markdown(
            '<div class="answer-box">😥 예기치 못한 작은 오류로 대화가 끊겼습니다. 마음에 안정을 취한 뒤 다른 상담사를 선택해 말을 건네어 보세요.</div>',
            unsafe_allow_html=True,
        )

    if not error_occurred and full_response:
        answer_placeholder.markdown(
            f'<div class="answer-box">{full_response}</div>',
            unsafe_allow_html=True,
        )
        st.success("상담사의 진심 어린 답신이 도착했습니다. 천천히 가슴으로 읽어보세요. 🌱")
    elif not error_occurred and not full_response:
        answer_placeholder.markdown(
            '<div class="answer-box">😥 깊은 고민 끝에 상담사가 말문을 열지 못했습니다. 다른 성향의 상담사를 선택해 한 번 더 말을 걸어주세요.</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# 7. 푸터
# =========================================================
st.markdown("---")
st.caption(
    "🌿 본 대화실은 AI 기반 자기이해 보조 도구이며, 전문 심리상담이나 "
    "정신건강 의학적 치료를 대체하지 못합니다. 마음의 짐이 너무 무거워 위태로울 때는 "
    "자살예방상담전화 (국내, 1393) 등 따뜻한 전문가의 손을 꼭 잡아주세요."
)
