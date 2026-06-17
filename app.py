# -*- coding: utf-8 -*-
"""
무료 심리상담 AI 웹 앱 (Free AI Psychological Counseling Web App)
--------------------------------------------------------------
- 사용자는 API Key, 토큰, 회원가입 등을 절대 입력하지 않습니다.
- huggingface_hub의 InferenceClient를 사용하여 무료 서버리스 인퍼런스(Inference Providers)로
  여러 LLM(Llama-3.1-8B, Qwen2.5-72B, Gemma-2-9B)을 선택해 호출합니다.
- 인증은 앱 내부에 보관된 토큰(st.secrets 또는 환경변수)으로 '조용히' 처리되며,
  사용자 화면에는 절대 노출되지 않습니다. (토큰이 없어도 앱이 죽지 않고
  친절한 안내 메시지를 보여줍니다.)
"""

import os
import time
import traceback

import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


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
# 1. 따뜻한 톤의 커스텀 CSS
# =========================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* 전체 배경 - 따뜻한 베이지/아이보리 톤 */
        .stApp {
            background-color: #FAF6F0;
        }

        /* 메인 타이틀 영역 */
        .main-title {
            text-align: center;
            padding: 1.2rem 0 0.4rem 0;
        }
        .main-title h1 {
            color: #6B4F3B;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .main-title p {
            color: #9C8268;
            font-size: 0.95rem;
            margin-top: 0;
        }

        /* 카드 느낌의 박스 */
        .soft-card {
            background-color: #FFFDF9;
            border: 1px solid #EFE4D8;
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(160, 130, 100, 0.08);
        }

        /* AI 답변 박스 */
        .answer-box {
            background-color: #FFF8EF;
            border: 1px solid #F0DCC0;
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            line-height: 1.75;
            color: #4A3B2C;
            font-size: 1.02rem;
        }

        /* 버튼 스타일 */
        div.stButton > button {
            background-color: #C8967A;
            color: #FFFFFF;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
            font-size: 1.05rem;
            width: 100%;
            transition: background-color 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #B17E5E;
            color: #FFFFFF;
        }

        /* 사이드바 */
        section[data-testid="stSidebar"] {
            background-color: #F3EAE0;
        }

        /* 안내 캡션 */
        .small-note {
            color: #A08D7C;
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# =========================================================
# 2. 모델 / 상담 이론 메타데이터 정의
# =========================================================
MODEL_OPTIONS = {
    "Llama-3.1-8B · 이성적·논리적 분석 스타일": {
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "desc": "사실과 논리를 차근차근 짚어가며 분석적으로 설명해주는 스타일이에요.",
    },
    "Qwen-2.5-72B · 상황 파악이 빠른 다재다능 스타일": {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "desc": "맥락을 빠르게 파악하고 다양한 관점에서 유연하게 답해주는 스타일이에요.",
    },
    "Gemma-2-9B · 부드럽고 친근한 대화 스타일": {
        "id": "google/gemma-2-9b-it",
        "desc": "다정하고 친근한 말투로 부담 없이 대화하듯 답해주는 스타일이에요.",
    },
}

THEORY_OPTIONS = {
    "옵션 A · 아론 벡의 인지행동치료 (CBT)": "CBT",
    "옵션 B · 칼 로저스의 인간중심치료 (PCT)": "PCT",
}

THEORY_DESCRIPTIONS = {
    "CBT": "생각의 왜곡(인지왜곡)을 함께 짚어보고, 더 균형 잡힌 시각을 찾아가도록 돕는 접근이에요.",
    "PCT": "옳고 그름을 판단하지 않고, 있는 그대로의 감정을 존중하며 곁에서 함께 들어주는 접근이에요.",
}


def build_system_prompt(theory_key: str) -> str:
    """선택된 상담 이론에 따라 시스템 프롬프트를 생성합니다."""
    if theory_key == "CBT":
        return (
            "너는 아론 벡의 인지행동치료(CBT) 전문 심리 상담사야. "
            "내담자의 사연에서 인지적 왜곡(흑백논리, 파국화, 과잉일반화 등)을 포착하고, "
            "1단계 [공감과 수용], 2단계 [심리학적 분석], 3단계 [소크라테스식 질문] "
            "순서로 답변해줘. 각 단계는 명확한 제목과 함께 한국어로 따뜻하지만 "
            "전문적인 어조로 작성해줘. 의학적 진단이나 약물 처방은 하지 말고, "
            "필요한 경우 전문가의 직접적인 상담을 권유해줘."
        )
    else:  # PCT
        return (
            "너는 칼 로저스의 인간중심치료 전문 심리 상담사야. "
            "무조건적 존중과 공감적 경청이 핵심이야. 절대 섣부른 조언이나 "
            "해결책을 지시하지 말고, 1단계 [감정 반영 및 경청], 2단계 [존중과 격려], "
            "3단계 [내면 탐색 유도] 순서로 답변해줘. 각 단계는 명확한 제목과 함께 "
            "한국어로 부드럽고 따뜻한 어조로 작성해줘. 평가하거나 판단하는 표현은 "
            "절대 사용하지 말고, 필요한 경우 전문가의 직접적인 상담을 권유해줘."
        )


# =========================================================
# 3. 인증 토큰 처리 (사용자에게는 절대 노출되지 않음)
#    - st.secrets["HF_TOKEN"] 또는 환경변수 HF_TOKEN을 사용합니다.
#    - 토큰이 없어도 InferenceClient는 생성되며, 호출 시점에만
#      필요한 예외처리를 통해 친절한 안내 메시지를 보여줍니다.
# =========================================================
def get_hidden_token() -> str | None:
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


@st.cache_resource(show_spinner=False)
def get_client(_token: str | None) -> InferenceClient:
    """
    InferenceClient를 생성합니다.
    provider='auto' 로 두면 huggingface_hub가 해당 모델을 서빙하는
    가용 Provider(서버리스 무료 라우팅 포함)를 자동으로 선택합니다.
    토큰이 없는 경우(_token=None)에도 객체 생성 자체는 가능하며,
    실제 요청 시점에 권한 문제가 발생하면 아래 except 블록에서 안내합니다.
    """
    return InferenceClient(provider="auto", token=_token)


# =========================================================
# 4. 사이드바 UI
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ 상담 설정")
    st.caption("아래 옵션을 선택한 뒤, 메인 화면에서 고민을 적어주세요.")

    st.markdown("#### 🤖 AI 상담사 엔진 선택")
    selected_model_label = st.selectbox(
        "상담을 진행할 AI 모델을 골라주세요.",
        options=list(MODEL_OPTIONS.keys()),
        index=2,
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]["id"]
    st.caption(f"💡 {MODEL_OPTIONS[selected_model_label]['desc']}")

    st.markdown("---")

    st.markdown("#### 📚 상담 이론 선택")
    selected_theory_label = st.selectbox(
        "어떤 심리상담 이론으로 상담받고 싶으신가요?",
        options=list(THEORY_OPTIONS.keys()),
        index=0,
    )
    selected_theory_key = THEORY_OPTIONS[selected_theory_label]
    st.caption(f"💡 {THEORY_DESCRIPTIONS[selected_theory_key]}")

    st.markdown("---")
    st.markdown(
        '<p class="small-note">🔒 별도의 회원가입이나 API Key 입력이 필요 없습니다.<br>'
        "본 서비스는 전문적인 의료/심리 치료를 대체하지 않으며, "
        "긴급한 위기 상황에는 반드시 전문기관(자살예방상담전화 1393 등)에 "
        "연락해주세요.</p>",
        unsafe_allow_html=True,
    )


# =========================================================
# 5. 메인 화면 UI
# =========================================================
st.markdown(
    """
    <div class="main-title">
        <h1>🌿 마음챙김 AI 상담실</h1>
        <p>가입도, 인증도 필요 없이 — 지금 떠오르는 고민을 편하게 적어보세요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="soft-card">
        <b>현재 선택된 상담사</b> · {selected_model_label.split(' · ')[0]}<br>
        <b>현재 상담 접근법</b> · {selected_theory_label.split(' · ')[1]}
    </div>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_area(
    label="지금 어떤 마음이신가요? 편하게 적어주세요. ✍️",
    placeholder=(
        "예) 요즘 회사에서 작은 실수를 했는데, 다들 나를 무능하다고 생각할 것 같아서 "
        "잠도 잘 못 자고 계속 그 생각만 나요..."
    ),
    height=220,
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    submit_clicked = st.button("AI 심리 상담사에게 털어놓기 💬", use_container_width=True)
with col_clear:
    clear_clicked = st.button("초기화", use_container_width=True)

if clear_clicked:
    st.rerun()


# =========================================================
# 6. 모델 호출 및 결과 출력
# =========================================================
def call_counseling_model(model_id: str, system_prompt: str, user_text: str):
    """
    huggingface_hub InferenceClient의 chat.completions.create를 사용해
    상담 답변을 스트리밍 방식으로 가져오는 제너레이터.
    """
    token = get_hidden_token()
    client = get_client(token)

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

    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError, TypeError):
            delta = None
        if delta:
            yield delta


if submit_clicked:
    cleaned_text = (user_input or "").strip()

    if not cleaned_text:
        st.warning("⚠️ 먼저 고민을 적어주셔야 상담사가 답변을 드릴 수 있어요.")
    else:
        system_prompt = build_system_prompt(selected_theory_key)

        st.markdown("#### 💌 AI 상담사의 답변")
        answer_placeholder = st.empty()
        answer_placeholder.markdown(
            '<div class="answer-box">상담사가 답변을 준비하고 있어요...</div>',
            unsafe_allow_html=True,
        )

        full_response = ""
        error_occurred = False

        try:
            with st.spinner(f"🌿 '{selected_model_label.split(' · ')[0]}' 상담사가 마음을 읽고 있어요..."):
                # 스트리밍 시작 전, 첫 토큰이 오기 전까지는 spinner가 보여지고
                # 첫 토큰부터는 실시간으로 텍스트를 갱신합니다.
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

        except HfHubHTTPError as e:
            error_occurred = True
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                friendly = (
                    "🔒 현재 무료 인퍼런스 서버의 인증/권한 문제로 답변을 받아오지 못했어요. "
                    "잠시 후 다시 시도해주시거나, 다른 AI 모델로 변경해 다시 시도해보세요."
                )
            elif status == 429:
                friendly = (
                    "⏳ 지금 무료 서버에 요청이 많아 잠시 대기가 필요해요. "
                    "30초~1분 후 다시 시도해주세요."
                )
            elif status == 503:
                friendly = (
                    "🛠️ 선택하신 AI 모델이 현재 워밍업(로딩) 중이에요. "
                    "잠시(약 20~30초) 후 다시 시도해주시면 정상적으로 응답이 와요."
                )
            else:
                friendly = "😥 모델 서버 호출 중 문제가 발생했어요. 잠시 후 다시 시도해주세요."

            answer_placeholder.markdown(
                f'<div class="answer-box">{friendly}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("자세한 오류 로그 보기 (개발자용)"):
                st.code(f"{type(e).__name__}: {e}")

        except Exception as e:
            error_occurred = True
            answer_placeholder.markdown(
                '<div class="answer-box">😥 답변을 가져오는 중 예상치 못한 문제가 발생했어요. '
                "잠시 후 다시 시도하거나 다른 AI 모델을 선택해보세요.</div>",
                unsafe_allow_html=True,
            )
            with st.expander("자세한 오류 로그 보기 (개발자용)"):
                st.code(traceback.format_exc())

        if not error_occurred and full_response:
            answer_placeholder.markdown(
                f'<div class="answer-box">{full_response}</div>',
                unsafe_allow_html=True,
            )
            st.success("상담사가 답변을 마쳤어요. 천천히 읽어보세요. 🌱")
        elif not error_occurred and not full_response:
            answer_placeholder.markdown(
                '<div class="answer-box">😥 모델이 빈 답변을 반환했어요. '
                "다시 시도하거나 다른 모델로 바꿔보세요.</div>",
                unsafe_allow_html=True,
            )


# =========================================================
# 7. 푸터
# =========================================================
st.markdown("---")
st.caption(
    "🌿 본 서비스는 AI 기반 자기이해 보조 도구이며, 전문 심리상담이나 "
    "정신건강 의학적 치료를 대체하지 않습니다. 위기 상황에는 자살예방상담전화 "
    "(국내, 1393) 등 전문기관에 즉시 연락해주세요."
)
