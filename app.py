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
# 1. 아늑하고 따뜻한 톤의 커스텀 CSS (UX/컬러 리뉴얼)
# =========================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* 전체 배경 - 눈이 편안하고 아늑한 웜 샌드 베이지 */
        .stApp {
            background-color: #F7F4EF;
        }

        /* 메인 타이틀 영역 */
        .main-title {
            text-align: center;
            padding: 1.5rem 0 0.8rem 0;
        }
        .main-title h1 {
            color: #3A4D39; /* 차분한 깊은 포레스트 그린 */
            font-weight: 800;
            margin-bottom: 0.4rem;
            letter-spacing: -0.03rem;
        }
        .main-title p {
            color: #736254; /* 부드러운 우드 톤 */
            font-size: 1.0rem;
            margin-top: 0;
            font-weight: 400;
        }

        /* 안내 및 현재 상태 카드 */
        .soft-card {
            background-color: #FFFFFF;
            border: 1px solid #E6DFD5;
            border-radius: 18px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 12px rgba(115, 98, 84, 0.04);
            color: #594E44;
            line-height: 1.6;
        }

        /* AI 답변 박스 - 편안한 양장지 느낌의 크림 웜화이트 */
        .answer-box {
            background-color: #FAF8F5;
            border: 1px solid #EAE4DA;
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            line-height: 1.85;
            color: #2F2A25;
            font-size: 1.05rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
        }

        /* 버튼 스타일 - 마음을 차분하게 만들어주는 포레스트 그린 */
        div.stButton > button {
            background-color: #4A6653;
            color: #FFFFFF;
            border: none;
            border-radius: 14px;
            padding: 0.7rem 1.4rem;
            font-weight: 600;
            font-size: 1.05rem;
            width: 100%;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 2px 6px rgba(74, 102, 83, 0.15);
        }
        div.stButton > button:hover {
            background-color: #3B5242;
            color: #FFFFFF;
            box-shadow: 0 4px 10px rgba(74, 102, 83, 0.25);
        }
        
        /* 초기화 버튼 스타일 (보조 버튼 느낌으로 은은하게) */
        div.stButton > button[key="clear_btn"] {
            background-color: #E6DFD5;
            color: #594E44;
        }
        div.stButton > button[key="clear_btn"]:hover {
            background-color: #DCD4C9;
            color: #4A3F35;
        }

        /* 사이드바 - 차분하고 정돈된 린넨 베이지 */
        section[data-testid="stSidebar"] {
            background-color: #EFEBE4;
        }

        /* 안내 캡션 및 하단 문구 */
        .small-note {
            color: #8C7E74;
            font-size: 0.85rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# =========================================================
# 2. 모델 / 상담 이론 메타데이터 정의 (텍스트 비-AI화 리뉴얼)
# =========================================================
MODEL_OPTIONS = {
    "이성적이고 깊이 있는 분석 대화 (Llama)": {
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "desc": "생각을 차근차근 정리하며 복잡한 감정의 실타래를 논리적으로 짚어주는 성향입니다.",
    },
    "상황을 다각도로 넓게 바라보는 대화 (Qwen)": {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "desc": "처하신 대화의 맥락을 넓게 이해하고 여러 시선에서 유연한 실마리를 열어주는 성향입니다.",
    },
    "부드럽고 따뜻하게 감싸주는 대화 (Gemma)": {
        "id": "google/gemma-2-9b-it",
        "desc": "가까운 벗처럼 다정하고 편안한 어조로 가만가만 이야기를 건네주는 성향입니다.",
    },
}

THEORY_OPTIONS = {
    "생각의 결을 짚어보는 대화 (인지행동치료)": "CBT",
    "마음 그대로를 안아주는 대화 (인간중심치료)": "PCT",
}

THEORY_DESCRIPTIONS = {
    "CBT": "나도 모르게 스스로를 아프게 했던 고정된 생각의 틀을 살펴보고, 조금 더 마음이 편해지는 균형 잡힌 시선을 찾아갑니다.",
    "PCT": "어떤 판단도, 조언도 하지 않습니다. 온전히 내담자의 편에 서서 그 마음에 머물며 깊이 공감하고 경청합니다.",
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
# 3. 인증 토큰 처리 (기능 유지)
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
# 4. 사이드바 UI (비-AI스러운 안내 텍스트로 수정)
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ 상담실 안내")
    st.caption("편안한 대화를 위해 원하시는 방식을 선택해주세요.")

    st.markdown("#### 🌿 대화 나눌 상담사 성향")
    selected_model_label = st.selectbox(
        "어떤 어조의 상담사와 대화하고 싶으신가요?",
        options=list(MODEL_OPTIONS.keys()),
        index=2,
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]["id"]
    st.caption(f"💡 {MODEL_OPTIONS[selected_model_label]['desc']}")

    st.markdown("---")

    st.markdown("#### 📚 마음을 나누는 방식")
    selected_theory_label = st.selectbox(
        "어떤 대화 접근 방식을 원하시나요?",
        options=list(THEORY_OPTIONS.keys()),
        index=0,
    )
    selected_theory_key = THEORY_OPTIONS[selected_theory_label]
    st.caption(f"💡 {THEORY_DESCRIPTIONS[selected_theory_key]}")

    st.markdown("---")
    st.markdown(
        '<p class="small-note">🔒 이곳에서의 대화는 흔적을 남기지 않으니 안심하세요.<br>'
        "본 공간은 스스로 마음을 들여다보도록 돕는 보조 도구이며, "
        "깊은 심리적 위기 상황에는 꼭 전문 상담 기관(자살예방상담전화 1393 등)의 "
        "도움을 받으시길 권합니다.</p>",
        unsafe_allow_html=True,
    )

    # 운영자용 진단 (기능 및 레이아웃 유지하되 톤 정리)
    with st.expander("🛠️ 시스템 연결 상태 (관리자 확인용)"):
        if _TOKEN_IS_SET:
            st.success("상담실 연결 통로가 활성화되어 있습니다.")
            st.caption(
                "연결 오류가 나타날 경우, "
                "huggingface.co/settings/inference-providers 메뉴에서 "
                "Featherless AI / Novita / Together 등의 제공처가 "
                "켜져 있는지 점검해 주세요."
            )
        else:
            st.error(
                "접속 토큰(HF_TOKEN)을 찾을 수 없습니다.\n\n"
                "Streamlit Cloud 배포 시: Settings → Secrets 메뉴에 HF_TOKEN을 등록해 주세요.\n\n"
                "로컬 실행 시: .streamlit/secrets.toml 파일에 토큰을 기입한 뒤 터미널을 다시 켜 주세요."
            )


# =========================================================
# 5. 메인 화면 UI (아늑한 감성 반영)
# =========================================================
st.markdown(
    """
    <div class="main-title">
        <h1>🌿 마음챙김 AI 상담실</h1>
        <p>복잡한 가입이나 인증 없이, 지금 마음에 머무는 고민을 편안히 털어놓으세요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="soft-card">
        🌱 <b>오늘의 상담사</b> : {selected_model_label.split(' (')[0]}<br>
        📖 <b>마음 조율 방식</b> : {selected_theory_label.split(' (')[0]}
    </div>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_area(
    label="지금 어떤 마음이신가요? 하고 싶으신 이야기를 자유롭게 적어주세요. ✍️",
    placeholder=(
        "예) 요즘 회사에서 작은 실수를 했는데, 다들 속으로 저를 무능하다고 생각할 것 같아서 "
        "자꾸만 위축되고 밤에도 쉽게 잠들지 못해요..."
    ),
    height=220,
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    submit_clicked = st.button("마음 털어놓고 대화 나누기 💬", use_container_width=True)
with col_clear:
    clear_clicked = st.button("글 지우기", use_container_width=True, key="clear_btn")

if clear_clicked:
    st.rerun()


# =========================================================
# 6. 모델 호출 및 결과 출력 (비-AI스러운 텍스트로 자연스럽게 정제)
# =========================================================
def call_counseling_model(model_id: str, system_prompt: str, user_text: str):
    if not _TOKEN_IS_SET:
        raise RuntimeError(
            "HF_TOKEN_NOT_CONFIGURED: 서버에 등록된 HF_TOKEN을 찾을 수 없습니다. "
            "배포 환경(Streamlit Cloud Secrets 또는 로컬 .streamlit/secrets.toml)에 "
            "HF_TOKEN이 설정되어 있는지 확인해주세요."
        )

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
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

    if last_error is not None:
        raise RuntimeError(
            "PROVIDER_NOT_AVAILABLE: 이 모델을 서빙할 수 있는 Provider를 "
            f"현재 계정에서 찾지 못했어요. (시도한 옵션: {', '.join(candidate_providers)}) "
            "huggingface.co/settings/inference-providers 에서 Provider를 "
            f"활성화해주세요. 원본 오류: {last_error}"
        )


if submit_clicked:
    cleaned_text = (user_input or "").strip()

    if not cleaned_text:
        st.warning("⚠️ 먼저 이야기를 들려주셔야 상담사가 마음을 나누어 드릴 수 있어요.")
    else:
        system_prompt = build_system_prompt(selected_theory_key)

        st.markdown("#### 💌 상담사의 따뜻한 편지")
        answer_placeholder = st.empty()
        answer_placeholder.markdown(
            '<div class="answer-box">상담사가 남겨주신 이야기를 깊이 읽어보며 답변을 준비하고 있어요...</div>',
            unsafe_allow_html=True,
        )

        full_response = ""
        error_occurred = False

        try:
            with st.spinner("🌿 보내주신 마음에 깊이 귀 기울이는 중입니다..."):
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
                friendly = (
                    "⚠️ 대화를 시작하기 위한 연결고리(토큰)가 아직 마련되지 않았습니다. "
                    "상담실 운영자에게 설정을 확인해달라고 이야기해 주세요."
                )
            elif "PROVIDER_NOT_AVAILABLE" in str(e):
                friendly = (
                    "🛠️ 지금은 선택하신 상담사와 연결이 잠시 지연되고 있습니다. "
                    "사이드바에서 다른 경향의 상담사를 선택해 보시거나, "
                    "잠시 후 다시 문을 두드려 주세요."
                )
            else:
                friendly = "😥 예상치 못한 정전이 발생한 것처럼 연결이 끊겼습니다. 잠시 후 다시 시도해 주세요."
            answer_placeholder.markdown(
                f'<div class="answer-box">{friendly}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("자세한 오류 로그 보기 (개발자용)"):
                st.code(f"{type(e).__name__}: {e}")

        except HfHubHTTPError as e:
            error_occurred = True
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                friendly = (
                    "🔒 대화 연결 권한에 문제가 생겨 답변을 전해드리지 못했습니다. "
                    "잠시 후 다시 시도해 주시거나 다른 상담사를 선택해 보세요."
                )
            elif status == 429:
                friendly = (
                    "⏳ 지금 상담실을 찾는 분들이 많아 잠시 대기가 필요합니다. "
                    "따뜻한 차 한 모금 드시며 30초 정도 후에 다시 이야기를 건네주세요."
                )
            elif status == 503:
                friendly = (
                    "🛠️ 상담사가 대화를 나누기 위해 마음을 고르는 중(서버 준비)입니다. "
                    "약 20초 정도만 기다려 주신 뒤 다시 대화를 건네주시면 정상적으로 응답합니다."
                )
            else:
                friendly = "😥 상담실 서버 연결에 문제가 생겼습니다. 잠시 후 다시 문을 두드려 주세요."

            answer_placeholder.markdown(
                f'<div class="answer-box">{friendly}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("자세한 오류 로그 보기 (개발자용)"):
                st.code(f"{type(e).__name__}: {e}")

        except Exception as e:
            error_occurred = True
            answer_placeholder.markdown(
                '<div class="answer-box">😥 이야기를 듣는 도중 예기치 못한 문제가 일어났습니다. '
                "잠시 후 다시 이야기를 들려주시거나 다른 상담사를 선택해 주세요.</div>",
                unsafe_allow_html=True,
            )
            with st.expander("자세한 오류 로그 보기 (개발자용)"):
                st.code(traceback.format_exc())

        if not error_occurred and full_response:
            answer_placeholder.markdown(
                f'<div class="answer-box">{full_response}</div>',
                unsafe_allow_html=True,
            )
            st.success("상담사가 온 정성을 다해 편지를 마쳤습니다. 천천히 음미하며 읽어보세요. 🌱")
        elif not error_occurred and not full_response:
            answer_placeholder.markdown(
                '<div class="answer-box">😥 상담사가 고개를 끄덕였지만 미처 말을 잇지 못했습니다. '
                "이야기를 다시 건네주시거나 다른 상담사로 바꾸어 대화해 보세요.</div>",
                unsafe_allow_html=True,
            )


# =========================================================
# 7. 푸터
# =========================================================
st.markdown("---")
st.caption(
    "🌿 본 공간은 마음을 편안히 가다듬도록 돕는 AI 기반 자기이해 보조 도구이며, 전문적인 정신건강 의학 치료나 상담을 대신할 수 없습니다. "
    "마음의 무거움이 깊어 긴급한 위로와 도움이 필요하실 때는 꼭 자살예방상담전화(국내 1393) 등의 전문 기관을 찾아주세요."
)
