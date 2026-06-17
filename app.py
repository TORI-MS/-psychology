import streamlit as st
from huggingface_hub import InferenceClient

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="심리학 논문 기반 AI 상담소", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    h1 { color: #2E4053; font-weight: 700; }
    .stTextArea textarea { font-size: 15px; }
    </style>
""", unsafe_allow_html=True)

# 2. 사이드바 구성 (논문 선택만 남김)
st.sidebar.title("🛠️ 상담 엔진 설정")
st.sidebar.markdown("---")
st.sidebar.subheader("📚 기반 심리학 논문 선택")
therapy_mode = st.sidebar.selectbox(
    "적용할 상담 이론을 선택하세요:",
    ["아론 벡의 인지행동치료 (CBT)", "칼 로저스의 인간중심치료 (PCT)"]
)

st.title("🧠 마음을 읽는 심리학 AI 상담소")
st.write("현대 심리학 연구 논문을 기반으로 설계된 에이전트입니다. 고민을 편안하게 적어주세요.")
st.markdown("---")

# 3. 프롬프트 정의
def get_system_prompt(mode):
    if mode == "아론 벡의 인지행동치료 (CBT)":
        return """
        너는 아론 벡(Aaron Beck)의 인지행동치료(CBT) 논문 및 임상 지침을 완벽히 마스터한 전문 심리 상담 에이전트야.
        1. 내담자의 사연에서 '인지적 왜곡(과도한 일반화, 흑백논리, 감정적 추론, 파국화)'이 있는지 예리하게 분석해줘.
        2. 1단계 [공감과 수용], 2단계 [심리학적 분석], 3단계 [소크라테스식 질문] 순서로 차분하고 따뜻한 격식체로 답변해줘.
        """
    else:
        return """
        너는 칼 로저스(Carl Rogers)의 인간중심치료 이론에 기반한 심리 상담 에이전트야.
        '무조건적 긍정적 존중'과 '공감적 경청'이 핵심 가치야. 절대 해결책이나 조언을 건네지 마.
        1단계 [감정 반영 및 경청], 2단계 [존중과 격려], 3단계 [내면 탐색 유도] 순서로 극진히 공감하며 답변해줘.
        """

# 4. 사용자 입력창
st.subheader("📝 오늘의 마음 기록")
user_input = st.text_area("상담받고 싶은 고민이나 마음 상태를 상세히 적어주세요:", placeholder="예시: 이번 팀 프로젝트 발표를 완전히 망쳤어요...", height=150)
submit_button = st.button("AI 심리 상담사에게 털어놓기 💬")

# 5. 답변 처리 (가입/로그인 없는 무료 우회 엔진)
if submit_button:
    if not user_input.strip():
        st.warning("⚠️ 고민 내용을 입력해주세요.")
    else:
        with st.spinner(f"💡 AI가 {therapy_mode} 논문을 기반으로 분석 중입니다..."):
            try:
                # 공용 무료 서버를 이용하여 구글의 고성능 오픈소스 모델을 직접 호출합니다.
                # 회원가입이나 API 키가 일절 필요 없는 프리패스 주소입니다.
                client = InferenceClient()
                
                messages = [
                    {"role": "system", "content": get_system_prompt(therapy_mode)},
                    {"role": "user", "content": user_input}
                ]
                
                # 구글의 최신 Gemma-2 모델을 사용하여 무료로 답변을 받아옵니다.
                response = client.chat.completions.create(
                    model="google/gemma-2-9b-it",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                st.markdown("---")
                st.subheader("✨ AI 심리 상담사의 마음 가이드")
                with st.chat_message("assistant", avatar="🧠"):
                    st.write(response.choices[0].message.content)
                st.success("✅ 분석 및 상담이 완료되었습니다.")
                
            except Exception as e:
                st.error(f"❌ 에러가 발생했습니다: {e}\n네트워크 연결 상태를 확인해주세요.")
