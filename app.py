import streamlit as st
import openai

# 1. 웹 페이지 기본 설정 및 디자인
st.set_page_config(
    page_title="심리학 논문 기반 AI 상담소",
    page_icon="🧠",
    layout="centered"
)

# 커스텀 스타일 입히기 (폰트 및 부드러운 배경 감성)
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    h1 { color: #2E4053; font-weight: 700; }
    .stTextArea textarea { font-size: 15px; }
    </style>
""", unsafe_allow_html=True)

# 2. 사이드바 구성 (심리학 논문/이론 선택 및 API 설정)
st.sidebar.title("🛠️ 상담 엔진 설정")
st.sidebar.markdown("---")

# OpenAI API 키 입력창 (보안을 위해 패스워드 형식으로 입력받음)
# 발표 시에는 여기에 본인의 키를 하드코딩해두면 시연이 매끄럽습니다.
# 예: openai_api_key = "sk-..."
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="OpenAI에서 발급받은 API Key를 입력하세요.")

st.sidebar.markdown("---")
st.sidebar.subheader("📚 기반 심리학 논문 선택")
therapy_mode = st.sidebar.selectbox(
    "적용할 상담 이론을 선택하세요:",
    [
        "아론 벡의 인지행동치료 (CBT)",
        "칼 로저스의 인간중심치료 (PCT)"
    ]
)

# 3. 메인 화면 타이틀 및 소개
st.title("🧠 마음을 읽는 심리학 AI 상담소")
st.write("현대 심리학 연구 논문을 기반으로 설계된 에이전트입니다. 고민이나 상담받고 싶은 내용을 편안하게 적어주세요.")
st.markdown("---")

# 4. 심리학 논문 기반 하드코딩 프롬프트 정의 (학습/지침 주입)
# RAG나 파인튜닝 없이도 논문 기반의 높은 정확도와 일관성을 보장하는 핵심 영역입니다.
def get_system_prompt(mode):
    if mode == "아론 벡의 인지행동치료 (CBT)":
        return """
        [역할 정의]
        너는 아론 벡(Aaron Beck)의 인지행동치료(Cognitive Behavioral Therapy, CBT) 논문 및 임상 지침을 완벽히 마스터한 전문 심리 상담 에이전트야.
        
        [치료적 접근 지침]
        1. 내담자의 사연에서 '인지적 왜곡(Cognitive Distortions)'을 예리하게 포착해야 해. 특히 '과도한 일반화', '흑백논리', '감정적 추론', '파국화'가 있는지 분석해줘.
        2. 내담자의 감정을 비난하지 않고 깊이 공감하되, 부정적 생각의 고리를 끊어낼 수 있도록 도와야 해.
        
        [답변 출력 규칙]
        - 1단계 [공감과 수용]: 내담자가 느꼈을 고통과 감정을 충분히 알아주고 정서적으로 지지해줄 것. (예: "그런 상황이었다면 정말 막막하고 힘드셨을 것 같아요.")
        - 2단계 [심리학적 분석]: 내담자가 무의식적으로 빠져 있는 '인지적 왜곡'이 무엇인지 부드러운 어조로 짚어줄 것. (예: "이야기를 들어보니, 이번 한 번의 사건으로 '앞으로 평생 안 될 것 같다'고 생각하시는 '과도한 일반화'의 함정에 빠지신 것일 수 있어요.")
        - 3단계 [소크라테스식 질문]: 내담자가 자신의 생각을 객관적으로 되돌아볼 수 있도록 유도하는 열린 질문을 하나 던질 것.
        
        [말투]
        따뜻하고 차분하며, 전문성이 느껴지는 존댓말을 사용할 것. 가벼운 위로나 뻔한 조언은 절대 금지.
        """
    elif mode == "칼 로저스의 인간중심치료 (PCT)":
        return """
        [역할 정의]
        너는 칼 로저스(Carl Rogers)의 인간중심치료(Person-Centered Therapy) 이론에 기반한 심리 상담 에이전트야.
        
        [치료적 접근 지침]
        1. '무조건적 긍정적 존중(Unconditional Positive Regard)'과 '공감적 경청'이 핵심 가치야. 내담자의 어떤 행동이나 감정도 판단하거나 재단(평가)하지 마.
        2. 해결책을 제시하거나 조언을 건네는 것은 인간중심치료 논문에 위배돼. 절대 "이렇게 하세요"라고 지시하지 마. 내담자 스스로 답을 찾을 힘이 있다고 믿어야 해.
        
        [답변 출력 규칙]
        - 1단계 [감정 반영 및 경청]: 내담자의 말 속에 담긴 감정의 알맹이를 그대로 비추어 주는 거울 역할을 해줄 것. (예: "~라는 감정이 드셨고, 그로 인해 마음이 많이 무거우셨군요.")
        - 2단계 [존중과 격려]: 내담자의 존재 자체를 긍정해주고, 혼자가 아님을 느끼게 해줄 것.
        - 3단계 [내면 탐색 유도]: 자신의 내면을 더 깊이 들여다볼 수 있도록 편안하게 유도할 것.
        
        [말투]
        한없이 따뜻하고, 포용력 있으며, 서두르지 않는 어조를 유지할 것.
        """

# 5. 사용자 입력창 구성
st.subheader("📝 오늘의 마음 기록")
user_input = st.text_area(
    "상담받고 싶은 고민이나 마음 상태를 상세히 적어주세요:",
    placeholder="예시: 이번 팀 프로젝트 발표를 완전히 망쳤어요. 동기들이 다 저를 한심하게 볼 것 같고, 앞으로 어떤 발표도 잘해낼 자신이 없어요...",
    height=150
)

# 상담 시작 버튼
submit_button = st.button("AI 심리 상담사에게 털어놓기 💬")

# 6. 답변 처리 및 OpenAI API 연동 영역
if submit_button:
    if not openai_api_key:
        st.error("⚠️ 시연 및 테스트를 위해 사이드바에 OpenAI API Key를 먼저 입력해주세요!")
    elif not user_input.strip():
        st.warning("⚠️ 고민 내용을 입력해주세요.")
    else:
        # 로딩 애니메이션 실행 (발표 시 긴장감과 몰입감을 주는 요소)
        with st.spinner(f"💡 AI가 {therapy_mode} 논문을 기반으로 사용자의 심리 상태를 분석 및 상담 중입니다..."):
            try:
                # OpenAI API 클라이언트 설정
                client = openai.OpenAI(api_key=openai_api_key)
                
                # 선택된 모드에 따른 프롬프트 주입 (핵심 학습 인젝션)
                system_instruction = get_system_prompt(therapy_mode)
                
                # GPT-4o 호출 (정확도와 문맥 이해도가 가장 높음)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7 # 너무 창의적이지도, 너무 딱딱하지도 않은 상담에 최적화된 수치
                )
                
                ai_answer = response.choices[0].message.content
                
                # 결과 출력 UI
                st.markdown("---")
                st.subheader("✨ AI 심리 상담사의 마음 가이드")
                
                # 상담소 느낌의 대화창 스타일로 출력
                with st.chat_message("assistant", avatar="🧠"):
                    st.write(ai_answer)
                    
                st.success("✅ 분석 및 상담이 완료되었습니다. 마음이 조금 편안해지셨기를 바랍니다.")
                
            except Exception as e:
                st.error(f"❌ 에러가 발생했습니다: {e}\nAPI 키가 올바른지 혹은 네트워크 상태를 확인해주세요.")
