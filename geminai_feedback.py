import google.generativeai as genai
import google.api_core.exceptions

# Gemini API 키 설정
genai.configure(api_key="AIzaSyAFw_pKqnO_aibkAhm5DZ0bkz_pLGT17sI")

# 모델 선택
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ✅ 1. 내부 RAG 지식 기반 문자열
SQUAT_KNOWLEDGE = """
# 스쿼트 자세 분석 기준

1. 무릎 각도
- 90도 미만: 딥 스쿼트 (강한 자극, 무릎 부담 주의)
- 90~135도: 풀/패러럴 스쿼트 (이상적인 깊이)
- 135도 이상: 부분 스쿼트 (자극 약함)

2. 무릎 정렬
- 발끝보다 무릎이 나가면 잘못된 정렬
- 발목 위쪽 정렬이 가장 안정적

3. 엉덩이 움직임
- 엉덩이가 먼저 빠져야 ‘힙 도미넌트’
- 무릎이 먼저 나가면 무릎 부담 ↑

4. 척추 & 골반 정렬
- 귀-어깨-엉덩이-발목이 수직에 가까워야 함
- 좌우 골반 높이 차이 주의

5. 속도
- 너무 빠른 동작은 반동으로 인해 관절에 무리
- 1초 하강-정지-상승 정도가 적절

6. 팔 위치
- 팔은 어깨보다 낮게, 몸 앞쪽에서 안정적으로 위치해야 함

7. 시작 자세
- 귀-어깨-엉덩이-발목이 좌우 대칭으로 정렬되었는지 평가
"""

# ✅ 2. 공통 Gemini 호출 함수
def generate_ai_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except google.api_core.exceptions.GoogleAPIError as e:
        return f"AI 서버 오류 발생: {e}"

# ✅ 3. 핵심 요약형 피드백 요청 (자연스러운 말투)
def get_feedback_rag(summary_data):
    """
    핵심 요약 피드백을 생성하는 RAG 기반 Gemini 호출
    """

    formatted_summary = f"""
- 총 반복 횟수: {summary_data['total_reps']}회
- 세트 수: {summary_data['sets']}세트
- 평균 무릎 각도: {summary_data['avg_knee_angle']:.2f}°
- 자주 발생한 문제:
{chr(10).join(['  • ' + issue for issue in summary_data['issues']])}
"""

    prompt = f"""
당신은 회원의 자세를 점검해주는 헬스 트레이너입니다.  
아래 기준과 사용자 운동 결과를 참고하여 핵심 피드백을 **짧은 단락 형태(2~3문장)**로 작성해주세요.

 [자세 기준 요약]
{SQUAT_KNOWLEDGE}

 [회원 운동 요약]
{formatted_summary}

 작성 형식 (예시):
- 팔 위치와 시작 자세는 조금 불안정했지만, 꾸준히 시도하는 모습은 아주 좋았습니다.
- 무릎 각도가 부족한 편이라 운동 효과가 떨어질 수 있어요. 엉덩이를 더 뒤로 빼고 속도를 천천히 조절해보세요.
- 지금처럼 계속 노력하신다면 분명 완벽한 자세를 갖추게 될 거예요. 화이팅입니다! 

💬 톤: 따뜻하고 격려하는 말투 / 요점만 짧고 자연스럽게 전달 / 한국어
"""

    return generate_ai_response(prompt)

