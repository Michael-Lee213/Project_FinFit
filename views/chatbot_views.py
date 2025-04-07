# ✅ chatbot_views.py 전체 코드 그대로 복붙용
from flask import Blueprint, request, jsonify
import google.generativeai as genai
import os

chatbot_bp = Blueprint('chatbot', __name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyChOw1hGe50i4gq0vmuFkM9UZxFrP127ug"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")
chat_session = model.start_chat(history=[])

page_responses = {
    "main": {
        "이 페이지": "여기는 메인 페이지입니다. 주요 기능과 서비스에 대한 링크를 제공합니다. 원하는 정보를 찾을 수 있도록 도와드릴게요.",
        "시작": "FinFit의 모든 기능은 이곳에서 시작할 수 있어요.",
        "서비스": "질병 예측, 병원 추천, 자세 분석, 운동 추천, 우울증 예측 등 다양한 기능을 제공해요.",
        "도움말": "궁금한 점이 있다면 언제든지 저에게 질문해주세요.",
    },
    "customer": {
        "이 페이지": "이곳은 건강 정보 입력 페이지입니다. 건강검진 데이터를 입력하여 질병 예측에 활용할 수 있어요.",
        "입력": "성별, 나이, 키, 체중 등 정보를 입력하면 질병 예측에 도움이 돼요.",
        "건강 정보": "입력한 건강 정보는 예측 결과에 활용되며, 더 정확한 결과를 위해 빠짐없이 입력하는 것이 좋아요.",
        "안녕": "안녕하세요! 여기는 건강 정보 입력 페이지입니다. 필요한 정보를 입력하시면 질병 예측 결과를 확인하실 수 있어요.",
    },
    "disease": {
        "이 페이지": "이곳은 질병 예측 페이지입니다. 건강검진 데이터를 바탕으로 질병 위험을 분석해요.",
        "고혈압": "혈압이 높을수록 심혈관 질환 위험이 커지니 주의가 필요해요.",
        "당뇨": "당뇨는 혈당 조절이 어려운 상태예요. 꾸준한 관리가 필요해요.",
        "고지혈증": "혈중 지방이 높아져 혈관 건강에 영향을 줄 수 있어요.",
        "암": "이 페이지에선 암 위험도도 예측해 볼 수 있어요.",
        "위험도" : "예측된 위험도는 참고용으로, 정확한 진단은 전문가와 상담해야 해요.",
        "점수" : ""
    },
    "squat": {
        "이 페이지": "이곳은 스쿼트 자세 분석 페이지입니다. 자세를 분석하고 피드백을 제공해요.",
        "스쿼트": "무릎이 발끝을 넘지 않게, 허리는 곧게 유지하는 것이 중요해요.",
        "자세 분석": "포즈 추정을 통해 동작의 정확도를 판단해줘요.",
        "피드백": "동작 오류 시 음성 피드백과 함께 개선 방향을 제시해요.",
        "TTS": "TTS를 통해 음성으로 피드백을 들을 수 있어요."
    },
    "hospital": {
        "이 페이지": "이곳은 병원 추천 페이지입니다. 질병 예측 결과를 기반으로 병원을 추천해줘요.",
        "병원 추천": "당신의 건강 상태에 따라 가까운 병원을 안내해드릴게요.",
        "내과": "고혈압, 당뇨, 고지혈증 위험이 있을 경우 내과 진료를 추천해요.",
        "근처 병원": "현재 위치 기준으로 주변 병원을 찾아볼 수 있어요."
    },
    "exercise": {
        "이 페이지": "이곳은 운동 추천 페이지입니다. 건강 상태에 맞는 맞춤 운동을 알려줘요.",
        "운동 추천": "질병 예방과 건강 증진을 위한 운동을 안내해드려요.",
        "운동 프로그램": "유산소, 근력, 스트레칭 등 다양한 운동이 포함돼 있어요.",
        "맞춤 운동": "개인 건강 상태에 따라 적절한 운동 강도와 종류를 조정해요."
    },
    "depression": {
        "이 페이지": "이곳은 우울증 예측 페이지입니다. 건강 데이터를 통해 정신건강 상태를 분석해요.",
        "우울증": "최근 우울감이 자주 느껴진다면 이 페이지를 참고해보세요.",
        "정신 건강": "몸과 마음 모두 중요해요. 이곳에서 위험도를 간단히 확인할 수 있어요.",
        "예측 결과": "예측된 우울증 위험도는 개선을 위한 시작점이 될 수 있어요."
    }
}

@chatbot_bp.route('/chatbot', methods=['POST'])
def chatbot_reply():
    user_msg = request.json.get('message', '').strip()
    page = request.json.get('page', '').strip().lower()

    if not user_msg:
        return jsonify({'reply': '무엇을 도와드릴까요?'})

    page_rules = page_responses.get(page, {})
    for keyword, reply in page_rules.items():
        if keyword in user_msg:
            return jsonify({'reply': reply})

    prompt = f"""지금 사용자가 보고 있는 페이지는 '{page}'입니다.
너무 장황하게 설명하지 말고 핵심만 간단하게 3~4줄 이내로 알려줘.
질문: \"{user_msg}\"
"""

    try:
        response = chat_session.send_message(prompt)
        return jsonify({'reply': response.text.strip()})
    except Exception as e:
        print("Gemini API 오류:", str(e))
        return jsonify({'reply': '죄송해요, 아직 그 내용은 잘 모르겠어요. 다른 질문을 해볼까요?'})
