from flask import Flask, Blueprint, request, render_template, flash, jsonify
import pickle
import pymysql
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import shap
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'

bp = Blueprint("depression", __name__, url_prefix="/depression")

MODEL_PATH = "models/depression_model.pkl"
STATIC_DIR = "C:/project/finfit/static"
model = None
scaler = None
features = None
encoder = None

# Gemini 초기화
GEMINI_API_KEY = "AIzaSyChOw1hGe50i4gq0vmuFkM9UZxFrP127ug"
genai.configure(api_key=GEMINI_API_KEY)
genai_model = genai.GenerativeModel("gemini-1.5-pro")

# PHQ 질문 리스트
PHQ_QUESTIONS = [
    "기분이 가라앉거나, 우울하거나, 희망이 없다고 느꼈다.",
    "평소 하던 일에 대한 흥미가 없어지거나, 즐거움을 느끼지 못했다.",
    "잠들기가 어렵거나 자주 깼다 혹은 너무 많이 잤다.",
    "평소보다 식욕이 줄었다 혹은 평소보다 많이 먹었다.",
    "평소보다 말과 행동이 느려졌다 혹은 너무 안절부절 못했다.",
    "피곤하고 기운이 없었다.",
    "내가 잘못 했거나 실패했다는 생각이 들었다.",
    "일상적인 일에도 집중을 할 수 없었다.",
    "자해하거나 죽고 싶다는 생각을 했다."
]


def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        db='finfit',
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )

def load_model():
    global model, scaler, features, encoder
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        model = model_data["model"]
        scaler = model_data["scaler"]
        features = model_data["features"]
        encoder = model_data["encoder"]
        return model_data
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        return None

model_data = load_model()
if model_data is None:
    raise ValueError("모델 로드 실패! 프로그램 종료.")

def analyze_sleep_habits(sleep_hours):
    if sleep_hours is None:
        return "데이터가 없습니다."
    if sleep_hours < 5:
        return "수면 시간이 매우 부족해요.<ㅠ 하루 5시간 미만의 수면은 피로와 집중력 저하를 유발하고, 우울감을 증가시킬 수 있어요."
    elif 5 <= sleep_hours < 7:
        return "수면 시간이 부족해요. 하루 평균 7시간 미만의 수면은 우울감에 영향을 줄 수 있어요."
    elif 7 <= sleep_hours <= 9:
        return "수면 시간이 적절해요. 현재 건강한 수면 패턴을 유지하고 있어요."
    else:
        return "수면 시간이 너무 길어요. 9시간 이상의 수면은 오히려 피로감을 증가시킬 수 있어요."

def calculate_sleep_risk(sleep):
    if sleep is None or sleep == 0:
        return 0.0
    if sleep < 5:
        return 3.74
    elif sleep > 9:
        return 2.53
    else:
        return 1.0

def plot_sleep_vs_depression(user_id, sleep_hours, depression_level):
    if sleep_hours is None:
        sleep_hours = 0

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["😄우울증 없음", "😐가벼운 우울증", "🥺중간 정도 우울증", "😭심한 우울증"]
    values = [0] * 4

    if depression_level is not None and 0 <= depression_level < len(values):
        values[int(depression_level)] = sleep_hours

    bars = ax.bar(categories, values, color=["#87CEEB", "#FFD700", "#FFA500", "#FF4500"])
    ax.set_ylabel("수면 시간 (시간)")
    ax.set_title("수면 시간과 우울증 단계 비교")

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}h',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='black')

    plot_path = os.path.join(STATIC_DIR, f"sleep_vs_depression_{user_id}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_shap_bar(input_df, user_id):
    try:
        explainer = shap.Explainer(model, pd.DataFrame(columns=features))
        shap_values = explainer(input_df)
        shap_array = shap_values[0].values
        top_idx = np.argsort(np.abs(shap_array))[::-1][:5]
        labels = [features[i] for i in top_idx]
        values = [shap_array[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(labels[::-1], values[::-1], color='#4682B4')
        ax.set_title("우울증 예측에 영향을 준 상위 5개 요인")
        ax.set_xlabel("SHAP 값")
        plot_path = os.path.join(STATIC_DIR, f"shap_bar_{user_id}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"SHAP 그래프 생성 오류: {e}")
        return None

def build_chat_prompt(user_data, predicted_label):
    prompt = f"""
[우울증 예측 분석 요청]

사용자의 건강 정보를 기반으로 우울증 단계와 관련된 분석을 해줘.
현재 예측된 우울증 단계는 '{predicted_label}'야.

아래는 사용자의 건강 정보야:
"""
    for key, value in user_data.items():
        prompt += f"- {key}: {value}\n"
    prompt += """
당신은 전문가로서, 사용자의 건강 정보를 바탕으로 우울증 단계에 대한 분석과 조언을 제공해야 해
2~3문장으로 따뜻하게 만들어줘. 인사 이모지하지마 입력하지 않은 값에 대해서는 멘트하지마.
1문장 후에 줄바꿈 해줘. 우울증 심리검사 안했으면, 거기에 관한 말 하지마.
"""
    return prompt

def get_chatbot_response(prompt):
    try:
        response = genai_model.generate_content(prompt)
        return response.text.replace('\n', '<br>')
    except Exception as e:
        print("Gemini 응답 오류:", e)
        return "챗봇 응답을 불러올 수 없습니다."

def fetch_latest_user_data():
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM customer ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    print("Fetched user data:", result)  # 디버깅용 출력
    return result or {}

def preprocess_input(user_data):
    input_dict = {f: user_data.get(f, 0) for f in features}
    if 'sleep_risk' in features and 'sleep' in user_data:
        input_dict['sleep_risk'] = calculate_sleep_risk(user_data['sleep'])
    input_df = pd.DataFrame([input_dict])
    return input_df

@bp.route("/", methods=["GET", "POST"], endpoint="depression")
def predict():
    if request.method == "POST":
        sleep_hours = request.form.get("sleep", type=int, default=0)
        phq_scores = [request.form.get(f"phq-{i}", type=int, default=0) for i in range(1, 10)]
        phq_total = sum(phq_scores)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM customer ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()

        if result:
            user_id = result["id"]
            update_query = """
                UPDATE customer 
                SET sleep=%s, phq_1=%s, phq_2=%s, phq_3=%s, phq_4=%s, phq_5=%s,
                    phq_6=%s, phq_7=%s, phq_8=%s, phq_9=%s, phq_total=%s
                WHERE id=%s
            """
            cursor.execute(update_query, (sleep_hours, *phq_scores, phq_total, user_id))
        else:
            insert_query = """
                INSERT INTO customer (sleep, phq_1, phq_2, phq_3, phq_4, phq_5,
                                      phq_6, phq_7, phq_8, phq_9, phq_total)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (sleep_hours, *phq_scores, phq_total))
        conn.commit()
        cursor.close()
        conn.close()

    user_data = fetch_latest_user_data()
    if not user_data:
        flash("기본 건강 정보를 먼저 입력해주세요!", "warning")
        return render_template("depression.html", base_data=None)
    print("User data for base_data:", user_data)  # 디버깅용 출력

    input_df = preprocess_input(user_data)
    prediction = model.predict(scaler.transform(input_df))
    predicted_label = encoder.inverse_transform(prediction)[0]

    sleep_vs_plot = plot_sleep_vs_depression(user_data["id"], user_data.get("sleep"), prediction[0])
    shap_plot = plot_shap_bar(input_df, user_data["id"])
    chatbot_message = get_chatbot_response(build_chat_prompt(user_data, predicted_label))
    sleep_advice = analyze_sleep_habits(user_data.get("sleep"))

    return render_template("depression.html", base_data={
        "classification": predicted_label,
        "chatbot_message": chatbot_message,
        "sleep_advice": sleep_advice,
        "sleep_vs_plot": sleep_vs_plot,
        "shap_plot": shap_plot,
        "phq_questions": PHQ_QUESTIONS
    }, result_ready=True)

@bp.route("/ajax_predict", methods=["POST"], endpoint="ajax_predict")
def ajax_predict():
    try:
        sleep_hours = request.form.get("sleep", type=int, default=0)
        phq_scores = [request.form.get(f"phq-{i}", type=int, default=0) for i in range(1, 10)]
        phq_total = sum(phq_scores)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM customer ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()

        if result:
            user_id = result["id"]
            update_query = """
                UPDATE customer 
                SET sleep=%s, phq_1=%s, phq_2=%s, phq_3=%s, phq_4=%s, phq_5=%s,
                    phq_6=%s, phq_7=%s, phq_8=%s, phq_9=%s, phq_total=%s
                WHERE id=%s
            """
            cursor.execute(update_query, (sleep_hours, *phq_scores, phq_total, user_id))
        else:
            insert_query = """
                INSERT INTO customer (sleep, phq_1, phq_2, phq_3, phq_4, phq_5,
                                      phq_6, phq_7, phq_8, phq_9, phq_total)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (sleep_hours, *phq_scores, phq_total))
        conn.commit()
        cursor.close()
        conn.close()

        user_data = fetch_latest_user_data()
        if not user_data:
            return jsonify({"success": False, "error": "사용자 데이터를 찾을 수 없습니다."})

        input_df = preprocess_input(user_data)
        prediction = model.predict(scaler.transform(input_df))
        predicted_label = encoder.inverse_transform(prediction)[0]

        sleep_advice = analyze_sleep_habits(user_data.get("sleep"))
        chatbot_message = get_chatbot_response(build_chat_prompt(user_data, predicted_label))

        return jsonify({
            "success": True,
            "classification": predicted_label,
            "chatbot_message": chatbot_message,
            "sleep_advice": sleep_advice,
        })

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"success": False, "error": str(e)})