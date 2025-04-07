from flask import Blueprint, render_template, request
import pymysql
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
import matplotlib
import plotly.graph_objects as go
import io
import base64
import seaborn as sns

bp = Blueprint("exercise", __name__, url_prefix="/exercise")

matplotlib.use("Agg")
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 모델 로드 (통합 저장된 파일)
bundle = joblib.load("models/exercise_models.pkl")
try:
    rf_model = bundle['model']
    scaler = bundle['scaler']
    label_encoder = bundle['label_encoder']
    print("✅ 모델 및 전처리기 로드 완료", flush=True)
except Exception as e:
    print("❌ 모델 로드 오류:", e, flush=True)
    traceback.print_exc()

try:
    df_videos = pd.read_csv("C:/project/finfit/static/dataset/exercise_video.csv", encoding="CP949")
    print("✅ 운동 영상 데이터 로드 완료", flush=True)
except Exception as e:
    print("❌ 운동 영상 데이터 로드 실패:", e, flush=True)
    df_videos = pd.DataFrame()

# 모델이 학습한 피처 리스트
model_features = scaler.feature_names_in_.tolist()
print("📌 모델이 학습한 피처:", model_features, flush=True)

def categorize_age_group(age):
        if age <= 3:
            return 1  # 유아
        elif age <= 13:
            return 2  # 유소년
        elif age <= 19:
            return 3  # 청소년
        elif age <= 65:
            return 4  # 성인
        else:
            return 5  # 어르신
        
def estimate_body_fat(sex, age, bmi):
    if sex == 1:  # 남자
        return round(1.20 * bmi + 0.23 * age - 16.2, 1)
    else:  # 여자
        return round(1.20 * bmi + 0.23 * age - 5.4, 1)

def calculate_fat_mass(weight, body_fat_percent):
    return round(weight * (body_fat_percent / 100), 1)

def calculate_lean_mass(weight, fat_mass):
    return round(weight - fat_mass, 1)

# body_activity 예측 함수
def calculate_body_activity(weight_day, weight_hour):
    """일주일 중강도 150분 이상이면 1, 아니면 0"""
    weekly_activity_minutes = (weight_day * weight_hour * 60)
    return 1 if weekly_activity_minutes >= 150 else 0


@bp.route("/", methods=["GET", "POST"])
def exercise_page():
    from finfit import get_db_connection

    db = get_db_connection()
    cursor = db.cursor(pymysql.cursors.DictCursor)

    difficulty_mapping = {
            "마른형": ["초급"],
            "마른 비만형": ["초급", "중급"],
            "표준형": ["중급"],
            "과체중형": ["중급", "고급"],
            "비만형": ["고급"]
        }

    cursor.execute("SELECT * FROM customer ORDER BY id DESC LIMIT 1")
    user_info = cursor.fetchone()

    if not user_info:
        cursor.close()
        db.close()
        return render_template("exercise.html", body_type=None, body_type_videos=[], effect_videos=[], error="사용자 정보가 없습니다.")
    
    # 📌 사용자 입력 데이터 디버깅
    print("📌 사용자 입력 데이터:", user_info, flush=True)  
    
    # 기본 처리
    age_group_id = user_info.get("age_group_id") or categorize_age_group(user_info["age"])
    sex = user_info["sex"]
    bmi = user_info["bmi"]
    
    # 사용자 데이터를 모델이 학습한 컬럼에 맞게 변환
    user_data = {col: user_info.get(col, 0) for col in model_features}
    user_df = pd.DataFrame([user_data])
    user_df = user_df[model_features]
    print("📊 모델 입력 데이터:", user_df.to_dict(orient="records"), flush=True)

    # 데이터 스케일링 적용
    user_df_scaled = scaler.transform(user_df)

    # 체형 예측
    predicted_body_type_encoded = rf_model.predict(user_df_scaled)[0]
    predicted_body_type = label_encoder.inverse_transform([predicted_body_type_encoded])[0]
    print(f"📌 예측된 체형: {predicted_body_type}", flush=True)

    # 🔹 GET 요청에서도 기본값 설정
    weight_day = user_info.get("weight_day", 0) or 0
    weight_hour = user_info.get("weight_hour", 0) or 0

    # 예측 수행
    predicted_body_activity = calculate_body_activity(weight_day, weight_hour)

    # 📌 group_avg_bmi를 여기서 한 번만 계산
    group_df = pd.read_csv("static/dataset/final_dataset_v6.csv")
    group_avg_bmi = (
        group_df[(group_df["sex"] == sex) & (group_df["age_group_id"] == age_group_id)]["bmi"]
        .mean()
    )
    group_avg_bmi = round(group_avg_bmi, 1)

    body_type_videos = []
    effect_videos = []

    # 체형 기반 추천
    recommended_difficulty = difficulty_mapping.get(predicted_body_type, ["초급"])
    shape_filter = df_videos[df_videos["difficulty"].isin(recommended_difficulty)]
    body_type_videos = (
        shape_filter[["title1", "title2", "difficulty", "video_url"]]
        .dropna()
        .sample(min(5, len(shape_filter)), random_state=42)
        .to_dict(orient="records")
    )
    print(f"📌 체형 기반 추천 영상 개수: {len(body_type_videos)}", flush=True)

    if request.method == "POST":
        selected_effects = request.form.get("exercise_effect", "").split(",")
        selected_effects = [e for e in selected_effects if e]

        print(f"📌 선택된 운동 효과: {selected_effects}", flush=True)

        # DB 업데이트
        cursor.execute(
            "UPDATE customer SET body_activity = %s WHERE id = %s",
            (predicted_body_activity, user_info["id"])
        )
        db.commit()

        update_query = """
            UPDATE customer 
            SET weight_day = %s, weight_hour = %s, exercise_effect = %s
            WHERE id = %s
        """
        update_values = [weight_day, weight_hour, ",".join(selected_effects), user_info["id"]]
        cursor.execute(update_query, tuple(update_values))
        db.commit()

        print(f"📌 DB 업데이트 완료: {update_values}", flush=True)

        df_videos["title1"] = df_videos["title1"].fillna("")
        df_videos["title2"] = df_videos["title2"].fillna("")
        df_videos["difficulty"] = df_videos["difficulty"].fillna("")
        df_videos["video_url"] = df_videos["video_url"].fillna("")
        df_videos.columns = df_videos.columns.str.strip()

        
        # 운동 효과 기반 추천 (최대 2개 효과 기반 필터링)
        if selected_effects:
            effect_filter = df_videos["title2"].apply(lambda x: any(effect in str(x) for effect in selected_effects))
            effect_videos_df = df_videos[effect_filter]
            if len(effect_videos_df) < 5:
                effect_videos_df = df_videos.sample(5, random_state=42)
            effect_videos = (
                effect_videos_df[["title1", "title2", "difficulty", "video_url"]]
                .dropna()
                .sample(min(5, len(effect_videos_df)), random_state=42)
                .to_dict(orient="records")
            )
            print(f"📌 운동 효과 기반 추천 영상 개수: {len(effect_videos)}", flush=True)

    context = {
        "body_type": predicted_body_type,
        "sex": user_info["sex"],
        "age": user_info["age"],
        "height": user_info["height"],
        "weight": user_info["weight"],
        "bmi": user_info["bmi"],
        "group_avg_bmi": group_avg_bmi,
        "body_type_videos": body_type_videos,
        "effect_videos": effect_videos
    }

    cursor.close()
    db.close()
    return render_template("exercise.html", **context)




