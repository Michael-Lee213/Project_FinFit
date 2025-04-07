import os 
from flask import Blueprint, render_template, current_app, flash
import pymysql
import pickle
import numpy as np

bp = Blueprint("hospital", __name__, url_prefix="/hospital")

# 현재 스크립트의 루트 디렉터리 가져오기
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # C:\project\finfit

# ✅ 1️⃣ DB 연결 함수
def get_db_connection():
    return pymysql.connect(
        host=current_app.config.get("DB_HOST", "localhost"),
        user=current_app.config.get("DB_USER", "root"),
        password=current_app.config.get("DB_PASSWORD", "1234"),
        database=current_app.config.get("DB_NAME", "finfit"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

# ✅ hospital_models.pkl 불러오기
with open(os.path.join(BASE_DIR, "models", "hospital_models.pkl"), "rb") as f:
    models = pickle.load(f)

# 모델을 딕셔너리에서 가져오기
diabetes_model = models["diabetes"]
dyslipidemia_model = models["dyslipidemia"]
high_blood_pressure_model = models["high_blood_pressure"]

# ✅ 2️⃣ 최신 사용자 데이터 가져오기
def get_latest_customer_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM customer ORDER BY id DESC LIMIT 1;"  # 최근 입력 데이터 1개 가져오기
        cursor.execute(query)
        result = cursor.fetchone()
    except Exception as e:
        print(f"❌ 사용자 데이터 조회 오류: {e}")
        result = None
    finally:
        cursor.close()
        conn.close()
    return result

# ✅ 3️⃣ 병원 추천 기능 포함 - 질병 예측
def predict_hospital(data):
    if not data:
        return None  # 데이터가 없으면 예측할 수 없음

    # ✅ 모델이 필요로 하는 feature 목록 가져오기
    expected_features = models["required_features"] + models["optional_features"]

    # ✅ 누락된 feature를 자동으로 평균값으로 채우기
    for feature in expected_features:
        if feature not in data or data[feature] is None:
            data[feature] = np.mean([v for v in data.values() if isinstance(v, (int, float))])  # 평균값 대체

    # ✅ feature 순서 맞추기
    input_data = np.array([[data[feature] for feature in expected_features]])

 # ✅ 예측 수행 (proba + 결과)
    prob_diabetes = diabetes_model.predict_proba(input_data)[0][1]
    prob_dyslipidemia = dyslipidemia_model.predict_proba(input_data)[0][1]
    prob_high_blood_pressure = high_blood_pressure_model.predict_proba(input_data)[0][1]

    diabetes_pred = diabetes_model.predict(input_data)[0]
    dyslipidemia_pred = dyslipidemia_model.predict(input_data)[0]
    high_blood_pressure_pred = high_blood_pressure_model.predict(input_data)[0]

    return {
        "diabetes": "당뇨" if diabetes_pred == 1 else "정상",
        "dyslipidemia": "고지혈증" if dyslipidemia_pred == 1 else "정상",
        "high_blood_pressure": "고협압" if high_blood_pressure_pred == 1 else "정상",
    }, round(prob_diabetes * 100), round(prob_dyslipidemia * 100), round(prob_high_blood_pressure * 100)


# ✅ 4️⃣ 사용자의 city, town 가져오기
def get_latest_user_city_town():
    customer_data = get_latest_customer_data()
    if customer_data:
        return customer_data["city"], customer_data["town"]
    return None, None

# ✅ 5️⃣ 특정 지역에서 "내과"가 포함된 병원 5곳 가져오기
def get_hospital_data(city, town):
    """
    특정 city, town에 위치한 병원 중 '내과'가 포함된 병원 5곳만 추천합니다.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
        SELECT name, address, phone_number, latitude, longitude 
        FROM hospital 
        WHERE city=%s AND town=%s AND name LIKE '%%내과%%'  -- ✅ "내과" 포함된 병원 필터링
        LIMIT 5;  -- ✅ 병원 5개만 추천 (기존 3 → 5로 변경)
        """
        cursor.execute(query, (city, town))
        hospitals = cursor.fetchall()
    except Exception as e:
        print(f"❌ 병원 데이터 조회 오류: {e}")
        hospitals = []
    finally:
        cursor.close()
        conn.close()
    
    return [
        {
            "name": h["name"],
            "address": h["address"],
            "phone": h["phone_number"],
            "latitude": h["latitude"],
            "longitude": h["longitude"]
        }
        for h in hospitals
    ]
# ✅ 6️⃣ 병원 추천 페이지 라우트
@bp.route("/")
def hospital_view():
    # ✅ 1️⃣ 가장 최근 사용자 데이터 가져오기
    customer_data = get_latest_customer_data()
    if not customer_data:
        return "❌ 사용자 건강 정보가 없습니다. 'customer.html'에서 정보를 입력해주세요.", 400

    # ✅ 2️⃣ 질병 예측 수행 (확률 + 결과)
    prediction_result, prediction_diabetes, prediction_dyslipidemia, prediction_high_blood_pressure = predict_hospital(customer_data)

    # ✅ 3️⃣ 사용자 지역 정보(city, town) 가져오기
    city, town = customer_data["city"], customer_data["town"]
    if not city or not town:
        return "❌ 사용자 지역 정보가 없습니다. 'customer.html'에서 정보를 입력해주세요.", 400

    # ✅ 4️⃣ 예측 결과가 모두 정상일 경우 병원 추천 생략
    if (
        prediction_result["diabetes"] == "정상"
        and prediction_result["dyslipidemia"] == "정상"
        and prediction_result["high_blood_pressure"] == "정상"
    ):
        hospitals = []  # 추천 병원 없음
        hospital_message = "모든 지표가 정상 범위 수치에 있어서 병원 추천이 필요없습니다.🎉"
    
    else:
        # ✅ 5️⃣ 예측 결과 중 하나라도 질병이면 병원 추천
        hospitals = get_hospital_data(city, town)
        hospital_message = ""

        # ✅ 6️⃣ 해당 지역에 내과 병원이 없을 경우 안내
        if not hospitals:
            return f"❌ {city} {town} 지역에서 '내과'가 포함된 병원 데이터를 찾을 수 없습니다.", 404

    # ✅ 7️⃣ hospital.html 템플릿에 필요한 정보 전달
    return render_template(
        "hospital.html",
        hospitals=hospitals,
        hospital_message=hospital_message,
        city=city,
        town=town,
        prediction=prediction_result,
        prediction_diabetes=prediction_diabetes,
        prediction_dyslipidemia=prediction_dyslipidemia,
        prediction_high_blood_pressure=prediction_high_blood_pressure,
    )

