import sys
import pickle
import pymysql
import numpy as np
from flask import Blueprint, render_template, flash
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from scipy.stats import percentileofscore

sys.stdout.reconfigure(line_buffering=True)

bp = Blueprint("disease", __name__, url_prefix="/disease")

def load_model():
    model_path = "models/disease_model.pkl"
    try:
        with open(model_path, "rb") as file:
            model_data = pickle.load(file)
        print("📌 모델 로드 완료:", model_data.keys(), flush=True)
        return model_data
    except Exception as e:
        print("❌ 모델 로드 실패:", str(e), flush=True)
        return None

@bp.route("/", methods=["GET", "POST"])
def predict_disease():
    from finfit import get_db_connection

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # ✅ 최근 고객 데이터 가져오기
        cursor.execute("""
            SELECT sex, age, height, weight, bmi, alchol, smoking_history, 
                   chol_total, chol_hdl, chol_ldl, chol_tg,
                   fasting_blood_sugar, glycated_hemoglobin,
                   sbp_average, dbp_average, cancer_diagnosis_fathers,
                   cancer_diagnosis_mother, cancer_diagnosis_sibling,
                   white_blood_cell_count, red_blood_cell_count, stress
            FROM customer ORDER BY id DESC LIMIT 1;
        """)
        customer_data = cursor.fetchone()

        if not customer_data:
            flash("❌ 예측할 데이터가 없습니다. 먼저 건강 정보를 입력하세요!", "danger")
            return render_template("disease.html", graph_html=None)

        # ✅ 모델 로드
        model_data = load_model()
        if model_data is None:
            flash("❌ 예측 모델을 불러오지 못했습니다!", "danger")
            return render_template("disease.html", graph_html=None)

        optimized_models = model_data["optimized_models"]
        scaler = model_data["scaler"]
        imputer = model_data["imputer"]

        # ✅ 2. 입력 데이터 전처리 (빈값 -> np.nan 처리)
        for key, value in customer_data.items():
            if value in ["", None, " "]:
                customer_data[key] = np.nan

        # ✅ 3. 학습 시 사용한 feature 순서에 맞춰 재정렬 (누락된 컬럼은 np.nan으로 자동 보완됨)
        input_df = pd.DataFrame([customer_data])
        for col in imputer.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[imputer.feature_names_in_]  # 학습 컬럼 순서 맞춤

        # ✅ 4. 결측값 로그 출력
        print("📌 Null 값 체크:\n", input_df.isnull().sum())

        # ✅ 5. 결측치 보간
        input_data = imputer.transform(input_df)

        # ✅ 6. 정규화
        input_data = scaler.transform(input_data)

        # ✅ 7. 최종 확인 로그
        print("📌 최종 input_data (모델 입력값):\n", input_data)

        # ✅ 유병 확률 예측 및 툴팁 설정
        disease_predictions = {}
        disease_risk_levels = {}
        tooltip_texts = {}

        for disease, model in optimized_models.items():
            probability = model.predict_proba(input_data)[:, 1] * 100  # %
            disease_predictions[disease] = round(probability[0], 2)

            # 🔹 위험도 분류
            if probability < 30:
                disease_risk_levels[disease] = "저위험"
            elif probability < 70:
                disease_risk_levels[disease] = "중위험"
            else:
                disease_risk_levels[disease] = "고위험"

            # 🔹 툴팁 데이터 설정
            if disease == "고지혈증":
                tooltip_texts[disease] = f"총 콜레스테롤: {customer_data['chol_total']}" if customer_data['chol_total'] else "사용자 데이터 부족으로 정확도가 떨어질 수 있음"
            elif disease == "고혈압":
                tooltip_texts[disease] = f"혈압(수축기/이완기): {customer_data['sbp_average']}/{customer_data['dbp_average']}" if customer_data['sbp_average'] else "사용자 데이터 부족으로 정확도가 떨어질 수 있음"
            elif disease == "당뇨":
                tooltip_texts[disease] = f"공복 혈당: {customer_data['fasting_blood_sugar']}" if customer_data['fasting_blood_sugar'] else "사용자 데이터 부족으로 정확도가 떨어질 수 있음"
            else:
                tooltip_texts[disease] = "추가 정보 없음"

        # ✅ 유병 확률 그래프 생성 (막대 내부에 확률 표시)
        # ✅ 유병 확률 그래프 생성 (막대 내부에 확률 표시)
        fig = go.Figure()

        for disease, probability in disease_predictions.items():
            risk_level = disease_risk_levels[disease]
            
            fig.add_trace(go.Bar(
                x=[disease],
                y=[probability],
                marker_color="#008b8b",  # ✅ 막대 색상 유지
                name=f"{disease} ({risk_level})",
                text=f"{probability:.2f}%",
                textposition="inside",
                hovertext=tooltip_texts[disease],
                hoverinfo="text+y",
                width=0.4
            ))

        # ✅ 위험도 레이블 추가 (텍스트 흰색으로 변경)
        fig.add_annotation(x=len(disease_predictions) - 0.5, y=15, text="저위험", showarrow=False, font=dict(color="white", size=12))
        fig.add_annotation(x=len(disease_predictions) - 0.5, y=50, text="중위험", showarrow=False, font=dict(color="white", size=12))
        fig.add_annotation(x=len(disease_predictions) - 0.5, y=85, text="고위험", showarrow=False, font=dict(color="white", size=12))

        # ✅ 다크 테마 레이아웃 설정
        fig.update_layout(
            title="유병 확률 및 위험도 분석",
            xaxis=dict(
                title=dict(text="질병", font=dict(color='white')),  # ✅ 수정된 titlefont 위치
                showgrid=True,
                gridcolor='#444',
                mirror=True,
                linecolor='white',
                linewidth=1.5,
                tickfont=dict(color='white')  # ✅ tick 글씨 색상
            ),
            yaxis=dict(
                title=dict(text="유병 확률 (%)", font=dict(color='white')),  # ✅ 수정된 titlefont 위치
                range=[0, 100],
                showgrid=True,
                gridcolor='#444',
                mirror=True,
                linecolor='white',
                linewidth=1.5,
                tickfont=dict(color='white')  # ✅ tick 글씨 색상
            ),
            barmode='group',
            plot_bgcolor='#1e2733',
            paper_bgcolor='#121820',  # ✅ 이건 layout에 넣는 게 맞음
            font=dict(color='white')  # 전체 폰트 색상 지정
        )


        # ✅ 상태 메시지 (이모지 + 질병명 + 위험도)
        status_icons = []
        for disease, risk in disease_risk_levels.items():
            emoji = "🟢" if risk == "저위험" else "🔵"
            status_icons.append(f"{emoji} {disease}: {risk}")

        disease_summary_message = "　".join(status_icons)  # 전각 공백

        # ✅ HTML 변환
        disease_graph_html = pio.to_html(fig, full_html=False)


    
        # ✅ 암 위험도 분석 (기존 데이터 기반)
        # ✅ 암 위험도 분석 (기존 데이터 기반)
        cursor.execute(""" SELECT * FROM medical """)
        medical_data = cursor.fetchall()
        df_medical = pd.DataFrame(medical_data)

        # ✅ 평균값 계산 (NaN 방지)
        column_means = df_medical.mean(numeric_only=True).fillna(0)

        # ✅ 수정된 방식 (값이 정수 0/1일 때 그대로 사용)
        cancer_diagnosis_fathers = customer_data["cancer_diagnosis_fathers"] or 0
        cancer_diagnosis_mother = customer_data["cancer_diagnosis_mother"] or 0
        cancer_diagnosis_sibling = customer_data["cancer_diagnosis_sibling"] or 0
        
        # ✅  유전여부확인
        print("🧬 유전 입력값 확인:",
            "부:", cancer_diagnosis_fathers,
            "모:", cancer_diagnosis_mother,
            "형제:", cancer_diagnosis_sibling)
        
        # ✅ 유전적 가중치 설정
        GENETIC_WEIGHT_FATHER = 5  
        GENETIC_WEIGHT_MOTHER = 5  
        GENETIC_WEIGHT_SIBLING = 4 

        # ✅ 현재 고객의 암 위험도 점수 계산
        customer_cancer_risk = (
            cancer_diagnosis_fathers * GENETIC_WEIGHT_FATHER +  
            cancer_diagnosis_mother * GENETIC_WEIGHT_MOTHER +  
            cancer_diagnosis_sibling * GENETIC_WEIGHT_SIBLING +  
            (customer_data["chol_total"] or column_means["chol_total"]) / 50 +  
            (customer_data["sbp_average"] or column_means["sbp_average"]) / 10 +  
            (customer_data["dbp_average"] or column_means["dbp_average"]) / 15 +  
            (customer_data["fasting_blood_sugar"] or column_means["fasting_blood_sugar"]) / 15 +  
            (customer_data["glycated_hemoglobin"] or column_means["glycated_hemoglobin"]) * 3 +  
            (customer_data["white_blood_cell_count"] or column_means["white_blood_cell_count"]) * 1.5 +  
            (customer_data["red_blood_cell_count"] or column_means["red_blood_cell_count"]) * 1.5 +  
            (customer_data["age"] or column_means["age"]) / 20 +  
            (customer_data["bmi"] or column_means["bmi"]) / 5 +  
            (customer_data["alchol"] or column_means["alchol"]) * 2 +  
            (customer_data["smoking_history"] or column_means["smoking_history"]) * 5 +
            ((5 - customer_data["stress"]) if customer_data["stress"] in [1, 2, 3, 4] else 0) * 2
        )

        # ✅ 모든 고객의 암 위험도 점수 계산
        all_cancer_scores = [
            (
                (row["cancer_diagnosis_fathers"] or column_means["cancer_diagnosis_fathers"]) * GENETIC_WEIGHT_FATHER +  
                (row["cancer_diagnosis_mother"] or column_means["cancer_diagnosis_mother"]) * GENETIC_WEIGHT_MOTHER +  
                (row["cancer_diagnosis_sibling"] or column_means["cancer_diagnosis_sibling"]) * GENETIC_WEIGHT_SIBLING +  
                (row["chol_total"] or column_means["chol_total"]) / 50 +  
                (row["sbp_average"] or column_means["sbp_average"]) / 10 +  
                (row["dbp_average"] or column_means["dbp_average"]) / 15 +  
                (row["fasting_blood_sugar"] or column_means["fasting_blood_sugar"]) / 15 +  
                (row["glycated_hemoglobin"] or column_means["glycated_hemoglobin"]) * 3 +  
                (row["white_blood_cell_count"] or column_means["white_blood_cell_count"]) * 1.5 +  
                (row["red_blood_cell_count"] or column_means["red_blood_cell_count"]) * 1.5 +  
                (row["age"] or column_means["age"]) / 20 +  
                (row["bmi"] or column_means["bmi"]) / 5 +  
                (row["alchol"] or column_means["alchol"]) * 2 +  
                (row["smoking_history"] or column_means["smoking_history"]) * 5 +
                ((5 - row["stress"]) if row["stress"] in [1, 2, 3, 4] else 0) * 2
            ) for _, row in df_medical.iterrows()
        ]

        # ✅ 기존 고객 점수를 오름차순으로 정렬
        sorted_scores = np.sort(all_cancer_scores)
        sorted_ranks = np.linspace(100, 1, len(sorted_scores))  # 100% (안전) ~ 1% (위험)

        # ✅ 현재 고객의 백분위 순위 계산
        percentile_rank = percentileofscore(all_cancer_scores, customer_cancer_risk, kind="rank")
        cancer_rank = 100 - percentile_rank  # 높은 점수가 낮은 순위 (위험)

        # ✅ 암 위험도 산점도 그래프 생성
        cancer_fig = go.Figure()

        # ✅ 기존 고객 데이터 (녹색)
        cancer_fig.add_trace(go.Scatter(
            x=sorted_scores,
            y=sorted_ranks,  
            mode='markers',
            marker=dict(color="teal", opacity=0.6, size=6),
            name="기존 데이터"
        ))

        # ✅ 현재 고객 Y축 값을 올바르게 매핑
        if customer_cancer_risk < min(sorted_scores):
            customer_y = 100
        elif customer_cancer_risk > max(sorted_scores):
            customer_y = 1
        else:
            customer_y = np.interp(customer_cancer_risk, sorted_scores, sorted_ranks)

        # ✅ 별표 위치가 유효한지 확인 (NaN 또는 이상치면 생략)
        if pd.isna(customer_y) or customer_y < 0 or customer_y > 100:
            print("⚠️ 암 점수는 계산되었지만 별을 표시할 수 없습니다.")
            cancer_status_message = "입력한 데이터가 부족하여 암 위험도 점수를 계산할 수 없습니다."
        else:
            # ✅ 신규 고객 별 표시
            cancer_fig.add_trace(go.Scatter(
                x=[customer_cancer_risk],
                y=[customer_y],
                mode='markers+text',
                marker=dict(color='darkblue', size=12, symbol="star"),
                text=[f"사용자 ({customer_cancer_risk:.2f}, {customer_y:.0f}%)"],
                textposition="top center",
                name="사용자"
            ))

            # ✅ 상태 메시지 출력
            if customer_y <= 30:
                cancer_status_message = "🔵 암 위험도는 상위권에 해당합니다. <br>건강에 주의가 필요하며, 빠른 시일 내에 정밀 검진을 받아보시길 권장드립니다."
            elif customer_y <= 70:
                cancer_status_message = "🔵 암 위험도는 중위권에 해당합니다. <br>정기적인 검진을 통해 건강 상태를 점검하세요."
            else:
                cancer_status_message = "🟢 암 위험도는 하위권에 해당합니다. <br>건강한 상태를 잘 유지하고 계십니다."

        # ✅ 그래프 레이아웃 수정
        cancer_fig.update_layout(
        title=dict(
            text="암 위험도 점수 vs. 암 위험도 순위",
            font=dict(color='white')
        ),
        xaxis=dict(
            title=dict(text="암 위험도 점수", font=dict(color='white')),
            showgrid=True,
            gridcolor="#444",
            linecolor="white",
            linewidth=1.5,
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title=dict(text="암 위험도 순위 (%) (위쪽이 가장 위험)", font=dict(color='white')),
            autorange="reversed",
            showgrid=True,
            gridcolor="#444",
            linecolor="white",
            linewidth=1.5,
            tickfont=dict(color='white')
        ),
        plot_bgcolor="#1e2733",
        paper_bgcolor="#121820",
        font=dict(color="white"),
        legend=dict(
        font=dict(color='white')
            )
        )


        # ✅ 그래프를 HTML로 변환하여 템플릿으로 전달
        cancer_graph_html = pio.to_html(cancer_fig, full_html=False)



        # ✅ 🚀 여기서부터 레이더 차트 🚀
        categories = ['BMI', '수축기 혈압', '이완기 혈압', '공복 혈당', '총 콜레스테롤']

        # ✅ 정상 범위 정의 (정육각형을 그래프의 절반 크기로 조정)
        normal_min = np.array([18.5, 90, 60, 70, 125])
        normal_max = np.array([24.9, 120, 80, 100, 200])

        # ✅ 사용자 입력값 (customer 테이블에서 원본 데이터 사용)
        user_data = np.array([
            customer_data["bmi"],
            customer_data["sbp_average"],
            customer_data["dbp_average"],
            customer_data["fasting_blood_sugar"],
            customer_data["chol_total"]
        ])

        # ✅ NaN 또는 None 값이 포함된 경우 0으로 대체
        user_data = np.array([0 if value is None else value for value in user_data])

        # ✅ 모든 값이 0일 경우 그래프를 표시하지 않음
        if all(value == 0 for value in user_data):
            flash("⚠️ 입력한 데이터가 부족하여 그래프를 표시할 수 없습니다.", "warning")
            return render_template(
                "disease.html",
                disease_predictions=disease_predictions,
                disease_risk_levels=disease_risk_levels,
                graph_html=None  # 🚨 그래프를 표시하지 않음
            )

        # ✅ 사용자 데이터 정규화 (정상 범위의 최대값을 0.5로 설정)
        normalized_user_data = user_data / normal_max * 0.5
        normalized_user_data = np.clip(normalized_user_data, 0, 1)  # ✅ 값이 1을 넘지 않도록 제한

        # ✅ 레이더 차트 생성
        fig = go.Figure()

        # ✅ 정상 범위 (정육각형 형태 유지, 크기를 그래프의 절반으로 조정)
        fig.add_trace(go.Scatterpolar(
            r=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # ✅ 정상 범위 크기를 0.5로 조정
            theta=categories + [categories[0]],
            fill='toself',
            opacity=0.3,
            line=dict(color='lightseagreen'),  # ✅ 여기!
            name="정상 범위"
        ))

        # ✅ 사용자 데이터 Trace (정규화된 값 적용, deepskyblue 적용)
        fig.add_trace(go.Scatterpolar(
            r=np.append(normalized_user_data, normalized_user_data[0]),
            theta=categories + [categories[0]],
            fill='toself',
            line=dict(color='deepskyblue'),  # ✅ 여기!
            marker=dict(color='deepskyblue', size=8, symbol='circle'),      
            name="사용자 데이터"
        ))

        # ✅ 그래프 레이아웃 조정 (정육각형 내부에 조정)
        fig.update_layout(
        title=dict(
            text="건강 상태 레이더 차트",
            font=dict(color='white', size=18)
        ),
        polar=dict(
            bgcolor='#1e2733',  # ✅ 내부 배경 어둡게
            angularaxis=dict(
                color='white',   # ✅ 각도축 글자 색
                linewidth=1,
                showline=True
            ),
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='gray',     # ✅ 눈금선 색
                linecolor='white',    # ✅ 축선 색
                tickfont=dict(color='white')
            )
        ),
        paper_bgcolor='#121820',  # ✅ 바깥 배경
        plot_bgcolor='#121820',
        font=dict(color='white'),
        legend=dict(
            font=dict(color='white')
        )
    )

        
        # ✅ 문자열 → float 변환 (없으면 0 처리)
        bmi = float(customer_data["bmi"]) if customer_data["bmi"] else 0
        sbp = float(customer_data["sbp_average"]) if customer_data["sbp_average"] else 0
        dbp = float(customer_data["dbp_average"]) if customer_data["dbp_average"] else 0
        sugar = float(customer_data["fasting_blood_sugar"]) if customer_data["fasting_blood_sugar"] else 0
        chol = float(customer_data["chol_total"]) if customer_data["chol_total"] else 0

        # ✅ 레이더 차트 상태메세지
        radar_abnormal = []
        # if bmi < 18.5 or bmi > 24.9:
        #     radar_abnormal.append("BMI")
        # if sbp < 90 or sbp > 120:
        #     radar_abnormal.append("수축기 혈압")
        # if dbp < 60 or dbp > 80:
        #     radar_abnormal.append("이완기 혈압")
        # if sugar < 70 or sugar > 100:
        #     radar_abnormal.append("공복 혈당")
        # if chol < 125 or chol > 200:
        #     radar_abnormal.append("총 콜레스테롤")
        if  bmi > 24.9:
            radar_abnormal.append("BMI")
        if  sbp > 120:
            radar_abnormal.append("수축기 혈압")
        if  dbp > 80:
            radar_abnormal.append("이완기 혈압")
        if  sugar > 100:
            radar_abnormal.append("공복 혈당")
        if  chol > 200:
            radar_abnormal.append("총 콜레스테롤")

        if radar_abnormal:
            radar_status_message = f"🔵 {' · '.join(radar_abnormal)} 수치가 정상 범위를 벗어났습니다."
        else:
            radar_status_message = "🟢 모든 지표가 정상 범위에 있습니다."


        # ✅ 그래프를 HTML로 변환하여 전달
        graph_html = pio.to_html(fig, full_html=False)

        # 🚀 여기서부터 3D 차트 🚀
        # ✅ 3D 정육면체 레이더 차트 생성
        categories_3d = ['수축기 혈압', '공복 혈당', 'BMI']

        # ✅ 정상 범위 설정
        normal_min_3d = np.array([90, 70, 18.5], dtype=np.float64)  # 최소 정상 값
        normal_max_3d = np.array([120, 100, 24.9], dtype=np.float64)  # 최대 정상 값

        # ✅ 사용자 데이터 (단일 값)
        user_data_3d = np.array([
            float(customer_data["sbp_average"]) if customer_data["sbp_average"] is not None else 0,
            float(customer_data["fasting_blood_sugar"]) if customer_data["fasting_blood_sugar"] is not None else 0,
            float(customer_data["bmi"]) if customer_data["bmi"] is not None else 0
        ], dtype=np.float64)

      # ✅ 데이터 정규화 (2D 레이더 차트와 동일한 방식: 값 / max * 0.5)
        normalized_user_data_3d = user_data_3d / normal_max_3d * 0.5
        normalized_user_data_3d = np.clip(normalized_user_data_3d, 0, 1)  # ✅ 값이 1을 넘지 않도록 제한

        # ✅ 정육면체 크기 설정 (2D와 맞춤: 0.5)
        cube_size = 0.5  # ✅ 박스 크기 (50% 기준)
        x_normal = np.array([0, cube_size, cube_size, 0, 0, cube_size, cube_size, 0], dtype=np.float64)
        y_normal = np.array([0, 0, cube_size, cube_size, 0, 0, cube_size, cube_size], dtype=np.float64)
        z_normal = np.array([0, 0, 0, 0, cube_size, cube_size, cube_size, cube_size], dtype=np.float64)

        # ✅ 사용자 데이터 (3D 점으로 표시) → cube_size 기준으로 적용
        x_user = [normalized_user_data_3d[0] * 1]
        y_user = [normalized_user_data_3d[1] * 1]
        z_user = [normalized_user_data_3d[2] * 1]

        # ✅ 3D 그래프 생성
        fig_3d = go.Figure()

        # ✅ 정상 범위 정육면체 (연한 초록색)
        fig_3d.add_trace(go.Mesh3d(
            x=x_normal, y=y_normal, z=z_normal,
            i=[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
            j=[1, 2, 3, 2, 6, 3, 7, 4, 5, 6, 7, 7],
            k=[5, 6, 7, 5, 5, 7, 6, 5, 6, 7, 6, 7],
            color="green",
            opacity=0.5,
            name="정상 범위"
        ))
        # ✅ 박스 위쪽 중앙에 '정상 범위' 라벨 추가
        fig_3d.add_trace(go.Scatter3d(
            x=[cube_size / 2],
            y=[cube_size / 2],
            z=[cube_size + 0.05],  # 살짝 위쪽에 띄움
            mode='text',
            text=["🟩 정상 범위"],
            textfont=dict(size=14, color="green"),
            showlegend=False
        ))
        # ✅ 사용자 데이터 (구 형태 Mesh 추가)
        theta = np.linspace(0, 2 * np.pi, 30)  # 구의 수평 좌표
        phi = np.linspace(0, np.pi, 30)  # 구의 수직 좌표
        theta, phi = np.meshgrid(theta, phi)

        r = 0.1  # 구의 반지름 (적절한 크기 조절)
        x_sphere = x_user + r * np.sin(phi) * np.cos(theta)
        y_sphere = y_user + r * np.sin(phi) * np.sin(theta)
        z_sphere = z_user + r * np.cos(phi)

        fig_3d.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale="Blues",  # ✅ → 예: 'YlGnBu', 'Viridis'도 고려 가능
            showscale=False,
            name="사용자 데이터",
            opacity=0.8
        ))
        # ✅ 사용자 구 위치에 텍스트 추가
        fig_3d.add_trace(go.Scatter3d(
            x=x_user,
            y=y_user,
            z=[z + 0.05 for z in z_user],  # 구 위에 살짝 띄움
            mode='text',
            text=["사용자"],
            textfont=dict(size=14, color="deepskyblue"),
            showlegend=False
        ))
        # ✅ 3D 그래프 레이아웃 조정
        fig_3d.update_layout(
        title=dict(
            text="3D 건강 상태 정육면체 차트",
            font=dict(color='white', size=18)
        ),
        scene=dict(
            xaxis=dict(
                title="혈압", range=[-0.5, 1.5],
                backgroundcolor="#1e2733",  # ✅ 축 배경
                gridcolor="gray",
                zerolinecolor="gray",
                color="white",              # ✅ 축 글씨
            ),
            yaxis=dict(
                title="혈당", range=[-0.5, 1.5],
                backgroundcolor="#1e2733",
                gridcolor="gray",
                zerolinecolor="gray",
                color="white",
            ),
            zaxis=dict(
                title="BMI", range=[-0.5, 1.5],
                backgroundcolor="#1e2733",
                gridcolor="gray",
                zerolinecolor="gray",
                color="white",
            ),
            aspectmode="cube"
        ),
        paper_bgcolor="#121820",  # ✅ 전체 배경
        font=dict(color="white"),
        legend=dict(font=dict(color="white"))
        )

        
        # ✅ 3D 그래프 상태메세지 분리 방식
        if (
            (0 <= normalized_user_data_3d[0] <= cube_size) and
            (0 <= normalized_user_data_3d[1] <= cube_size) and
            (0 <= normalized_user_data_3d[2] <= cube_size)
        ):
            health_status_color = "success"
            health_status_icon = "🟢"
            health_status_message = "현재 건강 지표는 정상 범위에 속해 있습니다."
        else:
            health_status_color = "blue"
            health_status_icon = "🔵"
            health_status_message = "현재 건강 지표가 정상 범위를 벗어났습니다. 주의가 필요합니다."

        # ✅ 그래프를 HTML로 변환하여 전달
        graph_3d_html = pio.to_html(fig_3d, full_html=False)


        # ✅ 최종 `return render_template`
        return render_template(
            "disease.html",
            disease_graph_html=disease_graph_html,
            cancer_graph_html=cancer_graph_html,
            graph_html=graph_html,
            graph_3d_html=graph_3d_html,
            disease_predictions=disease_predictions,
            disease_risk_levels=disease_risk_levels,
            disease_summary_message=disease_summary_message,
            cancer_risk_score=round(customer_cancer_risk, 2),
            cancer_rank=round(cancer_rank, 2),
            health_status_color=health_status_color,
            health_status_icon=health_status_icon,
            health_status_message=health_status_message,
            cancer_status_message=cancer_status_message,
            radar_status_message=radar_status_message
        )
    except pymysql.MySQLError as e:
        print("❌ 데이터베이스 오류 발생:", str(e), flush=True)
        flash(f"❌ 데이터베이스 오류: {str(e)}", "danger")
        return render_template("disease.html", graph_html=None)

    except Exception as e:
        print("❌ 예측 중 오류 발생:", str(e), flush=True)
        flash(f"❌ 예측 중 오류 발생: {str(e)}", "danger")
        return render_template("disease.html", graph_html=None)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

        