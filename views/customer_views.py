from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
import pymysql
from finfit import get_db_connection

bp = Blueprint("customer", __name__, url_prefix="/customer")

@bp.route("/", methods=["GET", "POST"])
def customer_form():
    conn = None
    cursor = None

    if request.method == "POST":
        try:
            # 사용자 입력 데이터 수집
            data = {
                "city": request.form.get("city"),
                "town": request.form.get("town"),
                "sex": request.form.get("sex"),
                "age": request.form.get("age"),
                "height": request.form.get("height"),
                "weight": request.form.get("weight"),
                "bmi": request.form.get("bmi"),
                "alchol": request.form.get("alchol"),
                "smoking_history": request.form.get("smoking_history"),
                "chol_total": request.form.get("chol_total"),
                "chol_hdl": request.form.get("chol_hdl"),
                "chol_ldl": request.form.get("chol_ldl"),
                "chol_tg": request.form.get("chol_tg"),
                "fasting_blood_sugar": request.form.get("fasting_blood_sugar"),
                "glycated_hemoglobin": request.form.get("glycated_hemoglobin"),
                "sbp_average": request.form.get("sbp_average"),
                "dbp_average": request.form.get("dbp_average"),
                "cancer_diagnosis_fathers": request.form.get("cancer_diagnosis_fathers"),
                "cancer_diagnosis_mother": request.form.get("cancer_diagnosis_mother"),
                "cancer_diagnosis_sibling": request.form.get("cancer_diagnosis_sibling"),
                "white_blood_cell_count": request.form.get("white_blood_cell_count"),
                "red_blood_cell_count": request.form.get("red_blood_cell_count"),
                "stress": request.form.get("stress")
            }

            # 🔧 빈 문자열 → None 처리
            for key in data:
                if data[key] == "":
                    data[key] = None

            # ✅ 유전 여부 문자열을 숫자로 변환
            for key in ["cancer_diagnosis_fathers", "cancer_diagnosis_mother", "cancer_diagnosis_sibling"]:
                if data[key] == "있음":
                    data[key] = 1
                else:
                    data[key] = 0

            print("📌 입력된 데이터:", data)


            # 🔹 BMI 자동 계산
            if not data["bmi"] and data["height"] and data["weight"]:
                try:
                    height_m = float(data["height"]) / 100
                    weight_kg = float(data["weight"])
                    data["bmi"] = round(weight_kg / (height_m * height_m), 2)
                    print(f"✅ 자동 계산된 BMI: {data['bmi']}", flush=True)
                except ValueError:
                    flash("❌ 키 또는 몸무게 값이 올바르지 않습니다.", "danger")
                    return redirect(url_for("customer.customer_form"))

            if data["bmi"]:
                try:
                    data["bmi"] = float(data["bmi"])
                except ValueError:
                    data["bmi"] = None

            # ✅ 필수 필드 (중요 피처 포함)
            required_fields = [
                "city", "town", "sex", "age", "height", "weight",
                "alchol", "smoking_history", "stress",
                "chol_ldl", "sbp_average", "glycated_hemoglobin"  # 질병별 중요 feature
            ]
            for field in required_fields:
                if not data[field]:
                    flash(f"❌ 필수 입력값이 누락되었습니다: {field}", "danger")
                    return redirect(url_for("customer.customer_form"))

            # DB 연결
            conn = get_db_connection()
            cursor = conn.cursor()

            # ✅ 평균값으로 대체할 필드 (필수 컬럼은 제외)
            average_fields = [
                "chol_total", "chol_hdl", "chol_tg",
                "fasting_blood_sugar", "dbp_average",
                "white_blood_cell_count", "red_blood_cell_count", "bmi"
            ]

            cursor.execute("""
                SELECT
                    AVG(chol_total) AS chol_total,
                    AVG(chol_hdl) AS chol_hdl,
                    AVG(chol_tg) AS chol_tg,
                    AVG(fasting_blood_sugar) AS fasting_blood_sugar,
                    AVG(dbp_average) AS dbp_average,
                    AVG(white_blood_cell_count) AS white_blood_cell_count,
                    AVG(red_blood_cell_count) AS red_blood_cell_count,
                    AVG(bmi) AS bmi
                FROM medical
            """)
            avg_values = cursor.fetchone()

            # 평균값으로 대체
            for field in average_fields:
                if data[field] is None and avg_values[field] is not None:
                    data[field] = round(avg_values[field], 2)

            # SQL INSERT 실행
            sql = """
            INSERT INTO customer 
            (city, town, sex, age, height, weight, bmi, alchol, smoking_history, 
             chol_total, chol_hdl, chol_ldl, chol_tg, 
             fasting_blood_sugar, glycated_hemoglobin, sbp_average, dbp_average, 
             cancer_diagnosis_fathers, cancer_diagnosis_mother, cancer_diagnosis_sibling, 
             white_blood_cell_count, red_blood_cell_count, stress)
            VALUES 
            (%(city)s, %(town)s, %(sex)s, %(age)s, %(height)s, %(weight)s, %(bmi)s, 
             %(alchol)s, %(smoking_history)s, %(chol_total)s, %(chol_hdl)s, %(chol_ldl)s, 
             %(chol_tg)s, %(fasting_blood_sugar)s, %(glycated_hemoglobin)s, %(sbp_average)s, 
             %(dbp_average)s, %(cancer_diagnosis_fathers)s, %(cancer_diagnosis_mother)s, 
             %(cancer_diagnosis_sibling)s, %(white_blood_cell_count)s, %(red_blood_cell_count)s, %(stress)s)
            """
            cursor.execute(sql, data)
            conn.commit()

            flash("✅ 건강 정보가 성공적으로 저장되었습니다!", "success")
            return redirect(url_for("main.main_page", submitted="true"))  # ✅ 여기에 추가!

        except pymysql.MySQLError as e:
            flash(f"❌ 데이터베이스 오류: {str(e)}", "danger")
            print(f"❌ MySQL 오류 발생: {e}")

        except Exception as e:
            flash(f"❌ 알 수 없는 오류 발생: {str(e)}", "danger")
            print(f"❌ 일반 오류 발생: {e}")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    return render_template("customer.html")
