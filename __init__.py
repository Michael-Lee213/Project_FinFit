from flask import Flask
import pymysql

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        db='finfit',
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )

def create_app():
    app = Flask(__name__)
    app.secret_key = 'mini'
    
    from .views import (
            customer_views, exercise_views, disease_views,
            hospital_views, depression_views, squat_views,
            main_views, company_views, chatbot_views
        )    
    
    # 블루프린트 등록
    app.register_blueprint(customer_views.bp)
    app.register_blueprint(disease_views.bp)
    app.register_blueprint(hospital_views.bp)
    app.register_blueprint(depression_views.bp)
    app.register_blueprint(exercise_views.bp)
    app.register_blueprint(squat_views.bp)
    app.register_blueprint(main_views.bp)
    app.register_blueprint(company_views.bp)
    app.register_blueprint(chatbot_views.chatbot_bp)

    
    return app