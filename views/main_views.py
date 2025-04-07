from flask import Blueprint, render_template

bp = Blueprint("main", __name__, url_prefix="/")  # ✅ 블루프린트 설정

@bp.route("/")
def main_page():
    return render_template("main.html")  # ✅ main.html을 렌더링