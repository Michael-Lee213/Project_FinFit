# company_views.py
from flask import Blueprint, render_template

bp = Blueprint('company', __name__, url_prefix='/company')

@bp.route('/')
def company_index():
    return render_template('company.html')
