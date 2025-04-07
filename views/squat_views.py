# FINFIT/views/squat_views.py

from flask import Blueprint, render_template, Response, request, redirect, url_for, flash
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import math
import time
import logging
logging.basicConfig(level=logging.DEBUG)
from ..geminai_feedback import get_feedback_rag
from ..utils import (
    calculate_angle,
    is_pose_detected,
    visualize_feedback,
    analyze_squat,
    generate_angle_plot,
    extract_major_issues,
    mp_pose
)

bp = Blueprint('squat', __name__, url_prefix='/squat')

# 전역 변수
camera = None
analysis_started = False
squat_count = 0
in_squat = False
prev_landmarks = None
target_count = 10
target_sets = 3
max_reps = target_count * target_sets

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

baseline_angle = None
down_offset = 55
up_offset = 10
last_down_angle = None
last_up_angle = None

# 분석 결과 저장
report_ready = False
feedback_text = ""
report_image_path = ""
angle_history = []
feedback_history = []

def is_user_fully_visible(landmarks):
    # visibility가 0.7 이상인 주요 랜드마크 개수 기반
    visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.7]
    return len(visible_landmarks) >= 25  # 전체 33개 중 약 75% 이상 보이면 True

def gen_frames():
    import numpy as np  # 꼭 상단에서 import 되어 있어야 합니다
    global camera, squat_count, in_squat, prev_landmarks
    global baseline_angle, last_down_angle, last_up_angle, max_reps
    global angle_history, feedback_history

    DESIRED_WIDTH = 640
    DESIRED_HEIGHT = 1080

    if camera is None:
        camera = cv2.VideoCapture(0)

    if not camera or not camera.isOpened():
        logging.debug("⚠️ Camera not initialized properly. Sending placeholder image.")
        while True:
            temp = np.zeros((DESIRED_HEIGHT, DESIRED_WIDTH, 3), dtype=np.uint8)
            msg = "Camera not available"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(msg, font, 1, 2)
            center_x = (DESIRED_WIDTH - text_w) // 2
            center_y = (DESIRED_HEIGHT + text_h) // 2
            cv2.putText(temp, msg, (center_x, center_y), font, 1, (255, 255, 255), 2)
            ret2, buffer = cv2.imencode('.jpg', temp)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    pose_estimator = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    while True:
        try:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            original_h, original_w = frame.shape[:2]

            # 업스케일 금지: 원본이 DESIRED_HEIGHT보다 크면 축소, 작으면 그대로
            if original_h > DESIRED_HEIGHT:
                scale = DESIRED_HEIGHT / original_h
                new_w = int(original_w * scale)
                resized = cv2.resize(frame, (new_w, DESIRED_HEIGHT), interpolation=cv2.INTER_AREA)
                if new_w > DESIRED_WIDTH:
                    left = (new_w - DESIRED_WIDTH) // 2
                    resized = resized[:, left:left + DESIRED_WIDTH]
            else:
                resized = frame.copy()

            image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(image_rgb)

            annotated = resized.copy()
            if results.pose_landmarks and is_pose_detected(results.pose_landmarks):
                landmarks = results.pose_landmarks.landmark

                if analysis_started:
                    left_knee_angle = calculate_angle(
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    )
                    right_knee_angle = calculate_angle(
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    )
                    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                    if baseline_angle is None:
                        baseline_angle = avg_knee_angle
                    elif avg_knee_angle > baseline_angle:
                        baseline_angle = avg_knee_angle

                    down_thresh = baseline_angle - down_offset
                    up_thresh = baseline_angle - up_offset

                    if not in_squat and avg_knee_angle < down_thresh:
                        in_squat = True
                        last_down_angle = avg_knee_angle
                    elif in_squat and avg_knee_angle > up_thresh:
                        if squat_count < max_reps:
                            squat_count += 1
                            last_up_angle = avg_knee_angle
                        in_squat = False

                    feedback, _ = analyze_squat(landmarks, prev_landmarks)
                    if not feedback:
                        feedback = {"INFO": "Keep your position steady!"}

                    angle_history.append(avg_knee_angle)
                    feedback_history.append(feedback)

                    progress = squat_count / max_reps if max_reps > 0 else 0
                    annotated = visualize_feedback(
                        resized.copy(),
                        feedback,
                        results.pose_landmarks,
                        progress=progress
                    )
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    h, w, _ = annotated.shape
                    texts = [
                        f"Squat Count: {squat_count}",
                        f"Avg Knee Angle: {avg_knee_angle:.1f}",
                        f"Last Down Angle: {last_down_angle:.1f}" if last_down_angle else "-",
                        f"Last Up Angle: {last_up_angle:.1f}" if last_up_angle else "-"
                    ]
                    y = 30
                    for t in texts:
                        cv2.putText(annotated, t, (w - 200, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (255, 255, 255), 1)
                        y += 20
                else:
                    msg = "Please stand straight to begin."
                    (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    center_x = (annotated.shape[1] - text_w) // 2
                    center_y = (annotated.shape[0] + text_h) // 2
                    cv2.putText(annotated, msg, (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
            else:
                msg = "Pose not detected! Please align with the camera."
                (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                center_x = (annotated.shape[1] - text_w) // 2
                center_y = (annotated.shape[0] + text_h) // 2
                cv2.putText(annotated, msg, (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            ret2, buffer = cv2.imencode('.jpg', annotated)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

        except Exception as e:
            logging.debug(f"📛 Error in frame generation loop: {e}")
            break

@bp.route('/')
def index():
    global analysis_started, squat_count, target_count, target_sets, angle_history, feedback_text, report_ready, camera

    # 웹캠을 미리 로딩해서 빠르게 구동 (웜업)
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if camera.isOpened():
            ret, frame = camera.read()
            if not ret or frame is None:
                logging.warning("Camera warmup read failed.")
        else:
            logging.warning("Camera not opened during warmup.")

    return render_template(
        'squat.html',
        analysis_started=analysis_started,
        squat_count=squat_count, 
        target_sets=target_sets,
        report_ready=report_ready,
        feedback_text=feedback_text,
        angle_history=angle_history  # angle_history 전달
    )

@bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/start', methods=['POST'])
def start():
    global camera, analysis_started, squat_count, in_squat
    global target_count, target_sets, max_reps
    global baseline_angle, last_down_angle, last_up_angle
    global angle_history, feedback_history, report_ready, feedback_text, report_image_path

    target_count = int(request.form.get('target_count', 10))
    target_sets = int(request.form.get('target_sets', 3))
    max_reps = target_count * target_sets

    squat_count = 0
    in_squat = False
    baseline_angle = None
    last_down_angle = None
    last_up_angle = None
    angle_history = []
    feedback_history = []
    report_ready = False
    feedback_text = ""
    report_image_path = ""

    analysis_started = True  # 분석 시작

    # 동영상 업로드 기능 활성화
    if 'video_file' in request.files:
        file = request.files['video_file']
        if file and file.filename != '':
            # 임시 파일에 저장 (동영상 확장자를 명시할 수 있음)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            file.save(temp_file.name)
            if camera is not None:
                camera.release()
            camera = cv2.VideoCapture(temp_file.name)
            logging.debug(f"Video file uploaded and opened: {temp_file.name}")
        else:
            # 업로드된 파일이 없으면 기본 웹캠 사용
            if camera is not None:
                camera.release()
            camera = cv2.VideoCapture(0)
            logging.debug("No valid file uploaded, using webcam instead.")
    else:
        # 파일 업로드 필드가 없으면 기본적으로 웹캠 사용
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(0)
        logging.debug("No file upload found, using webcam.")

    return redirect(url_for('squat.index'))

@bp.route('/stop', methods=['POST'])
def stop():
    global analysis_started, squat_count, camera
    global feedback_text, report_image_path, report_ready
    global angle_history, feedback_history, target_sets

    analysis_started = False

    if camera is not None:
        camera.release()
        camera = None

    if squat_count > 0 and angle_history and feedback_history:
        summary_data = {
            'total_reps': squat_count,
            'sets': target_sets,
            'avg_knee_angle': sum(angle_history) / len(angle_history),
            'issues': extract_major_issues(feedback_history)
        }

        try:
            feedback_raw = get_feedback_rag(summary_data)
            if "AI 서버 오류 발생" in feedback_raw or "API key not valid" in feedback_raw:
                feedback_text = "❗ 현재 AI 분석 서버에 문제가 있습니다. 잠시 후 다시 시도해주세요."
            else:
                feedback_text = feedback_raw
        except Exception as e:
            feedback_text = "❗ Gemini 서버와의 연결 중 문제가 발생했습니다."

        report_image_path = generate_angle_plot(angle_history)
        report_ready = True

    return redirect(url_for('squat.index'))

@bp.route('/reset', methods=['POST'])
def reset():
    global squat_count, report_ready, feedback_text, report_image_path
    global angle_history, feedback_history
    squat_count = 0
    report_ready = False
    feedback_text = ""
    report_image_path = ""
    angle_history = []
    feedback_history = []
    return redirect(url_for('squat.index'))
