# utils.py
import matplotlib
matplotlib.use('Agg')
import math
import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def count_repetitions(landmarks, prev_landmarks, count):
    if prev_landmarks and hasattr(prev_landmarks, 'landmark'):
        left_knee_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
        )
        prev_left_knee_angle = calculate_angle(
            prev_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
            prev_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
            prev_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value],
        )
        if left_knee_angle < 100 and prev_left_knee_angle > 160:
            count += 1
    return count

def is_start_pose(pose_landmarks):
    if pose_landmarks and hasattr(pose_landmarks, 'landmark'):
        left_knee_angle = calculate_angle(
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
            pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value],
        )
        right_knee_angle = calculate_angle(
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        )
        if left_knee_angle > 160 and right_knee_angle > 160:
            return True
    return False

def is_pose_detected(landmarks):
    if landmarks and hasattr(landmarks, 'landmark'):
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ]
        for idx in required_landmarks:
            if not landmarks.landmark[idx]:
                return False
        return True
    return False

def visualize_feedback(frame, feedback, landmarks, progress=None, down_threshold=None, up_threshold=None):
    """
    왼쪽 상단에 작게 피드백(초록/빨간) + 게이지 바 표시
    """
    overlay = frame.copy()
    feedback_box_width = 150   # 좁은 폭
    feedback_box_height = 25 * (len(feedback) + 2)

    # 왼쪽 상단 (0,0)
    cv2.rectangle(overlay, (0, 0), (feedback_box_width, feedback_box_height), (50, 50, 50), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y_pos = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45  # 폰트 작게
    thickness = 1

    for key, value in feedback.items():
        good_values = ["Ok", "Good", "Stable", "Wide", "Narrow",
                       "Full Squat", "Parallel Squat", "Knee behind ankle", "Deep Squat"]
        is_good = (value in good_values)
        status_text = "OK" if is_good else "Check"
        color = (0, 255, 0) if is_good else (0, 0, 255)
        text = f"{key}: {value} ({status_text})"
        cv2.putText(frame, text, (5, y_pos), font, font_scale, color, thickness)
        y_pos += 20

    # 진행률 바(게이지바)도 작게
    if progress is not None:
        progress_bar_x = 5
        progress_bar_y = y_pos + 5
        progress_bar_width = feedback_box_width - 10
        progress_bar_height = 12
        cv2.rectangle(frame, (progress_bar_x, progress_bar_y),
                      (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height),
                      (100, 100, 100), -1)
        filled_width = int(progress_bar_width * min(progress, 1.0))
        cv2.rectangle(frame, (progress_bar_x, progress_bar_y),
                      (progress_bar_x + filled_width, progress_bar_y + progress_bar_height),
                      (0, 255, 0), -1)
        cv2.rectangle(frame, (progress_bar_x, progress_bar_y),
                      (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height),
                      (255, 255, 255), 1)
        percentage_text = f"{int(progress * 100)}%"
        cv2.putText(frame, percentage_text, (progress_bar_x + 3, progress_bar_y + 10),
                    font, 0.4, (255, 255, 255), 1)

    # 랜드마크 시각화 (기존 로직 유지)
    if landmarks and hasattr(landmarks, 'landmark'):
        for idx, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            color = (0, 255, 0)
            if feedback.get("Knee Align (Side)") not in ["Good", ""] and idx in [
                mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value
            ]:
                color = (0, 0, 255)
                cv2.putText(frame, "!?", (x - 10, y - 10), font, 0.45, color, 1)
            cv2.circle(frame, (x, y), 3, color, -1)

    return frame


def calculate_angle_op(a, b, c):
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_speed_fast(landmarks, prev_landmarks):
    if prev_landmarks and hasattr(prev_landmarks, 'landmark'):
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        prev_left_knee = prev_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
        distance = calculate_distance(left_knee, prev_left_knee)
        speed = distance / 0.033  # 약 30 FPS 기준
        if speed > 0.05:
            return True
    return False

def draw_prepared_pose(frame):
    height, width, _ = frame.shape
    cv2.line(frame, (width // 4, height // 2), (width // 4, height), (0, 255, 0), 2)
    cv2.line(frame, (width * 3 // 4, height // 2), (width * 3 // 4, height), (0, 255, 0), 2)
    cv2.circle(frame, (width // 4, height // 2), 5, (0, 255, 0), -1)
    cv2.circle(frame, (width * 3 // 4, height // 2), 5, (0, 255, 0), -1)
    return frame

# ===== 신규 추가 함수: Precise Starting Pose 평가 =====
def is_precise_start_pose(landmarks, tolerance=0.05):
    """
    시작 자세에서 왼쪽/오른쪽 귀, 어깨, 엉덩이, 발목 중앙의 x 좌표가 
    일정 오차(tolerance) 이내에 있는지 평가합니다.
    """
    try:
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        ear_center = (left_ear.x + right_ear.x) / 2.0
        shoulder_center = (left_shoulder.x + right_shoulder.x) / 2.0
        hip_center = (left_hip.x + right_hip.x) / 2.0
        ankle_center = (left_ankle.x + right_ankle.x) / 2.0

        if (abs(ear_center - shoulder_center) > tolerance or
            abs(shoulder_center - hip_center) > tolerance or
            abs(hip_center - ankle_center) > tolerance):
            return False
        else:
            return True
    except Exception as e:
        return False

# ===== 수정된 analyze_squat 함수 =====
def analyze_squat(landmarks, prev_landmarks):
    """
    기존 분석 로직에 아래 항목들을 추가합니다.
      1. 시작 자세 정밀 평가 (귀, 어깨, 엉덩이, 발목 수직 정렬)
      2. 척추 정렬 체크 (어깨 중앙과 엉덩이 중앙 비교)
      3. 엉덩이 움직임의 세밀한 분석 (하강/상승 시 hip의 x 좌표 변화)
      4. 골반 수평 체크 (좌우 엉덩이 y 좌표 차이)
      6. 스쿼트 깊이 평가 보완 (hip의 y 위치 활용)
    """
    feedback = {}
    avg_knee_angle = None

    if landmarks:
        # 주요 랜드마크 취득
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Ankle Stability
        left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        right_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        if left_ankle.y > left_foot_index.y or right_ankle.y > right_foot_index.y:
            feedback["Ankle Stability"] = "Unstable"
        else:
            feedback["Ankle Stability"] = "Stable"

        # Hip Hinge 체크 (간단 비교)
        hip_x = (left_hip.x + right_hip.x) / 2.0
        knee_x = (left_knee.x + right_knee.x) / 2.0
        if (knee_x - hip_x) > 0.10:
            feedback["Hip Hinge"] = "Excessive hip hinge"
        else:
            feedback["Hip Hinge"] = "Ok"

        # 무릎 각도 기반 Squat Depth 평가
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # 향상된 Squat Depth 평가: hip의 y 위치도 고려 (hip이 knee보다 낮은 정도)
        hip_y = (left_hip.y + right_hip.y) / 2.0
        knee_y = (left_knee.y + right_knee.y) / 2.0
        if avg_knee_angle < 90:
            if hip_y > knee_y + 0.05:
                feedback["Squat Depth"] = "Deep Squat"
            else:
                feedback["Squat Depth"] = "Full Squat"
        elif avg_knee_angle < 135:
            if abs(hip_y - knee_y) < 0.05:
                feedback["Squat Depth"] = "Parallel Squat"
            else:
                feedback["Squat Depth"] = "Full Squat"
        else:
            feedback["Squat Depth"] = "Partial Squat"

        # Knee Alignment (Side)
        if left_knee.x < left_ankle.x - 0.05:
            feedback["Knee Align (Side)"] = "Knee far forward"
        elif left_knee.x > left_ankle.x + 0.05:
            feedback["Knee Align (Side)"] = "Knee behind ankle"
        else:
            feedback["Knee Align (Side)"] = "Good"
        if right_knee.x < right_ankle.x - 0.05:
            feedback["Knee Align (Side)"] = "Knee far forward"
        elif right_knee.x > right_ankle.x + 0.05:
            feedback["Knee Align (Side)"] = "Knee behind ankle"

        # Trunk Lean 체크
        if (left_shoulder.y - left_hip.y) < -0.15 or (right_shoulder.y - right_hip.y) < -0.15:
            feedback["Trunk Lean"] = "Excessive forward lean"
        else:
            feedback["Trunk Lean"] = "Good"

        # Movement Speed 체크
        feedback["Movement Speed"] = "N/A"
        if prev_landmarks:
            if is_speed_fast(landmarks, prev_landmarks):
                feedback["Movement Speed"] = "Too fast"
            else:
                feedback["Movement Speed"] = "Ok"

        # Arm Position 체크
        if left_shoulder.y < 0.6 or right_shoulder.y < 0.6:
            feedback["Arm Position"] = "Stable"
        else:
            feedback["Arm Position"] = "Check arms"

        # Overall Posture 평가
        feedback["Posture"] = "Needs evaluation"

        # 추가 체크 항목

        # 1. 시작 자세 정밀 평가 (귀, 어깨, 엉덩이, 발목 중앙 x 좌표 비교)
        if is_precise_start_pose(landmarks):
            feedback["Starting Pose Alignment"] = "Good"
        else:
            feedback["Starting Pose Alignment"] = "Misaligned starting pose"

        # 2. 척추 정렬 체크 (어깨 중앙과 엉덩이 중앙 비교)
        shoulder_center = (left_shoulder.x + right_shoulder.x) / 2.0
        hip_center = (left_hip.x + right_hip.x) / 2.0
        if abs(shoulder_center - hip_center) > 0.05:
            feedback["Spine Alignment"] = "Misaligned spine"
        else:
            feedback["Spine Alignment"] = "Good"

        # 3. 엉덩이 움직임의 세밀한 분석 (하강/상승 시 hip의 x 좌표 변화)
        if prev_landmarks:
            prev_landmarks_list = prev_landmarks.landmark if hasattr(prev_landmarks, 'landmark') else prev_landmarks
            prev_left_hip = prev_landmarks_list[mp_pose.PoseLandmark.LEFT_HIP.value]
            prev_right_hip = prev_landmarks_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
            prev_hip_x = (prev_left_hip.x + prev_right_hip.x) / 2.0

            prev_left_knee = prev_landmarks_list[mp_pose.PoseLandmark.LEFT_KNEE.value]
            prev_right_knee = prev_landmarks_list[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            prev_left_ankle = prev_landmarks_list[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            prev_right_ankle = prev_landmarks_list[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            prev_left_knee_angle = calculate_angle(prev_left_hip, prev_left_knee, prev_left_ankle)
            prev_right_knee_angle = calculate_angle(prev_right_hip, prev_right_knee, prev_right_ankle)
            prev_avg_knee_angle = (prev_left_knee_angle + prev_right_knee_angle) / 2.0

            if avg_knee_angle < prev_avg_knee_angle:
                if hip_x > prev_hip_x + 0.01:
                    feedback["Hip Movement"] = "Hip not moving backward during descent"
                else:
                    feedback["Hip Movement"] = "Good"
            elif avg_knee_angle > prev_avg_knee_angle:
                if hip_x < prev_hip_x - 0.01:
                    feedback["Hip Movement"] = "Hip not moving forward during ascent"
                else:
                    feedback["Hip Movement"] = "Good"
            else:
                feedback["Hip Movement"] = "Stable"
        else:
            feedback["Hip Movement"] = "N/A"

        # 4. 골반 수평 체크 (좌우 엉덩이 y 좌표 차이)
        if abs(left_hip.y - right_hip.y) > 0.05:
            feedback["Pelvis Alignment"] = "Pelvis not level"
        else:
            feedback["Pelvis Alignment"] = "Good"

    return feedback, avg_knee_angle

def generate_angle_plot(angle_history):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform

    # 한글 폰트 설정 (운영체제에 따라 다르게 처리)
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False

    if not angle_history:
        return ""

    plt.figure(figsize=(8, 4))
    plt.plot(angle_history, label="무릎 각도 (°)", color='royalblue', linewidth=2)
    plt.axhline(90, color='red', linestyle='--', linewidth=1.5, label='기준선: 90°')

    plt.title("스쿼트 중 무릎 각도 변화", fontsize=14, pad=15)
    plt.xlabel("시간 흐름 (프레임)", fontsize=11)
    plt.ylabel("무릎 각도 (도)", fontsize=11)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()

    save_path = "static/images/knee_plot.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


def extract_major_issues(feedback_history):
    """
    여러 프레임에서 등장한 '나쁜 피드백' 항목을 집계해서 핵심 문제 도출
    """
    from collections import Counter
    bad_keywords = ["Check", "Unstable", "Too fast", "Misaligned", "Pelvis not level"]

    issue_list = []

    for feedback in feedback_history:
        for key, value in feedback.items():
            if any(bad in value for bad in bad_keywords):
                issue_list.append(key)

    counter = Counter(issue_list)
    return [f"{k} ({v}회)" for k, v in counter.most_common(3)]
