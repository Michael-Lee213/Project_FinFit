import utils
import mediapipe as mp

mp_pose = mp.solutions.pose

def analyze_squat(landmarks, prev_landmarks):
    """
    Foot Width, Ankle Stability, Knee Align (Front) 제거.
    Hip Hinge, Squat Depth, Knee Align (Side), Trunk Lean, Movement Speed, Arm Position, Posture만 유지.
    """
    feedback = {}
    avg_knee_angle = None

    if landmarks:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Hip Hinge
        hip_x = (left_hip.x + right_hip.x) / 2.0
        knee_x = (left_knee.x + right_knee.x) / 2.0
        if (knee_x - hip_x) > 0.10:
            feedback["Hip Hinge"] = "Excessive hip hinge"
        else:
            feedback["Hip Hinge"] = "Ok"

        # Squat Depth
        left_knee_angle = utils.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = utils.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        if avg_knee_angle < 90:
            feedback["Squat Depth"] = "Full Squat"
        elif avg_knee_angle < 135:
            feedback["Squat Depth"] = "Parallel Squat"
        else:
            feedback["Squat Depth"] = "Partial Squat"

        # Knee Align (Side)
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

        # Trunk Lean
        if (left_shoulder.y - left_hip.y) < -0.15 or (right_shoulder.y - right_hip.y) < -0.15:
            feedback["Trunk Lean"] = "Excessive forward lean"
        else:
            feedback["Trunk Lean"] = "Good"

        # Movement Speed
        feedback["Movement Speed"] = "N/A"
        if prev_landmarks:
            if utils.is_speed_fast(landmarks, prev_landmarks):
                feedback["Movement Speed"] = "Too fast"
            else:
                feedback["Movement Speed"] = "Ok"

        # Arm Position
        if left_shoulder.y < 0.6 or right_shoulder.y < 0.6:
            feedback["Arm Position"] = "Stable"
        else:
            feedback["Arm Position"] = "Check arms"

        # Overall Posture
        feedback["Posture"] = "Needs evaluation"

    return feedback, avg_knee_angle
