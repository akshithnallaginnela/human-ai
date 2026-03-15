from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image
import math
import time
from collections import deque

app = Flask(__name__)
CORS(app)

# ── MediaPipe Setup ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Activity History (in-memory ring buffer) ──────────────────────────────────
activity_history = deque(maxlen=30)
frame_count = 0

# ── Landmark Indices ──────────────────────────────────────────────────────────
NOSE          = 0
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW    = 13; RIGHT_ELBOW    = 14
LEFT_WRIST    = 15; RIGHT_WRIST    = 16
LEFT_HIP      = 23; RIGHT_HIP      = 24
LEFT_KNEE     = 25; RIGHT_KNEE     = 26
LEFT_ANKLE    = 27; RIGHT_ANKLE    = 28


def lm(landmarks, idx):
    """Return (x, y, z, visibility) for a landmark index."""
    p = landmarks[idx]
    return p.x, p.y, p.z, p.visibility


def angle_between(a, b, c):
    """Angle (degrees) at point b formed by segments b→a and b→c."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot   = ax * cx + ay * cy
    cross = ax * cy - ay * cx
    return abs(math.degrees(math.atan2(abs(cross), dot)))


def torso_inclination(lms):
    """Angle of the torso from vertical (0 = upright, ~90 = lying)."""
    lsx, lsy, *_ = lm(lms, LEFT_SHOULDER)
    rsx, rsy, *_ = lm(lms, RIGHT_SHOULDER)
    lhx, lhy, *_ = lm(lms, LEFT_HIP)
    rhx, rhy, *_ = lm(lms, RIGHT_HIP)
    shoulder_cx = (lsx + rsx) / 2
    shoulder_cy = (lsy + rsy) / 2
    hip_cx      = (lhx + rhx) / 2
    hip_cy      = (lhy + rhy) / 2
    dx = shoulder_cx - hip_cx
    dy = shoulder_cy - hip_cy
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


def classify_activity(lms):
    """Rule-based activity classification from pose landmarks."""
    try:
        lax, lay, *_ = lm(lms, LEFT_ANKLE)
        rax, ray, *_ = lm(lms, RIGHT_ANKLE)
        lkx, lky, *_ = lm(lms, LEFT_KNEE)
        rkx, rky, *_ = lm(lms, RIGHT_KNEE)
        lhx, lhy, *_ = lm(lms, LEFT_HIP)
        rhx, rhy, *_ = lm(lms, RIGHT_HIP)
        lsx, lsy, *_ = lm(lms, LEFT_SHOULDER)
        rsx, rsy, *_ = lm(lms, RIGHT_SHOULDER)
        lex, ley, *_ = lm(lms, LEFT_ELBOW)
        rex, rey, *_ = lm(lms, RIGHT_ELBOW)
        lwx, lwy, *_ = lm(lms, LEFT_WRIST)
        rwx, rwy, *_ = lm(lms, RIGHT_WRIST)
        nx,  ny,  *_ = lm(lms, NOSE)

        # Computed values
        incl        = torso_inclination(lms)
        hip_cy      = (lhy + rhy) / 2
        knee_cy     = (lky + rky) / 2
        ankle_cy    = (lay + ray) / 2
        shoulder_cy = (lsy + rsy) / 2

        left_knee_angle  = angle_between((lhx, lhy), (lkx, lky), (lax, lay))
        right_knee_angle = angle_between((rhx, rhy), (rkx, rky), (rax, ray))
        avg_knee_angle   = (left_knee_angle + right_knee_angle) / 2

        left_elbow_angle  = angle_between((lsx, lsy), (lex, ley), (lwx, lwy))
        right_elbow_angle = angle_between((rsx, rsy), (rex, rey), (rwx, rwy))

        wrist_avg_y = (lwy + rwy) / 2

        # ── Lying Down ────────────────────────────────────────────
        if incl > 55:
            return "Lying Down", 0.90, "🛌"

        # ── Sitting ───────────────────────────────────────────────
        if avg_knee_angle < 120 and hip_cy < knee_cy:
            return "Sitting", 0.87, "🪑"

        # ── Squatting ─────────────────────────────────────────────
        if avg_knee_angle < 110 and hip_cy > knee_cy * 0.85:
            return "Squatting", 0.83, "🏋️"

        # ── Push-up ───────────────────────────────────────────────
        if incl > 30 and avg_knee_angle > 150 and shoulder_cy > hip_cy:
            return "Push-up", 0.80, "💪"

        # ── Arms Raised ───────────────────────────────────────────
        if wrist_avg_y < shoulder_cy - 0.05:
            return "Arms Raised", 0.85, "🙌"

        # ── Walking / Running heuristic (hip sway) ────────────────
        hip_diff = abs(lhy - rhy)
        if hip_diff > 0.04:
            if hip_diff > 0.08:
                return "Running", 0.78, "🏃"
            return "Walking", 0.75, "🚶"

        # ── Standing (default) ────────────────────────────────────
        return "Standing", 0.88, "🧍"

    except Exception:
        return "Unknown", 0.0, "❓"


def extract_keypoints(lms):
    """Return 33×3 flat list of (x, y, z) normalised coords."""
    pts = []
    for lmk in lms:
        pts.extend([round(lmk.x, 4), round(lmk.y, 4), round(lmk.z, 4)])
    return pts


def decode_image(b64_string):
    """Decode a base64 data-URL or raw base64 string to an OpenCV BGR image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def encode_image(cv_img):
    """Encode an OpenCV BGR image to a base64 JPEG data-URL."""
    _, buf    = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64       = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Human Activity Detection API is running"})


@app.route("/detect", methods=["POST"])
def detect():
    global frame_count
    data = request.get_json(force=True)
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        frame        = decode_image(data["image"])
        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results      = pose.process(rgb_frame)
        frame_count += 1

        if not results.pose_landmarks:
            return jsonify({
                "activity":    "No Person Detected",
                "confidence":  0,
                "emoji":       "👁️",
                "landmarks":   [],
                "annotated_image": encode_image(frame),
                "history":     list(activity_history),
                "frame":       frame_count
            })

        # Classify
        activity, confidence, emoji = classify_activity(results.pose_landmarks.landmark)

        # Record history
        activity_history.append({
            "activity":   activity,
            "confidence": round(confidence, 2),
            "emoji":      emoji,
            "timestamp":  time.time(),
            "frame":      frame_count
        })

        # Draw skeleton
        annotated = frame.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
        )

        # Overlay label
        h, w = annotated.shape[:2]
        label_text = f"{emoji} {activity} ({int(confidence*100)}%)"
        cv2.putText(annotated, label_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)

        return jsonify({
            "activity":        activity,
            "confidence":      round(confidence, 2),
            "emoji":           emoji,
            "landmarks":       extract_keypoints(results.pose_landmarks.landmark),
            "annotated_image": encode_image(annotated),
            "history":         list(activity_history)[-10:],
            "frame":           frame_count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    return jsonify({
        "history":      list(activity_history),
        "total_frames": frame_count
    })


@app.route("/reset", methods=["POST"])
def reset():
    global frame_count
    activity_history.clear()
    frame_count = 0
    return jsonify({"message": "Session reset"})


if __name__ == "__main__":
    print("🚀 Human Activity Detection API starting on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
