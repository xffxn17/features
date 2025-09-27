"""
all_in_one_activity.py

All-in-one human activity prototype:
- MediaPipe Pose for body
- MediaPipe Hands for gestures (finger count, thumbs up, fist, pointing)
- MediaPipe FaceDetection for face bbox + simple emotion heuristics
- Draw green bounding box around person
- Sliding window of pose keypoints for action recognition (TFLite optional)
- Heuristic fallback classifiers included

Run:
    python all_in_one_activity.py
    python all_in_one_activity.py --model action_lstm.tflite

Notes:
- Use Python 3.10 in a venv on Windows for MediaPipe compatibility.
- The heuristics are simple but fast; train a TFLite model for real action recognition.
"""

import argparse
import collections
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---- CONFIG ----
WINDOW_SIZE = 30      # frames for action recognition
KEYPOINTS = 33        # BlazePose landmarks
LABELS = ["standing", "sitting", "walking", "waving", "unknown"]
FPS_AVG_ALPHA = 0.9

# ---- TRY LOAD TFLITE INTERPRETER ----
try:
    import tflite_runtime.interpreter as tflite_rt
    TFLITE_AVAILABLE = True
    TFLITE_INTERPRETER = tflite_rt.Interpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter as TFLITE_INTERPRETER
        TFLITE_AVAILABLE = True
    except Exception:
        TFLITE_AVAILABLE = False
        TFLITE_INTERPRETER = None


def load_tflite_model(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print("TFLite model not found:", path)
        return None
    interp = TFLITE_INTERPRETER(model_path=str(p))
    interp.allocate_tensors()
    return interp


def tflite_predict(interp, input_data):
    # input_data shaped (1, seq, features)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    # Resize dynamic dims if needed
    try:
        interp.resize_tensor_input(input_details[0]["index"], input_data.shape)
        interp.allocate_tensors()
    except Exception:
        pass
    interp.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
    interp.invoke()
    out = interp.get_tensor(output_details[0]["index"])
    return out

# ---- UTILITIES: bounding box & pose vector ----


def landmarks_to_bbox(lms, w, h, pad=0.08):
    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    minx = max(0.0, min(xs) - pad*(max(xs)-min(xs)))
    miny = max(0.0, min(ys) - pad*(max(ys)-min(ys)))
    maxx = min(1.0, max(xs) + pad*(max(xs)-min(xs)))
    maxy = min(1.0, max(ys) + pad*(max(ys)-min(ys)))
    return int(minx*w), int(miny*h), int(maxx*w), int(maxy*h)


def pose_landmarks_to_vector(lms):
    # flattened [x,y,z] per landmark, length KEYPOINTS * 3
    arr = []
    for lm in lms:
        arr.extend([lm.x, lm.y, lm.z if hasattr(lm, "z") else 0.0])
    return np.array(arr, dtype=np.float32)

# ---- HAND GESTURE HEURISTICS (fast) ----


def hand_gesture_from_landmarks(hand_landmarks, handedness):
    """
    Simple heuristics:
    - finger_count: 0..5 (thumb included roughly)
    - gestures: 'fist', 'open', 'pointing', 'thumbs_up' (best-effort)
    """
    # landmark indices: 4 thumb tip, 8 index tip, 12 middle, 16 ring, 20 pinky
    tips = [4, 8, 12, 16, 20]
    coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    # compute whether tip is above (y smaller) than pip joint for fingers 8,12,16,20

    def is_finger_up(tip_idx, pip_idx):
        # smaller y = higher on image
        return coords[tip_idx][1] < coords[pip_idx][1]
    fingers = []
    # rough thumb: compare x to IP joint (handedness dependent)
    fingers.append(coords[4][0] < coords[3][0])
    fingers.append(is_finger_up(8, 6))
    fingers.append(is_finger_up(12, 10))
    fingers.append(is_finger_up(16, 14))
    fingers.append(is_finger_up(20, 18))
    count = sum(1 for f in fingers if f)
    # detect some gestures
    if count == 0:
        gesture = "fist"
    elif count == 5:
        gesture = "open"
    else:
        # thumbs up heuristic: thumb up and others down, and thumb x position relative to hand
        if fingers[0] and not any(fingers[1:]):
            # further check thumb direction: y lower (up) than wrist?
            wrist_y = coords[0][1]
            thumb_tip_y = coords[4][1]
            if thumb_tip_y < wrist_y:
                gesture = "thumbs_up"
            else:
                gesture = f"{count}_fingers"
        elif fingers[1] and not any(fingers[2:]):  # only index
            gesture = "pointing"
        else:
            gesture = f"{count}_fingers"
    return {"count": count, "gesture": gesture, "handedness": handedness.classification[0].label if handedness else "Unknown"}

# ---- FACE EMOTION HEURISTICS (fast, no model) ----


def face_emotion_from_landmarks(face_rect, frame_gray, face_crop=None):
    """
    Lightweight heuristics using facial region:
    - smile if mouth corners pulled (approx via pixel intensity / simple proxy)
    - mouth open (surprised) if mouth height is large relative to width
    This is not a trained model but gives quick feedback; for production use train a small model.
    """
    # We'll do simple pixel-based: detect mouth area brightness / distance heuristics if face_crop provided
    if face_crop is None:
        return "neutral"
    h, w = face_crop.shape[:2]
    # convert to gray if needed
    crop_gray = cv2.cvtColor(
        face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
    # approximate mouth by lower third of bbox
    mouth_region = crop_gray[int(h*0.6):int(h*0.9), int(w*0.25):int(w*0.75)]
    if mouth_region.size == 0:
        return "neutral"
    # simple mouth openness: count dark pixels (open mouth usually darker)
    mean_val = np.mean(mouth_region)
    std_val = np.std(mouth_region)
    # smile heuristic: left-right symmetry with slightly brighter corners – fragile; return simple classes
    if mean_val < 90 and std_val > 25:
        return "surprised"   # mouth dark + high variance
    if mean_val > 120:
        return "smile"       # bright mouth area -> likely grin (rough)
    return "neutral"

# ---- ACTION HEURISTIC (fallback) ----


def heuristic_action_classify(seq_np):
    # seq_np: (seq_len, features)
    if seq_np.shape[0] < 5:
        return "unknown"
    nose_y = seq_np[:, 1]
    var = float(np.var(nose_y))
    if var > 0.0008:
        # moving up/down -> walking or waving
        left_shoulder_x = seq_np[:, (11*3)]
        right_shoulder_x = seq_np[:, (12*3)]
        shoulder_var = float(np.var(left_shoulder_x - right_shoulder_x))
        if shoulder_var > 0.00025:
            return "walking"
        return "waving"
    else:
        left_hip_y = float(np.mean(seq_np[:, (23*3+1)]))
        right_hip_y = float(np.mean(seq_np[:, (24*3+1)]))
        hip_y = (left_hip_y + right_hip_y) / 2
        if hip_y < 0.6:
            return "standing"
        else:
            return "sitting"

# ---- MAIN REAL-TIME LOOP ----


def run_all(model_path=None, camera_id=0, show_fps=True):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        enable_segmentation=False, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face = mp_face.FaceDetection(min_detection_confidence=0.5)

    interpreter = load_tflite_model(model_path)
    use_tflite = interpreter is not None and TFLITE_AVAILABLE

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera", camera_id)
        return

    seq = collections.deque(maxlen=WINDOW_SIZE)
    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detectors
        pose_res = pose.process(rgb)
        hands_res = hands.process(rgb)
        face_res = face.process(rgb)

        action_label = "unknown"

        # If pose present: draw landmarks, compute bbox, append to sequence
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks.landmark
            # bbox covering pose
            x1, y1, x2, y2 = landmarks_to_bbox(lms, w, h, pad=0.12)
            # green bounding box for the PERSON
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # pose vector
            vec = pose_landmarks_to_vector(lms)
            seq.append(vec)

            # draw pose
            mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(
                                       color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 128, 255), thickness=1))

            # run action recognizer if enough frames
            if len(seq) == WINDOW_SIZE:
                seq_np = np.stack(seq, axis=0)  # (WINDOW, features)
                if use_tflite:
                    inp = seq_np.reshape(
                        1, seq_np.shape[0], seq_np.shape[1]).astype(np.float32)
                    out = tflite_predict(interpreter, inp)
                    pred_idx = int(np.argmax(out))
                    action_label = LABELS[pred_idx] if pred_idx < len(
                        LABELS) else "unknown"
                else:
                    action_label = heuristic_action_classify(seq_np)
            else:
                action_label = "warming_up"

            cv2.putText(frame, f"Action: {action_label}", (x1, max(10, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            seq.clear()
            cv2.putText(frame, "No person detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Hands heuristic gestures
        if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
            for i, hlm in enumerate(hands_res.multi_hand_landmarks):
                label = hands_res.multi_handedness[i]
                res = hand_gesture_from_landmarks(hlm, label)
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
                # show gesture text near wrist
                wrist = hlm.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                txt = f"{res['gesture']} ({res['count']})"
                cv2.putText(frame, txt, (cx+10, cy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # Face detection + emotion heuristic
        if face_res.detections:
            for det in face_res.detections:
                bbox = det.location_data.relative_bounding_box
                fx, fy, fw_rel, fh_rel = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                x, y, bw, bh = int(
                    fx*w), int(fy*h), int(fw_rel*w), int(fh_rel*h)
                # safe crop
                x0, y0 = max(0, x), max(0, y)
                x1c, y1c = min(w, x + bw), min(h, y + bh)
                face_crop = frame[y0:y1c,
                                  x0:x1c] if y1c > y0 and x1c > x0 else None
                emotion = face_emotion_from_landmarks(bbox, None, face_crop)
                cv2.rectangle(frame, (x0, y0), (x1c, y1c), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x0, max(
                    10, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # FPS display
        if show_fps:
            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps = FPS_AVG_ALPHA*fps + \
                    (1-FPS_AVG_ALPHA)*(1.0/dt) if fps else 1.0/dt
            last_time = now
            cv2.putText(frame, f"FPS: {int(fps)}", (
                20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("All-in-one Activity Detector (q to quit)", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    face.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to TFLite action model (optional).")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_all(args.model, camera_id=args.camera)
