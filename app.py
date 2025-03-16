import yaml
from flask import Flask, Response, render_template
import cv2
import numpy as np
from utils.mediapipe import MediaPipePose
from utils.classifier import ClassifierOnlineTest
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading

# Загрузка конфигурации
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

app = Flask(__name__)

# Инициализация модели позы
pose_config = config["POSE"]
pose_estimator = MediaPipePose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_estimator.min_total_joints = pose_config["min_total_joints"]
pose_estimator.min_leg_joints = pose_config["min_leg_joints"]
pose_estimator.include_head = pose_config["include_head"]

# Инициализация классификатора действий
classifier_config = config["CLASSIFIER"]
action_classifier = ClassifierOnlineTest(
    model_path="/".join(classifier_config["model_path"]),
    action_labels=classifier_config["classes"],
    window_size=classifier_config["window_size"],
    threshold=classifier_config["threshold"]
)

# Инициализация трекера
tracker_config = config["TRACKER"]
object_tracker = DeepSort(
    max_cosine_distance=tracker_config["max_dist"],  # Исправлено
    max_iou_distance=tracker_config["max_iou_distance"],
    max_age=tracker_config["max_age"],
    n_init=tracker_config["n_init"],
    nn_budget=tracker_config["nn_budget"]
)

frame_lock = threading.Lock()

def process_frame(frame):
    with frame_lock:
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_estimator.pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks  # Исправлено: убрана итерация
                pose_estimator.mp_drawing.draw_landmarks(
                    frame, landmarks,
                    pose_estimator.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_estimator.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=pose_estimator.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                keypoints = np.zeros((18, 3), dtype=np.float64)
                mapping = {0: 0, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16}
                for mp_idx, coco_idx in mapping.items():
                    if mp_idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[mp_idx]
                        keypoints[coco_idx] = np.array([coco_idx, landmark.y, landmark.x])
                
                action = action_classifier.predict(keypoints)
                if action[0]:
                    label = f"Action: {action[0]} ({action[1]:.2f})"
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                keypoints_xy = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in landmarks.landmark])
                x_min, y_min = np.min(keypoints_xy, axis=0)
                x_max, y_max = np.max(keypoints_xy, axis=0)
                bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                
                detections = [(bbox, 0.9, None)]
                tracks = object_tracker.update_tracks(detections, frame=frame)
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    x1, y1, x2, y2 = track.to_tlbr()
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
        except Exception as e:
            print(f"Ошибка при обработке кадра: {e}")
    
    return frame

def generate_frames():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось захватить кадр")
            break
        processed_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)