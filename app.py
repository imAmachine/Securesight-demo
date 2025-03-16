from flask import Flask, Response, render_template
import cv2
import numpy as np
from utils.mediapipe import MediaPipePose
from utils.classifier import ClassifierOnlineTest
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading

app = Flask(__name__)

# Инициализация моделей
pose_estimator = MediaPipePose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

action_classifier = ClassifierOnlineTest(
    model_path="utils/weights/action_classifier3.pt",
    action_labels=['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave'],
    window_size=5,
    threshold=0.5
)

object_tracker = DeepSort(max_age=70, n_init=5)

# Блокировка для потокобезопасного доступа к обработке кадров
frame_lock = threading.Lock()

def process_frame(frame):
    with frame_lock:
        try:
            # Обработка кадра с помощью MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_estimator.pose.process(image_rgb)
            
            # Отрисовка скелета
            if results.pose_landmarks:
                # Отрисовка скелета средствами MediaPipe
                pose_estimator.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks,
                    pose_estimator.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_estimator.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=pose_estimator.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Преобразование в формат COCO для классификатора
                keypoints = np.zeros((18, 3), dtype=np.float64)
                mapping = {
                    0: 0,  # nose
                    11: 5, # left shoulder
                    12: 6, # right shoulder
                    13: 7, # left elbow
                    14: 8, # right elbow
                    15: 9, # left wrist
                    16: 10, # right wrist
                    23: 11, # left hip
                    24: 12, # right hip
                    25: 13, # left knee
                    26: 14, # right knee
                    27: 15, # left ankle
                    28: 16, # right ankle
                }
                
                for mp_idx, coco_idx in mapping.items():
                    if mp_idx < len(results.pose_landmarks.landmark):
                        landmark = results.pose_landmarks.landmark[mp_idx]
                        keypoints[coco_idx] = np.array([coco_idx, landmark.y, landmark.x])
                
                # Временно отключим классификацию для отладки
                action = ["debugging", 0.9]
                # Раскомментируйте следующую строку, когда будете готовы тестировать классификацию
                # action = action_classifier.predict(keypoints)
                
                if action[0]:
                    label = f"Action: {action[0]} ({action[1]:.2f})"
                    cv2.putText(frame, label, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Расчет bbox для трекера
                keypoints_xy = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                                    for lm in results.pose_landmarks.landmark])
                x_min, y_min = np.min(keypoints_xy, axis=0)
                x_max, y_max = np.max(keypoints_xy, axis=0)
                
                # Подготовка данных для трекера
                bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                
                # Обновление трекера
                try:
                    detections = [(bbox, 0.9, None)]  # формат: bbox, confidence, feature
                    tracks = object_tracker.update_tracks(detections, frame=frame)
                    
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        
                        track_id = track.track_id
                        x1, y1, x2, y2 = track.to_tlbr()
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Ошибка при обновлении трекера: {e}")
                
        except Exception as e:
            print(f"Ошибка при обработке кадра: {e}")
            import traceback
            traceback.print_exc()
    
    return frame

def generate_frames():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось захватить кадр")
            break

        # Обработка кадра
        processed_frame = process_frame(frame)

        # Конвертация кадра в JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)