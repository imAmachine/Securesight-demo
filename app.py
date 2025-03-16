import yaml
from flask import Flask, Response, render_template
import cv2
import threading
import numpy as np

from utils.action_classifier import get_classifier
from utils.pose_estimation import get_pose_estimator
from utils.tracker import get_tracker
from utils.utils.config import Config
from utils.utils.drawer import Drawer
from utils.utils.utils import convert_to_openpose_skeletons

# Загрузка конфигурации
config = Config("config.yaml")

app = Flask(__name__)

# Инициализация модели позы
pose_estimator = get_pose_estimator(**config.POSE)
tracker = get_tracker(**config.TRACKER)
action_classifier = get_classifier(**config.CLASSIFIER)

frame_lock = threading.Lock()

def process_frame(rgb_frame):
    predictions = pose_estimator.predict(rgb_frame, get_bbox=True)
    if len(predictions) == 0:
        tracker.increment_ages()
        return rgb_frame

    # Ограничение количества объектов
    max_objects = 8  # Начните с небольшого числа
    if len(predictions) > max_objects:
        predictions = predictions[:max_objects]
    
    # Преобразование всех bbox в numpy массивы
    for pred in predictions:
        if hasattr(pred, "bbox") and pred.bbox is not None:
            if not isinstance(pred.bbox, np.ndarray):
                pred.bbox = np.array(pred.bbox)

    predictions = convert_to_openpose_skeletons(predictions)
    
    try:
        # Обернем вызов трекера в try-except
        predictions, _ = tracker.predict(rgb_frame, predictions)
    except Exception as e:
        print(f"Ошибка трекера: {e}")
        # В случае ошибки трекера, просто увеличиваем возраст треков
        tracker.increment_ages()
    
    if len(predictions) > 0:
        try:
            predictions = action_classifier.classify(predictions)
        except Exception as e:
            print(f"Ошибка классификатора: {e}")
    
    # Еще раз проверяем bbox перед рендерингом
    for pred in predictions:
        if hasattr(pred, "bbox") and pred.bbox is not None:
            if not isinstance(pred.bbox, np.ndarray):
                pred.bbox = np.array(pred.bbox)
    
    drawer = Drawer()
    try:
        annotated_frame = drawer.render_frame(rgb_frame, predictions, text_color='green', add_blank=False)
    except Exception as e:
        print(f"Ошибка рендеринга: {e}")
        # В случае ошибки рендеринга возвращаем исходный кадр
        return rgb_frame
    
    return annotated_frame

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