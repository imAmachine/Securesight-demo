import cv2
import threading
import numpy as np

from utils.action_classifier import get_classifier
from utils.pose_estimation import get_pose_estimator
from utils.tracker import get_tracker
from utils.utils.config import Config
from utils.utils.drawer import Drawer
from utils.utils.utils import convert_to_openpose_skeletons
from utils.emotion_classifier import get_emotion_classifier

class ActionDetectionSystem:
    def __init__(self, config_path="config.yaml", max_objects=8):
        # Загрузка конфигурации
        self.config = Config(config_path)
        self.initialize_models()
        self.frame_lock = threading.Lock()
        self.max_objects = max_objects
        self.processing_methods = {
            'pose': self.process_pose_only,
            'action': self.process_action,
            'emotion': self.process_emotion,
            'full': self.process_full
        }
        self.current_method = 'action'
    
    def initialize_models(self):
        # Инициализация моделей
        self.pose_estimator = get_pose_estimator(**self.config.POSE)
        self.tracker = get_tracker(**self.config.TRACKER)
        self.action_classifier = get_classifier(**self.config.CLASSIFIER)
        self.emotion_classifier = get_emotion_classifier(**self.config.get('EMOTION', {}))
        self.drawer = Drawer()
    
    def set_processing_method(self, method_name):
        if method_name in self.processing_methods:
            self.current_method = method_name
            print(f"Switched processing method to: {method_name}")
            return True
        return False
    
    def predict_poses(self, rgb_frame):
        predictions = self.pose_estimator.predict(rgb_frame, get_bbox=True)
        if len(predictions) == 0:
            self.tracker.increment_ages()
            return []
        return self.limit_predictions(predictions)
    
    def limit_predictions(self, predictions):
        # Ограничение количества объектов
        if len(predictions) > self.max_objects:
            predictions = predictions[:self.max_objects]
        return predictions
    
    def normalize_bboxes(self, predictions):
        # Преобразование всех bbox в numpy массивы
        for pred in predictions:
            if hasattr(pred, "bbox") and pred.bbox is not None:
                if not isinstance(pred.bbox, np.ndarray):
                    pred.bbox = np.array(pred.bbox)
        return predictions
    
    def track_objects(self, rgb_frame, predictions):
        predictions = convert_to_openpose_skeletons(predictions)
        
        try:
            predictions, _ = self.tracker.predict(rgb_frame, predictions)
            return predictions
        except Exception as e:
            print(f"Ошибка трекера: {e}")
            self.tracker.increment_ages()
            return predictions
    
    def classify_actions(self, predictions):
        if len(predictions) == 0:
            return predictions
            
        try:
            predictions = self.action_classifier.classify(predictions)
        except Exception as e:
            print(f"Ошибка классификатора действий: {e}")
        
        return predictions
    
    def classify_emotions(self, rgb_frame, predictions):
        if len(predictions) == 0:
            return predictions
            
        try:
            # Предполагаем, что классификатор эмоций принимает кадр и предсказания с bbox
            predictions = self.emotion_classifier.classify(rgb_frame, predictions)
        except Exception as e:
            print(f"Ошибка классификатора эмоций: {e}")
        
        return predictions
    
    def render_frame(self, rgb_frame, predictions):
        try:
            # Use the drawer instance instead of calling the method directly
            annotated_frame = self.drawer.render_frame(
                rgb_frame, 
                predictions, 
                text_color='green', 
                add_blank=False
            )
            return annotated_frame
        except Exception as e:
            print(f"Ошибка рендеринга: {e}")
            return rgb_frame
    
    def process_pose_only(self, rgb_frame):
        # Только определение позы
        predictions = self.predict_poses(rgb_frame)
        if len(predictions) == 0:
            return rgb_frame
        
        predictions = self.normalize_bboxes(predictions)
        predictions = self.track_objects(rgb_frame, predictions)
        return self.render_frame(rgb_frame, predictions)
    
    def process_action(self, rgb_frame):
        # Определение позы + классификация действий
        predictions = self.predict_poses(rgb_frame)
        if len(predictions) == 0:
            return rgb_frame
        
        predictions = self.normalize_bboxes(predictions)
        predictions = self.track_objects(rgb_frame, predictions)
        predictions = self.classify_actions(predictions)
        predictions = self.normalize_bboxes(predictions)
        return self.render_frame(rgb_frame, predictions)
    
    def process_emotion(self, rgb_frame):
        # Определение позы + классификация эмоций
        predictions = self.predict_poses(rgb_frame)
        if len(predictions) == 0:
            return rgb_frame
        
        predictions = self.normalize_bboxes(predictions)
        predictions = self.track_objects(rgb_frame, predictions)
        predictions = self.classify_emotions(rgb_frame, predictions)
        predictions = self.normalize_bboxes(predictions)
        return self.render_frame(rgb_frame, predictions)
    
    def process_full(self, rgb_frame):
        # Полная обработка: поза + действия + эмоции
        predictions = self.predict_poses(rgb_frame)
        if len(predictions) == 0:
            return rgb_frame
        
        predictions = self.normalize_bboxes(predictions)
        predictions = self.track_objects(rgb_frame, predictions)
        predictions = self.classify_actions(predictions)
        predictions = self.classify_emotions(rgb_frame, predictions)
        predictions = self.normalize_bboxes(predictions)
        return self.render_frame(rgb_frame, predictions)
    
    def process_frame(self, rgb_frame):
        # Выбор метода обработки
        return self.processing_methods[self.current_method](rgb_frame)


def generate_frames(system, camera_id=1):
    cap = cv2.VideoCapture(camera_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось захватить кадр")
            break
        
        processed_frame = system.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')