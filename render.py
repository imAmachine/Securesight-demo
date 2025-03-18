import cv2
import threading
import numpy as np
import platform
import emoji
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace

from utils.action_classifier import get_classifier
from utils.pose_estimation import get_pose_estimator
from utils.tracker import get_tracker
from utils.utils.config import Config
from utils.utils.drawer import Drawer
from utils.utils.utils import convert_to_openpose_skeletons

class ActionDetectionSystem:
    def __init__(self, config_path="config.yaml", max_objects=8):
        # Загрузка конфигурации
        self.config = Config(config_path)
        self.current_objects = []
        self.initialize_models()
        self.frame_lock = threading.Lock()
        self.max_objects = max_objects
        
        # Словарь для перевода действий на русский язык
        self.action_names = {
            'stand': 'Стоит',
            'walk': 'Ходьба',
            'run': 'Бег',
            'jump': 'Прыжок',
            'sit': 'Сидит',
            'squat': 'Приседает',
            'kick': 'Удар ногой',
            'punch': 'Удар рукой',
            'wave': 'Махание',
            'unknown': 'Неизвестно'
        }
    
    def initialize_models(self):
        # Инициализация моделей
        self.pose_estimator = get_pose_estimator(**self.config.POSE)
        self.tracker = get_tracker(**self.config.TRACKER)
        self.action_classifier = get_classifier(**self.config.CLASSIFIER)
        self.drawer = Drawer()
    
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
            # Добавляем русские названия действий
            for pred in predictions:
                if hasattr(pred, 'action') and pred.action and pred.action[0]:
                    action_name = pred.action[0]
                    confidence = pred.action[1]
                    # Переводим название действия на русский (если есть в словаре)
                    translated_name = self.action_names.get(action_name, action_name)
                    pred.action = (translated_name, confidence)
        except Exception as e:
            print(f"Ошибка классификатора: {e}")
        
        return predictions
    
    def render_frame(self, frame, predictions):
        try:
            return self.drawer.render_frame(
                frame, 
                predictions, 
                text_color='green', 
                add_blank=False
            )
        except Exception as e:
            print(f"Ошибка рендеринга: {e}")
            return frame
        
    def get_current_objects(self):
        objects_data = []
        with self.frame_lock:
            for pred in self.current_objects:
                try:
                    track_id = getattr(pred, 'id', 'N/A')
                    action_data = getattr(pred, 'action', ['unknown', 0.0])

                    obj = {
                        "id": track_id,
                        "action": str(action_data[0]),
                        "confidence": float(action_data[1])
                    }
                    objects_data.append(obj)
                except Exception as e:
                    print(f"Error processing prediction: {e}")
        return objects_data
    
    def process_frame(self, frame):
        if frame is None:
            return frame
            
        # Определение позы
        predictions = self.predict_poses(frame)
        if len(predictions) == 0:
            with self.frame_lock:
                self.current_objects = []
            return frame
        
        # Нормализация bbox
        predictions = self.normalize_bboxes(predictions)
        
        # Трекинг объектов
        predictions = self.track_objects(frame, predictions)
        
        # Классификация действий
        predictions = self.classify_actions(predictions)
        
        # Повторная нормализация bbox перед рендерингом
        predictions = self.normalize_bboxes(predictions)

        with self.frame_lock:
            self.current_objects = predictions
        
        # Рендеринг кадра
        return self.render_frame(frame, predictions)

def generate_frames(system, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось захватить кадр")
            break
        
        # Обработка кадра системой
        processed_frame = system.process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

class EmotionDetectionSystem:
    def __init__(self, config_path="config.yaml"):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.drawer = Drawer()
        self.emotion_data = {
            "happy": {
                "text": "Счастье",
                "emoji": "😊",
                "bgr": (0, 255, 0),    # Зеленый для рамки (BGR)
                "rgb": (0, 255, 0)     # Зеленый для текста (RGB)
            },
            "sad": {
                "text": "Грусть",
                "emoji": "😢",
                "bgr": (0, 0, 255),    # Красный для рамки (BGR)
                "rgb": (255, 0, 0)     # Красный для текста (RGB)
            },
            "angry": {
                "text": "Злость",
                "emoji": "😠",
                "bgr": (255, 0, 0),    # Синий для рамки (BGR)
                "rgb": (0, 0, 255)     # Синий для текста (RGB)
            },
            "surprise": {
                "text": "Удивление",
                "emoji": "😲",
                "bgr": (0, 255, 255),  # Желтый для рамки (BGR)
                "rgb": (255, 255, 0)   # Желтый для текста (RGB)
            },
            "neutral": {
                "text": "Нейтрально",
                "emoji": "😐",
                "bgr": (128, 128, 128),# Серый для рамки (BGR)
                "rgb": (128, 128, 128) # Серый для текста (RGB)
            }
        }
        self.current_objects = []
        self.frame_lock = threading.Lock()

    def process_frame(self, frame):
        try:
            # Преобразуем в оттенки серого для обнаружения лиц
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Обнаружение лиц
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50)
            )
            
            # Создаем список предикций в формате, совместимом с Drawer
            predictions = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Анализ эмоции
                emotion = self._process_face(frame, x, y, w, h)
                if not emotion:
                    continue
                    
                # Получаем информацию о эмоции
                emotion_info = self.emotion_data.get(emotion, self.emotion_data["neutral"])
                
                # Создаем объект предикции
                from types import SimpleNamespace
                pred = SimpleNamespace()
                
                # Устанавливаем keypoints = [] для совместимости с Drawer
                pred.keypoints = np.zeros((18, 3), dtype=np.int16)
                
                # Устанавливаем bbox
                pred.bbox = np.array([x, y, x+w, y+h])
                
                # Устанавливаем ID
                pred.id = i + 1
                
                # Устанавливаем эмоцию
                pred.emotion = emotion_info["text"]
                pred.emotion_score = 0.95
                pred.emotion_emoji = emotion_info["emoji"]
                
                # Устанавливаем цвет (в BGR)
                pred.color = emotion_info["bgr"]
                
                predictions.append(pred)
            
            with self.frame_lock:
                self.current_objects = predictions
            
            # Используем drawer для отрисовки
            if predictions:
                return self.drawer.render_frame(frame, predictions)
            else:
                return frame
            
        except Exception as e:
            print(f"Emotion processing error: {e}")
            return frame

    def _process_face(self, frame, x, y, w, h):
        try:
            face_roi = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            return result[0]['dominant_emotion']
        except Exception as e:
            print("Ошибка анализа эмоции:", e)
            return None

    def get_current_objects(self):
        objects_data = []
        with self.frame_lock:
            for pred in self.current_objects:
                objects_data.append({
                    "id": pred.id,
                    "emotion": getattr(pred, 'emotion', 'unknown'),
                    "confidence": getattr(pred, 'emotion_score', 0.0)
                })
        return objects_data