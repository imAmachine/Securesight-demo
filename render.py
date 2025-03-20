import cv2
import threading
import numpy as np
import platform
import emoji
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
from types import SimpleNamespace

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
        self.dangerous_actions = ['kick', 'punch']
    
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
            # Добавляем русские названия действий и выделяем опасные действия
            for pred in predictions:
                if hasattr(pred, 'action') and pred.action and pred.action[0]:
                    action_name = pred.action[0]
                    confidence = pred.action[1]
                    # Переводим название действия на русский (если есть в словаре)
                    translated_name = self.action_names.get(action_name, action_name)
                    pred.action = (translated_name, confidence)
                    
                    # Выделяем опасные действия красным цветом
                    if action_name in self.dangerous_actions:
                        pred.color = (0, 0, 255)  # BGR формат - красный цвет
                    else:
                        pred.color = (0, 255, 0)  # BGR формат - зеленый цвет
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
                    action_data = getattr(pred, 'action', None)
                    
                    # Проверка на None и корректность типа
                    if action_data is None or not isinstance(action_data, (list, tuple)) or len(action_data) < 2:
                        action_data = ['unknown', 0.0]
                    
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

def generate_frames(system, camera_id=0, vedeo_res=(640, 480)):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vedeo_res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vedeo_res[1])
    
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
                "bgr": (255, 0, 0),    # Красный для рамки (BGR)
                "rgb": (0, 0, 255)     # Красный для текста (RGB)
            },
            "angry": {
                "text": "Злость",
                "emoji": "😠",
                "bgr": (0, 0, 255),    # Синий для рамки (BGR)
                "rgb": (255, 0, 0)     # Синий для текста (RGB)
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
            },
        }
        self.current_objects = []
        self.frame_lock = threading.Lock()

        self.emotion_history = {}  # {track_id: [emotions]}
        self.current_tracks = {}  # {track_id: {bbox, last_seen, emotion}}
        self.next_track_id = 1
        self.CONFIDENCE_THRESHOLD = 0.6  # Порог уверенности для смены эмоции
        self.HISTORY_LENGTH = 5  # Количество кадров для усреднения

        self.MAX_TRACK_AGE = 5  # Максимальное время жизни трека без обновления (в кадрах)
        self.MIN_IOU = 0.3  # Минимальное пересечение для обновления трека
        self.MAX_TRACKS = 5  # Максимальное количество одновременно отслеживаемых лиц

    def process_frame(self, frame):
        try:
            # Преобразуем в оттенки серого для обнаружения лиц
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Обнаружение лиц
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            self.current_tracks = {
                track_id: track 
                for track_id, track in self.current_tracks.items()
                if track['last_seen'] < self.MAX_TRACK_AGE
            }

            updated_tracks = {}
            used_faces = set()

            sorted_tracks = sorted(
                self.current_tracks.items(),
                key=lambda x: x[1]['last_seen']
            )

            for track_id, track in sorted_tracks:
                best_match = None
                best_iou = self.MIN_IOU
                
                for i, (x, y, w, h) in enumerate(faces):
                    if i in used_faces:
                        continue
                    
                    iou = self._calculate_iou(track['bbox'], (x, y, w, h))
                    if iou > best_iou:
                        best_match = (x, y, w, h)
                        best_iou = iou
                        best_index = i
                
                if best_match:
                    x, y, w, h = best_match
                    updated_tracks[track_id] = {
                        'bbox': (x, y, w, h),
                        'last_seen': 0,
                        'emotion': track['emotion']
                    }
                    used_faces.add(best_index)
                else:
                    # Увеличиваем счетчик пропущенных кадров
                    updated_tracks[track_id] = {
                        **track,
                        'last_seen': track['last_seen'] + 1
                    }


            new_faces = [f for i, f in enumerate(faces) if i not in used_faces]
            for (x, y, w, h) in new_faces[:self.MAX_TRACKS - len(updated_tracks)]:
                self.current_tracks[self.next_track_id] = {
                    'bbox': (x, y, w, h),
                    'last_seen': 0,
                    'emotion': 'neutral'
                }
                self.emotion_history[self.next_track_id] = []
                self.next_track_id += 1

            # Объединяем обновленные треки
            self.current_tracks = {**self.current_tracks, **updated_tracks}
            
            # Создаем список предикций в формате, совместимом с Drawer
            predictions = []
            for track_id, track in self.current_tracks.items():
                x, y, w, h = track['bbox']
                emotion = self._process_face(frame, x, y, w, h)
                
                if emotion:
                    # Добавляем эмоцию в историю
                    self.emotion_history[track_id].append(emotion)
                    if len(self.emotion_history[track_id]) > self.HISTORY_LENGTH:
                        self.emotion_history[track_id].pop(0)
                    
                    # Определяем доминирующую эмоцию
                    counts = {e: self.emotion_history[track_id].count(e) for e in set(self.emotion_history[track_id])}
                    dominant_emotion = max(counts, key=counts.get)
                    
                    # Обновляем только если уверенность выше порога
                    if counts[dominant_emotion]/len(self.emotion_history[track_id]) >= self.CONFIDENCE_THRESHOLD:
                        track['emotion'] = dominant_emotion

                # Создаем объект предсказания
                emotion_info = self.emotion_data.get(track['emotion'], self.emotion_data["neutral"])
                
                pred = SimpleNamespace()
                pred.keypoints = np.zeros((18, 3), dtype=np.int16)
                pred.bbox = np.array([x, y, x+w, y+h])
                pred.id = track_id
                pred.emotion = emotion_info["text"]
                pred.emotion_score = 0.95
                pred.emotion_emoji = emotion_info["emoji"]
                pred.color = emotion_info["bgr"]
                
                predictions.append(pred)

            with self.frame_lock:
                self.current_objects = predictions

            return self.drawer.render_frame(frame, predictions) if predictions else frame

        except Exception as e:
            print(f"Emotion processing error: {e}")
            return frame
        
    @staticmethod
    def _calculate_iou(box1, box2):
        # box: (x, y, w, h)
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi = max(x1, x2)
        yi = max(y1, y2)
        xu = min(x1 + w1, x2 + w2)
        yu = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xu - xi) * max(0, yu - yi)
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / union_area if union_area != 0 else 0

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
