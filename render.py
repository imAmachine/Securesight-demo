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
        except Exception as e:
            print(f"Ошибка классификатора: {e}")
        
        return predictions
    
    def render_frame(self, rgb_frame, predictions):
        try:
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
        
    def get_current_objects(self):
        objects_data = []
        for pred in self.current_objects:
            try:
                track_id = getattr(pred, 'id', 'N/A'),

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
    
    def process_frame(self, rgb_frame):
        # Определение позы
        predictions = self.predict_poses(rgb_frame)
        if len(predictions) == 0:
            self.current_objects = []
            return rgb_frame
        
        # Нормализация bbox
        predictions = self.normalize_bboxes(predictions)
        
        # Трекинг объектов
        predictions = self.track_objects(rgb_frame, predictions)
        
        # Классификация действий
        predictions = self.classify_actions(predictions)
        
        # Повторная нормализация bbox перед рендерингом
        predictions = self.normalize_bboxes(predictions)

        self.current_objects = predictions
        
        # Рендеринг кадра
        return self.render_frame(rgb_frame, predictions)

def generate_frames(system, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось захватить кадр")
            break
        
        if isinstance(system, ActionDetectionSystem):
            processed_frame = system.process_frame(frame)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = system.process_frame(rgb_frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

class EmotionDetectionSystem:
    def __init__(self, config_path="config.yaml"):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.fonts = self._load_fonts()
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
            "fear": {
                "text": "Страх",
                "emoji": "😨",
                "bgr": (255, 255, 0),
                "rgb": (0, 255, 255)
            },
            "disgust": {
                "text": "Отвращение",
                "emoji": "🤢",
                "bgr": (0, 128, 0),
                "rgb": (0, 128, 0)
            }
        }
        self.current_objects = []
        self.frame_lock = threading.Lock()

    def _load_fonts(self):
        fonts = {
            'text': ImageFont.truetype("arial.ttf", 28),
            'emoji': self._get_emoji_font()
        }
        return fonts

    def _get_emoji_font(self):
        try:
            if platform.system() == "Windows":
                return ImageFont.truetype("seguiemj.ttf", 40)
            elif platform.system() == "Darwin":
                return ImageFont.truetype("Apple Color Emoji.ttf", 40)
            else:
                return ImageFont.truetype("NotoColorEmoji.ttf", 40)
        except:
            return self.fonts['text']

    def process_frame(self, rgb_frame):
        try:
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50))
            
            emotions_data = []
            for (x, y, w, h) in faces:
                emotion = self._process_face(bgr_frame, x, y, w, h)
                emotion_info = self.emotion_data.get(
                    emotion if emotion else "neutral", 
                    self.emotion_data["neutral"]
                )
                
                # Рисуем рамку в BGR
                cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), emotion_info["bgr"], 2)

                if emotion:
                    emotions_data.append((x, y, emotion))

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(rgb_frame)

            for x, y, emotion in emotions_data:
                frame_pil = self._draw_emotion_info(frame_pil, x, y, emotion)

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            
            with self.frame_lock:
                self.current_objects = emotions_data
                
            return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Emotion processing error: {e}")
            return rgb_frame

    def _process_face(self, frame, x, y, w, h):
        try:
            face_roi = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            return result[0]['dominant_emotion']
        except Exception as e:
            print("Ошибка анализа эмоции:", e)
            return None

    def _draw_emotion_info(self, frame_pil, x, y, emotion):
        emotion_info = self.emotion_data.get(emotion, self.emotion_data["neutral"])
        draw = ImageDraw.Draw(frame_pil)
        
        text_position = (x, y - 45)
        draw.text(text_position, emotion_info["text"], 
             font=self.fonts['text'], fill=emotion_info["rgb"])
        
        text_width = self.fonts['text'].getlength(emotion_info["text"])
        emoji_position = (x + text_width + 5, y - 50)
        draw.text(emoji_position, emotion_info["emoji"], 
             font=self.fonts['emoji'], fill=emotion_info["rgb"])
        
        return frame_pil

    def get_current_objects(self):
        objects = []
        for idx, (_, _, emotion) in enumerate(self.current_objects):
            emotion_info = self.emotion_data.get(
                emotion, 
                self.emotion_data["neutral"]
            )
            objects.append({
                    "id": idx,
                    "emotion": f"{emotion} ({emotion_info['text']})",
                    "confidence": 0.95
                })
        return objects