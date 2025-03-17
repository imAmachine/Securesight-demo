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
            "happy": ("Счастье", "😊", (0, 255, 0)),
            "sad": ("Грусть", "😢", (255, 0, 0)),
            "angry": ("Злость", "😠", (0, 0, 255)),
            "surprise": ("Удивление", "😲", (0, 255, 255)),
            "neutral": ("Нейтрально", "😐", (128, 128, 128))
        }
        self.current_objects = []
        self.frame_lock = threading.Lock()
        self.qr_code = self._load_qr_code("qrcode.png")

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

    def _load_qr_code(self, path):
        qr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if qr is None:
            print(f"Ошибка загрузки QR-кода: {path}")
        return qr

    def process_frame(self, rgb_frame):
        try:
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50))
            
            emotions_data = []
            for (x, y, w, h) in faces:
                emotion = self._process_face(frame, x, y, w, h)
                if emotion:
                    emotions_data.append((x, y, emotion))

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for x, y, emotion in emotions_data:
                frame_pil = self._draw_emotion_info(frame_pil, x, y, emotion)

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            frame = self._add_ui_elements(frame)
            
            with self.frame_lock:
                self.current_objects = emotions_data
                
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
        text_part, emoji_part, color = self.emotion_data.get(
            emotion, self.emotion_data["neutral"]
        )
        draw = ImageDraw.Draw(frame_pil)
        
        text_position = (x, y - 45)
        draw.text(text_position, text_part, font=self.fonts['text'], fill=color)
        
        text_width = self.fonts['text'].getlength(text_part)
        emoji_position = (x + text_width + 5, y - 50)
        draw.text(emoji_position, emoji_part, font=self.fonts['emoji'], fill=color)
        
        return frame_pil

    def _add_ui_elements(self, frame):
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, 
                                 cv2.BORDER_CONSTANT, value=(255, 0, 0))
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        text = emoji.emojize("Я люблю ВШИТиАС :red_heart:", language='alias')
        text_part, emoji_part = text.split()[:-1], text.split()[-1]
        
        text_width = self.fonts['text'].getlength(' '.join(text_part) + ' ')
        position = ((frame.shape[1] - text_width) // 2, 5)
        
        draw.text(position, ' '.join(text_part) + ' ', 
                 font=self.fonts['text'], fill=(0, 0, 0))
        draw.text((position[0] + text_width, 0), emoji_part,
                 font=self.fonts['emoji'], fill=(0, 0, 0))
        
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return self._handle_qr_code(frame)

    def _handle_qr_code(self, frame):
        if self.qr_code is None:
            return frame
            
        qr_h, qr_w = self.qr_code.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        max_qr_size = min(frame_w // 4, frame_h // 4)
        
        if qr_h > max_qr_size:
            qr = cv2.resize(self.qr_code, (max_qr_size, max_qr_size))
        else:
            qr = self.qr_code

        x_offset = frame_w - qr.shape[1] - 10
        y_offset = frame_h - qr.shape[0] - 10

        if qr.shape[2] == 4:
            alpha = qr[:, :, 3] / 255.0
            for c in range(3):
                frame[y_offset:y_offset+qr.shape[0], x_offset:x_offset+qr.shape[1], c] = \
                    alpha * qr[:, :, c] + (1 - alpha) * frame[y_offset:y_offset+qr.shape[0], x_offset:x_offset+qr.shape[1], c]
        else:
            frame[y_offset:y_offset+qr.shape[0], x_offset:x_offset+qr.shape[1]] = qr
        
        return frame

    def get_current_objects(self):
        return [{
            "id": idx,
            "emotion": emotion,
            "confidence": 0.95  # Можно добавить реальное значение из DeepFace
        } for idx, (_, _, emotion) in enumerate(self.current_objects)]