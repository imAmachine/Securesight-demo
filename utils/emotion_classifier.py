# utils/emotion_classifier.py
import cv2
import numpy as np
import os

class SimpleEmotionClassifier:
    def __init__(self, classes=None, threshold=0.6, **kwargs):
        self.classes = classes or ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.threshold = threshold
        
        # Загрузка предобученной модели для обнаружения лиц
        face_model_path = kwargs.get('face_model_path', 'utils/weights/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_model_path)
        
        # Загрузка модели для распознавания эмоций
        emotion_model_path = kwargs.get('emotion_model_path', 'utils/weights/emotion_model.xml')
        if os.path.exists(emotion_model_path):
            self.emotion_model = cv2.face.FisherFaceRecognizer_create()
            self.emotion_model.read(emotion_model_path)
        else:
            print(f"Emotion model not found at {emotion_model_path}")
            self.emotion_model = None
            
        # Можно использовать DNN для обнаружения лиц (более точно)
        self.use_dnn = kwargs.get('use_dnn', False)
        if self.use_dnn:
            prototxt = kwargs.get('prototxt_path', 'utils/weights/deploy.prototxt')
            model = kwargs.get('caffemodel_path', 'utils/weights/res10_300x300_ssd_iter_140000.caffemodel')
            if os.path.exists(prototxt) and os.path.exists(model):
                self.face_net = cv2.dnn.readNetFromCaffe(prototxt, model)
            else:
                print("DNN face detection model not found.")
                self.use_dnn = False
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.use_dnn:
            # DNN-based face detection
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                          (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Фильтр по уверенности
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    faces.append((x, y, x2-x, y2-y))
            return faces
        else:
            # Haar cascade detection
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
    
    def classify(self, frame, predictions):
        """
        Classify emotions from faces detected in the predictions.
        
        Args:
            frame: RGB image
            predictions: List of prediction objects with bbox attribute
            
        Returns:
            Updated predictions with emotion attributes
        """
        # Если нет модели эмоций, используем только обнаружение лиц
        if self.emotion_model is None:
            # Обнаруживаем лица
            try:
                faces = self.detect_faces(frame)
                results = []
                
                for (x, y, w, h) in faces:
                    # Обрезаем и обрабатываем лицо
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue
                        
                    # Конвертируем в grayscale для модели
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (48, 48))
                    
                    # Предсказание эмоции
                    if self.emotion_model:
                        label, confidence = self.emotion_model.predict(resized)
                        emotion = self.classes[label]
                    else:
                        emotion = "unknown"
                        confidence = 0.0
                        
                    results.append((
                        emotion,
                        confidence / 100.0,  # Нормализуем confidence
                        (x, y, w, h)
                    ))
                    
                return results
                
            except Exception as e:
                print(f"Emotion classification error: {e}")
                return []

def get_emotion_classifier(**config):
    """Factory function to create an emotion classifier based on config."""
    if config.get('name', '') == 'facial_emotion':
        return SimpleEmotionClassifier(**config)
    else:
        raise ValueError(f"Unknown emotion classifier: {config.get('name')}")