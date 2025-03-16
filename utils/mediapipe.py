import cv2
import mediapipe as mp
import numpy as np
from utils.annotation import Annotation

class MediaPipePose:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def predict(self, image, get_bbox=False):
        """
        Функция для совместимости с существующим кодом
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = image.shape[:2]
        results = self.pose.process(image_rgb)
        
        predictions = []
        if results.pose_landmarks:
            # Преобразуем результаты MediaPipe в формат COCO
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
            
            ann = Annotation(keypoints)
            
            if get_bbox:
                # Вычисление bbox
                keypoints_xy = np.array([[lm.x * self.img_w, lm.y * self.img_h] 
                                        for lm in results.pose_landmarks.landmark])
                x_min, y_min = np.min(keypoints_xy, axis=0)
                x_max, y_max = np.max(keypoints_xy, axis=0)
                
                # Расширение bbox
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - width * 0.1)
                y_min = max(0, y_min - height * 0.1)
                x_max = min(self.img_w, x_max + width * 0.1)
                y_max = min(self.img_h, y_max + height * 0.1)
                
                ann.bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
            predictions.append(ann)
            
        return predictions  