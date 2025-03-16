import cv2
import numpy as np
import torch

from .sort.detection import Detection
from .sort.tracker import Tracker
from .reid_feature_extractor import FeatureExtractor
from .sort.nn_matching import NearestNeighborDistanceMetric

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, **kwargs):
        self.extractor = FeatureExtractor(**kwargs)
        metric = NearestNeighborDistanceMetric(
            "cosine",
            max_dist,
            nn_budget
            )
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init
        )

    def predict(self, rgb_img, predictions, debug=False):
        """Update tracker state via analyis of current keypoint's bboxes with previous tracked bbox.
        args:
            predictions (list): list of annotations object with keypoints bboxes, (xmin, ymin, w, h).
            img (np.ndarray): original rgb image.
        return:
            tracked_predictions (list): Filtered tracked list of annotations object filled with
                    tracked id, tracked color and bbox (top,left,btm,right) attributes.
            debug_img (np.ndarray or None)
        """
        try:
            # generate detections
            bbox_list = []
            for i, pred in enumerate(predictions):
                # Проверка наличия атрибута bbox
                if not hasattr(pred, "bbox"):
                    print(f"[WARNING] Объект predictions[{i}] не содержит атрибута 'bbox'. Пропускаем.")
                    continue

                # Преобразуем pred.bbox в numpy-массив и выпрямляем его
                try:
                    bbox_array = np.array(pred.bbox, dtype=float).flatten()
                except Exception as e:
                    print(f"[WARNING] Не удалось преобразовать bbox для predictions[{i}]: {pred.bbox}. Ошибка: {e}")
                    continue

                # Если в bbox_array больше 4 элементов, берем только первые 4
                if bbox_array.size > 4:
                    bbox_array = bbox_array[:4]
                elif bbox_array.size < 4:
                    print(f"[WARNING] bbox в predictions[{i}] имеет недостаточную длину ({bbox_array.size}). Пропускаем.")
                    continue

                bbox_list.append(bbox_array)

            if len(bbox_list) == 0:
                print("[ERROR] Нет корректных bbox для обработки.")
                bbox_tlwh = np.empty((0, 4), dtype=float)
            else:
                # Гарантируем, что все элементы имеют форму (4,)
                bbox_tlwh = np.stack(bbox_list)
            bbox_tlbr = self.tlwh_to_tlbr(bbox_tlwh)
            features = self._get_features(bbox_tlbr, rgb_img)
            detections = [Detection(bbox, features[i]) for i, bbox in enumerate(bbox_tlwh)]

            # update tracker and predictions object
            self.tracker.predict() # update track_id's time_since_update and age increasement
            self.tracker.update(detections, predictions) # update predictions with tracked ID and Color
            # filter untracked persons' keypoints
            tracked_predictions = list(filter(lambda x: x.id, predictions))
            if debug:
                debug_img = rgb_img[...,::-1].copy()
                self.debug_bboxes(debug_img, self.tracker.tracks, bbox_tlbr)
                return tracked_predictions, debug_img

            return tracked_predictions, None
        except Exception as e:
            print(f"Ошибка в DeepSort.predict: {e}")
            self.tracker.increment_ages()
            # Важно: возвращаем исходные предсказания без изменений
            return predictions, None

    def increment_ages(self):
        self.tracker.increment_ages()

    def _get_features(self, bbox_tlbr, ori_img):
        im_crops = []
        for box in bbox_tlbr:
            x1, y1, x2, y2 = map(int, box)
            # Проверка границ, чтобы избежать ошибок при обрезке
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ori_img.shape[1], x2), min(ori_img.shape[0], y2)
            
            # Проверка, что область обрезки не пуста
            if x1 >= x2 or y1 >= y2:
                continue
                
            im = ori_img[y1:y2, x1:x2]
            if im.size == 0:
                continue
            im_crops.append(im)
        
        if not im_crops:
            return np.array([])
        
        # Обработка небольшими партиями
        batch_size = 2  # Очень маленький размер партии для надежности
        features_list = []
        
        for i in range(0, len(im_crops), batch_size):
            batch = im_crops[i:i+batch_size]
            try:
                batch_features = self.extractor(batch)
                # Преобразуем тензор в numpy массив
                if hasattr(batch_features, 'cpu'):
                    batch_features = batch_features.cpu().numpy()
                features_list.append(batch_features)
            except Exception as e:
                print(f"Ошибка обработки партии {i}/{len(im_crops)}: {e}")
                # Добавляем заглушки для неудачных изображений
                dummy_features = np.zeros((len(batch), 128))  # Предполагаем размерность признаков 128
                features_list.append(dummy_features)
        
        if features_list:
            return np.vstack(features_list)
        else:
            return np.array([])

    @staticmethod
    def tlwh_to_tlbr(bbox_tlwh):
        if isinstance(bbox_tlwh, np.ndarray):
            bbox_tlbr = bbox_tlwh.copy()
        elif isinstance(bbox_tlwh, torch.Tensor):
            bbox_tlbr = bbox_tlwh.clone()

        bbox_tlbr[:, 2] += bbox_tlwh[:, 0]
        bbox_tlbr[:, 3] += bbox_tlwh[:, 1]
        return bbox_tlbr

    @staticmethod
    def debug_bboxes(image, tracks, detections):
        for track in tracks:
            # if track.is_comfirmed: continue
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            text = f'{track.track_id}: update[{track.time_since_update}]'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, ((x1+x2)//2, (y1+y2)//2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det)
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(image, f'{idx}: detect', (x1,y1-5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)
