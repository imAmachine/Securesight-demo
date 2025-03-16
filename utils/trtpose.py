import os
import json
from collections import OrderedDict

import torch
try: import torch2trt
except: print('torch2trt not installed.')
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from trt_pose import models, coco
from trt_pose.parse_objects import ParseObjects

from utils.annotation import Annotation


POSE_META = {
    'num_parts': 18,
    'num_links': 21,
    'skeleton': [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],
        [5, 7], [18, 1], [18, 6], [18, 7], [18, 12], [18, 13]
    ]
}


class TrtPose:
    """trtpose wrapper for pose prediction"""

    _params = OrderedDict(
            backbone='densenet121',
            cmap_threshold=0.1,
            link_threshold=0.1,
            )

    def __init__(self, size, model_path, min_leg_joints, min_total_joints, include_head=True):
        self.min_total_joints = min_total_joints
        self.min_leg_joints = min_leg_joints
        self.include_head = include_head

        if not isinstance(size, (tuple, list)):
            size = (size, size)
        self.height, self.width = size
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка топологии
        self.topology = coco.coco_category_to_topology(POSE_META)
        self.parse_objects = ParseObjects(self.topology, cmap_threshold=0.1, link_threshold=0.1)

        # Автоматическая загрузка модели densenet121
        self.model = self._load_torch_model(self.model_path, backbone='densenet121')

        # Трансформации изображения
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_torch_model(self, model_file, backbone='densenet121'):
        """Загрузка модели densenet121 из файла или автоматически"""
        num_parts = POSE_META['num_parts']
        num_links = POSE_META['num_links']

        if backbone == 'densenet121':
            print(f"[INFO] Загрузка модели densenet121 из {model_file}")
            model = models.densenet121_baseline_att(
                cmap_channels=num_parts,
                paf_channels=2 * num_links
            )
        else:
            raise ValueError(f"Неподдерживаемая модель: {backbone}")

        # Загрузка весов модели
        if os.path.isfile(model_file):
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
        else:
            print(f"[WARNING] Файл с весами {model_file} не найден. Используется модель без весов.")

        return model.to(self.device).eval()

    def _load_trt_model(self, model_file):
        """load converted tensorRT model"""
        print(f'[INFO] Loading TensorRT trtpose model : {model_file}')
        model_trt = torch2trt.TRTModule()
        model_trt.load_state_dict(torch.load(model_file))
        model_trt.eval()
        return model_trt

    def _load_torch_model(self, model_file, backbone='densenet121'):
        """load pytorch model with resnet18 encoder or densenet121"""

        print(f'[INFO] Loading pytorch trtpose model with "{model_file}"')
        num_parts = POSE_META['num_parts']
        num_links = POSE_META['num_links']

        if backbone=='resnet18':
            model = models.resnet18_baseline_att(
                cmap_channels=num_parts,
                paf_channels=2 * num_links
                )
        elif backbone=='densenet121':
            model = models.densenet121_baseline_att(
                cmap_channels=num_parts,
                paf_channels= 2 * num_links
                )
        else:
            print('not supported model type "{}"'.format(backbone))
            return -1

        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        return model.to(self.device).eval()

    def _preprocess(self, image):
        """Resize image and transform to tensor image"""

        assert isinstance(image, np.ndarray), 'image type need to be array'
        image = Image.fromarray(image).resize(
            (self.width, self.height),
            resample=Image.BILINEAR
            )
        tensor = self.transforms(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        return image, tensor

    @torch.no_grad()
    def predict(self, image, get_bbox=False):
        """predict pose estimation on rgb image
        args:
            image (np.ndarray[r,g,b]): rgb input image.
        return:
            predictions (list): list of annotation object with only good person keypoints
        """
        self.img_h, self.img_w = image.shape[:2]
        pil_img, tensor_img = self._preprocess(image)
        #print(tensor_img.shape)
        cmap, paf = self.model(tensor_img)
        cmap, paf = cmap.cpu(), paf.cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf) # cmap threhold=0.15, link_threshold=0.15
        predictions = self.get_keypoints(objects, counts, peaks, get_bbox=get_bbox)
        return predictions

    def get_bbox_from_keypoints(self, keypoints):
        def expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height):
            """expand bbox for containing more background"""
            width = xmax - xmin
            height = ymax - ymin
            ratio = 0.1   # expand ratio
            new_xmin = np.clip(xmin - ratio * width, 0, img_width)
            new_xmax = np.clip(xmax + ratio * width, 0, img_width)
            new_ymin = np.clip(ymin - ratio * height, 0, img_height)
            new_ymax = np.clip(ymax + ratio * height, 0, img_height)
            new_width = new_xmax - new_xmin
            new_height = new_ymax - new_ymin
            return [new_xmin, new_ymin, new_width, new_height]

        keypoints = np.where(keypoints[:, 1:] !=0, keypoints[:, 1:], np.nan)
        keypoints[:, 0] *= self.img_w
        keypoints[:, 1] *= self.img_h
        xmin = np.nanmin(keypoints[:,0])
        ymin = np.nanmin(keypoints[:,1])
        xmax = np.nanmax(keypoints[:,0])
        ymax = np.nanmax(keypoints[:,1])
        bbox = expand_bbox(xmin, xmax, ymin, ymax, self.img_w, self.img_h)
        # discard bbox with width and height == 0
        if bbox[2] < 1 or bbox[3] < 1 :
            return None
        return bbox

    def get_keypoints(self, humans, counts, peaks, get_bbox=False):
        """Get all persons keypoint from predictions obtained via TRT pose."""
        def is_good_person_keypoints(keypoints):
            # Если не включаем голову, берем ключевые точки начиная с индекса 5
            total_keypoints = keypoints[5:, 1:] if not self.include_head else keypoints[:, 1:]
            # Определяем количество валидных (ненулевых) точек
            num_valid_joints = np.count_nonzero(total_keypoints)
            # Для ног берем последние 6-7 точек (зависит от вашей топологии)
            num_leg_joints = np.count_nonzero(total_keypoints[-7:-1])
            return num_valid_joints >= self.min_total_joints and num_leg_joints >= self.min_leg_joints

        predictions = []
        for cnt in range(counts):
            keypoints = np.zeros((18, 3), dtype=np.float64)
            # Получаем данные для текущего человека
            human = humans[0][cnt]
            C = human.shape[0]
            for j in range(C):
                k = int(human[j])
                if k >= 0:
                    # Пробуем извлечь данные для j-й ключевой точки
                    try:
                        peak_val = peaks[0][j][k]
                        # Убедимся, что peak_val имеет хотя бы 2 элемента (координаты)
                        if len(peak_val) >= 2:
                            peak = (j, float(peak_val[1]), float(peak_val[0]))
                        else:
                            # Если данные короче ожидаемого, используем нули
                            peak = (j, 0.0, 0.0)
                    except Exception as e:
                        print(f"[WARNING] Не удалось получить данные для ключевой точки {j} у человека {cnt}: {e}")
                        peak = (j, 0.0, 0.0)
                else:
                    peak = (j, 0.0, 0.0)
                # Преобразуем peak в массив фиксированной длины и записываем в keypoints
                try:
                    # Если peak – кортеж с 3 элементами, он корректно преобразуется в массив
                    keypoints[j] = np.array(peak, dtype=np.float64)
                except Exception as e:
                    print(f"[WARNING] Не удалось записать keypoint для индекса {j} у человека {cnt}: {e}")
                    keypoints[j] = np.array([j, 0.0, 0.0], dtype=np.float64)

            if is_good_person_keypoints(keypoints):
                ann = Annotation(keypoints)
                if get_bbox:
                    ann.bbox = self.get_bbox_from_keypoints(ann.keypoints)
                predictions.append(ann)

        return predictions

