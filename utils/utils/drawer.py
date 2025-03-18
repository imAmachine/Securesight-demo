import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import platform
from PIL import Image, ImageDraw, ImageFont

from .commons import *


class Drawer:
    def __init__(self, draw_points=True, draw_numbers=False, color='green', thickness=1):
        self.draw_points = draw_points
        self.draw_numbers = draw_numbers
        self.color = COLORS[color]
        self.scale = 0.6 if thickness <= 2 else 0.8
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.pil_fonts = self._load_fonts()

    def _load_fonts(self):
        """Загрузка шрифтов для корректного отображения русских символов и эмодзи"""
        fonts = {}
        try:
            # Шрифт для текста (желательно поддерживающий кириллицу)
            fonts['text'] = ImageFont.truetype("arial.ttf", 28)
            
            # Шрифт для эмодзи
            if platform.system() == "Windows":
                fonts['emoji'] = ImageFont.truetype("seguiemj.ttf", 40)
            elif platform.system() == "Darwin":
                fonts['emoji'] = ImageFont.truetype("Apple Color Emoji.ttf", 40)
            else:
                fonts['emoji'] = ImageFont.truetype("NotoColorEmoji.ttf", 40)
        except Exception as e:
            print(f"Ошибка загрузки шрифтов: {e}")
            fonts['text'] = ImageFont.load_default()
            fonts['emoji'] = None
        return fonts

    def render_frame(self, image, predictions, **user_text_kwargs):
        """Draw all persons [skeletons / tracked_id / action] annotations on image
        in trtpose keypoint format.
        """
        render_frame = image.copy()
        def _scale_keypoints(pred):
            if pred.keypoints[..., 1:].max() <= 1:
                pred.keypoints[..., 1:] *= render_frame.shape[:2][::-1]
            pred.keypoints = pred.keypoints.astype(np.int16)
            return pred

        predictions = [_scale_keypoints(pred) for pred in predictions]
        
        # Сначала рисуем скелеты и bounding boxes
        for pred in predictions:
            if pred.color is not None: self.color = pred.color
            self.draw_trtpose(render_frame, pred)
            if pred.bbox is not None:
                self.draw_bbox(render_frame, pred)
        
        # Конвертируем в RGB для PIL
        rgb_frame = cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Теперь рисуем текстовые метки
        for pred in predictions:
            if pred.bbox is not None:
                self.draw_labels_pil(pil_image, pred)
        
        # Конвертируем обратно в BGR для OpenCV
        render_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if len(user_text_kwargs)>0:
            render_frame = self.add_user_text(render_frame, **user_text_kwargs)
        return render_frame

    def draw_trtpose(self, image, pred):
        """Draw skeletons on image with trtpose keypoint format"""

        visibilities = []
        # draw circle and keypoint numbers
        for kp in pred.keypoints:
            if kp[1]==0 or kp[2]==0:
                visibilities.append(kp[0])
                continue
            if self.draw_points:
                cv2.circle(image, (kp[1],kp[2]), self.thickness, self.color, self.thickness+2)
            if self.draw_numbers:
                cv2.putText(image, str(kp[0]), (kp[1],kp[2]), self.font,
                            self.scale - 0.2, COLORS['blue'], self.thickness)

        # draw skeleton connections
        for pair in LIMB_PAIRS:
            if pair[0] in visibilities or pair[1] in visibilities: continue
            start, end = map(tuple, [pred.keypoints[pair[0]], pred.keypoints[pair[1]]])
            cv2.line(image, start[1:], end[1:], self.color, self.thickness)

    def draw_bbox(self, image, pred):
        """Рисует только bbox без текстовых меток"""
        # Преобразование bbox в numpy массив, если это список или другой тип
        if not isinstance(pred.bbox, np.ndarray):
            bbox = np.array(pred.bbox)
        else:
            bbox = pred.bbox
        
        # draw person bbox
        x1, y1, x2, y2 = bbox.astype(np.int16)
        cv2.rectangle(image, (x1,y1), (x2,y2), self.color, self.thickness)

    def draw_labels_pil(self, pil_image, pred):
        """Рисует текстовые метки с использованием PIL для поддержки русского языка"""
        draw = ImageDraw.Draw(pil_image)
        
        # Если шрифты не загружены, выходим
        if 'text' not in self.pil_fonts or self.pil_fonts['text'] is None:
            return
        
        # Преобразование bbox
        if not isinstance(pred.bbox, np.ndarray):
            bbox = np.array(pred.bbox)
        else:
            bbox = pred.bbox
        
        x1, y1, x2, y2 = bbox.astype(np.int16)
        
        # Цвета для фона и текста
        bg_color = (50, 50, 50)
        text_color = (255, 255, 255)
        
        # Расстояние между строками текста
        line_height = 32
        
        # Отрисовка ID если есть
        current_y = y1
        if hasattr(pred, 'id') and pred.id:
            track_id_text = f'ID: {pred.id}'
            text_width = self.pil_fonts['text'].getlength(track_id_text)
            
            # Рисуем фон
            draw.rectangle([(x1, current_y), (x1 + text_width + 10, current_y + line_height)], 
                          fill=bg_color)
            
            # Рисуем текст
            draw.text((x1 + 5, current_y + 2), track_id_text, 
                     font=self.pil_fonts['text'], fill=text_color)
            
            current_y += line_height
        
        # Отрисовка действия если есть
        if hasattr(pred, 'action') and pred.action and pred.action[0]:
            action_text = f'{pred.action[0]}: {pred.action[1]:.2f}'
            text_width = self.pil_fonts['text'].getlength(action_text)
            
            # Рисуем фон
            draw.rectangle([(x1, current_y), (x1 + text_width + 10, current_y + line_height)], 
                          fill=bg_color)
            
            # Рисуем текст
            draw.text((x1 + 5, current_y + 2), action_text, 
                     font=self.pil_fonts['text'], fill=text_color)
            
            current_y += line_height
        
        # Отрисовка эмоции если есть
        if hasattr(pred, 'emotion') and pred.emotion and pred.emotion != 'unknown':
            emotion_text = f'Эмоция: {pred.emotion}'
            if hasattr(pred, 'emotion_score'):
                emotion_text += f' ({pred.emotion_score:.2f})'
            
            text_width = self.pil_fonts['text'].getlength(emotion_text)
            
            # Рисуем фон
            draw.rectangle([(x1, current_y), (x1 + text_width + 10, current_y + line_height)], 
                          fill=bg_color)
            
            # Рисуем текст
            draw.text((x1 + 5, current_y + 2), emotion_text, 
                     font=self.pil_fonts['text'], fill=text_color)
            
            # Добавляем эмодзи если есть
            if 'emoji' in self.pil_fonts and self.pil_fonts['emoji'] and hasattr(pred, 'emotion_emoji'):
                emoji_text = pred.emotion_emoji
                text_width = self.pil_fonts['text'].getlength(emotion_text)
                draw.text((x1 + text_width + 15, current_y - 5), emoji_text, 
                         font=self.pil_fonts['emoji'], fill=text_color)

    def draw_bbox_label(self, image, pred):
        """Устаревший метод - оставлен для обратной совместимости"""
        scale = self.scale - 0.1
        
        # Преобразование bbox в numpy массив, если это список или другой тип
        if not isinstance(pred.bbox, np.ndarray):
            bbox = np.array(pred.bbox)
        else:
            bbox = pred.bbox
        
        # draw person bbox
        x1, y1, x2, y2 = bbox.astype(np.int16)
        cv2.rectangle(image, (x1,y1), (x2,y2), self.color, self.thickness)

        def get_label_position(label, is_track=False):
            w, h = cv2.getTextSize(label, self.font, scale, self.thickness)[0]
            offset_w, offset_h = w + 3, h + 5
            xmax = x1 + offset_w
            is_upper_pos = True
            if (y1 - offset_h) < 0 or is_track:
                ymax = y1 + offset_h
                y_text = ymax - 2
            else:
                ymax = y1 - offset_h
                y_text = y1 - 2
                is_upper_pos = False
            return xmax, ymax, y_text, is_upper_pos

        # Draw text with white color on dark background for better visibility
        bg_color = (50, 50, 50)
        text_color = (255, 255, 255)

        if pred.id:
            track_label = f'{pred.id}'
            *track_loc, is_upper_pos = get_label_position(track_label, is_track=True)
            cv2.rectangle(image, (x1, y1), (track_loc[0], track_loc[1]), bg_color, -1)
            cv2.putText(image, track_label, (x1+1, track_loc[2]), self.font,
                        scale, text_color, self.thickness)

            # Draw action label if available
            if hasattr(pred, 'action') and pred.action and pred.action[0]:
                action_label = '{}: {:.2f}'.format(*pred.action)
                if not is_upper_pos:
                    action_label = f'{track_label}-{action_label}'
                action_loc = get_label_position(action_label)
                cv2.rectangle(image, (x1, y1), (action_loc[0], action_loc[1]), bg_color, -1)
                cv2.putText(image, action_label, (x1+1, action_loc[2]), self.font,
                            scale, text_color, self.thickness)
            
            # Draw emotion label if available
            if hasattr(pred, 'emotion') and pred.emotion and pred.emotion != 'unknown':
                emotion_text = f"Emotion: {pred.emotion}"
                if hasattr(pred, 'emotion_score'):
                    emotion_text += f" ({pred.emotion_score:.2f})"
                emotion_loc = get_label_position(emotion_text, is_track=True)
                cv2.rectangle(image, (x1, y1+20), (emotion_loc[0], emotion_loc[1]+20), bg_color, -1)
                cv2.putText(image, emotion_text, (x1+1, emotion_loc[2]+20), self.font,
                            scale, text_color, self.thickness)


    def add_user_text(self, image, text_color='red', add_blank=True, **user_text):
        h, w, d = image.shape
        if add_blank:
            size = (h, 200, d) if h > w/1.5 else (200, w, d)
            blank = np.zeros(size, dtype=np.uint8)
            image = np.hstack((blank, image)) if h > w/1.5 else np.vstack((blank, image))
        # draw texts
        if len(user_text) > 0:
            x, y0, dy = 5, 25, 30
            cnt = 0
            for key, value in user_text.items():
                text = f'{key}: {value}'
                y = y0 + cnt * dy
                if y > 200 and h < w/1.5 and add_blank:
                    cnt = 0
                    x = w // 2
                    y = y0 + cnt * dy
                cnt += 1
                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, COLORS[text_color], 2)
        return image