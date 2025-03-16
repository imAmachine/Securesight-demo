import numpy as np

class FeatureGenerator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.features = []

    def add_cur_skeleton(self, skeleton):
        """
        Добавляет текущий скелетон и извлекает признаки.
        Возвращает:
            - is_features_good: bool, достаточно ли данных для классификации
            - features: извлечённые признаки
        """
        self.features.append(skeleton)
        if len(self.features) < self.window_size:
            return False, None

        # Преобразуем данные в формат, который ожидает классификатор
        # Вместо простого усреднения создаем полный вектор признаков
        window_features = np.array(self.features)
        
        # Соединяем все позиции точек скелета в один вектор (x, y для каждой точки)
        flat_features = []
        for frame in window_features:
            for kp in frame:
                flat_features.extend([kp[1], kp[2]])  # Добавляем координаты y, x
        
        # Преобразуем в numpy массив
        features = np.array(flat_features)
        
        # Если размер не совпадает с ожидаемым (314), заполняем нулями
        if len(features) < 314:
            features = np.pad(features, (0, 314 - len(features)), 'constant')
        elif len(features) > 314:
            features = features[:314]
        
        self.features.pop(0)  # Удаляем старый скелетон
        return True, features