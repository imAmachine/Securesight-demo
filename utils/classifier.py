import os
import torch
import joblib
from torch import nn
import numpy as np
from collections import deque
from .feature_generator import FeatureGenerator  # Импорт FeatureGenerator

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPClassifier, self).__init__()
        self.hidden_layers = nn.Sequential()
        for in_size, out_size in zip([input_size] + hidden_sizes[:-1], hidden_sizes):
            self.hidden_layers.add_module(
                f"layer_{len(self.hidden_layers)}",
                nn.Sequential(
                    nn.Linear(in_size, out_size),
                    nn.BatchNorm1d(out_size),
                    nn.ReLU()
                )
            )
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

class ClassifierOnlineTest:
    def __init__(self, model_path, action_labels, window_size, human_id=0, threshold=0.7):
        self.model = None
        self.human_id = human_id
        self.model_path = model_path
        self.action_labels = action_labels
        self.threshold = threshold
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()

        self.feature_generator = FeatureGenerator(window_size)
        with open(os.path.join(os.path.dirname(model_path), "pca.pkl"), 'rb') as f:
            self.pca = joblib.load(f)
        self.reset()

    def load_model(self):
        self.model = MLPClassifier(
            input_size=50, hidden_sizes=[1024, 512], output_size=len(self.action_labels)
        ).to(self.device)

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at path: {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def reset(self):
        self.feature_generator.reset()
        self.scores_hist = deque()
        self.scores = None

    def predict(self, skeleton):
        is_features_good, features = self.feature_generator.add_cur_skeleton(skeleton)

        if is_features_good:
            features_pca = self.pca.transform(features.reshape(1, -1))
            features_tensor = torch.tensor(features_pca, dtype=torch.float32)
            with torch.no_grad():
                curr_scores = self.model(features_tensor.to(self.device))
            self.scores = self.smooth_scores(curr_scores.cpu().numpy()[0])

            if self.scores.max() < self.threshold:
                predicted_label = ['', 0]
            else:
                predicted_idx = self.scores.argmax()
                predicted_label = [self.action_labels[predicted_idx], self.scores.max()]
        else:
            predicted_label = ['', 0]

        return predicted_label

    def smooth_scores(self, curr_scores):
        self.scores_hist.append(curr_scores)
        DEQUE_MAX_SIZE = 2
        if len(self.scores_hist) > DEQUE_MAX_SIZE:
            self.scores_hist.popleft()

        score_sums = np.zeros((len(self.action_labels),))
        for score in self.scores_hist:
            score_sums += score
        score_sums /= len(self.scores_hist)
        return score_sums