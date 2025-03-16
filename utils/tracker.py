from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=70, n_init=5)

    def update(self, bboxes, scores, frame):
        # Обновление треков
        tracks = self.tracker.update_tracks(bboxes, scores, frame)
        return tracks