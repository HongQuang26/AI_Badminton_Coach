from ultralytics import YOLO
import cv2
import numpy as np


class PoseDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = 'yolov8n-pose.pt'

        print(f"[INFO] Đang tải model từ: {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("[INFO] Tải model thành công!")
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải model: {e}")
            print("[INFO] Đang thử tải model mặc định...")
            self.model = YOLO('yolov8n-pose.pt')

    def process_frame(self, frame):
        results = self.model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        return results, annotated_frame

    def get_keypoint_by_index(self, results, person_index=0, keypoint_index=10):
        if results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None

        try:
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            person = keypoints_data[person_index]
            point = person[keypoint_index]
            x, y, conf = point
            if conf < 0.5:
                return None
            return (int(x), int(y))

        except IndexError:
            return None