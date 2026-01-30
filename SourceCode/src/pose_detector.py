from ultralytics import YOLO
import cv2
import numpy as np


class PoseDetector:
    def __init__(self, model_path=None):
        """
        Khởi tạo model YOLO.
        """
        # Sử dụng model Large (l) để nhận diện người ở xa tốt nhất
        # Tuy nặng hơn bản Medium nhưng độ chính xác cao hơn hẳn
        if model_path is None:
            model_path = 'yolov8l-pose.pt'

        print(f"[INFO] Đang tải model từ: {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("[INFO] Tải model thành công!")
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải model: {e}")
            print("[INFO] Đang thử tải model mặc định 'yolov8l-pose.pt'...")
            self.model = YOLO('yolov8l-pose.pt')

    def process_frame(self, frame):
        """
        Xử lý 1 khung hình: Nhận diện và vẽ xương.
        Sử dụng chế độ TRACKING để ổn định ID.
        """
        # persist=True: Giúp AI "nhớ" người chơi qua các khung hình
        # conf=0.25: Hạ ngưỡng thấp một chút để không bỏ sót người ở góc xa
        # tracker="bytetrack.yaml": Thuật toán theo dõi người tốt nhất
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)

        annotated_frame = results[0].plot()

        return results, annotated_frame

    def get_keypoint_by_index(self, results, person_index=0, keypoint_index=10):
        """
        Lấy tọa độ (x, y) của một khớp cụ thể.
        """
        # Kiểm tra an toàn: Có keypoints không?
        if results[0].keypoints is None:
            return None

        if len(results[0].keypoints) == 0:
            return None

        # Kiểm tra index người chơi có hợp lệ không
        if person_index >= len(results[0].keypoints):
            return None

        try:
            # Chuyển dữ liệu sang CPU/Numpy
            keypoints_data = results[0].keypoints.data.cpu().numpy()

            # Lấy thông tin người thứ 'person_index'
            person = keypoints_data[person_index]

            # Lấy khớp 'keypoint_index' (x, y, conf)
            point = person[keypoint_index]

            x, y, conf = point

            # Nếu độ tin cậy quá thấp (< 0.25) thì bỏ qua
            if conf < 0.25:
                return None

            return (int(x), int(y))

        except IndexError:
            return None