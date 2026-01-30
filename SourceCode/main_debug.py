# SourceCode/main_debug.py
import cv2
import os
import time  # Import thêm thư viện thời gian để đo FPS
from src.pose_detector import PoseDetector

# CẤU HÌNH ĐƯỜNG DẪN
VIDEO_PATH = 'data/badminton_test.mp4'


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"LỖI: Không tìm thấy file video tại {VIDEO_PATH}")
        return

    # Khởi tạo AI
    detector = PoseDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)

    # --- KỸ THUẬT TÍNH FPS ---
    prev_frame_time = 0
    new_frame_time = 0

    print("Bắt đầu xử lý... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video.")
            break

        # --- BƯỚC QUAN TRỌNG: RESIZE ẢNH ---
        # Nếu video 4K (3840px) mà đưa vào AI thì rất chậm.
        # Ta thu nhỏ về chiều ngang 960px (giữ nguyên tỉ lệ khung hình)
        target_width = 960
        height, width = frame.shape[:2]

        # Chỉ resize nếu video gốc lớn hơn 960px
        if width > target_width:
            scaling_factor = target_width / float(width)
            new_height = int(height * scaling_factor)
            frame = cv2.resize(frame, (target_width, new_height))

        # --- GỌI AI XỬ LÝ (Trên ảnh đã thu nhỏ) ---
        results, annotated_frame = detector.process_frame(frame)

        # --- TÍNH FPS (Frame Per Second) ---
        new_frame_time = time.time()
        # Công thức: 1 / (thời gian xử lý 1 khung hình)
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Chuyển FPS thành số nguyên (int) và vẽ lên màn hình
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- VẼ CHẤM CỔ CHÂN TRÁI (BÀI TẬP VỀ NHÀ) ---
        # Index 15 là Cổ chân trái (Left Ankle)
        left_ankle = detector.get_keypoint_by_index(results, keypoint_index=15)
        if left_ankle:
            lx, ly = left_ankle
            # Vẽ chấm Xanh Lá (0, 255, 0)
            cv2.circle(annotated_frame, (lx, ly), 10, (0, 255, 0), -1)

        # --- HIỂN THỊ ---
        cv2.imshow('Badminton AI Coach - Debug', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()