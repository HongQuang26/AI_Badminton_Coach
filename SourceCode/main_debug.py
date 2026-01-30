import cv2
import os
import numpy as np
from src.pose_detector import PoseDetector
from src.court_mapper import CourtMapper

# Cập nhật đúng tên video TrackNet
VIDEO_PATH = 'data/tracknet_test.mp4'


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Lỗi: Không tìm thấy file {VIDEO_PATH}")
        return

    # 1. Khởi tạo
    # Không truyền tham số -> Tự động tải 'yolov8m-pose.pt' (Medium)
    detector = PoseDetector()
    mapper = CourtMapper()

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Màu sắc cho P1, P2, P3...
    PLAYER_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]

    print("Đang chạy... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kết thúc video.")
            break

        # --- SỬA LỖI RESIZE (Quan trọng) ---
        target_width = 1280  # Dùng 1280 để nhìn rõ người ở xa hơn
        height, width = frame.shape[:2]
        new_height = height  # Khởi tạo giá trị mặc định

        if width > target_width:
            scaling_factor = target_width / float(width)
            new_height = int(height * scaling_factor)
            frame = cv2.resize(frame, (target_width, new_height))

        # --- A. XỬ LÝ AI ---
        results, annotated_frame = detector.process_frame(frame)

        num_people = 0
        if results[0].keypoints is not None:
            num_people = len(results[0].keypoints)

        # --- B. MINIMAP ---
        minimap_img = mapper.court_img.copy()

        for i in range(num_people):
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]

            # Lấy toạ độ 2 chân
            ankle_l = detector.get_keypoint_by_index(results, person_index=i, keypoint_index=15)
            ankle_r = detector.get_keypoint_by_index(results, person_index=i, keypoint_index=16)

            player_pos_video = None
            if ankle_l and ankle_r:
                mid_x = (ankle_l[0] + ankle_r[0]) // 2
                mid_y = (ankle_l[1] + ankle_r[1]) // 2
                player_pos_video = (mid_x, mid_y)
            elif ankle_l:
                player_pos_video = ankle_l
            elif ankle_r:
                player_pos_video = ankle_r

            if player_pos_video:
                # Vẽ lên video gốc
                cv2.circle(annotated_frame, player_pos_video, 8, color, -1)
                cv2.putText(annotated_frame, f"P{i + 1}", (player_pos_video[0], player_pos_video[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Vẽ lên Minimap
                minimap_point = mapper.convert_to_minimap(player_pos_video)
                if minimap_point:
                    mx, my = minimap_point
                    if 0 <= mx < mapper.map_width and 0 <= my < mapper.map_height:
                        cv2.circle(minimap_img, (mx, my), 8, color, -1)
                        cv2.circle(minimap_img, (mx, my), 10, (255, 255, 255), 2)
                        cv2.putText(minimap_img, f"P{i + 1}", (mx + 10, my),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- C. HIỂN THỊ ---
        # Resize Minimap khớp chiều cao video
        scale_mini = new_height / minimap_img.shape[0]
        minimap_resized = cv2.resize(minimap_img, None, fx=scale_mini, fy=scale_mini)

        final_display = np.hstack((annotated_frame, minimap_resized))

        cv2.imshow('Badminton AI Coach (Tracking Mode)', final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()