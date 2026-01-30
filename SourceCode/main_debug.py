import cv2
import os
import numpy as np
from src.pose_detector import PoseDetector
from src.court_mapper import CourtMapper

# ĐƯỜNG DẪN VIDEO
VIDEO_PATH = 'data/tracknet_test.mp4'


def is_inside_court(point, court_polygon):
    if point is None: return False
    return cv2.pointPolygonTest(court_polygon, point, False) >= 0


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Lỗi: Không tìm thấy file {VIDEO_PATH}")
        return

    # 1. Khởi tạo
    print("Đang khởi tạo hệ thống (Resolution: 960px)...")
    detector = PoseDetector(model_path='yolov8l-pose.pt')
    mapper = CourtMapper()

    cap = cv2.VideoCapture(VIDEO_PATH)

    PLAYER_COLORS = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0)
    ]

    # --- TẠO VÙNG ROI DỰA TRÊN TOẠ ĐỘ 960PX ---
    src_pts = mapper.src_points

    # Mở rộng vùng sân ra 40px (với độ phân giải 960 thì 40px là đủ)
    roi_polygon = np.array([
        [src_pts[0][0] - 40, src_pts[0][1] + 40],  # Dưới-Trái
        [src_pts[1][0] + 40, src_pts[1][1] + 40],  # Dưới-Phải
        [src_pts[2][0] + 40, src_pts[2][1] - 40],  # Trên-Phải
        [src_pts[3][0] - 40, src_pts[3][1] - 40]  # Trên-Trái
    ], np.int32)
    roi_polygon = roi_polygon.reshape((-1, 1, 2))

    print("Đang chạy... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video.")
            break

        # --- RESIZE VỀ 960PX ĐỂ KHỚP VỚI TOẠ ĐỘ ---
        target_width = 960
        height, width = frame.shape[:2]
        new_height = height

        if width > target_width:
            scaling_factor = target_width / float(width)
            new_height = int(height * scaling_factor)
            frame = cv2.resize(frame, (target_width, new_height))

        # --- A. XỬ LÝ AI ---
        results, annotated_frame = detector.process_frame(frame)

        # Vẽ khung ROI màu xanh lá
        cv2.polylines(annotated_frame, [roi_polygon], True, (0, 255, 0), 2)

        # --- B. LỌC NGƯỜI & MINIMAP ---
        minimap_img = mapper.court_img.copy()
        valid_players = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for i, track_id in enumerate(track_ids):
                # Lấy toạ độ chân
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

                # BỘ LỌC TRỌNG TÀI
                if player_pos_video:
                    if is_inside_court(player_pos_video, roi_polygon):
                        valid_players.append((track_id, player_pos_video))

        # --- C. VẼ KẾT QUẢ ---
        for track_id, pos in valid_players:
            color = PLAYER_COLORS[track_id % len(PLAYER_COLORS)]

            # Vẽ lên Video
            cv2.circle(annotated_frame, pos, 6, color, -1)
            cv2.putText(annotated_frame, f"ID:{track_id}", (pos[0], pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Vẽ lên Minimap
            minimap_point = mapper.convert_to_minimap(pos)
            if minimap_point:
                mx, my = minimap_point
                if 0 <= mx < mapper.map_width and 0 <= my < mapper.map_height:
                    cv2.circle(minimap_img, (mx, my), 8, color, -1)
                    cv2.circle(minimap_img, (mx, my), 10, (255, 255, 255), 2)
                    cv2.putText(minimap_img, f"{track_id}", (mx + 10, my),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Hiển thị
        scale_mini = new_height / minimap_img.shape[0]
        minimap_resized = cv2.resize(minimap_img, None, fx=scale_mini, fy=scale_mini)

        final_display = np.hstack((annotated_frame, minimap_resized))
        cv2.imshow('Badminton AI (960px Synced)', final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()