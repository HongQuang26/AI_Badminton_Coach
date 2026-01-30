import cv2
import os
import numpy as np
from src.pose_detector import PoseDetector
from src.court_mapper import CourtMapper

# ĐƯỜNG DẪN VIDEO TRACKNET
VIDEO_PATH = 'data/tracknet_test.mp4'


def is_inside_court(point, court_polygon):
    """
    Kiểm tra xem một điểm (x, y) có nằm trong vùng sân thi đấu không.
    Dùng để loại bỏ trọng tài và khán giả.
    """
    if point is None: return False
    # cv2.pointPolygonTest trả về:
    # > 0: Bên trong
    # = 0: Trên cạnh
    # < 0: Bên ngoài
    # measureDist=False để chạy nhanh hơn
    result = cv2.pointPolygonTest(court_polygon, point, False)
    return result >= 0


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Lỗi: Không tìm thấy file {VIDEO_PATH}")
        return

    # 1. Khởi tạo
    # Dùng model L (Large) để bắt người ở xa tốt nhất (chấp nhận chậm hơn chút)
    print("Đang tải model Large để tăng độ chính xác...")
    detector = PoseDetector(model_path='yolov8l-pose.pt')
    mapper = CourtMapper()

    cap = cv2.VideoCapture(VIDEO_PATH)

    PLAYER_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]  # Đỏ, Xanh, Vàng

    # --- TẠO VÙNG ROI (REGION OF INTEREST) ---
    # Lấy 4 góc sân từ mapper để làm mốc
    src_pts = mapper.src_points

    # Mở rộng vùng sân ra một chút (Padding) để VĐV chạy ra biên vẫn bắt được
    # Nhưng không được rộng quá kẻo dính trọng tài biên
    # Logic: Kéo rộng sang 2 bên và trên dưới
    # Đây là toạ độ thủ công dựa trên video TrackNet (Bạn có thể tinh chỉnh)
    roi_polygon = np.array([
        [src_pts[0][0] - 50, src_pts[0][1] + 50],  # Góc Dưới-Trái (Mở rộng)
        [src_pts[1][0] + 50, src_pts[1][1] + 50],  # Góc Dưới-Phải
        [src_pts[2][0] + 50, src_pts[2][1] - 50],  # Góc Trên-Phải
        [src_pts[3][0] - 50, src_pts[3][1] - 50]  # Góc Trên-Trái
    ], np.int32)
    roi_polygon = roi_polygon.reshape((-1, 1, 2))

    print("Đang chạy... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Resize chuẩn HD
        target_width = 1280
        height, width = frame.shape[:2]
        new_height = height

        if width > target_width:
            scaling_factor = target_width / float(width)
            new_height = int(height * scaling_factor)
            frame = cv2.resize(frame, (target_width, new_height))

        # --- A. XỬ LÝ AI ---
        results, annotated_frame = detector.process_frame(frame)

        # Vẽ vùng ROI màu xanh lá để bạn dễ debug (Xem ai nằm trong vạch này)
        cv2.polylines(annotated_frame, [roi_polygon], True, (0, 255, 0), 2)

        # --- B. LỌC NGƯỜI & MINIMAP ---
        minimap_img = mapper.court_img.copy()

        # Danh sách chứa những người hợp lệ (Là VĐV)
        valid_players = []

        if results[0].keypoints is not None and results[0].keypoints.id is not None:
            # Lấy ID của các đối tượng (Tracking ID)
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Duyệt qua từng người
            for i, track_id in enumerate(track_ids):
                # 1. Lấy toạ độ chân
                ankle_l = detector.get_keypoint_by_index(results, person_index=i, keypoint_index=15)
                ankle_r = detector.get_keypoint_by_index(results, person_index=i, keypoint_index=16)

                player_pos_video = None

                # Logic: Ưu tiên lấy điểm thấp nhất (gần mặt đất nhất) để chính xác hơn
                if ankle_l and ankle_r:
                    # Lấy trung điểm
                    mid_x = (ankle_l[0] + ankle_r[0]) // 2
                    mid_y = (ankle_l[1] + ankle_r[1]) // 2
                    player_pos_video = (mid_x, mid_y)
                elif ankle_l:
                    player_pos_video = ankle_l
                elif ankle_r:
                    player_pos_video = ankle_r

                # 2. BỘ LỌC TRỌNG TÀI (QUAN TRỌNG NHẤT)
                if player_pos_video:
                    if is_inside_court(player_pos_video, roi_polygon):
                        # Nếu nằm trong vùng xanh lá -> Là VĐV
                        valid_players.append((track_id, player_pos_video))
                    else:
                        # Nằm ngoài -> Trọng tài -> Bỏ qua
                        continue

        # --- C. VẼ KẾT QUẢ ---
        for track_id, pos in valid_players:
            # Dùng track_id để cố định màu (P1 luôn đỏ, P2 luôn xanh)
            color = PLAYER_COLORS[track_id % len(PLAYER_COLORS)]

            # Vẽ lên Video
            cv2.circle(annotated_frame, pos, 8, color, -1)
            cv2.putText(annotated_frame, f"ID:{track_id}", (pos[0], pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Vẽ lên Minimap
            minimap_point = mapper.convert_to_minimap(pos)
            if minimap_point:
                mx, my = minimap_point
                # Vẽ lên minimap
                cv2.circle(minimap_img, (mx, my), 8, color, -1)
                cv2.circle(minimap_img, (mx, my), 10, (255, 255, 255), 2)
                cv2.putText(minimap_img, f"{track_id}", (mx + 10, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize Minimap
        scale_mini = new_height / minimap_img.shape[0]
        minimap_resized = cv2.resize(minimap_img, None, fx=scale_mini, fy=scale_mini)
        final_display = np.hstack((annotated_frame, minimap_resized))

        cv2.imshow('Badminton AI - Referee Filtered', final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()