# SourceCode/utils/coordinate_picker.py
import cv2
import os

# --- CẤU HÌNH ---
VIDEO_PATH = '../data/tracknet_test.mp4'  # Đường dẫn đến video của bạn


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Toạ độ: ({x}, {y})")
        # Vẽ một chấm đỏ để đánh dấu
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Chon 4 goc san', img_display)


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Lỗi: Không tìm thấy file {VIDEO_PATH}")
        exit()

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Đọc frame đầu tiên (hoặc frame thứ 100 cho rõ sân)
    # Nếu sân bị che lúc đầu, bạn có thể loop để tìm frame đẹp
    success, frame = cap.read()

    if not success:
        print("Không đọc được video.")
        exit()

    # Resize về 960px (giống hệt lúc chạy AI để toạ độ khớp nhau)
    target_width = 960
    height, width = frame.shape[:2]
    if width > target_width:
        scaling_factor = target_width / float(width)
        new_height = int(height * scaling_factor)
        frame = cv2.resize(frame, (target_width, new_height))

    img_display = frame.copy()

    print("--- HƯỚNG DẪN ---")
    print("1. Click chuột vào 4 góc sân theo thứ tự:")
    print("   (1) Góc Dưới-Trái  (Bottom-Left)")
    print("   (2) Góc Dưới-Phải  (Bottom-Right)")
    print("   (3) Góc Trên-Phải  (Top-Right)")
    print("   (4) Góc Trên-Trái  (Top-Left)")
    print("2. Nhìn cửa sổ Terminal (Console) để chép toạ độ.")
    print("3. Bấm phím bất kỳ để thoát.")

    cv2.imshow('Chon 4 goc san', img_display)
    cv2.setMouseCallback('Chon 4 goc san', mouse_callback)

    cv2.waitKey(0)
    cv2.destroyAllWindows()