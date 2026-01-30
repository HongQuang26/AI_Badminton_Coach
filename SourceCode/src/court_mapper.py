import cv2
import numpy as np


class CourtMapper:
    def __init__(self):
        # =================================================================
        # INPUT: TOẠ ĐỘ TỪ VIDEO TRACKNET (BẠN ĐÃ CUNG CẤP)
        # =================================================================
        self.src_points = np.float32([
            [199, 504],  # Bottom-Left
            [761, 503],  # Bottom-Right
            [645, 270],  # Top-Right
            [313, 269]  # Top-Left
        ])

        # =================================================================
        # CẤU HÌNH SÂN CHUẨN BWF
        # =================================================================
        # Kích thước thực tế (mét)
        REAL_LENGTH = 13.40
        REAL_WIDTH_DOUBLES = 6.10
        REAL_WIDTH_SINGLES = 5.18
        SERVICE_LINE_DIST = 1.98
        BACK_SERVICE_DIST = 0.76

        # Tỷ lệ quy đổi (Pixels per Meter)
        self.SCALE = 60
        self.PADDING = 60

        # Kích thước trên Minimap
        self.court_w_px = int(REAL_WIDTH_DOUBLES * self.SCALE)
        self.court_h_px = int(REAL_LENGTH * self.SCALE)
        self.singles_margin_px = int(((REAL_WIDTH_DOUBLES - REAL_WIDTH_SINGLES) / 2) * self.SCALE)
        self.service_dist_px = int(SERVICE_LINE_DIST * self.SCALE)
        self.back_service_dist_px = int(BACK_SERVICE_DIST * self.SCALE)

        self.map_width = self.court_w_px + 2 * self.PADDING
        self.map_height = self.court_h_px + 2 * self.PADDING

        # =================================================================
        # TÍNH MA TRẬN HOMOGRAPHY
        # =================================================================
        p1 = [self.PADDING, self.map_height - self.PADDING]
        p2 = [self.map_width - self.PADDING, self.map_height - self.PADDING]
        p3 = [self.map_width - self.PADDING, self.PADDING]
        p4 = [self.PADDING, self.PADDING]

        self.dst_points = np.float32([p1, p2, p3, p4])
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.court_img = self.create_detailed_court()

    def create_detailed_court(self):
        """Vẽ sân cầu lông chi tiết"""
        COLOR_BG = (0, 100, 0)
        COLOR_OUT = (34, 34, 34)
        COLOR_LINES = (255, 255, 255)
        THICKNESS = 2

        img = np.full((self.map_height, self.map_width, 3), COLOR_OUT, dtype=np.uint8)

        # Vẽ nền sân
        cv2.rectangle(img, (self.PADDING, self.PADDING),
                      (self.map_width - self.PADDING, self.map_height - self.PADDING),
                      COLOR_BG, -1)

        def draw_line(p1, p2):
            cv2.line(img, p1, p2, COLOR_LINES, THICKNESS)

        # Toạ độ các đường
        xl_out = self.PADDING
        xl_in = self.PADDING + self.singles_margin_px
        xr_in = self.map_width - self.PADDING - self.singles_margin_px
        xr_out = self.map_width - self.PADDING

        yt = self.PADDING
        yb = self.map_height - self.PADDING
        ynet = self.map_height // 2
        ys_t = ynet - self.service_dist_px
        ys_b = ynet + self.service_dist_px
        yb_t = yt + self.back_service_dist_px
        yb_b = yb - self.back_service_dist_px

        # Vẽ đường dọc
        draw_line((xl_out, yt), (xl_out, yb))
        draw_line((xr_out, yt), (xr_out, yb))
        draw_line((xl_in, yt), (xl_in, yb))
        draw_line((xr_in, yt), (xr_in, yb))

        # Vẽ đường ngang
        draw_line((xl_out, yt), (xr_out, yt))
        draw_line((xl_out, yb), (xr_out, yb))
        draw_line((xl_out, ynet), (xr_out, ynet))
        draw_line((xl_out, ys_t), (xr_out, ys_t))
        draw_line((xl_out, ys_b), (xr_out, ys_b))
        draw_line((xl_out, yb_t), (xr_out, yb_t))
        draw_line((xl_out, yb_b), (xr_out, yb_b))

        # Đường giữa
        draw_line((self.map_width // 2, yt), (self.map_width // 2, ys_t))
        draw_line((self.map_width // 2, ys_b), (self.map_width // 2, yb))

        return img

    def convert_to_minimap(self, point):
        if point is None: return None
        x, y = point
        original_point = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(original_point, self.matrix)
        return (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))

    def draw_minimap(self, current_pos):
        img_display = self.court_img.copy()
        if current_pos:
            x, y = current_pos
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                cv2.circle(img_display, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(img_display, (x, y), 10, (255, 255, 255), 2)
        return img_display