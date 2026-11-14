import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import textwrap
import os


class MangaRenderer:
    def __init__(self, font_path="/Library/Fonts/Comic Sans MS.ttf"):
        self.font_path = font_path

    def process_chapter(self, json_path: str, chapter_dir: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        output_dir = os.path.join(chapter_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        for page in data:
            img_filename = page["image_path"]
            img_path = os.path.join(chapter_dir, img_filename)

            if not os.path.exists(img_path):
                continue

            img_cv = cv2.imread(img_path)
            img_pil = Image.open(img_path).convert("RGBA")
            h, w = img_cv.shape[:2]
            page_area = h * w

            contours = self._get_contours(img_cv)

            for bubble in page["bubbles"]:
                try:
                    ymin, xmin, ymax, xmax = bubble["bounding_box"]
                    y1, x1 = int(ymin / 1000 * h), int(xmin / 1000 * w)
                    y2, x2 = int(ymax / 1000 * h), int(xmax / 1000 * w)
                    gemini_box = (x1, y1, x2 - x1, y2 - y1)

                    # Detect if we found a natural bubble contour
                    target_box = self._get_best_bounding_box(gemini_box, contours, page_area)

                    # If target_box is the same as gemini_box, it's "NotSquare" (no bubble found)
                    is_natural_bubble = (target_box != gemini_box)

                    img_pil = self._draw_text(
                        img_pil,
                        target_box,
                        bubble["translated_text"],
                        bubble["text_color"],
                        bubble["bg_color"],
                        is_solidified=not is_natural_bubble
                    )
                except Exception as e:
                    print(f"Error in {img_filename} at {bubble.get('id')}: {e}")
                    continue

            output_path = os.path.join(output_dir, f"rendered_{img_filename}")
            img_pil.convert("RGB").save(output_path)
            print(f"Saved: {output_path}")

    def _get_contours(self, img_cv):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(blurred, kernel, iterations=1)
        edges = cv2.Canny(eroded, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _get_best_bounding_box(self, gemini_box, contours, page_area):
        gx, gy, gw, gh = gemini_box
        g_area = gw * gh
        best_green, best_yellow = None, None
        max_green_area, min_yellow_area = -1, float('inf')

        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            c_area = cw * ch
            if c_area == 0: continue

            ix_min, iy_min = max(gx, cx), max(gy, cy)
            ix_max, iy_max = min(gx + gw, cx + cw), min(gy + gh, cy + ch)
            intersection_area = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)

            encloses = (cx <= gx and cy <= gy and cx + cw >= gx + gw and cy + ch >= gy + gh)
            if encloses and c_area <= (4 * g_area) and c_area < (0.15 * page_area):
                if c_area < min_yellow_area:
                    min_yellow_area = c_area
                    best_yellow = (cx, cy, cw, ch)

            if intersection_area / c_area >= 0.8:
                if c_area > (0.4 * g_area) and c_area > max_green_area:
                    max_green_area = c_area
                    best_green = (cx, cy, cw, ch)

        return best_green if best_green else (best_yellow if best_yellow else gemini_box)

    def _draw_text(self, img_pil, box, text, text_color, bg_color, is_solidified=False):
        x, y, w, h = box
        draw = ImageDraw.Draw(img_pil)

        # 1. Calculate optimal font size and wrapping
        font_size = min(40, max(14, min(h, w) // 3))
        wrapped_text = []

        while font_size >= 12:
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except:
                font = ImageFont.load_default()
                break

            chars_per_line = max(1, int((w * 0.95) / (font_size * 0.5)))
            wrapped_text = textwrap.wrap(text, width=chars_per_line, break_long_words=False)

            line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in wrapped_text]
            total_h = sum([b[3] - b[1] for b in line_bboxes]) + (len(wrapped_text) - 1) * 4
            max_w = max([b[2] - b[0] for b in line_bboxes] + [0])

            if total_h <= h * 0.9 and max_w <= w * 0.9:
                break
            font_size -= 1

        # 2. Calculate the "Solidified" Tight Box (10px padding)
        line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in wrapped_text]
        text_block_w = max([b[2] - b[0] for b in line_bboxes] + [0])
        text_block_h = sum([b[3] - b[1] for b in line_bboxes]) + (len(wrapped_text) - 1) * 4

        # Center the tight box within the target area
        center_x = x + (w // 2)
        center_y = y + (h // 2)

        padding = 10
        tight_box = [
            center_x - (text_block_w // 2) - padding,
            center_y - (text_block_h // 2) - padding,
            center_x + (text_block_w // 2) + padding,
            center_y + (text_block_h // 2) + padding
        ]

        # 3. Draw Background Mask
        if is_solidified:
            # Rounded rectangle for a natural, non-square look
            draw.rounded_rectangle(tight_box, radius=8, fill=tuple(bg_color), outline=(0, 0, 0), width=1)
        else:
            # Fill the entire detected bubble contour area
            draw.rectangle([x, y, x + w, y + h], fill=tuple(bg_color))

        # 4. Render Text
        current_y = center_y - (text_block_h // 2)
        for line in wrapped_text:
            l_bbox = draw.textbbox((0, 0), line, font=font)
            l_w = l_bbox[2] - l_bbox[0]
            l_h = l_bbox[3] - l_bbox[1]
            draw.text((center_x - (l_w // 2), current_y), line, font=font, fill=tuple(text_color))
            current_y += l_h + 4

        return img_pil

class BoundingBoxTester:
    def __init__(self):
        pass

    def debug_chapter(self, json_path: str, chapter_dir: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for page in data:
            img_filename = page["image_path"]
            img_path = os.path.join(chapter_dir, img_filename)

            if not os.path.exists(img_path):
                print(f"Not found: {img_path}")
                continue

            img_cv = cv2.imread(img_path)
            h, w = img_cv.shape[:2]
            page_area = h * w
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Pre-processing: Blur and Erosion to clean screentone noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(blurred, kernel, iterations=1)

            # Edge Detection
            edges = cv2.Canny(eroded, 100, 200)

            # Find all contours for shape tracing
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # BGR mask for visualization
            debug_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Process bubbles from Gemini JSON
            for bubble in page["bubbles"]:
                # Convert normalized 0–1000 coordinates to absolute pixel values
                try:
                    ymin, xmin, ymax, xmax = bubble["bounding_box"]
                except Exception as e:
                    print(e)

                y1, x1 = int(ymin / 1000 * h), int(xmin / 1000 * w)
                y2, x2 = int(ymax / 1000 * h), int(xmax / 1000 * w)

                gx, gy, gw, gh = x1, y1, x2 - x1, y2 - y1
                g_area = gw * gh

                best_inside_cnt = None
                max_inside_area = -1

                best_outside_cnt = None
                min_outside_area = float('inf')

                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    c_area = cw * ch

                    if c_area == 0:
                        continue

                    # Intersection logic for association
                    ix_min, iy_min = max(gx, cx), max(gy, cy)
                    ix_max, iy_max = min(gx + gw, cx + cw), min(gy + gh, cy + ch)
                    intersection_area = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)

                    # --- YELLOW LOGIC: Outside Traces ---
                    # 1. Must completely enclose red box
                    # 2. Must be <= 5x the area of the red box
                    # 3. Must be < 20% of the total page area
                    completely_encloses = (cx <= gx and cy <= gy and
                                           cx + cw >= gx + gw and
                                           cy + ch >= gy + gh)

                    if completely_encloses and c_area <= (5 * g_area) and c_area < (0.17 * page_area):
                        if c_area < min_outside_area:
                            min_outside_area = c_area
                            best_outside_cnt = cnt

                    # --- GREEN LOGIC: Inside Traces ---
                    # Must be > 0.4x the area of the red box
                    if intersection_area / c_area >= 0.8:
                        if c_area > (0.4 * g_area) and c_area > max_inside_area:
                            max_inside_area = c_area
                            best_inside_cnt = cnt

                # 1. Draw Outside Trace (Yellow)
                if best_outside_cnt is not None:
                    cv2.drawContours(debug_mask, [best_outside_cnt], -1, (0, 255, 255), 2)

                # 2. Draw Inside Trace (Green) - Smoothed to rounded/geometric outline
                if best_inside_cnt is not None:
                    # Use Convex Hull to create a smooth, non-jittery rounded outline
                    hull = cv2.convexHull(best_inside_cnt)
                    # Further approximate to smooth out minor edge noise
                    epsilon = 0.01 * cv2.arcLength(hull, True)
                    smoothed_cnt = cv2.approxPolyDP(hull, epsilon, True)

                    cv2.drawContours(debug_mask, [smoothed_cnt], -1, (0, 255, 0), 2)

                # 3. Draw Original Gemini Box (Red)
                cv2.rectangle(debug_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

            output_path = os.path.join(chapter_dir, f"debug_bbox_{img_filename}")
            cv2.imwrite(output_path, debug_mask)
            print(f"Saved: {output_path}")

# Usage implementation:
# renderer = MangaRenderer()
# renderer.process_chapter("chapter-8/imagedata.json", "chapter-8/")
#
# tester = BoundingBoxTester()
# tester.debug_chapter("chapter-8/imagedata.json", "chapter-8/")