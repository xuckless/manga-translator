import json
import re
import math
from pathlib import Path
from PIL import Image

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


class MangaOCRExtractor:
    def __init__(self, directory_path: str, proximity_threshold: int = 25):
        """
        Initializes the OCR extraction engine optimized for Apple Silicon.

        :param directory_path: Root directory containing manga images.
        :param proximity_threshold: Max distance in pixels ($N$) to merge bounding boxes.
        """
        self.directory_path = Path(directory_path)
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        self.proximity_threshold = proximity_threshold

        # Load models for M2 MacBook Pro optimization
        self.foundation_predictor = FoundationPredictor()
        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
        self.detection_predictor = DetectionPredictor()

    def _is_valid_japanese_text(self, text: str) -> bool:
        """
        Filters noise by checking for Japanese characters or numbers.
        Discards boxes containing only ASCII letters or symbols.
        """
        # Unicode ranges: Hiragana, Katakana, Kanji, and Numbers (Standard & Full-width)
        jp_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF0-9\uFF10-\uFF19]')

        if not jp_pattern.search(text):
            return False

        # Explicitly discard if it's purely ASCII letters/symbols (artifacts)
        if re.fullmatch(r'[a-zA-Z\s\W\d]+', text) and not any(ord(c) > 127 for c in text):
            # Check if it contains digits; if it's just letters/symbols, it's noise
            if not any(c.isdigit() for c in text):
                return False

        return True

    def _get_distance(self, box1: list, box2: list) -> float:
        """Calculates the minimum distance between two bounding boxes [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        left = x2_2 < x1_1
        right = x1_2 > x2_1
        bottom = y2_2 < y1_1
        top = y1_2 > y2_1

        if top and left:
            return math.hypot(x1_1 - x2_2, y1_2 - y2_1)
        elif left and bottom:
            return math.hypot(x1_1 - x2_2, y1_1 - y2_2)
        elif bottom and right:
            return math.hypot(x1_2 - x2_1, y1_1 - y2_2)
        elif right and top:
            return math.hypot(x1_2 - x2_1, y1_2 - y2_1)
        elif left:
            return x1_1 - x2_2
        elif right:
            return x1_2 - x2_1
        elif bottom:
            return y1_1 - y2_2
        elif top:
            return y1_2 - y2_1
        else:  # Overlap
            return 0.0

    def _merge_bubbles(self, raw_bubbles: list) -> list:
        """Groups nearby bounding boxes into single contextual bubbles."""
        if not raw_bubbles:
            return []

        merged = []
        visited = [False] * len(raw_bubbles)

        for i in range(len(raw_bubbles)):
            if visited[i]:
                continue

            # Start a new cluster
            cluster = [i]
            visited[i] = True

            # Find all connected boxes within threshold
            found_new = True
            while found_new:
                found_new = False
                for j in range(len(raw_bubbles)):
                    if not visited[j]:
                        if any(self._get_distance(raw_bubbles[idx]['bounding_box'],
                                                  raw_bubbles[j]['bounding_box']) <= self.proximity_threshold
                               for idx in cluster):
                            cluster.append(j)
                            visited[j] = True
                            found_new = True

            # Merge cluster data
            cluster_boxes = [raw_bubbles[idx]['bounding_box'] for idx in cluster]
            cluster_texts = [raw_bubbles[idx]['original_text'] for idx in cluster]

            x1 = min(b[0] for b in cluster_boxes)
            y1 = min(b[1] for b in cluster_boxes)
            x2 = max(b[2] for b in cluster_boxes)
            y2 = max(b[3] for b in cluster_boxes)

            merged.append({
                "id": f"bubble_{len(merged) + 1}",
                "original_text": "".join(cluster_texts),  # Contextual joining
                "translated_text": None,
                "bounding_box": [x1, y1, x2, y2],
                "text_color": [255, 255, 255],  # Placeholder for rendering
                "bg_color": [0, 0, 0],  # Placeholder for rendering
                "is_filled": False
            })

        return merged

    def _process_image(self, file_path: Path) -> dict:
        image = Image.open(file_path).convert("RGB")
        predictions = self.recognition_predictor([image], det_predictor=self.detection_predictor)
        page_prediction = predictions[0]

        raw_bubbles = []
        for text_line in page_prediction.text_lines:
            text = text_line.text.strip()

            # Step 1: Character-Based Noise Filtering
            if self._is_valid_japanese_text(text):
                raw_bubbles.append({
                    "original_text": text,
                    "bounding_box": text_line.bbox
                })

        # Step 2: Proximity-Based Bounding Box Merging
        final_bubbles = self._merge_bubbles(raw_bubbles)

        return {
            "image_path": str(file_path.relative_to(self.directory_path).as_posix()),
            "bubbles": final_bubbles
        }

    def process_directory(self):
        """Recursively scans and processes images, saving to imagedata.json."""
        image_paths = sorted(
            p for p in self.directory_path.rglob('*')
            if p.suffix.lower() in self.image_extensions
        )

        all_pages_data = []
        for file_path in image_paths:
            print(f"Processing: {file_path.name}")
            page_data = self._process_image(file_path)
            all_pages_data.append(page_data)

        output_file = self.directory_path / 'imagedata.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_pages_data, f, ensure_ascii=False, indent=2)

        print(f"Extraction complete. Data saved to {output_file}")


if __name__ == "__main__":
    # Adjust directory_path to your manga source folder
    extractor = MangaOCRExtractor(directory_path="./manga_folder", proximity_threshold=30)
    extractor.process_directory()