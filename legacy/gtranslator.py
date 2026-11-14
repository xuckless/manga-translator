import os
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict
import google.generativeai as genai


class MangaTranslationPipeline:
    """
    A pipeline to automate the localization of manga pages using Google's Gemini Vision models.
    Updated to skip previously processed images and append new results to existing JSON files.
    """

    def __init__(self, api_key: str, repo_path: str, output_filename: str = "imagedata.json"):
        """
        Initializes the pipeline with API credentials and localization rules.
        """
        self.repo_path = Path(repo_path)
        self.output_filename = output_filename
        genai.configure(api_key=api_key)

        system_instruction = """You are a specialized Manga Localization Agent. Your goal is to perform high-fidelity OCR and context-aware translation on manga pages.
                            Rules for Extraction:
                            Exhaustive Detection: Detect every text element, including large speech bubbles, small side-text, date markers (e.g., "5 months ago"), and handwritten sound effects.
                            Reading Order: Strictly follow the Japanese reading order (Top-to-Bottom, Right-to-Left). Do not fragment lines within a single bubble. When encountering new line \n characters incorporate them into translation as well at accurate spaces where you encountered them. 
                            Contextual Translation: Use the visual context to inform translations. Specifically, if "孕んでいた" appears in a context involving danger, dungeons, or abstract concepts, translate it as "fraught with" or "contained" rather than "pregnant."
                            Bounding Boxes: Provide one bounding box per cohesive text block/bubble. Bounding box should encapsulate the text.
                            Use the Google native normalized coordinate format: [ymin, xmin, ymax, xmax] where 0-1000 represents the image dimensions.
                            Colors: Estimate the average text_color and bg_color in RGB format for each bubble.
                            text_color: Is the color of the text in the image
                            bg_color: Is the color of the background in the text
                            ID: give a unique ID for every bubble
                            Image path: Use the file name provided to you.
                            IsFilled: Only enter "false" here.
                            Output Format:
                            Output ONLY a valid JSON object.
                            Do not include markdown code blocks or any conversational filler.
                            [
                              {
                                "image_path": "image_path.jpg",
                                "bubbles": [
                                  {
                                    "id": "bubble_1",
                                    "original_text": "Japanese text here",
                                    "translated_text": "Natural English translation here",
                                    "bounding_box": [ymin, xmin, ymax, xmax],
                                    "text_color": [r, g, b],
                                    "bg_color": [r, g, b],
                                    "is_filled": false
                                  }
                                ]
                              }
                            ]
                            """

        self.model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction=system_instruction
        )

    def process_repository(self):
        """
        Scans repo_path for images, skips already translated images, and appends new results.
        """
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp')
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.repo_path.rglob(ext))

        directories = defaultdict(list)
        for img_path in image_paths:
            directories[img_path.parent].append(img_path)

        for directory, imgs in directories.items():
            output_file = directory / self.output_filename
            all_results = []
            processed_images = set()

            # Load existing data to avoid re-processing and to support appending
            if output_file.exists():
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        all_results = json.load(f)
                        # Create a set of filenames already present in the JSON
                        processed_images = {entry['image_path'] for entry in all_results if 'image_path' in entry}
                    print(f"Found existing {self.output_filename}. {len(processed_images)} images already processed.")
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Could not read existing {output_file}. Starting fresh. Error: {e}")
                    all_results = []

            new_results_count = 0
            for img_path in imgs:
                # Feature: Skip if image path is already in the JSON data
                if img_path.name in processed_images:
                    continue

                print(f"Processing: {img_path.name} in {directory}")
                result = self._process_single_image(img_path)

                if result:
                    if isinstance(result, list):
                        all_results.extend(result)
                        new_results_count += len(result)
                    else:
                        all_results.append(result)
                        new_results_count += 1

            # Feature: Save if new results were added
            if new_results_count > 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"Updated {output_file} with {new_results_count} new image records.")
            else:
                print(f"No new images to process in {directory}.")

    def _process_single_image(self, img_path: Path):
        """
        Sends a single image to the Gemini API and parses the JSON response.
        """
        try:
            image = Image.open(img_path)
            prompt = f"Image path: {img_path.name}"

            response = self.model.generate_content(
                [image, prompt],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                )
            )

            text_response = response.text.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:-3].strip()
            elif text_response.startswith("```"):
                text_response = text_response[3:-3].strip()

            return json.loads(text_response)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            return None