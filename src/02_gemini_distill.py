import os
import time
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# 1. Environment & Client Setup
load_dotenv()
API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_GENAI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

# 2. Schema Definition
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "document_type": {"type": "STRING", "enum": ["official_letter", "regulation", "report", "unknown"]},
        "language": {"type": "STRING", "enum": ["arabic", "mixed"]},
        "dates_mentioned": {"type": "ARRAY", "items": {"type": "STRING"}},
        "entities": {
            "type": "OBJECT",
            "properties": {
                "organizations": {"type": "ARRAY", "items": {"type": "STRING"}},
                "signatures": {"type": "ARRAY", "items": {"type": "STRING"}},
                "stamps_or_watermarks": {"type": "ARRAY", "items": {"type": "STRING"}}
            }
        },
        "full_text": {"type": "STRING"},
        "tables": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["document_type", "language", "full_text"]
}

SYSTEM_PROMPT = "Extract structured Arabic OCR data from this image. Return strictly valid JSON."

def process_image_and_call_model(image_path: Path, source_pdf: str, max_retries=5) -> Optional[dict]:
    retries = 0
    while retries < max_retries:
        try:
            img = Image.open(image_path)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img, SYSTEM_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SCHEMA,
                    temperature=0.1
                )
            )

            if not response.text:
                return None

            extracted_data = json.loads(response.text)
            extracted_data["source_pdf"] = source_pdf
            extracted_data["image_path"] = str(image_path)
            
            return extracted_data

        except Exception as e:
            error_msg = str(e)
            # Catch BOTH 429 (Rate Limit) and 503/500 (Google Server Overload)
            if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                print(f"\n[API BUSY] Hit {error_msg[:3]} error (Server/Quota). Cooling down for 60 seconds before retrying... (Attempt {retries + 1}/{max_retries})")
                time.sleep(60)
                retries += 1
                continue
            else:
                # For local errors (corrupted image, bad JSON), print and skip
                print(f"\n[ERROR] {image_path.name}: {e}")
                return None
                
    print(f"\n[ERROR] Max retries hit for {image_path.name}. Skipping.")
    return None

def main():
    input_root = Path("data/02_raw_images")
    output_file = Path("data/03_synthetic_json/ocr_dataset.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_images = list(input_root.rglob("*.jpg"))
    if not all_images:
        print("No images found.")
        return

    processed_images = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_images.add(data.get("image_path"))
                except json.JSONDecodeError:
                    continue
                    
    remaining_images = [img for img in all_images if str(img) not in processed_images]
    
    print(f"Total images: {len(all_images)}")
    print(f"Already processed: {len(processed_images)}. Skipping.")
    print(f"Resuming with {len(remaining_images)} images...")
    
    if not remaining_images:
        print("All images processed!")
        return

    with open(output_file, "a", encoding="utf-8") as f:
        for image_path in tqdm(remaining_images, desc="Distilling PDF pages"):
            source_pdf = image_path.parent.name 
            
            result = process_image_and_call_model(image_path, source_pdf)
            
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

            time.sleep(4.5)

if __name__ == "__main__":
    main()