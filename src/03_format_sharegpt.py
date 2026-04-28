# src/03_format_sharegpt.py

import json
from pathlib import Path

INPUT_PATH = Path("data/03_synthetic_json/legacy_dataset.jsonl")
OUTPUT_PATH = Path("data/04_training_data/sharegpt_dataset.json")


def main() -> None:
    """
    Convert a legacy JSONL dataset into LLaMA-Factory's Multimodal ShareGPT format.
    """
    output_dir = OUTPUT_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sharegpt_records: list[dict] = []

    with open(INPUT_PATH, encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON at line {line_num}: {e}")
                continue

            # Requirement 3: Extract image_path and output strings.
            image_path_str = record.get("image_path")
            output_str = record.get("output")

            if image_path_str is None or output_str is None:
                print(f"Skipping record at line {line_num}: missing image_path or output.")
                continue

            # Requirement 4: Path Correction.
            # The original `image_path` is a Google Drive path (e.g.,
            # "drive/0001/page_001.jpg"). Extract the last two parts.
            path_parts = image_path_str.strip("/").split("/")
            if len(path_parts) < 2:
                print(f"Skipping record at line {line_num}: invalid path format '{image_path_str}'.")
                continue

            folder_name = path_parts[-2]
            file_name = path_parts[-1]
            local_image_path = Path(f"data/02_raw_images/{folder_name}/{file_name}")

            if not local_image_path.exists():
                print(f"Warning: Local file not found at {local_image_path}. Skipping record.")
                continue

            # Requirement 5: Output Parsing.
            try:
                parsed_output = json.loads(output_str)
            except json.JSONDecodeError as e:
                print(f"Skipping record at line {line_num}: output is not valid JSON: {e}")
                continue

            # Re-format the parsed dictionary into a clean, pretty-printed string.
            clean_output_string = json.dumps(parsed_output, indent=2, ensure_ascii=True)

            # Requirement 6-8: Construct ShareGPT record.
            sharegpt_record = {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>\nExtract the structured Arabic OCR data from this document. Output strictly valid JSON."
                    },
                    {
                        "role": "assistant",
                        "content": clean_output_string
                    }
                ],
                "images": [str(local_image_path)]
            }

            sharegpt_records.append(sharegpt_record)

    # Requirement 9: Save master list to output JSON.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        json.dump(sharegpt_records, f_out, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
