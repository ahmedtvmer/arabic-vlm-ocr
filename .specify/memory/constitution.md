# Constitution: Arabic OCR VLM Data Pipeline

**Core Tech Stack:**
- Python 3.10+
- `pdf2image` and `Pillow` (Image Processing)
- `google-genai` (Synthetic data via Gemini API)
- `json` and `pydantic` (Strict schema validation)

**Engineering Rules:**
1. Write clean, modular, production-ready Python code.
2. Always include robust error handling (try/except blocks).
3. Use explicit type hinting.
4. Do not hallucinate external libraries; stick to the provided stack.
5. Write local-first, memory-efficient code.

**Target JSON Schema:**
All synthetic data generated must strictly adhere to this exact JSON schema:
{
  "document_type": "official_letter | regulation | report | unknown",
  "language": "arabic | mixed",
  "dates_mentioned": ["YYYY-MM-DD", "Hijri Date"],
  "entities": {
    "organizations": [],
    "signatures": [],
    "stamps_or_watermarks": []
  },
  "full_text": "The exact, word-for-word Arabic text extracted.",
  "tables": []
}