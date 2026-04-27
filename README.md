# Arabic VLM OCR

Fine-tuned **Qwen2-VL 2B** for structured OCR on Arabic government documents.  
The model takes a scanned page image as input and returns a structured JSON object containing the full text, document type, dates, entities, and table data.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline at a Glance](#pipeline-at-a-glance)
- [Stage 1 — PDF to Images](#stage-1--pdf-to-images)
- [Stage 2 — Gemini Distillation](#stage-2--gemini-distillation)
- [Stage 3 — ShareGPT Formatting](#stage-3--sharegpt-formatting)
- [Stage 4 — LoRA Fine-Tuning](#stage-4--lora-fine-tuning)
- [Stage 5 — Adapter Merging](#stage-5--adapter-merging)
- [Stage 6 — GGUF Quantization](#stage-6--gguf-quantization)
- [Stage 7 — Local Inference](#stage-7--local-inference)
- [Stage 8 — Running agent](#stage-8--running-agent)
- [Output Schema](#output-schema)
- [Evaluation & Results](#evaluation--results)
- [Setup](#setup)
- [License](#license)

---

## Overview

Standard OCR engines struggle with complex Arabic document layouts — stamps, signatures, mixed headers, and right-to-left text flow.  
This project solves that by fine-tuning a small vision-language model (Qwen2-VL 2B) on a synthetic dataset distilled from Gemini 2.5 Flash, then quantizing it to GGUF for fast, offline inference via `llama.cpp`.

**Key characteristics:**

- **2,059 training images** across 71 PDF sources (official letters, regulations, reports)
- **Structured JSON output** — not raw text; every response includes document type classification, entity extraction, and full transcription
- **Runs offline** — the final Q4_K_M GGUF weighs ~940 MB and serves locally through `llama.cpp` with no API calls or cloud dependencies
- **End-to-end pipeline** — from raw PDFs to a deployed local server, every step is scripted and reproducible

---

## Project Structure

```
arabic-vlm-ocr/
├── src/
│   ├── 01_pdf_to_images.py        # PDF → grayscale JPEG pages
│   ├── 02_gemini_distill.py        # Image → structured JSON via Gemini API
│   ├── 03_format_sharegpt.py       # JSONL → LLaMA-Factory ShareGPT format
│   └── utils/
│       └── vision_helpers.py       # Resize, grayscale, base64 encoding
├── training/
│   └── lora_config.yaml            # LoRA hyperparameters for LLaMA-Factory
├── data/
│   ├── 01_raw_pdfs/                # Source PDF documents
│   ├── 02_raw_images/              # Extracted page images (71 folders)
│   ├── 03_synthetic_json/          # Gemini distillation output (JSONL)
│   └── 04_training_data/           # Final ShareGPT-formatted dataset
├── model_weights/
│   ├── base_model/                 # Qwen2-VL-2B-Instruct base weights
│   ├── adapters/                   # Trained LoRA adapter weights
│   ├── qwen2-vl-arabic-ocr-merged/ # Merged full-precision model
│   ├── qwen2-vl-arabic-ocr-f16.gguf
│   ├── qwen2-vl-arabic-ocr-q4_k_m.gguf
│   └── qwen2-vl-arabic-mmproj.gguf # Vision projection for llama.cpp
├── llama.cpp/                      # llama.cpp (git submodule)
├── merge.py                        # Merge LoRA adapters into base model
├── test_gguf.py                    # Test quantized model via OpenAI-compatible API
├── test_data/                      # Sample images for manual testing
├── requirements.txt
└── pyproject.toml
```

---

## Pipeline at a Glance

```
 Raw PDFs
    │
    ▼  src/01_pdf_to_images.py
 Grayscale JPEGs (1024px max)
    │
    ▼  src/02_gemini_distill.py
 Structured JSON labels (JSONL)
    │
    ▼  src/03_format_sharegpt.py
 ShareGPT training dataset
    │
    ▼  LLaMA-Factory (external)
 LoRA adapters
    │
    ▼  merge.py
 Merged full-precision model
    │
    ▼  llama.cpp/convert & quantize
 GGUF model files
    │
    ▼  llama.cpp/llama-server
 Local OpenAI-compatible API
```

---

## Stage 1 — PDF to Images

**Script:** `src/01_pdf_to_images.py`

Converts every PDF in `data/01_raw_pdfs/` into individual page images.

**Processing details:**
- Uses `pdf2image` (wraps Poppler) to rasterize each page
- Each page is passed through `vision_helpers.process_image()` which:
  - Converts to **grayscale** (`L` mode) to remove color noise
  - Downscales to a **max dimension of 1024px** while preserving aspect ratio (uses Lanczos resampling)
- Output is saved as JPEG to `data/02_raw_images/<pdf_name>/page_<N>.jpg`

```bash
python -m src.01_pdf_to_images
```

---

## Stage 2 — Gemini Distillation

**Script:** `src/02_gemini_distill.py`

Sends each page image to **Gemini 2.5 Flash** with a constrained JSON schema to generate structured OCR annotations. This is the synthetic labeling step — Gemini acts as the teacher model.

**Processing details:**
- Loads the Google GenAI API key from `.env` (`GOOGLE_GENAI_API_KEY`)
- For each image, sends a `generate_content` request with:
  - `response_mime_type="application/json"` — forces JSON output
  - `response_schema=SCHEMA` — enforces the exact output structure (see [Output Schema](#output-schema))
  - `temperature=0.1` — near-deterministic extraction
- Implements automatic retry logic with 60-second cooldown for `429` / `503` / `RESOURCE_EXHAUSTED` errors (up to 5 retries per image)
- Writes results incrementally to `data/03_synthetic_json/ocr_dataset.jsonl` (one JSON object per line)
- **Resumable** — on restart, skips images that already appear in the output file
- Throttles at **4.5 seconds between requests** to respect Gemini rate limits

```bash
python -m src.02_gemini_distill
```

---

## Stage 3 — ShareGPT Formatting

**Script:** `src/03_format_sharegpt.py`

Transforms the raw JSONL annotations into [LLaMA-Factory's multimodal ShareGPT format](https://github.com/hiyouga/LLaMA-Factory) for direct use in fine-tuning.

**Processing details:**
- Reads from `data/03_synthetic_json/legacy_dataset.jsonl`
- For each record:
  - Extracts `image_path` and `output` fields
  - Corrects the path from the original Google Drive structure to the local `data/02_raw_images/` layout
  - Validates the local image file exists on disk
  - Parses the `output` string as JSON and re-serializes it with `indent=2` for clean formatting
- Constructs a ShareGPT conversation pair:
  - **User turn:** `<image>\nExtract the structured Arabic OCR data from this document. Output strictly valid JSON.`
  - **Assistant turn:** the clean JSON string
  - **Images array:** path to the local image file
- Writes the complete dataset to `data/04_training_data/sharegpt_dataset.json`

```bash
python -m src.03_format_sharegpt
```

---

## Stage 4 — LoRA Fine-Tuning

Fine-tuning is performed externally using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

**Configuration:** `training/lora_config.yaml`

**Base model:** `Qwen/Qwen2-VL-2B-Instruct`

LLaMA-Factory handles:
- Loading the base Qwen2-VL model with the ShareGPT dataset
- Applying LoRA adapters to the model's attention layers
- Multi-modal training with image + text pairs
- Saving the trained adapter weights to `model_weights/adapters/`

---

## Stage 5 — Adapter Merging

**Script:** `merge.py`

Merges the trained LoRA adapter weights back into the base model to produce a single, standalone model.

**Processing details:**
- Loads the base `Qwen2VLForConditionalGeneration` model on CPU in `float16`
- Attaches the LoRA adapter from `model_weights/adapters/` using `PeftModel`
- Calls `merge_and_unload()` to fuse the adapter weights into the base model permanently
- Saves the merged model with `safe_serialization=True` (safetensors format) to `model_weights/qwen2-vl-arabic-ocr-merged/`
- Also copies the `AutoProcessor` (tokenizer + image processor) to the merged directory

```bash
python merge.py
```

---

## Stage 6 — GGUF Quantization

Conversion is done using **llama.cpp**'s built-in tooling (included as a git submodule).

**Generated files:**

| File | Size | Description |
|---|---|---|
| `qwen2-vl-arabic-ocr-f16.gguf` | ~2.9 GB | Full float16 precision |
| `qwen2-vl-arabic-ocr-q4_k_m.gguf` | ~940 MB | 4-bit quantized (recommended) |
| `qwen2-vl-arabic-mmproj.gguf` | ~1.2 GB | Vision encoder projection |

**Typical conversion commands:**

```bash
# Convert merged HF model to f16 GGUF
python llama.cpp/convert_hf_to_gguf.py \
    model_weights/qwen2-vl-arabic-ocr-merged \
    --outfile model_weights/qwen2-vl-arabic-ocr-f16.gguf \
    --outtype f16

# Quantize to Q4_K_M
./llama.cpp/build/bin/llama-quantize \
    model_weights/qwen2-vl-arabic-ocr-f16.gguf \
    model_weights/qwen2-vl-arabic-ocr-q4_k_m.gguf \
    Q4_K_M
```

---

## Stage 7 — Local Inference

### Serving with llama.cpp

Start the OpenAI-compatible server using the quantized model and vision projection:

```bash
./llama.cpp/build/bin/llama-server \
    -m model_weights/qwen2-vl-arabic-ocr-q4_k_m.gguf \
    --mmproj model_weights/qwen2-vl-arabic-mmproj.gguf \
    -c 4096 \
    --port 8080
```

### Testing with the OpenAI client

**Script:** `test_gguf.py`

Sends a test image to the local server using the standard OpenAI Python client:

```bash
python test_gguf.py
```

The script:
- Reads and base64-encodes the test image from `test_data/page_002.jpg`
- Sends a `chat.completions.create` request to `http://localhost:8080/v1`
- Uses `temperature=0.1` and `max_tokens=2048`
- Prints the extracted structured JSON to stdout

---

## Stage 8 — Running the Agentic Workflow

This stage integrates the local OCR engine into a **LangGraph** orchestration. It uses a two-node architecture to ensure high-fidelity results even on hardware-constrained environments.

### 1. Prerequisites
* **Local Server:** Ensure the `llama-server` is running on port `8080` (see the "Usage" section above).
* **API Key:** Ensure you have your `GOOGLE_API_KEY` (for Gemini) or `OPENAI_API_KEY` exported in your `.env` file to power the Reasoning Node.

### 2. Execution
Run the orchestrator script using `uv`:
```bash
uv run python agent.py
```

## Output Schema

Every model response follows this structured JSON schema:

```json
{
  "document_type": "official_letter | regulation | report | unknown",
  "language": "arabic | mixed",
  "dates_mentioned": ["١٤٤٠/١٠/٢٨هـ"],
  "entities": {
    "organizations": ["وزارة العدل"],
    "signatures": ["سعد بن محمد السيف"],
    "stamps_or_watermarks": ["شعار وزارة العدل (ميزان العدالة)"]
  },
  "full_text": "بسم الله الرحمن الرحيم ...",
  "tables": ["..."]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `document_type` | `string` | ✅ | Classification: `official_letter`, `regulation`, `report`, or `unknown` |
| `language` | `string` | ✅ | Detected language: `arabic` or `mixed` |
| `dates_mentioned` | `string[]` | ❌ | All dates found in the document (Hijri or Gregorian) |
| `entities.organizations` | `string[]` | ❌ | Government bodies, ministries, courts mentioned |
| `entities.signatures` | `string[]` | ❌ | Names appearing in signature blocks |
| `entities.stamps_or_watermarks` | `string[]` | ❌ | Visual seals or watermarks described |
| `full_text` | `string` | ✅ | Complete transcription of all visible text |
| `tables` | `string[]` | ❌ | Table data extracted as strings |

---

## Evaluation & Results

The fine-tuning process was executed using **Parameter-Efficient Fine-Tuning (PEFT)** via LoRA. The model demonstrated strong convergence, specifically mastering the complex nested JSON schema and the nuances of right-to-left (RTL) Arabic text alignment in official documents.

### Training Metrics

| Metric | Start | End | Improvement |
| :--- | :--- | :--- | :--- |
| **Training Loss** | 1.2 | **0.01** | ~99% |
| **Evaluation Loss** | 1.5 | **0.15** | ~90% |

### Qualitative Analysis

* **Schema Adherence:** The base Qwen2-VL model frequently failed to close JSON brackets or hallucinated keys. The fine-tuned version maintains **100% schema validity** across test samples.
* **Entity Extraction:** Significant improvement in identifying Saudi governmental entities (e.g., Ministry of Justice, Royal Court) and distinguishing between stamps and signatures.
* **Tabular Data:** Resolved the "text bleeding" issue where text from adjacent columns would merge. The model now preserves column integrity in the `tables` output field.

### Inference Performance (Local GGUF)

| Hardware | Backend | Quantization | Latency (tokens/s) |
| :--- | :--- | :--- | :--- |
| **Laptop GPU (4GB VRAM)** | llama.cpp | Q4_K_M | ~12-15 t/s |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Poppler (`pdf2image` dependency) — install via `apt install poppler-utils` or `brew install poppler`
- CMake and a C++ compiler (for building llama.cpp)

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/ahmedtvmer/arabic-vlm-ocr.git
cd arabic-vlm-ocr

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Set up environment variables
cp ".env example" .env
# Edit .env and add your GOOGLE_GENAI_API_KEY
```

### Building llama.cpp

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release -j$(nproc)
cd ..
./llama.cpp/build/bin/llama-server \
    -m model_weights/qwen2-vl-arabic-ocr-q4_k_m.gguf \
    --mmproj model_weights/qwen2-vl-arabic-mmproj.gguf \
    --alias qwen2-vl-arabic-ocr-merged \
    -ngl 99 --port 8080
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.