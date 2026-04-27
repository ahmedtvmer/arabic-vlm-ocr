import torch
import warnings
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Suppress the non-critical CUDA driver warning since we are merging on CPU
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

# Define relative paths based on your project structure
BASE_PATH = "./model_weights/base_model"
ADAPTER_PATH = "./model_weights/adapters"
EXPORT_PATH = "./model_weights/qwen2-vl-arabic-ocr-merged"

def run_merge():
    print(f"Step 1: Loading Qwen2-VL base model on CPU...")
    try:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_PATH,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print("Step 2: Attaching LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except Exception as e:
        print(f"Error loading adapters: {e}")
        return

    print("Step 3: Fusing weights (merge_and_unload)...")
    model = model.merge_and_unload()

    print(f"Step 4: Serializing unified weights to {EXPORT_PATH}...")
    model.save_pretrained(EXPORT_PATH, safe_serialization=True)

    print("Step 5: Loading and saving AutoProcessor...")
    # NOTE: This step explicitly requires 'torchvision' to be installed
    try:
        processor = AutoProcessor.from_pretrained(BASE_PATH, trust_remote_code=True)
        processor.save_pretrained(EXPORT_PATH)
    except ImportError as e:
        print(f"Processor Error: Missing dependency. {e}")
        return
    except Exception as e:
        print(f"Processor Error: {e}")
        return

    print("\nSuccess. The model is merged and ready for standalone inference.")

if __name__ == "__main__":
    run_merge()