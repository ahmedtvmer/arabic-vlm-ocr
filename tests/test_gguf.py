from openai import OpenAI
import base64

# Point the OpenAI client to your local llama.cpp server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "./test_data/page_002.jpg"
base64_image = encode_image(image_path)

print("Sending document to local GGUF engine...")

response = client.chat.completions.create(
    model="qwen2-vl-arabic-ocr", # The name doesn't actually matter for the local server
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Extract the structured Arabic OCR data from this document. Output strictly valid JSON."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.1,
    max_tokens=2048
)

print("\n--- MODEL OUTPUT ---")
print(response.choices[0].message.content)