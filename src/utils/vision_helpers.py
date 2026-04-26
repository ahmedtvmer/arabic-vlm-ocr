from PIL import Image
import io
import base64

def process_image(img, max_dim=1024):
    # If a string path is passed by mistake, handle it
    if isinstance(img, str):
        img = Image.open(img)
        
    # Convert to grayscale
    img = img.convert("L")
    
    # Resize the image while maintaining aspect ratio, only if it exceeds max_dim
    width, height = img.size
    if width > max_dim or height > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    return img

def encode_image_to_base64(pil_image):
    # Convert PIL image to base64 string
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")