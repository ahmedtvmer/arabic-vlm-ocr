import pdf2image
import os
from tqdm import tqdm
from src.utils.vision_helpers import process_image

if __name__ == "__main__":
    input_directory = "data/01_raw_pdfs"
    output_directory = "data/02_raw_images"

    os.makedirs(output_directory, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_directory) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDFs found in {input_directory}. Exiting.")
        exit()

    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        pdf_path = os.path.join(input_directory, pdf_file)
        
        # Create a specific folder for this PDF's pages
        pdf_out_dir = os.path.join(output_directory, pdf_file.replace(".pdf", ""))
        os.makedirs(pdf_out_dir, exist_ok=True)

        # pdf2image returns a list of PIL Images
        images = pdf2image.convert_from_path(pdf_path)
        
        for i, image in enumerate(images):
            processed_image = process_image(image)
            save_path = os.path.join(pdf_out_dir, f"page_{i+1}.jpg")
            processed_image.save(save_path, "JPEG")