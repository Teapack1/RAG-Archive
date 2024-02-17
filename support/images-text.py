import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import os

def extract_images_from_pdf(pdf_path, output_folder, dpi=300):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img_bytes = pix.tobytes("ppm")
        
        # Convert PPM to JPEG using PIL and save
        image = Image.open(io.BytesIO(img_bytes))
        output_filename = f"{output_folder}/page_{page_num + 1}.jpg"
        image.save(output_filename, "JPEG")
        
        # Also store the image bytes for OCR processing
        img_converted_bytes = io.BytesIO()
        image.save(img_converted_bytes, format='JPEG')
        images.append(img_converted_bytes.getvalue())
        
    print(f"Images saved to {output_folder}")
    return images

def ocr_images(images):
    text_from_images = []
    for image_bytes in images:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang='ces')
        text_from_images.append(text)
    return text_from_images

def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

# Path to your PDF file and output locations
pdf_path = "norm.pdf"
output_folder = "extracted_images"
output_text_file = "ocr_results.txt"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Extract images from the PDF and save them as JPEG
images = extract_images_from_pdf(pdf_path, output_folder)

# Transcribe text from the extracted images with Czech language
transcribed_texts = ocr_images(images)

# Combine all texts into a single string
combined_text = "\n".join(transcribed_texts)

# Save OCR results to a text file
save_text_to_file(combined_text, output_text_file)

print(f"OCR results saved to {output_text_file}")
