# -*- coding: utf-8 -*-
import os
import torch
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import urllib.request

# Initialize the PaddleOCR and the Hugging Face model
bnb_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
device_map = {"transformer.word_embeddings": 0, "transformer.h": 0, "model.layers": 0}

model_id = "mychen76/mistral7b_ocr_to_json_v1"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map=device_map,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Initialize PaddleOCR
paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", use_gpu=True, show_log=False)

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to perform OCR using PaddleOCR
def paddle_scan(paddleocr, img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray, cls=True)
    result = result[0]  # Assuming only one image is processed
    boxes = [line[0] for line in result]       # Bounding boxes
    txts = [line[1][0] for line in result]     # Raw text
    scores = [line[1][1] for line in result]   # Scores
    return txts, result

# Function to process the invoice image and extract JSON
def process_invoice(image_path):
    try:
        # Perform OCR
        receipt_texts, receipt_boxes = paddle_scan(paddleocr, image_path)
        # Prepare prompt for Hugging Face model
        prompt = f"""### Instruction:
        You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object.
        Don't make up value not in the Input. Output must be a well-formed JSON object.```json


        ### Input:
        {receipt_boxes}

        ### Output:
        """
        # Generate JSON output using Hugging Face model
        with torch.inference_mode():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=512)
            result_text = tokenizer.batch_decode(outputs)[0]
        return {"json_output": result_text}
    except Exception as e:
        return {"error": str(e)}

# Flask routes
@app.route("/", methods=["GET"])
def home():
    return "Flask app is running!"

@app.route('/upload_ocr', methods=['POST'])
def upload_and_ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform OCR on the uploaded file
        receipt_image = Image.open(file_path)
        receipt_image_array = np.array(receipt_image.convert('RGB'))
        ocr_result = process_invoice(receipt_image_array)

        # Extract JSON output from the model's response
        prompt_str = str(ocr_result)
        output_index = prompt_str.find("### Output:")
        if output_index != -1:
            output_content = prompt_str[output_index + len("### Output:"):]
        else:
            output_content = "### Output: not found in the prompt."

        return jsonify({"message": "File successfully uploaded", "filename": filename, "ocr_result": output_content}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
