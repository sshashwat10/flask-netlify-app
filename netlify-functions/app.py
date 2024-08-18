from flask import Flask, jsonify, request
import os
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from paddleocr import PaddleOCR
import io

# Initialize Flask app
app = Flask(__name__)

# Set up PaddleOCR
paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", use_gpu=True, show_log=False)

# Model configuration
bnb_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "mychen76/mistral7b_ocr_to_json_v1"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map={"transformer.word_embeddings": 0, "transformer.word_embeddings_layernorm": 0, "lm_head": 0, "transformer.h": 0, "transformer.ln_f": 0}
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def paddle_scan(img):
    result = paddleocr.ocr(img, cls=True)
    result = result[0]  # Assuming only one image is processed
    boxes = [line[0] for line in result]  # Bounding boxes
    txts = [line[1][0] for line in result]  # Raw text
    scores = [line[1][1] for line in result]  # Scores
    return txts, boxes

def process_invoice(image_file):
    try:
        # Perform OCR
        receipt_texts, receipt_boxes = paddle_scan(image_file)
        prompt = f"""### Instruction:
        You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object.
        Don't make up value not in the Input. Output must be a well-formed JSON object.```json


        ### Input:
        {receipt_boxes}

        ### Output:
        """
        # Generate JSON output using Hugging Face model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        result_text = tokenizer.batch_decode(outputs)[0]

        return {
            "json_output": result_text
        }
    except Exception as e:
        return {
            "error": str(e)
        }

@app.route('/upload_ocr', methods=['POST'])
def upload_and_ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        result = process_invoice(image)
        return jsonify({"message": "File successfully uploaded", "ocr_result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
