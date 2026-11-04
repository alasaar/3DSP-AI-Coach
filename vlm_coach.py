"""
vlm_coach.py
Loads Qwen2.5-VL-7B-Instruct in 4-bit quantization and generates a 
structured coaching report based on the biomechanics JSON and a skeleton image.
"""
import torch
import json
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

def load_vlm():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_quant_type="nf4"
    )
    print("Loading Qwen2.5-VL-7B (4-bit)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        device_map="auto", 
        quantization_config=quantization_config, 
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_coaching_report(model, processor, report_json, image_path):
    report = json.loads(report_json)
    flagged = [f for f in report.get("findings", []) if f.get("flagged")]
    
    if flagged:
        flagged_text = "\n".join([
            f"- {f['id']}: {f['description']} (Measured: {f['value']}, Threshold: {f['threshold']})" 
            for f in flagged
        ])
    else:
        flagged_text = "None. Technique was perfect according to thresholds."

    image = Image.open(image_path)

    messages = [
        {
            "role": "system", 
            "content": "You are an elite football biomechanics coach. Be professional, specific, and actionable. CRITICAL: Only discuss errors explicitly listed in the FLAGGED ISSUES. Do not invent mistakes."
        },
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text", 
                    "text": f"""Biomechanical Analysis:
[FLAGGED ISSUES]
{flagged_text}

[RAW DATA]
{report_json}

Provide a report with:
1. Overall Technique Summary (2 sentences)
2. Identified Mistakes (ONLY discuss flagged issues and explain the biomechanical 'why')
3. Actionable Corrections (2 specific drills/cues)
4. Strengths (Mention one thing done well, e.g., good trunk stability)
"""
                }
            ]
        }
    ]
    
    # CRITICAL FIX: Qwen2.5-VL requires process_vision_info for dynamic resolution
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    print("Generating report...")
    output_ids = model.generate(**inputs, max_new_tokens=512)
    
    # Decode ONLY the generated tokens (strips out the prompt)
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response