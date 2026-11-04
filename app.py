"""
app.py
Gradio UI for the AI Coaching Prototype.
"""
import gradio as gr
import json
import numpy as np
import os
from PIL import Image
from biomechanics import generate_report, smooth
from vlm_coach import load_vlm, generate_coaching_report

def mock_inference():
    traj = np.random.rand(20, 17, 2) * 200 + 100 
    traj[:, 5, 1] = np.linspace(100, 300, 20) 
    traj_s = smooth(traj)
    conf = np.ones((20, 17)) * 0.85
    report = generate_report(traj_s, conf)
    return json.dumps(report, indent=2), "assets/sample_skeleton.png"

def run_coach_pipeline():
    report_json, img_path = mock_inference()
    model, processor = load_vlm()
    report = generate_coaching_report(model, processor, report_json, img_path)
    return report_json, report

with gr.Blocks(title="3DSP AI Coach") as demo:
    gr.Markdown("# 3DSP AI Coaching Prototype")
    gr.Markdown("Click 'Analyze Technique' to receive an AI biomechanical analysis (uses mock data for demonstration).")
    
    with gr.Row():
        with gr.Column():
            btn = gr.Button("Analyze Technique")
            json_out = gr.JSON(label="Biomechanical Data (Grounding Layer)")
        with gr.Column():
            report_out = gr.Markdown(label="AI Coaching Report (VLM Layer)")
            
    btn.click(run_coach_pipeline, outputs=[json_out, report_out])

if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    Image.new('RGB', (192, 256), color = 'gray').save("assets/sample_skeleton.png")
    demo.launch()