"""
app.py
Gradio UI for the 3DSP AI Coaching Prototype.
Features a video upload interface and a dashboard for biomechanical data and AI feedback.
"""
import gradio as gr
import json
import numpy as np
import os
from PIL import Image
from biomechanics import generate_report, smooth
from vlm_coach import load_vlm, generate_coaching_report

# Global variables to prevent reloading the heavy 7B model on every click
MODEL = None
PROCESSOR = None

def get_vlm():
    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None:
        MODEL, PROCESSOR = load_vlm()
    return MODEL, PROCESSOR

def mock_inference(video_path):
    """
    Simulates the ViTPose pipeline. 
    In the final deployment, this function will extract frames from `video_path`
    and run the real pose estimation model.
    """
    traj = np.random.rand(20, 17, 2) * 200 + 100 
    traj[:, 5, 1] = np.linspace(100, 300, 20) 
    traj_s = smooth(traj)
    conf = np.ones((20, 17)) * 0.85
    report = generate_report(traj_s, conf)
    return json.dumps(report, indent=2)

def run_pipeline(video):
    if video is None:
        gr.Warning("Please upload a video first.")
        return "No video provided.", "No video provided."

    # 1. Generate Biomechanical Data (Mock for prototype)
    report_json = mock_inference(video)
    
    # 2. Generate AI Coaching Report
    model, processor = get_vlm()
    # We use the sample skeleton image for the VLM context in this prototype
    skeleton_path = "assets/sample_skeleton.png" 
    report_text = generate_coaching_report(model, processor, report_json, skeleton_path)
    
    return report_json, report_text

# --- Gradio UI Layout ---
with gr.Blocks(title="3DSP AI Coach", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 3DSP AI Coaching Prototype")
    gr.Markdown("Upload a video of a football kick to receive an AI biomechanical analysis and coaching feedback.")
    
    with gr.Row():
        # Left Column: Input and Visuals
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload", "webcam"], height=400)
            analyze_btn = gr.Button("Analyze Technique", variant="primary", size="lg")
            skeleton_out = gr.Image(
                label="Analyzed Skeleton", 
                value="assets/sample_skeleton.png",
                interactive=False
            )
            
        # Right Column: Data and Language Outputs
        with gr.Column(scale=2):
            with gr.Tab("AI Coaching Report"):
                report_out = gr.Markdown(label="AI Coaching Report (VLM Layer)", height=600)
            with gr.Tab("Biomechanical Data"):
                json_out = gr.JSON(label="Structured JSON (Grounding Layer)", height=600)

    # Connect the button to the pipeline
    analyze_btn.click(
        fn=run_pipeline,
        inputs=[video_input],
        outputs=[json_out, report_out]
    )

if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    if not os.path.exists("assets/sample_skeleton.png"):
        Image.new('RGB', (192, 256), color='gray').save("assets/sample_skeleton.png")
    
    # Launch the app
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)