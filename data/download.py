"""
Download and extract 3DSP and CrowdPose datasets.
Run once before training:
    python -m data.download
"""

import os
import subprocess
import sys

from configs.default import DSP_ROOT, CROWD_ROOT


def clone_3dsp():
    """Clone the 3D-Shot-Posture-Dataset repo and extract the zip."""
    if not os.path.isdir(DSP_ROOT):
        print("[3DSP] Cloning repository …")
        subprocess.run(
            ["git", "clone",
             "https://github.com/calvinyeungck/3D-Shot-Posture-Dataset.git",
             DSP_ROOT],
            check=True,
        )

    zip_path  = os.path.join(DSP_ROOT, "3dsp.zip")
    train_dir = os.path.join(DSP_ROOT, "3dsp", "train")

    if not os.path.isdir(train_dir):
        print("[3DSP] Extracting 3dsp.zip …")
        subprocess.run(
            ["unzip", "-q", "-o", zip_path, "-d", DSP_ROOT],
            check=True,
        )

    n_shots = len(os.listdir(train_dir))
    n_imgs  = len(os.listdir(os.path.join(train_dir, "00001", "img")))
    print(f"[3DSP] {n_shots} shots  |  {n_imgs} frames/shot")


def download_crowdpose():
    """Download CrowdPose images + annotations via gdown."""
    images_dir = os.path.join(CROWD_ROOT, "images")
    ann_path   = os.path.join(CROWD_ROOT, "json", "crowdpose_train.json")

    if (os.path.isdir(images_dir)
            and len(os.listdir(images_dir)) > 19_000
            and os.path.exists(ann_path)):
        print("[CrowdPose] Already present – skipping.")
        return

    os.makedirs(CROWD_ROOT, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"],
                   check=False)

    urls = {
        "images.zip":      "https://drive.google.com/uc?id=1VprytECcLtU4tKP32SYi_7oDRbw7yUTL",
        "annotations.zip": "https://drive.google.com/uc?id=196vsma1uuLLCcUt1NrXp1K8PBU6tVH8w",
    }
    for fname, url in urls.items():
        dest = os.path.join(CROWD_ROOT, fname)
        print(f"[CrowdPose] Downloading {fname} …")
        subprocess.run(["gdown", "-O", dest, url], check=True)
        subprocess.run(["unzip", "-q", "-o", dest, "-d", CROWD_ROOT],
                       check=True)

    print("[CrowdPose] Download complete.")


if __name__ == "__main__":
    clone_3dsp()
    download_crowdpose()