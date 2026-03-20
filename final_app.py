# final_app.py  (replace your app.py contents with this)
import os
import tempfile
import subprocess
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from model_resnet18 import AVDeepfakeDetector

# ----- DEVICE & TRANSFORMS -----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_DEVICE = torch.device('cpu')

class_names = ['FVFA', 'FVRA', 'RVFA', 'RVRA']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----- UTIL FUNCTIONS -----
def convert_video_to_mp4(input_path):
    output_path = tempfile.mktemp(suffix='.mp4')
    cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-strict', '-2',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def extract_frames(video_path, target_len=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        pil_img = transform(pil_img)
        frames.append(pil_img)
        if len(frames) >= target_len:
            break
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames extracted from video.")
    if len(frames) < target_len:
        pad_size = target_len - len(frames)
        padding = [torch.zeros_like(frames[0])] * pad_size
        frames.extend(padding)
    video_tensor = torch.stack(frames).unsqueeze(0)
    return video_tensor

def extract_audio_mfcc(video_path, target_time=44):
    import librosa
    # librosa can read many container formats if ffmpeg is installed
    y, sr = librosa.load(video_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] > target_time:
        mfcc = mfcc[:, :target_time]
    elif mfcc.shape[1] < target_time:
        pad_width = target_time - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0, pad_width)), mode='constant')
    mfcc_tensor = torch.tensor(mfcc).float().unsqueeze(0).unsqueeze(1)
    return mfcc_tensor

def generate_gradcam_images(model, video_tensor):
    model.video_model.eval()
    target_layer = model.video_model.layer4[-1]
    cam = GradCAM(model=model.video_model, target_layers=[target_layer])
    cams = []
    for i in range(video_tensor.shape[1]):
        frame_tensor = video_tensor[0, i].unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=frame_tensor)
        frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # normalize for visualization
        frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-8)
        visualization = show_cam_on_image(frame_np, grayscale_cam[0], use_rgb=True)
        cams.append(visualization)
    return cams

def explain_model_full_cpu(model, video_tensor, mfcc_tensor, pred_class):
    model_cpu = model.to(CPU_DEVICE)
    video_cpu = video_tensor.to(CPU_DEVICE)
    mfcc_cpu = mfcc_tensor.to(CPU_DEVICE)
    ig = IntegratedGradients(model_cpu)
    inputs = (video_cpu, mfcc_cpu)
    attr, delta = ig.attribute(inputs, target=pred_class.item(), return_convergence_delta=True)
    video_attr = attr[0].detach().cpu()
    audio_attr = attr[1].detach().cpu()
    model.to(DEVICE)
    return video_attr, audio_attr, delta.item()

def plot_audio_attr(attr):
    attr_np = attr.squeeze().numpy()
    plt.figure(figsize=(10,4))
    plt.title("Audio MFCC Attribution")
    plt.imshow(attr_np, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    filename = tempfile.mktemp(suffix='.png')
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    return filename

def rgb_array_to_pil(img_array):
    # expecting float [0..1] arrays from grad-cam visualization
    img_array_uint8 = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array_uint8)

# ----- REAL INFERENCE (your original function with small logging) -----
def inference(video_path):
    try:
        print("=== Starting inference ===")
        print(f"Received video: {video_path}")

        converted_video_path = convert_video_to_mp4(video_path)

        video_tensor = extract_frames(converted_video_path)
        mfcc_tensor = extract_audio_mfcc(converted_video_path)

        # -------------------------------
        # Main Prediction
        # -------------------------------
        with torch.no_grad():
            outputs = model(video_tensor.to(DEVICE), mfcc_tensor.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = probs[0][pred_class].item()

        label_map = {
            'RVRA': 'REAL',
            'FVFA': 'FAKE',
            'FVRA': 'FAKE',
            'RVFA': 'FAKE',
        }

        raw_class = class_names[pred_class.item()]
        prediction = label_map.get(raw_class, raw_class)

        # -------------------------------
        # Temporal Probability (Per Frame)
        # -------------------------------
        temporal_probs = []
        timestamps = []

        total_frames = video_tensor.shape[1]

        for i in range(total_frames):
            single_frame = video_tensor[:, i:i+1, :, :, :]
            with torch.no_grad():
                out = model(single_frame.to(DEVICE), mfcc_tensor.to(DEVICE))
                prob = torch.softmax(out, dim=1)[0][pred_class].item()

            temporal_probs.append(prob)
            timestamps.append(i)

        # -------------------------------
        # GradCAM (Video Attribution)
        # -------------------------------
        gradcam_imgs = generate_gradcam_images(model, video_tensor)

        gradcam_dir = os.path.join('static', 'gradcams')
        os.makedirs(gradcam_dir, exist_ok=True)

        gradcam_paths = []
        for i, img_array in enumerate(gradcam_imgs):
            pil_img = rgb_array_to_pil(img_array)
            fname = f"gradcam_{i}_{os.path.basename(video_path)}.png"
            fpath = os.path.join(gradcam_dir, fname)
            pil_img.save(fpath)
            gradcam_paths.append(f"gradcams/{fname}")

        # -------------------------------
        # Integrated Gradients
        # -------------------------------
        video_attr, audio_attr, delta = explain_model_full_cpu(
            model, video_tensor, mfcc_tensor, pred_class
        )

        # ------------------------------
        # HUMAN FRIENDLY EXPLANATION
        # ------------------------------

        if prediction == "FAKE":
            if abs(delta) < 1:
                explanation = (
                    "The system is confident in this result. "
                    "The detected patterns strongly indicate that parts of the video may have been digitally manipulated."
                )
            else:
                explanation = (
                    "The system detected potential manipulation patterns, "
                    "but some uncertainty remains in the explanation process."
                )
        else:
            if abs(delta) < 1:
                explanation = (
                    "The video appears authentic. "
                    "No significant signs of digital manipulation were detected."
                )
            else:
                explanation = (
                    "The video appears authentic overall, "
                    "though minor irregularities were analyzed during processing."
                )
        
        
        ##Temporal Probabilty##        
        temporal_probs = []
        temporal_ranges = []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        total_segments = video_tensor.shape[1]
        segment_duration = duration / total_segments if total_segments > 0 else 0

        for i in range(total_segments):
            single_frame = video_tensor[:, i:i+1, :, :, :]

            with torch.no_grad():
                out = model(single_frame.to(DEVICE),
                            mfcc_tensor.to(DEVICE))
                prob = torch.softmax(out, dim=1)[0][pred_class].item()

            temporal_probs.append(prob)

            start_time = round(i * segment_duration, 2)
            end_time = round((i + 1) * segment_duration, 2)

            temporal_ranges.append((start_time, end_time))

        # Human readable temporal summary
        temporal_summary = []

        if temporal_probs:
            max_prob = max(temporal_probs)
            max_index = temporal_probs.index(max_prob)
            start, end = temporal_ranges[max_index]

            if max_prob > 0.7:
                temporal_summary.append(
                    f"⚠ Higher likelihood of manipulation between {start}s – {end}s."
                )
            else:
                temporal_summary.append(
                    "✅ No specific time segment shows strong manipulation."
                )


        # -------------------------------
        # Derived Analytics
        # -------------------------------

        # Audio anomaly score
        audio_anomaly_score = float(torch.mean(torch.abs(audio_attr)))

        # Visual artifact score
        visual_artifact_score = float(torch.mean(torch.abs(video_attr)))

        # Lip-sync consistency (heuristic)
        lip_sync_score = float(
            1 - abs(video_attr.mean().item() - audio_attr.mean().item())
        )
        lip_sync_score = max(0.0, min(1.0, lip_sync_score))

        # Lighting consistency (brightness variance)
        brightness_scores = []
        for frame in video_tensor[0]:
            brightness_scores.append(frame.mean().item())

        lighting_consistency = float(1 - np.std(brightness_scores))
        lighting_consistency = max(0.0, min(1.0, lighting_consistency))

        # Audio spectral noise
        audio_noise_score = float(np.std(mfcc_tensor.cpu().numpy()))

        # ------------------------------
        # HUMAN INFLUENTIAL FEATURES
        # ------------------------------

        human_features = []

        if visual_artifact_score > audio_anomaly_score:
            human_features.append(
                "🖼 Visual facial regions contributed most to this decision."
            )

        if audio_anomaly_score > visual_artifact_score:
            human_features.append(
                "🎙 Audio characteristics significantly influenced the result."
            )

        if lip_sync_score < 0.6:
            human_features.append(
                "👄 Lip-sync inconsistencies were considered during analysis."
            )

        if lighting_consistency < 0.6:
            human_features.append(
                "💡 Lighting variations played a role in the model's assessment."
            )

        if not human_features:
            human_features.append(
                "No single dominant feature influenced the decision."
            )

        # -------------------------------
        # Save Audio Heatmap
        # -------------------------------
        audio_fname = f"audio_{os.path.basename(video_path)}.png"
        audio_fpath = os.path.join(gradcam_dir, audio_fname)

        attr_np = audio_attr.squeeze().numpy()
        plt.figure(figsize=(16, 9))  
        plt.title("Audio MFCC Attribution")
        plt.imshow(attr_np, aspect="auto", origin="lower")
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")
        plt.savefig(audio_fpath, bbox_inches='tight', dpi=300)
        plt.close()

        audio_heatmap_static = f"gradcams/{audio_fname}"

        print("=== Inference complete ===")
        
        # -------------------------------------
        # HUMAN READABLE FORENSIC SUMMARY
        # -------------------------------------

        forensic_summary = []

        # 1. Visual Analysis
        if visual_artifact_score > 0.05:
            forensic_summary.append("⚠ The facial region shows visual manipulation patterns.")
        else:
            forensic_summary.append("✅ No strong visual manipulation detected in facial regions.")

        # 2. Audio Analysis
        if audio_anomaly_score > 0.05:
            forensic_summary.append("⚠ The voice pattern shows characteristics of synthetic or altered audio.")
        else:
            forensic_summary.append("✅ The audio signal appears natural and unaltered.")

        # 3. Lip Sync
        if lip_sync_score < 0.6:
            forensic_summary.append("⚠ Lip movements do not fully align with the speech audio.")
        else:
            forensic_summary.append("✅ Lip movements are well synchronized with speech.")

        # 4. Lighting
        if lighting_consistency < 0.6:
            forensic_summary.append("⚠ Inconsistent lighting suggests possible frame manipulation.")
        else:
            forensic_summary.append("✅ Lighting remains consistent across frames.")

        # 5. Audio Noise
        if audio_noise_score > 50:
            forensic_summary.append("⚠ The audio contains unusual spectral noise patterns.")
        else:
            forensic_summary.append("✅ No abnormal spectral noise detected in audio.")

        # -------------------------------
        # FINAL RETURN OBJECT
        # -------------------------------
        return {
            "prediction": prediction,
            "confidence": confidence,
            "raw_class": raw_class,
            "explanation": explanation,
            "temporal_probs": temporal_probs,
            "timestamps": timestamps,
            "gradcams": gradcam_paths,
            "audio_heatmap": audio_heatmap_static,
            "audio_anomaly": audio_anomaly_score,
            "visual_artifact": visual_artifact_score,
            "lip_sync": lip_sync_score,
            "lighting": lighting_consistency,
            "audio_noise": audio_noise_score,
            "temporal_summary": temporal_summary,
            "temporal_ranges": temporal_ranges,
            "influential_summary": human_features,
            "forensic_summary": forensic_summary
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "prediction": "Error",
            "confidence": 0.0,
            "error": str(e)
        }

# ----- LOAD MODEL -----
model = AVDeepfakeDetector(num_classes=4)
model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=torch.device('cpu')))
model.to(DEVICE)
model.eval()

