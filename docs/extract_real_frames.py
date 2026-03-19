"""
Extract real movie frames with InsightFace face detection for infographic.
Run on GPU machine: python3 docs/extract_real_frames.py

Outputs:
  docs/real_frames/frame_*.png - raw frames with face bboxes drawn
  docs/real_frames/detections.json - face detection metadata
"""

import os
import sys
import json
import numpy as np

# Target timestamps: pick moments likely to have multiple visible faces
# Chunk 15: t=1275-1365s (7 characters!) and Chunk 17: t=1445-1535s (6 chars)
# Sample a few specific timestamps within those windows
# Timestamps for filmstrip + face detection/mask panels
# Filmstrip: sample across movie to show variety of scenes and characters
# Multi-face scenes for detection/mask panels
TARGET_TIMESTAMPS_LONG = [
    # Filmstrip: varied scenes across the movie showing different characters
    300.0, 600.0, 900.0, 1200.0, 1480.0, 1800.0,
    2400.0, 3000.0, 3600.0, 4200.0, 4800.0, 5400.0,
    6000.0, 6600.0, 7200.0, 7800.0,
    # Multi-face scenes for detection/mask panels
    1290.0, 1320.0, 1350.0, 1500.0,
]
# Will be replaced with evenly-spaced samples if video is short
TARGET_TIMESTAMPS = TARGET_TIMESTAMPS_LONG

VIDEO_PATH = None
# Try to find the Aliens video
for path in [
    "/data/huggingface/Aliens.1986.mkv",
    "/mnt/data/Aliens.1986.mkv",
    "/mnt/data/video-sources/Aliens.1986.mkv",
    os.path.expanduser("~/Aliens.1986.mkv"),
    os.path.expanduser("~/chapter_2.mkv"),  # fallback
]:
    if os.path.exists(path):
        VIDEO_PATH = path
        break

if VIDEO_PATH is None:
    # List what's available
    for d in ["/mnt/data", "/mnt/data/video-sources", os.path.expanduser("~")]:
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith((".mkv", ".mp4", ".mov"))]
            if files:
                print(f"Available in {d}: {files}")
    print("No video file found. Set VIDEO_PATH manually.")
    sys.exit(1)

print(f"Using video: {VIDEO_PATH}")

OUT_DIR = os.path.join(os.path.dirname(__file__), "real_frames")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Extract frames ──────────────────────────────────────────────────
try:
    from torchcodec.decoders import VideoDecoder
    import torch
    decoder = VideoDecoder(VIDEO_PATH)
    duration = float(decoder.metadata.duration_seconds)
    print(f"Video duration: {duration:.1f}s")

    # Filter timestamps to valid range; if none fit, sample evenly
    timestamps = [t for t in TARGET_TIMESTAMPS if t < duration - 1]
    if len(timestamps) < 3:
        # Sample 12 evenly spaced frames across the video
        timestamps = [duration * i / 13 for i in range(1, 13)]
        print(f"  Short video, sampling {len(timestamps)} evenly-spaced frames")
    print(f"Extracting {len(timestamps)} frames...")

    frames = []
    for ts in timestamps:
        frame_data = decoder.get_frame_played_at(ts)
        frame_np = frame_data.data.permute(1, 2, 0).numpy()  # HWC, uint8
        frames.append((ts, frame_np))
        print(f"  t={ts:.1f}s -> {frame_np.shape}")

except ImportError:
    print("torchcodec not available, trying cv2...")
    import cv2
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    print(f"Video duration: {duration:.1f}s, fps: {fps}")

    timestamps = [t for t in TARGET_TIMESTAMPS if t < duration - 1]
    if len(timestamps) < 3:
        timestamps = [duration * i / 13 for i in range(1, 13)]
    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((ts, frame_rgb))
            print(f"  t={ts:.1f}s -> {frame_rgb.shape}")
    cap.release()

print(f"Extracted {len(frames)} frames")

# ── Face detection ──────────────────────────────────────────────────
print("Running InsightFace detection...")
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

all_detections = {}

for ts, frame in frames:
    faces = app.get(frame)
    dets = []
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        score = float(face.det_score)
        embedding = face.embedding.tolist() if face.embedding is not None else None
        dets.append({
            "bbox": bbox,  # [x1, y1, x2, y2]
            "score": round(score, 3),
            "embedding_norm": round(float(np.linalg.norm(face.embedding)), 2) if face.embedding is not None else None,
        })
    all_detections[str(ts)] = dets
    print(f"  t={ts:.1f}s: {len(faces)} faces detected")

# ── Save frames as PNG with bboxes drawn ────────────────────────────
from PIL import Image, ImageDraw, ImageFont

COLORS = [
    (88, 166, 255),   # blue
    (63, 185, 80),    # green
    (210, 153, 34),   # orange
    (188, 140, 255),  # purple
    (247, 120, 186),  # pink
    (57, 210, 192),   # cyan
    (248, 81, 73),    # red
    (255, 166, 87),   # amber
]

for ts, frame in frames:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    faces = all_detections[str(ts)]
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]
        color = COLORS[i % len(COLORS)]
        # Draw bbox
        for offset in range(3):  # thick line
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                           outline=color)
        # Score label
        label = f"{face['score']:.2f}"
        draw.text((x1, y1 - 15), label, fill=color)

    # Save raw frame (no bbox) and annotated frame
    raw_img = Image.fromarray(frame)
    raw_img.save(os.path.join(OUT_DIR, f"frame_{ts:.0f}_raw.png"))
    img.save(os.path.join(OUT_DIR, f"frame_{ts:.0f}_det.png"))
    print(f"  Saved frame_{ts:.0f}_raw.png and frame_{ts:.0f}_det.png")

# ── Save detections metadata ────────────────────────────────────────
meta = {
    "video": VIDEO_PATH,
    "duration_seconds": duration,
    "frame_shape": list(frames[0][1].shape) if frames else None,
    "detections": all_detections,
}
meta_path = os.path.join(OUT_DIR, "detections.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved metadata to {meta_path}")
print("Done!")
