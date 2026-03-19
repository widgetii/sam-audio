"""
Extract face crops with embeddings for clustering visualization.
Samples densely to get many face detections, saves crops and embeddings.

Run on GPU machine: python3 docs/extract_face_crops.py
Output: docs/real_frames/face_crops/ and docs/real_frames/face_embeddings.json
"""

import os
import sys
import json
import numpy as np

VIDEO_PATH = None
for path in [
    "/data/huggingface/Aliens.1986.mkv",
    "/mnt/data/Aliens.1986.mkv",
    os.path.expanduser("~/chapter_2.mkv"),
]:
    if os.path.exists(path):
        VIDEO_PATH = path
        break

if VIDEO_PATH is None:
    print("No video file found")
    sys.exit(1)

print(f"Using video: {VIDEO_PATH}")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_frames", "face_crops")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Extract frames every 4s ─────────────────────────────────────────
from torchcodec.decoders import VideoDecoder
decoder = VideoDecoder(VIDEO_PATH)
duration = float(decoder.metadata.duration_seconds)
# Sample two 5-minute windows with lots of characters, every 4s
# Chunk 15-17 area (~1275-1535s) and timeline window (~4680-4980s)
timestamps = []
for start, end in [(1275, 1540), (4680, 4980)]:
    timestamps.extend([t + 1.0 for t in range(start, min(end, int(duration)), 4)])
print(f"Video duration: {duration:.1f}s, extracting {len(timestamps)} frames")

# ── Face detection ──────────────────────────────────────────────────
from insightface.app import FaceAnalysis
from PIL import Image

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

all_faces = []  # list of {crop_file, embedding, score, timestamp}
crop_idx = 0

for ts in timestamps:
    if ts >= duration - 0.5:
        continue
    frame_data = decoder.get_frame_played_at(ts)
    frame_np = frame_data.data.permute(1, 2, 0).numpy()
    h, w = frame_np.shape[:2]

    faces = app.get(frame_np)
    for face in faces:
        if face.det_score < 0.5:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        # Pad by 15%
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * 0.15), int(bh * 0.15)
        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px)
        cy2 = min(h, y2 + py)

        crop = frame_np[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        fname = f"face_{crop_idx:04d}.jpg"
        Image.fromarray(crop).save(os.path.join(OUT_DIR, fname), quality=85)

        all_faces.append({
            "crop_file": fname,
            "timestamp": round(ts, 2),
            "score": round(float(face.det_score), 3),
            "embedding": face.embedding.tolist(),
        })
        crop_idx += 1

    if len(timestamps) > 10 and timestamps.index(ts) % 10 == 0:
        print(f"  t={ts:.0f}s: {crop_idx} total crops so far")

print(f"Total face crops: {crop_idx}")

# ── Cluster embeddings ──────────────────────────────────────────────
from sklearn.cluster import AgglomerativeClustering

if crop_idx >= 2:
    embeddings = np.array([f["embedding"] for f in all_faces])
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)

    # Cosine distance clustering (same as pipeline: threshold=0.6)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.6,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings_norm)

    # t-SNE for 2D visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, crop_idx // 3)))
    coords_2d = tsne.fit_transform(embeddings_norm)

    for i, face in enumerate(all_faces):
        face["cluster_id"] = int(labels[i])
        face["tsne_x"] = float(coords_2d[i, 0])
        face["tsne_y"] = float(coords_2d[i, 1])
        del face["embedding"]  # don't save full embeddings to JSON

    n_clusters = len(set(labels))
    print(f"Clusters: {n_clusters}")
    for cid in sorted(set(labels)):
        count = sum(1 for f in all_faces if f["cluster_id"] == cid)
        print(f"  Cluster {cid}: {count} faces")

# ── Save metadata ───────────────────────────────────────────────────
meta_path = os.path.join(os.path.dirname(OUT_DIR), "face_embeddings.json")
with open(meta_path, "w") as f:
    json.dump({"faces": all_faces, "n_clusters": n_clusters}, f, indent=2)
print(f"Saved to {meta_path}")
