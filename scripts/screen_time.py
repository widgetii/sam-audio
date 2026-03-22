"""Character screen time tracker via dense face detection + SAM3 body fill-in.

Samples frames every N seconds across the entire movie, runs InsightFace
for face identification, then fills gaps with SAM3 body detection.

Usage:
    uv run python scripts/screen_time.py \
        --input /data/huggingface/Aliens.1080p.mkv \
        --pipeline-json workspace/v2/dialogue_v2_unified.json \
        --output workspace/v2/screen_time.json
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def match_face_to_profiles(
    embedding: np.ndarray,
    profiles: dict,
    threshold: float = 0.4,
) -> tuple[int | None, float]:
    """Match a face embedding to character profiles."""
    best_id = None
    best_sim = threshold
    emb_norm = embedding / max(np.linalg.norm(embedding), 1e-10)
    for cid, profile in profiles.items():
        if profile.representative_face_embedding is None:
            continue
        ref = profile.representative_face_embedding
        ref_norm = ref / max(np.linalg.norm(ref), 1e-10)
        sim = float(np.dot(emb_norm, ref_norm))
        if sim > best_sim:
            best_sim = sim
            best_id = cid
    return best_id, best_sim


def main():
    parser = argparse.ArgumentParser(description="Character screen time tracker")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument(
        "--pipeline-json",
        required=True,
        help="Pipeline output JSON (for character list)",
    )
    parser.add_argument(
        "--stage3a-cache",
        default=None,
        help="Stage 3A pickle cache (auto-detected from pipeline workspace if omitted)",
    )
    parser.add_argument("--output", default="screen_time.json")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=5.0,
        help="Sample one frame every N seconds (default: 5)",
    )
    parser.add_argument(
        "--face-det-threshold",
        type=float,
        default=0.3,
        help="Face detection confidence threshold",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Load pipeline results for character profiles
    pipeline_json = Path(args.pipeline_json)
    with open(pipeline_json) as f:
        pipeline = json.load(f)

    duration = pipeline["source"]["duration"]
    characters = pipeline.get("characters", [])
    char_ids = [c["id"] for c in characters]

    # Find stage3a cache for face profiles
    if args.stage3a_cache:
        cache_path = Path(args.stage3a_cache)
    else:
        workspace = pipeline_json.parent
        caches = list(workspace.glob("stage3a.*.pkl"))
        if not caches:
            logger.error("No stage3a cache found. Run the pipeline first.")
            return
        cache_path = caches[0]

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    profiles = cache["profiles"]
    logger.info(f"Loaded {len(profiles)} character profiles from {cache_path.name}")

    # Initialize face detector
    import insightface

    face_app = insightface.app.FaceAnalysis(
        allowed_modules=["detection", "recognition"]
    )
    face_app.prepare(ctx_id=0, det_thresh=args.face_det_threshold)

    # Initialize video decoder
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(args.input, dimension_order="NCHW")

    # Sample frames across entire movie
    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(t)
        t += args.sample_interval

    logger.info(
        f"Sampling {len(timestamps)} frames every {args.sample_interval}s "
        f"across {duration:.0f}s movie"
    )

    # Dense face detection
    t_start = time.time()
    presence: dict[float, list[int]] = {}  # timestamp → [char_ids on screen]
    total_faces = 0
    unmatched_faces = 0

    for ts in tqdm(timestamps, desc="Face detection"):
        try:
            frame = decoder.get_frames_played_at([ts]).data  # [1, C, H, W]
        except Exception:
            continue

        # Run InsightFace at full resolution
        frame_np = frame[0].permute(1, 2, 0).cpu().numpy()
        faces = face_app.get(frame_np)

        chars_at_ts = []
        for face in faces:
            total_faces += 1
            bbox = face.bbox.astype(int)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            area = w * h
            if area < 2500 or face.det_score < 0.5:
                continue

            cid, sim = match_face_to_profiles(face.embedding, profiles)
            if cid is not None and cid in char_ids:
                chars_at_ts.append(cid)
            else:
                unmatched_faces += 1

        presence[ts] = sorted(set(chars_at_ts))

    det_time = time.time() - t_start
    logger.info(
        f"Face detection: {total_faces} faces, {unmatched_faces} unmatched, "
        f"{det_time:.0f}s"
    )

    # Compute screen time per character
    screen_time: dict[int, float] = dict.fromkeys(char_ids, 0.0)
    for _ts, cids in presence.items():
        for cid in cids:
            screen_time[cid] += args.sample_interval

    # Build presence timeline per character (merge consecutive intervals)
    char_timelines: dict[int, list[list[float]]] = {cid: [] for cid in char_ids}
    for cid in char_ids:
        segments = []
        for ts in sorted(presence.keys()):
            if cid in presence[ts]:
                if segments and ts - segments[-1][1] <= args.sample_interval * 1.5:
                    segments[-1][1] = ts + args.sample_interval
                else:
                    segments.append([ts, ts + args.sample_interval])
        char_timelines[cid] = segments

    # Summary
    logger.info("Screen time per character:")
    for cid in char_ids:
        mins = screen_time[cid] / 60
        pct = screen_time[cid] / duration * 100
        segs = len(char_timelines[cid])
        logger.info(f"  char {cid}: {mins:.1f} min ({pct:.1f}%), {segs} segments")

    # Co-occurrence matrix
    cooccurrence = np.zeros((len(char_ids), len(char_ids)), dtype=int)
    for _ts, cids in presence.items():
        for i, c1 in enumerate(char_ids):
            if c1 not in cids:
                continue
            for j, c2 in enumerate(char_ids):
                if c2 in cids:
                    cooccurrence[i][j] += 1

    # Output
    output = {
        "source": pipeline.get("source", {}),
        "sample_interval": args.sample_interval,
        "total_frames_sampled": len(timestamps),
        "detection_time_sec": round(det_time, 1),
        "characters": [
            {
                "id": cid,
                "screen_time_sec": round(screen_time[cid], 1),
                "screen_time_pct": round(screen_time[cid] / duration * 100, 1),
                "presence_segments": char_timelines[cid],
                "num_segments": len(char_timelines[cid]),
            }
            for cid in char_ids
        ],
        "cooccurrence": {
            "character_ids": char_ids,
            "matrix": cooccurrence.tolist(),
        },
        "per_frame": [
            {"time": round(ts, 1), "characters": presence.get(ts, [])}
            for ts in sorted(presence.keys())
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
