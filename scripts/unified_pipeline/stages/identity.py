"""Stage 3: Face identity on SAM3 person crops + clustering.

For each SAM3 track, crop the person region from 1 keyframe per shot at 1080p,
run InsightFace to get ArcFace embeddings, then cluster into characters.
"""

import logging
import subprocess
import tempfile

import numpy as np

from unified_pipeline.db import AnalysisDB, embed_to_blob

log = logging.getLogger(__name__)

STAGE = "stage3"


def _extract_frame_jpeg(video_path: str, sec: float, output_path: str):
    """Extract a single frame at given timestamp."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(sec),
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-qscale:v",
            "2",
            output_path,
        ],
        check=True,
    )


def run_stage3(db: AnalysisDB, video_path: str, cluster_threshold: float = 0.6):
    """Run InsightFace on SAM3 person crops, cluster into characters.

    Args:
        db: Analysis database.
        video_path: Source video path (for full-res frame extraction).
        cluster_threshold: Agglomerative clustering distance threshold.
    """
    shots = db.get_shots()
    if not shots:
        raise RuntimeError("Stage 3: no shots in DB")

    progress = db.get_progress(STAGE)

    # Collect shots that have person tracks but no face detections yet
    shots_with_tracks = set()
    for row in db.conn.execute("SELECT DISTINCT shot_id FROM person_tracks").fetchall():
        shots_with_tracks.add(row["shot_id"])

    remaining = [
        s
        for s in shots
        if s["shot_id"] in shots_with_tracks
        and progress.get(str(s["shot_id"])) != "done"
    ]

    if not remaining:
        log.info("Stage 3: face detection already complete for all tracked shots")
        _cluster_faces(db, cluster_threshold)
        return

    log.info(f"Stage 3: detecting faces in {len(remaining)} shots with person tracks")

    # Load InsightFace
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    _detect_faces_in_shots(db, face_app, video_path, remaining)
    _cluster_faces(db, cluster_threshold)


def _detect_faces_in_shots(
    db: AnalysisDB, face_app, video_path: str, shots: list[dict]
):
    """For each shot, extract keyframe at full res, detect faces in SAM3 person crops."""
    import cv2

    total_detections = 0

    for shot_idx, shot in enumerate(shots):
        shot_id = shot["shot_id"]
        # Use middle of shot as keyframe
        keyframe_sec = (shot["start_sec"] + shot["end_sec"]) / 2

        # Get person tracks at this keyframe
        tracks = db.get_person_tracks_at_sec(keyframe_sec)
        shot_tracks = [t for t in tracks if t["shot_id"] == shot_id]
        if not shot_tracks:
            db.mark_progress(STAGE, str(shot_id), "done")
            continue

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            _extract_frame_jpeg(video_path, keyframe_sec, tmp.name)
            frame = cv2.imread(tmp.name)

        if frame is None:
            db.mark_progress(STAGE, str(shot_id), "done")
            continue

        h, w = frame.shape[:2]
        detections = []

        for track in shot_tracks:
            # Crop person region from full-res frame
            x1 = max(0, int(track["bbox_x1"]))
            y1 = max(0, int(track["bbox_y1"]))
            x2 = min(w, int(track["bbox_x2"]))
            y2 = min(h, int(track["bbox_y2"]))

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            person_crop = frame[y1:y2, x1:x2]

            # Run InsightFace on the crop
            faces = face_app.get(person_crop)
            if not faces:
                continue

            # Take the largest face in the crop
            best = max(
                faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )

            # Translate face bbox back to full-frame coordinates
            fx1 = x1 + best.bbox[0]
            fy1 = y1 + best.bbox[1]
            fx2 = x1 + best.bbox[2]
            fy2 = y1 + best.bbox[3]

            detections.append(
                {
                    "shot_id": shot_id,
                    "sam3_obj_id": track["sam3_obj_id"],
                    "frame_sec": keyframe_sec,
                    "face_bbox_x1": float(fx1),
                    "face_bbox_y1": float(fy1),
                    "face_bbox_x2": float(fx2),
                    "face_bbox_y2": float(fy2),
                    "face_score": float(best.det_score),
                    "embedding": embed_to_blob(best.embedding),
                }
            )

        if detections:
            db.insert_face_detections(detections)
            total_detections += len(detections)

        db.mark_progress(STAGE, str(shot_id), "done")

        if (shot_idx + 1) % 100 == 0 or shot_idx == len(shots) - 1:
            log.info(
                f"Stage 3: {shot_idx + 1}/{len(shots)} shots, "
                f"{total_detections} faces total"
            )


def _cluster_faces(db: AnalysisDB, threshold: float):
    """Cluster face embeddings into characters, assign track identities."""
    from sklearn.cluster import AgglomerativeClustering

    if db.count_rows("characters") > 0:
        log.info("Stage 3: characters already clustered, skipping")
        _assign_identities(db)
        return

    face_dets = db.get_face_detections()
    if not face_dets:
        log.warning("Stage 3: no face detections to cluster")
        return

    # Filter to high-quality detections
    good_dets = [d for d in face_dets if d["face_score"] >= 0.5]
    if len(good_dets) < 2:
        good_dets = face_dets

    log.info(
        f"Stage 3: clustering {len(good_dets)} face embeddings (threshold={threshold})"
    )

    from unified_pipeline.db import blob_to_embed

    embeddings = np.stack([blob_to_embed(d["embedding"]) for d in good_dets])

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms

    if len(embeddings) < 2:
        labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

    # Create characters from cluster centroids
    cluster_ids = sorted(set(labels))
    log.info(f"Stage 3: found {len(cluster_ids)} character clusters")

    label_to_char: dict[int, int] = {}
    for cid in cluster_ids:
        mask = labels == cid
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-8)

        char_id = db.insert_character(
            name=f"Character {cid}",
            face_embedding=centroid,
        )
        label_to_char[cid] = char_id

    # Map each detection to character
    det_to_char: dict[tuple[int, int], int] = {}
    for det, label in zip(good_dets, labels, strict=True):
        key = (det["shot_id"], det["sam3_obj_id"])
        det_to_char[key] = label_to_char[label]

    # Also assign remaining detections by nearest centroid
    characters = db.get_characters()
    char_embeddings = {
        c["character_id"]: c["face_embedding"]
        for c in characters
        if c["face_embedding"] is not None
    }

    for det in face_dets:
        key = (det["shot_id"], det["sam3_obj_id"])
        if key not in det_to_char and det["embedding"]:
            emb = blob_to_embed(det["embedding"])
            emb = emb / max(np.linalg.norm(emb), 1e-8)
            best_char = None
            best_sim = -1.0
            for cid, cemb in char_embeddings.items():
                sim = float(np.dot(emb, cemb))
                if sim > best_sim:
                    best_sim = sim
                    best_char = cid
            if best_char is not None and best_sim > 0.4:
                det_to_char[key] = best_char

    # Insert track identities
    for (shot_id, sam3_obj_id), char_id in det_to_char.items():
        db.insert_track_identity(shot_id, sam3_obj_id, char_id, method="face")

    log.info(f"Stage 3: assigned {len(det_to_char)} track identities")


def _assign_identities(db: AnalysisDB):
    """If characters exist but track identities are missing, re-assign."""
    existing = db.get_track_identities()
    if existing:
        return

    from unified_pipeline.db import blob_to_embed

    characters = db.get_characters()
    char_embeddings = {
        c["character_id"]: c["face_embedding"]
        for c in characters
        if c["face_embedding"] is not None
    }

    if not char_embeddings:
        return

    face_dets = db.get_face_detections()
    count = 0
    for det in face_dets:
        if not det["embedding"]:
            continue
        emb = blob_to_embed(det["embedding"])
        emb = emb / max(np.linalg.norm(emb), 1e-8)

        best_char = None
        best_sim = -1.0
        for cid, cemb in char_embeddings.items():
            sim = float(np.dot(emb, cemb))
            if sim > best_sim:
                best_sim = sim
                best_char = cid

        if best_char is not None and best_sim > 0.4:
            db.insert_track_identity(
                det["shot_id"], det["sam3_obj_id"], best_char, method="face"
            )
            count += 1

    log.info(f"Stage 3: re-assigned {count} track identities")
