"""Face detection, embedding, clustering, and SAM3 body tracking for character tracking."""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from character_profile import CharacterProfile, FaceDetection
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


@dataclass
class CharacterInfo:
    character_id: int
    representative_embedding: np.ndarray
    total_frames: int = 0  # screen time proxy
    frame_indices: list[int] = field(default_factory=list)


class FaceTracker:
    """Detect faces, compute embeddings, cluster into characters, track bodies via SAM3."""

    def __init__(
        self,
        sam3_predictor=None,
        det_threshold: float = 0.3,
        cluster_threshold: float = 0.6,
    ):
        import insightface

        self.face_app = insightface.app.FaceAnalysis(
            allowed_modules=["detection", "recognition"]
        )
        self.face_app.prepare(ctx_id=0, det_thresh=det_threshold)
        self.sam3 = sam3_predictor
        self.det_threshold = det_threshold
        self.cluster_threshold = cluster_threshold

    def detect_faces(
        self, frames: torch.Tensor, frame_indices: list[int] | None = None
    ) -> list[list[FaceDetection]]:
        """Detect faces in video frames.

        Args:
            frames: Video frames tensor [N, C, H, W] in uint8 0-255 range.
            frame_indices: Optional original frame indices for tracking.

        Returns:
            List of detections per frame.
        """
        if frame_indices is None:
            frame_indices = list(range(frames.shape[0]))

        all_detections = []
        for i in range(frames.shape[0]):
            # InsightFace expects BGR HWC numpy array
            frame_np = frames[i].permute(1, 2, 0).cpu().numpy()
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            frame_bgr = frame_np[:, :, ::-1].copy()

            faces = self.face_app.get(frame_bgr)
            frame_dets = []
            for face in faces:
                bbox = tuple(int(v) for v in face.bbox)
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                area = w * h
                ar = w / h if h > 0 else 0.0
                frame_dets.append(
                    FaceDetection(
                        bbox=bbox,
                        embedding=face.normed_embedding,
                        confidence=float(face.det_score),
                        frame_index=frame_indices[i],
                        bbox_area=area,
                        aspect_ratio=round(ar, 3),
                    )
                )
            all_detections.append(frame_dets)
        return all_detections

    @staticmethod
    def filter_detections(
        all_detections: list[list[FaceDetection]],
        min_area: int = 2500,
        min_confidence: float = 0.5,
        min_aspect_ratio: float = 0.4,
        max_aspect_ratio: float = 2.5,
    ) -> list[list[FaceDetection]]:
        """Filter out garbage detections before clustering.

        Removes partial faces, tiny reflections, and poster faces.
        """
        total_before = sum(len(fd) for fd in all_detections)
        filtered = []
        for frame_dets in all_detections:
            good = []
            for det in frame_dets:
                x1, y1, x2, y2 = det.bbox
                w, h = x2 - x1, y2 - y1
                area = w * h if det.bbox_area == 0 else det.bbox_area
                ar = (
                    (w / h if h > 0 else 0.0)
                    if det.aspect_ratio == 0.0
                    else det.aspect_ratio
                )
                if area < min_area:
                    continue
                if det.confidence < min_confidence:
                    continue
                if ar < min_aspect_ratio or ar > max_aspect_ratio:
                    continue
                good.append(det)
            filtered.append(good)
        total_after = sum(len(fd) for fd in filtered)
        logger.info(
            f"Quality filter: {total_before} → {total_after} detections "
            f"(removed {total_before - total_after})"
        )
        return filtered

    def cluster_characters(
        self, all_detections: list[list[FaceDetection]]
    ) -> dict[int, CharacterInfo]:
        """Two-phase clustering: tight clusters then merge small into large.

        Phase 1: Agglomerative clustering at threshold 0.5 (tight).
        Phase 2: Merge clusters with <3 detections into nearest large cluster.
        Renumber IDs to 0..K-1 sorted by screen time.

        Returns:
            character_id -> CharacterInfo, sequential IDs sorted by screen time (desc).
        """
        # Collect all embeddings
        flat_dets = []
        embeddings = []
        for frame_dets in all_detections:
            for det in frame_dets:
                flat_dets.append(det)
                embeddings.append(det.embedding)

        if len(embeddings) == 0:
            return {}

        embeddings = np.stack(embeddings)

        if len(embeddings) == 1:
            flat_dets[0].character_id = 0
            return {
                0: CharacterInfo(
                    character_id=0,
                    representative_embedding=embeddings[0],
                    total_frames=1,
                    frame_indices=[flat_dets[0].frame_index],
                )
            }

        # Phase 1: Tight clustering
        tight_threshold = min(self.cluster_threshold, 0.5)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tight_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Phase 2: Merge small clusters into nearest large cluster
        min_cluster_size = 3
        unique_labels, counts = np.unique(labels, return_counts=True)
        large_clusters = set(unique_labels[counts >= min_cluster_size])
        small_clusters = set(unique_labels[counts < min_cluster_size])

        if large_clusters and small_clusters:
            # Compute centroids of large clusters
            large_centroids = {}
            for lbl in large_clusters:
                large_centroids[lbl] = embeddings[labels == lbl].mean(axis=0)

            # Merge each small cluster into nearest large one
            large_ids = sorted(large_clusters)
            centroid_matrix = np.stack([large_centroids[lid] for lid in large_ids])

            for small_lbl in small_clusters:
                small_embs = embeddings[labels == small_lbl]
                small_centroid = small_embs.mean(axis=0)
                # Cosine similarity to all large centroids
                norms_c = np.linalg.norm(centroid_matrix, axis=1, keepdims=True).clip(
                    1e-10
                )
                norm_s = max(np.linalg.norm(small_centroid), 1e-10)
                sims = (centroid_matrix @ small_centroid) / (norms_c.squeeze() * norm_s)
                best_idx = int(np.argmax(sims))
                best_large = large_ids[best_idx]
                labels[labels == small_lbl] = best_large

            logger.info(
                f"Cluster merge: {len(unique_labels)} → {len(large_clusters)} "
                f"(merged {len(small_clusters)} small clusters)"
            )

        # Renumber to 0..K-1 sorted by cluster size (descending)
        unique_labels, counts = np.unique(labels, return_counts=True)
        size_order = np.argsort(-counts)
        old_to_new = {}
        for new_id, idx in enumerate(size_order):
            old_to_new[unique_labels[idx]] = new_id

        for i, det in enumerate(flat_dets):
            det.character_id = old_to_new[labels[i]]

        new_labels = np.array([old_to_new[lbl] for lbl in labels])

        # Build CharacterInfo per cluster
        characters: dict[int, CharacterInfo] = {}
        for det in flat_dets:
            cid = det.character_id
            if cid not in characters:
                characters[cid] = CharacterInfo(
                    character_id=cid,
                    representative_embedding=det.embedding,
                    total_frames=0,
                    frame_indices=[],
                )
            characters[cid].total_frames += 1
            characters[cid].frame_indices.append(det.frame_index)

        # Compute representative embedding as mean of cluster
        for cid, info in characters.items():
            cluster_embs = embeddings[new_labels == cid]
            info.representative_embedding = cluster_embs.mean(axis=0)

        # Already sorted by screen time via renumbering
        sorted_chars = dict(sorted(characters.items(), key=lambda x: x[0]))
        logger.info(
            f"Clustering: {len(flat_dets)} detections → {len(sorted_chars)} characters "
            f"(IDs 0..{len(sorted_chars) - 1})"
        )
        return sorted_chars

    def get_best_detection_per_character(
        self,
        all_detections: list[list[FaceDetection]],
        sample_shot_indices: list[int],
        chunk_start_sec: float,
        chunk_end_sec: float,
        character_ids: list[int],
        sample_timestamps: list[float],
    ) -> dict[int, FaceDetection]:
        """Find the highest-confidence face detection per character within a chunk.

        For each character, the detection's frame_index becomes the SAM3 prompt frame.
        If no detection in chunk range, uses nearest detection from adjacent frames.

        Args:
            all_detections: All detections per sampled frame.
            sample_shot_indices: Shot index for each sampled frame.
            chunk_start_sec: Chunk start time.
            chunk_end_sec: Chunk end time.
            character_ids: Characters to find.
            sample_timestamps: Timestamp for each sampled frame.

        Returns:
            char_id -> best FaceDetection (only for characters found).
        """
        best: dict[int, FaceDetection] = {}

        # Find frames within chunk time range
        for frame_i, frame_dets in enumerate(all_detections):
            if frame_i >= len(sample_timestamps):
                break
            t = sample_timestamps[frame_i]
            if t < chunk_start_sec or t >= chunk_end_sec:
                continue
            for det in frame_dets:
                if det.character_id not in character_ids:
                    continue
                cid = det.character_id
                if cid not in best or det.confidence > best[cid].confidence:
                    best[cid] = det

        # For missing characters, search nearest frames outside chunk
        missing = set(character_ids) - set(best.keys())
        if missing:
            chunk_mid = (chunk_start_sec + chunk_end_sec) / 2
            for cid in missing:
                nearest_det = None
                nearest_dist = float("inf")
                for frame_i, frame_dets in enumerate(all_detections):
                    if frame_i >= len(sample_timestamps):
                        break
                    t = sample_timestamps[frame_i]
                    for det in frame_dets:
                        if det.character_id == cid:
                            dist = abs(t - chunk_mid)
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest_det = det
                if nearest_det is not None:
                    best[cid] = nearest_det

        return best

    def track_characters_in_chunk(
        self,
        video_file: str,
        chunk_start_frame: int,
        chunk_end_frame: int,
        character_detections: dict[int, FaceDetection],
        frame_height: int,
        frame_width: int,
    ) -> dict[int, torch.Tensor]:
        """Track multiple characters through a video chunk using SAM3.

        Uses body box prompts (expanded from face bboxes) and SAM3's multi-object
        video tracking with temporal state propagation.

        Args:
            video_file: Path to video file.
            chunk_start_frame: Start frame index (in video frames).
            chunk_end_frame: End frame index (in video frames).
            character_detections: char_id -> best FaceDetection for prompting.
            frame_height: Video frame height.
            frame_width: Video frame width.

        Returns:
            char_id -> inverted mask tensor [N_frames, 1, H, W] (target=0, bg=1).
        """
        if self.sam3 is None:
            raise RuntimeError("SAM3 predictor is required for body tracking")

        # 1. Start SAM3 session
        response = self.sam3.handle_request(
            {
                "type": "start_session",
                "resource_path": video_file,
            }
        )
        session_id = response["session_id"]

        # 2. Add body box prompts for each character on their best-detected frame
        for char_id, det in character_detections.items():
            x1, y1, x2, y2 = det.bbox
            face_w, face_h = x2 - x1, y2 - y1
            # Expand face bbox to approximate body bbox
            body_box = [
                max(0, x1 - face_w),  # left: 1 face-width padding
                max(0, y1 - face_h // 2),  # top: half face-height above face
                min(frame_width, x2 + face_w),  # right: 1 face-width padding
                min(frame_height, y2 + face_h * 4),  # bottom: 4 face-heights below
            ]
            self.sam3.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": det.frame_index,
                    "box": body_box,
                    "obj_id": char_id,
                }
            )

        # 3. Propagate through video — SAM3 tracks all bodies simultaneously
        all_masks: dict[int, list] = {}  # char_id -> list of per-frame masks
        frame_count = 0
        for result in self.sam3.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "both",
                "start_frame_index": chunk_start_frame,
                "max_frame_num_to_track": chunk_end_frame - chunk_start_frame,
            }
        ):
            frame_count += 1
            for obj_id, mask in zip(
                result["object_ids"], result["pred_masks"], strict=True
            ):
                all_masks.setdefault(obj_id, []).append(mask)

        logger.debug(
            f"SAM3 tracked {len(character_detections)} bodies across {frame_count} frames"
        )

        # 4. Convert to per-character mask tensors, invert (target=0, bg=1)
        result_masks = {}
        for char_id, masks in all_masks.items():
            # Stack masks: each is [1, H, W] boolean
            stacked = np.stack(masks)  # [N, 1, H, W]
            mask_tensor = torch.from_numpy(stacked)
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(1)
            # Invert: SAM3 mask is True where object is, we need 0=target, 1=bg
            inverted = (~mask_tensor.bool()).float()
            result_masks[char_id] = inverted

        return result_masks

    def characters_in_range(
        self,
        all_detections: list[list[FaceDetection]],
        start_sec: float,
        end_sec: float,
        sample_fps: float,
        sample_interval: float = 0.5,
    ) -> list[CharacterInfo]:
        """Find unique characters visible in a time range.

        Args:
            all_detections: Detections per sampled frame.
            start_sec: Start time in seconds.
            end_sec: End time in seconds.
            sample_fps: Effective fps of sampled frames (1/sample_interval).
            sample_interval: Seconds between sampled frames.

        Returns:
            List of CharacterInfo for characters visible in this range.
        """
        start_frame = int(start_sec / sample_interval)
        end_frame = int(end_sec / sample_interval)
        start_frame = max(0, start_frame)
        end_frame = min(len(all_detections), end_frame)

        seen_chars: dict[int, CharacterInfo] = {}
        for frame_dets in all_detections[start_frame:end_frame]:
            for det in frame_dets:
                if det.character_id >= 0 and det.character_id not in seen_chars:
                    seen_chars[det.character_id] = CharacterInfo(
                        character_id=det.character_id,
                        representative_embedding=det.embedding,
                        total_frames=1,
                    )
                elif det.character_id in seen_chars:
                    seen_chars[det.character_id].total_frames += 1

        return list(seen_chars.values())

    def get_detections_for_range(
        self,
        all_detections: list[list[FaceDetection]],
        start_frame: int,
        end_frame: int,
    ) -> list[list[FaceDetection]]:
        """Extract detections for a frame range."""
        start_frame = max(0, start_frame)
        end_frame = min(len(all_detections), end_frame)
        return all_detections[start_frame:end_frame]

    def detect_faces_for_shots(
        self,
        video_decoder,
        shots: list,
        sample_interval: float = 0.5,
        min_frames_per_shot: int = 2,
        batch_size: int = 32,
    ) -> list[list[FaceDetection]]:
        """Detect faces with shot-aware sampling.

        Ensures at least min_frames_per_shot are sampled per shot,
        even for very short shots.

        Args:
            video_decoder: torchcodec VideoDecoder instance.
            shots: List of Shot objects with start_sec/end_sec.
            sample_interval: Seconds between sample frames.
            min_frames_per_shot: Minimum frames to sample per shot.
            batch_size: Batch size for face detection.

        Returns:
            List of detections per sampled frame (same format as detect_faces).
        """
        from tqdm import tqdm

        # Build sample timestamps ensuring coverage per shot
        sample_timestamps = []
        sample_shot_indices = []
        for shot in shots:
            duration = shot.end_sec - shot.start_sec
            n_regular = max(1, int(duration / sample_interval))
            n_frames = max(min_frames_per_shot, n_regular)
            for j in range(n_frames):
                t = shot.start_sec + j * duration / n_frames
                if t < shot.end_sec:
                    sample_timestamps.append(t)
                    sample_shot_indices.append(shot.index)

        logger.info(
            f"Shot-aware sampling: {len(sample_timestamps)} frames "
            f"from {len(shots)} shots (interval={sample_interval}s, "
            f"min_per_shot={min_frames_per_shot})"
        )

        all_detections = []
        for range_start in tqdm(
            range(0, len(sample_timestamps), batch_size),
            desc="Face detection (shot-aware)",
        ):
            batch_ts = sample_timestamps[range_start : range_start + batch_size]
            batch_shot_idx = sample_shot_indices[range_start : range_start + batch_size]
            batch_result = video_decoder.get_frames_played_at(batch_ts)
            frames_batch = batch_result.data

            dets = self.detect_faces(
                frames_batch,
                frame_indices=list(range(range_start, range_start + len(batch_ts))),
            )

            # Annotate detections with timestamp and shot index
            for i, frame_dets in enumerate(dets):
                for det in frame_dets:
                    det.timestamp = batch_ts[i]
                    det.shot_index = batch_shot_idx[i]

            all_detections.extend(dets)
            del frames_batch, batch_result

        return all_detections, sample_timestamps, sample_shot_indices

    def cluster_to_profiles(
        self, all_detections: list[list[FaceDetection]]
    ) -> dict[int, CharacterProfile]:
        """Cluster face detections into CharacterProfile objects.

        Same clustering as cluster_characters but returns CharacterProfile.
        """
        characters = self.cluster_characters(all_detections)
        profiles = {}
        for cid, info in characters.items():
            profile = CharacterProfile(
                character_id=cid,
                face_detections_count=info.total_frames,
                frame_indices=info.frame_indices,
                identity_sources=["face"],
            )
            profile.representative_face_embedding = info.representative_embedding

            # Collect face embeddings for this cluster
            for frame_dets in all_detections:
                for det in frame_dets:
                    if det.character_id == cid:
                        profile.face_embeddings.append(det.embedding)

            profiles[cid] = profile

        return profiles
