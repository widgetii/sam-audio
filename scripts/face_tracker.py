"""Face detection, embedding, clustering, and mask generation for character tracking."""

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
    """Detect faces, compute embeddings, cluster into characters, generate masks."""

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
                frame_dets.append(
                    FaceDetection(
                        bbox=bbox,
                        embedding=face.normed_embedding,
                        confidence=float(face.det_score),
                        frame_index=frame_indices[i],
                    )
                )
            all_detections.append(frame_dets)
        return all_detections

    def cluster_characters(
        self, all_detections: list[list[FaceDetection]]
    ) -> dict[int, CharacterInfo]:
        """Cluster face detections into characters using agglomerative clustering.

        Returns:
            character_id -> CharacterInfo, sorted by total screen time (descending).
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

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.cluster_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Assign character_ids to detections
        for det, label in zip(flat_dets, labels, strict=True):
            det.character_id = int(label)

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
            cluster_embs = embeddings[labels == cid]
            info.representative_embedding = cluster_embs.mean(axis=0)

        # Sort by total screen time (descending)
        sorted_chars = dict(
            sorted(characters.items(), key=lambda x: x[1].total_frames, reverse=True)
        )
        return sorted_chars

    def generate_masks(
        self,
        frames: torch.Tensor,
        all_detections: list[list[FaceDetection]],
        character_id: int,
        video_file: str | None = None,
    ) -> torch.Tensor:
        """Generate binary masks for a specific character.

        With SAM3: uses face bbox center as point prompt for precise segmentation.
        Without SAM3: creates rectangular masks from face bboxes (padded 20%).

        Args:
            frames: Video frames [N, C, H, W].
            all_detections: Detections per frame (aligned to frames).
            character_id: Which character to generate masks for.
            video_file: Path to video file (needed for SAM3).

        Returns:
            Binary mask tensor [N, 1, H, W] where target=0, background=1.
        """
        N, C, H, W = frames.shape
        masks = torch.ones(N, 1, H, W, dtype=frames.dtype)

        if self.sam3 is not None and video_file is not None:
            masks = self._generate_sam3_masks(
                frames, all_detections, character_id, video_file
            )
        else:
            masks = self._generate_bbox_masks(frames, all_detections, character_id)

        return masks

    def _generate_bbox_masks(
        self,
        frames: torch.Tensor,
        all_detections: list[list[FaceDetection]],
        character_id: int,
    ) -> torch.Tensor:
        """Fallback: rectangular masks from face bboxes with 20% padding."""
        N, C, H, W = frames.shape
        # background=1 (non-zero), target region=0
        masks = torch.ones(N, 1, H, W, dtype=frames.dtype)

        for i, frame_dets in enumerate(all_detections):
            for det in frame_dets:
                if det.character_id != character_id:
                    continue
                x1, y1, x2, y2 = det.bbox
                # Pad by 20% of bbox size
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(W, x2 + pad_x)
                y2 = min(H, y2 + pad_y)
                masks[i, 0, y1:y2, x1:x2] = 0
        return masks

    def _generate_sam3_masks(
        self,
        frames: torch.Tensor,
        all_detections: list[list[FaceDetection]],
        character_id: int,
        video_file: str,
    ) -> torch.Tensor:
        """Use SAM3 video predictor with face bbox center as point prompt."""
        N, C, H, W = frames.shape

        response = self.sam3.handle_request(
            request={"type": "start_session", "resource_path": video_file}
        )
        session_id = response["session_id"]

        output_masks = []
        prev_mask = np.zeros((1, H, W), dtype=bool)

        for i, frame_dets in enumerate(all_detections):
            # Find this character's bbox in this frame
            char_det = None
            for det in frame_dets:
                if det.character_id == character_id:
                    char_det = det
                    break

            if char_det is not None:
                x1, y1, x2, y2 = char_det.bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                response = self.sam3.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": i,
                        "points": [[cx, cy]],
                        "labels": [1],
                    }
                )
                mask = response["outputs"]["out_binary_masks"]
                if mask.shape[0] == 0:
                    mask = prev_mask
                else:
                    prev_mask = mask
            else:
                mask = prev_mask

            output_masks.append(mask)

        # Convert: SAM3 mask is True where object is, we need 0=target, 1=background
        mask_tensor = torch.from_numpy(np.stack(output_masks))  # [N, 1, H, W]
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(1)
        # Invert: target=0, background=1
        inverted = (~mask_tensor.bool()).float()
        return inverted

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
