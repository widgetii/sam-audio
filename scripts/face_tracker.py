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

    def cluster_characters(
        self, all_detections: list[list[FaceDetection]]
    ) -> dict[int, CharacterInfo]:
        """Cluster face detections into characters.

        Single-pass agglomerative clustering at the configured threshold.
        Small clusters (< 2 detections) are merged into the nearest larger
        cluster only if the cosine similarity exceeds a minimum (0.15).

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

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.cluster_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Merge singletons into nearest cluster if similarity is high enough
        min_merge_similarity = 0.15
        unique_labels, counts = np.unique(labels, return_counts=True)
        large_clusters = set(unique_labels[counts >= 2])
        singletons = set(unique_labels[counts < 2])

        if large_clusters and singletons:
            large_centroids = {}
            for lbl in large_clusters:
                emb = embeddings[labels == lbl]
                centroid = emb.mean(axis=0)
                centroid /= max(np.linalg.norm(centroid), 1e-10)
                large_centroids[lbl] = centroid

            large_ids = sorted(large_clusters)
            centroid_matrix = np.stack([large_centroids[lid] for lid in large_ids])
            merged = 0
            for s_lbl in singletons:
                s_emb = embeddings[labels == s_lbl][0]
                s_emb_n = s_emb / max(np.linalg.norm(s_emb), 1e-10)
                sims = centroid_matrix @ s_emb_n
                best_idx = int(np.argmax(sims))
                if sims[best_idx] >= min_merge_similarity:
                    labels[labels == s_lbl] = large_ids[best_idx]
                    merged += 1

            remaining = len(np.unique(labels))
            logger.info(
                f"Clustering: {len(unique_labels)} initial → "
                f"merged {merged} singletons → {remaining} characters"
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

    def track_characters_in_chunk(
        self,
        chunk_frames: torch.Tensor,
        character_detections: dict[int, FaceDetection],
        prompt_frame_indices: dict[int, int],
    ) -> dict[int, torch.Tensor]:
        """Track multiple characters through pre-decoded video frames using SAM3.

        Accepts pre-decoded tensor frames (shared with SAM-Audio) and passes
        them as PIL images to SAM3 — no temp files, no re-decoding.

        Uses text prompt "person" for auto-detection, then maps SAM3's
        auto-assigned obj_ids to character IDs via face center overlap.

        Args:
            chunk_frames: Pre-decoded video frames [N, C, H, W] uint8 at 480p.
            character_detections: char_id -> FaceDetection with bbox at
                frame resolution (480p).
            prompt_frame_indices: char_id -> chunk-local frame index for
                prompting SAM3.

        Returns:
            char_id -> inverted mask tensor [N_frames, 1, H, W] (target=0, bg=1).
        """
        from PIL import Image

        if self.sam3 is None:
            raise RuntimeError("SAM3 predictor is required for body tracking")

        n_frames, _, frame_h, frame_w = chunk_frames.shape

        # Convert tensor frames to PIL images — SAM3's supported input format.
        # SAM3 resizes to 1008x1008 internally and outputs masks at orig resolution.
        pil_frames = []
        for i in range(n_frames):
            frame_np = chunk_frames[i].permute(1, 2, 0).cpu().numpy()
            pil_frames.append(Image.fromarray(frame_np))

        # Start SAM3 session with PIL images (no disk I/O)
        response = self.sam3.handle_request(
            {
                "type": "start_session",
                "resource_path": pil_frames,
            }
        )
        session_id = response["session_id"]
        del pil_frames

        # Pick prompt frame — use the most common index among characters
        prompt_frame = max(
            set(prompt_frame_indices.values()),
            key=list(prompt_frame_indices.values()).count,
            default=0,
        )
        prompt_frame = max(0, min(prompt_frame, n_frames - 1))

        # Add text prompt "person" — SAM3 detects all people on prompt frame
        response = self.sam3.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame,
                "text": "person",
            }
        )

        # Map SAM3 auto-assigned obj_ids to character IDs via face center overlap
        outputs = response.get("outputs")
        sam3_to_char: dict[int, int] = {}
        if outputs is not None:
            prompt_masks: dict[int, np.ndarray] = {}
            for obj_id, mask in zip(
                outputs["out_obj_ids"],
                outputs["out_binary_masks"],
                strict=True,
            ):
                prompt_masks[int(obj_id)] = mask  # [H, W] bool

            used_sam3_ids: set[int] = set()
            for char_id, det in character_detections.items():
                x1, y1, x2, y2 = det.bbox
                face_cx = min(int((x1 + x2) / 2), frame_w - 1)
                face_cy = min(int((y1 + y2) / 2), frame_h - 1)

                best_obj_id = None
                best_area = float("inf")
                for sam3_obj_id, mask in prompt_masks.items():
                    if sam3_obj_id in used_sam3_ids:
                        continue
                    if mask[face_cy, face_cx]:
                        area = int(mask.sum())
                        if area < best_area:
                            best_obj_id = sam3_obj_id
                            best_area = area

                if best_obj_id is not None:
                    sam3_to_char[best_obj_id] = char_id
                    used_sam3_ids.add(best_obj_id)

        if not sam3_to_char:
            self.sam3.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
            logger.warning("No SAM3 detections matched character face centers")
            return {}

        logger.debug(
            f"SAM3 mapped {len(sam3_to_char)} objects to characters "
            f"on prompt frame {prompt_frame}"
        )

        # Propagate through all frames
        per_frame_masks: dict[int, dict[int, np.ndarray]] = {}
        frame_count = 0
        for result in self.sam3.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "both",
                "start_frame_index": 0,
            }
        ):
            frame_count += 1
            outputs = result.get("outputs")
            if outputs is None:
                continue
            frame_idx = result["frame_index"]
            for obj_id, mask in zip(
                outputs["out_obj_ids"],
                outputs["out_binary_masks"],
                strict=True,
            ):
                obj_id = int(obj_id)
                if obj_id in sam3_to_char:
                    char_id = sam3_to_char[obj_id]
                    per_frame_masks.setdefault(frame_idx, {})[char_id] = mask

        self.sam3.handle_request({"type": "close_session", "session_id": session_id})

        logger.debug(
            f"SAM3 tracked {len(sam3_to_char)} bodies across {frame_count} frames"
        )

        # Assemble per-character mask tensors, inverted (target=0, bg=1)
        result_masks = {}
        for char_id in sam3_to_char.values():
            frame_masks = []
            for fi in range(n_frames):
                if fi in per_frame_masks and char_id in per_frame_masks[fi]:
                    m = per_frame_masks[fi][char_id]
                else:
                    # No detection — use all-ones (no mask, full background)
                    m = np.ones((frame_h, frame_w), dtype=bool)
                frame_masks.append(m)

            stacked = np.stack(frame_masks)  # [N, H, W]
            mask_tensor = torch.from_numpy(stacked).unsqueeze(1)  # [N, 1, H, W]
            inverted = (~mask_tensor.bool()).float()
            result_masks[char_id] = inverted

        return result_masks

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
