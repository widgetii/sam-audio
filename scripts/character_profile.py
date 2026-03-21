"""Character profile with multi-modal identity (face + voice)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceDetection:
    """A single face detection in a video frame."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray  # 512-dim ArcFace
    confidence: float
    frame_index: int = -1
    timestamp: float = -1.0  # seconds into the movie
    character_id: int = -1  # assigned after clustering
    shot_index: int = -1  # which shot this detection belongs to
    bbox_area: int = 0  # width * height in pixels
    aspect_ratio: float = 0.0  # width / height


@dataclass
class CharacterProfile:
    """Multi-modal character identity combining face and voice signals."""

    character_id: int

    # Face modality
    face_embeddings: list[np.ndarray] = field(default_factory=list)
    representative_face_embedding: np.ndarray | None = None
    face_detections_count: int = 0

    # Voice modality
    voice_embeddings: list[np.ndarray] = field(default_factory=list)
    representative_voice_embedding: np.ndarray | None = None
    voice_confidence: float = 0.0  # num clean voice samples

    # Tracking metadata
    frame_indices: list[int] = field(default_factory=list)
    identity_sources: list[str] = field(
        default_factory=list
    )  # ["face"], ["voice"], ["face", "voice"]

    @property
    def total_frames(self) -> int:
        return self.face_detections_count

    def update_face_centroid(self):
        """Recompute representative face embedding from all samples."""
        if self.face_embeddings:
            self.representative_face_embedding = np.mean(
                np.stack(self.face_embeddings), axis=0
            )
            if "face" not in self.identity_sources:
                self.identity_sources.append("face")

    def add_voice_embedding(self, embedding: np.ndarray):
        """Add a voice sample and update the centroid."""
        self.voice_embeddings.append(embedding)
        self.voice_confidence = len(self.voice_embeddings)
        self.representative_voice_embedding = np.mean(
            np.stack(self.voice_embeddings), axis=0
        )
        if "voice" not in self.identity_sources:
            self.identity_sources.append("voice")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def match_voice_to_character(
    voice_embedding: np.ndarray,
    profiles: dict[int, CharacterProfile],
    threshold: float = 0.7,
) -> int | None:
    """Match a voice embedding to the best matching character profile.

    Returns character_id if match found above threshold, else None.
    """
    best_id = None
    best_sim = threshold
    for cid, profile in profiles.items():
        if profile.representative_voice_embedding is None:
            continue
        sim = cosine_similarity(voice_embedding, profile.representative_voice_embedding)
        if sim > best_sim:
            best_sim = sim
            best_id = cid
    return best_id


def match_face_to_character(
    face_embedding: np.ndarray,
    profiles: dict[int, CharacterProfile],
    threshold: float = 0.5,
) -> int | None:
    """Match a face embedding to the best matching character profile.

    Returns character_id if match found above threshold, else None.
    """
    best_id = None
    best_sim = threshold
    for cid, profile in profiles.items():
        if profile.representative_face_embedding is None:
            continue
        sim = cosine_similarity(face_embedding, profile.representative_face_embedding)
        if sim > best_sim:
            best_sim = sim
            best_id = cid
    return best_id


def fuse_identity(
    face_match: int | None,
    voice_match: int | None,
    face_confidence: float = 1.0,
    voice_confidence: float = 0.8,
) -> tuple[int | None, float, str]:
    """Fuse face and voice identity signals.

    Returns (character_id, confidence, source).

    Identity fusion rules:
    1. Face match exists -> use it (high precision)
    2. Face + voice agree -> boost confidence
    3. Only voice match -> use it (lower confidence)
    4. Face and voice disagree -> trust face
    5. No match -> None
    """
    if face_match is not None and voice_match is not None:
        if face_match == voice_match:
            # Both agree — highest confidence
            return face_match, min(1.0, face_confidence + 0.1), "face+voice"
        else:
            # Disagree — trust face
            return face_match, face_confidence * 0.9, "face (voice disagrees)"
    elif face_match is not None:
        return face_match, face_confidence, "face"
    elif voice_match is not None:
        return voice_match, voice_confidence, "voice"
    else:
        return None, 0.0, "unknown"
