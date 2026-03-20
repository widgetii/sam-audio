"""Voice fingerprinting using ECAPA-TDNN speaker embeddings."""

import logging
import math

import numpy as np
import torch
import torchaudio
from character_profile import CharacterProfile, cosine_similarity

logger = logging.getLogger(__name__)


class VoiceTracker:
    """Extract and manage speaker voice embeddings using ECAPA-TDNN."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self._model = None

    def _load_model(self):
        """Lazy-load the ECAPA-TDNN model."""
        if self._model is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier

        logger.info("Loading ECAPA-TDNN speaker encoder")
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )

    def extract_embedding(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
        target_rate: int = 16000,
    ) -> np.ndarray | None:
        """Extract a 192-dim speaker embedding from audio.

        Args:
            audio: Audio tensor [1, samples] or [samples].
            sample_rate: Input sample rate.
            target_rate: ECAPA-TDNN expects 16kHz.

        Returns:
            192-dim numpy embedding, or None if audio too short/quiet.
        """
        self._load_model()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Resample to 16kHz
        if sample_rate != target_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_rate)

        # Minimum 0.5s of audio
        min_samples = target_rate // 2
        if audio.shape[-1] < min_samples:
            return None

        # Check RMS — skip if too quiet
        rms = audio.float().pow(2).mean().sqrt()
        if rms < 1e-5:
            return None

        with torch.inference_mode():
            embedding = self._model.encode_batch(audio.to(self.device))
            return embedding.squeeze().cpu().numpy()

    def extract_from_separation(
        self,
        target_audio: torch.Tensor,
        residual_audio: torch.Tensor,
        sample_rate: int = 48000,
        min_target_to_residual_db: float = 5.0,
    ) -> np.ndarray | None:
        """Extract voice embedding from a SAM-Audio separation result.

        Only extracts if the separation quality is good enough
        (target significantly louder than residual).

        Args:
            target_audio: Separated target audio [1, samples].
            residual_audio: Residual audio [1, samples].
            sample_rate: Audio sample rate.
            min_target_to_residual_db: Minimum dB difference for quality gate.

        Returns:
            192-dim embedding or None if quality too low.
        """
        t_rms = _rms_db(target_audio)
        r_rms = _rms_db(residual_audio)
        t_to_r = t_rms - r_rms

        if t_to_r < min_target_to_residual_db:
            return None

        return self.extract_embedding(target_audio, sample_rate)

    def match_to_profiles(
        self,
        embedding: np.ndarray,
        profiles: dict[int, CharacterProfile],
        threshold: float = 0.7,
    ) -> tuple[int | None, float]:
        """Match a voice embedding to known character profiles.

        Returns (character_id, similarity) or (None, 0.0).
        """
        best_id = None
        best_sim = threshold
        for cid, profile in profiles.items():
            if profile.representative_voice_embedding is None:
                continue
            sim = cosine_similarity(embedding, profile.representative_voice_embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def update_profile(
        self,
        profile: CharacterProfile,
        embedding: np.ndarray,
    ):
        """Add a voice sample to a character's profile."""
        profile.add_voice_embedding(embedding)
        logger.debug(
            f"Character {profile.character_id}: "
            f"{len(profile.voice_embeddings)} voice samples"
        )


def _rms_db(audio: torch.Tensor) -> float:
    """Compute RMS in dB."""
    rms = audio.float().pow(2).mean().sqrt()
    if rms < 1e-10:
        return -100.0
    return 20 * math.log10(rms.item())
