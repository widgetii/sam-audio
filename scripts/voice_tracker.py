"""Voice fingerprinting using Resemblyzer d-vector speaker embeddings."""

import logging
import math

import numpy as np
import torch
import torchaudio
from character_profile import CharacterProfile, cosine_similarity

logger = logging.getLogger(__name__)


class VoiceTracker:
    """Extract and manage speaker voice embeddings using Resemblyzer."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self._encoder = None

    def _load_model(self):
        """Lazy-load the Resemblyzer voice encoder."""
        if self._encoder is not None:
            return
        from resemblyzer import VoiceEncoder

        logger.info("Loading Resemblyzer voice encoder")
        self._encoder = VoiceEncoder(device=str(self.device))

    def extract_embedding(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
    ) -> np.ndarray | None:
        """Extract a 256-dim speaker embedding from audio.

        Args:
            audio: Audio tensor [1, samples] or [samples].
            sample_rate: Input sample rate.

        Returns:
            256-dim numpy embedding, or None if audio too short/quiet.
        """
        self._load_model()

        if audio.ndim == 2:
            audio = audio.squeeze(0)

        # Resample to 16kHz (Resemblyzer requirement)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)

        # Minimum 0.5s of audio
        if audio.shape[-1] < 8000:
            return None

        # Check RMS — skip if too quiet
        rms = audio.float().pow(2).mean().sqrt()
        if rms < 1e-5:
            return None

        wav = audio.float().cpu().numpy()
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(wav, source_sr=16000)
        if len(wav) < 8000:
            return None

        embedding = self._encoder.embed_utterance(wav)
        return embedding

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
        threshold: float = 0.55,
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
