"""TalkNet ASD vendor wrapper.

Wraps the TalkNet Active Speaker Detection model for use with the unified
pipeline's Stage 4 enrichment. The actual TalkNet source lives in an external
directory and is imported via sys.path injection at load time.

TalkNet inference requires:
- Face crops at 25fps: grayscale 112x112 center-cropped from 224x224 resize
- Audio: 16kHz mono WAV → MFCC (13 cepstral, 25ms window, 10ms step → 100 frames/sec)
- Multiple durations averaged for robust scoring
"""

import logging
import math
import sys

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

log = logging.getLogger(__name__)

# Durations (in seconds) to average over for stable scores
DURATION_SET = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]


class TalkNetASD:
    """Active Speaker Detection using TalkNet."""

    def __init__(self, model_path: str, talknet_root: str, device: str = "cuda"):
        """Load TalkNet model.

        Args:
            model_path: Path to pretrained checkpoint (.model file).
            talknet_root: Path to TalkNet source directory (contains talkNet.py).
            device: CUDA device string.
        """
        self.device = device

        # TalkNet source uses mixed import styles:
        #   `from talknet.model.xxx import ...` (package imports — need parent on path)
        #   `from talkNet import talkNet` (module imports — need talknet dir on path)
        import os

        if os.path.isfile(os.path.join(talknet_root, "talkNet.py")):
            # talknet_root IS the package dir — add both it and its parent
            parent = os.path.dirname(talknet_root)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            if talknet_root not in sys.path:
                sys.path.insert(0, talknet_root)
        else:
            if talknet_root not in sys.path:
                sys.path.insert(0, talknet_root)

        from talkNet import talkNet

        self.model = talkNet(device=device)
        self.model.loadParameters(model_path)
        self.model.eval()
        log.info(f"TalkNet loaded from {model_path}")

    def _prepare_video_features(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """Convert face crops to TalkNet visual features.

        Args:
            face_crops: List of BGR face crop images (any size).

        Returns:
            Array of shape (N, 112, 112) — grayscale center-cropped faces.
        """
        features = []
        for crop in face_crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (224, 224))
            # Center crop 112x112
            center = resized[56:168, 56:168]
            features.append(center)
        return np.array(features)

    def _prepare_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from audio file.

        Args:
            audio_path: Path to 16kHz mono WAV.

        Returns:
            Array of shape (N, 13) — MFCC features at 100fps.
        """
        sr, audio = wavfile.read(audio_path)
        if sr != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sr}Hz")
        mfcc = python_speech_features.mfcc(
            audio, sr, numcep=13, winlen=0.025, winstep=0.010
        )
        return mfcc

    def score_track(
        self,
        face_crops: list[np.ndarray],
        audio_path: str,
        audio_offset: float = 0.0,
        audio_duration: float | None = None,
    ) -> np.ndarray:
        """Score a face track for speaking activity.

        Args:
            face_crops: Face crop images at 25fps (BGR, any resolution).
            audio_path: Path to 16kHz mono WAV for the full shot.
            audio_offset: Start offset in seconds within the audio file.
            audio_duration: Duration in seconds. If None, derived from face_crops.

        Returns:
            Per-frame speaking scores (one per face crop). Higher = more likely speaking.
        """
        if not face_crops:
            return np.array([])

        video_feat = self._prepare_video_features(face_crops)
        audio_mfcc = self._prepare_audio_features(audio_path)

        # Slice audio to match the track's time range
        start_frame = int(audio_offset * 100)
        if audio_duration is not None:
            end_frame = start_frame + int(audio_duration * 100)
        else:
            end_frame = start_frame + int(len(face_crops) / 25.0 * 100)
        audio_mfcc = audio_mfcc[start_frame:end_frame]

        # Align lengths: 100 audio frames per second, 25 video frames per second
        length = min(
            (audio_mfcc.shape[0] - audio_mfcc.shape[0] % 4) / 100,
            video_feat.shape[0] / 25,
        )
        if length < 0.04:  # Less than 1 video frame
            return np.zeros(len(face_crops))

        audio_mfcc = audio_mfcc[: int(round(length * 100))]
        video_feat = video_feat[: int(round(length * 25))]

        # Multi-duration scoring for robustness
        all_scores = []
        for duration in DURATION_SET:
            if duration > length:
                continue
            batch_size = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batch_size):
                    a_start = i * duration * 100
                    a_end = (i + 1) * duration * 100
                    v_start = i * duration * 25
                    v_end = (i + 1) * duration * 25

                    input_a = (
                        torch.FloatTensor(audio_mfcc[a_start:a_end])
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    input_v = (
                        torch.FloatTensor(video_feat[v_start:v_end])
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    if input_a.shape[1] == 0 or input_v.shape[1] == 0:
                        continue

                    embed_a = self.model.model.forward_audio_frontend(input_a)
                    embed_v = self.model.model.forward_visual_frontend(input_v)
                    embed_a, embed_v = self.model.model.forward_cross_attention(
                        embed_a, embed_v
                    )
                    out = self.model.model.forward_audio_visual_backend(
                        embed_a, embed_v
                    )
                    score = self.model.lossAV.forward(out, labels=None)
                    scores.extend(score)
            if scores:
                all_scores.append(scores)

        if not all_scores:
            return np.zeros(len(face_crops))

        # Average across durations, pad/truncate to match video length
        min_len = min(len(s) for s in all_scores)
        avg_scores = np.mean([s[:min_len] for s in all_scores], axis=0)

        # Scores are at 25fps granularity per duration chunk — expand to per-frame
        # Each score covers `duration` seconds = `duration*25` frames
        # We have one score per chunk, expand to per-frame
        result = np.zeros(len(face_crops))
        n = len(avg_scores)
        frames_per_score = max(1, len(video_feat) // n) if n > 0 else 1
        for i, s in enumerate(avg_scores):
            start = i * frames_per_score
            end = min((i + 1) * frames_per_score, len(result))
            result[start:end] = s

        return result
