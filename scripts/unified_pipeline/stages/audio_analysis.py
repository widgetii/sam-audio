"""Stage 5: Audio analysis — SAM-Audio separation + center channel dialogue detection.

5A: Extract 5.1 center channel + full downmix
5B: SAM-Audio text-only separation + center channel → dialogue_segments
5C: High-fps SAM3 for multi-speaker visual separation → character_audio
5D: Voice fingerprinting → characters.voice_embedding
5E: Voice matching for unattributed chunks
"""

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)

STAGE = "stage5"
SAMPLE_RATE = 48000
CHUNK_DURATION = 90  # seconds per processing chunk


def rms_db(audio: torch.Tensor) -> float:
    rms = audio.float().pow(2).mean().sqrt()
    if rms < 1e-10:
        return -100.0
    return 20 * math.log10(rms.item())


def _extract_center_channel(video_path: str, stream_index: int) -> torch.Tensor | None:
    """Extract center channel from 5.1 surround audio."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "stream=channels",
            "-select_streams",
            str(stream_index),
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        channels = int(probe.stdout.strip())
    except ValueError:
        return None

    if channels < 6:
        log.info(
            f"Audio stream {stream_index} has {channels} channels, no center channel"
        )
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-map",
                f"0:{stream_index}",
                "-af",
                "pan=mono|c0=FC",
                "-ar",
                str(SAMPLE_RATE),
                tmp_path,
            ],
            capture_output=True,
            check=True,
        )
        wav, sr = torchaudio.load(tmp_path)
        log.info(f"Extracted center channel ({wav.shape[-1] / sr:.0f}s)")
        return wav[:1]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_full_audio(video_path: str, stream_index: int) -> torch.Tensor:
    """Extract full downmix audio as mono 48kHz."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-map",
                f"0:{stream_index}",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                tmp_path,
            ],
            capture_output=True,
            check=True,
        )
        wav, sr = torchaudio.load(tmp_path)
        return wav[:1]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_stage5(
    db: AnalysisDB,
    video_path: str,
    audio_stream: int = 0,
    device: str = "cuda",
    rms_threshold_db: float = -50.0,
    center_threshold_db: float = -40.0,
    workspace_dir: str | None = None,
):
    """Run all audio analysis sub-stages.

    Args:
        db: Analysis database.
        video_path: Source video.
        audio_stream: Audio stream index (e.g. 6 for English in Aliens).
        device: CUDA device.
        rms_threshold_db: SAM-Audio dialogue detection threshold.
        center_threshold_db: Center channel dialogue threshold.
        workspace_dir: Directory for intermediate audio files.
    """
    if workspace_dir is None:
        workspace_dir = str(Path(db.db_file).parent / "audio_workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    # 5A: Extract audio
    log.info("Stage 5A: extracting audio...")
    center_audio = _extract_center_channel(video_path, audio_stream)
    full_audio = _extract_full_audio(video_path, audio_stream)
    total_duration = full_audio.shape[-1] / SAMPLE_RATE
    log.info(f"Stage 5A: {total_duration:.0f}s audio extracted")

    # 5B: SAM-Audio text-only separation + dialogue detection
    if db.count_rows("dialogue_segments") == 0:
        _run_stage5b(
            db,
            full_audio,
            center_audio,
            video_path,
            device,
            rms_threshold_db,
            center_threshold_db,
        )
    else:
        log.info("Stage 5B: dialogue segments already in DB, skipping")

    # 5D: Voice fingerprinting from dialogue segments
    _run_stage5d(db, full_audio, workspace_dir)

    # 5E: Voice matching
    _run_stage5e(db, full_audio, workspace_dir)


def _run_stage5b(
    db: AnalysisDB,
    full_audio: torch.Tensor,
    center_audio: torch.Tensor | None,
    video_path: str,
    device: str,
    rms_threshold_db: float,
    center_threshold_db: float,
):
    """SAM-Audio text-only separation + center channel → dialogue segments."""
    log.info(
        "Stage 5B: running SAM-Audio text-only separation for dialogue detection..."
    )

    from sam_audio import SAMAudio, SAMAudioProcessor

    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-base")
    model = SAMAudio.from_pretrained("facebook/sam-audio-base").to(device)

    total_samples = full_audio.shape[-1]
    chunk_samples = CHUNK_DURATION * SAMPLE_RATE
    num_chunks = math.ceil(total_samples / chunk_samples)

    progress = db.get_progress("stage5b")
    all_segments = []

    for ci in range(num_chunks):
        chunk_key = str(ci)
        if progress.get(chunk_key) == "done":
            continue

        start_sample = ci * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        start_sec = start_sample / SAMPLE_RATE

        chunk_audio = full_audio[..., start_sample:end_sample]

        # SAM-Audio text-only separation
        try:
            batch = processor(audio=chunk_audio, text="speech", sample_rate=SAMPLE_RATE)
            result = model.separate(batch.to(device))
            target = result.target.cpu()
            residual = result.residual.cpu()
        except Exception as e:
            log.warning(f"SAM-Audio error on chunk {ci}: {e}")
            target = chunk_audio
            residual = torch.zeros_like(chunk_audio)

        # Center channel slice
        center_chunk = None
        if center_audio is not None:
            center_chunk = center_audio[..., start_sample:end_sample]

        # Analyze into 1-second segments
        seg_samples = SAMPLE_RATE
        for seg_start in range(0, end_sample - start_sample, seg_samples):
            seg_end = min(seg_start + seg_samples, end_sample - start_sample)
            t_seg = target[..., seg_start:seg_end].flatten()
            r_seg = residual[..., seg_start:seg_end].flatten()

            t_rms = rms_db(t_seg)
            r_rms = rms_db(r_seg)
            sam_dialogue = t_rms > rms_threshold_db and (t_rms - r_rms) > 0

            center_rms_val = -100.0
            if center_chunk is not None:
                c_seg = center_chunk[..., seg_start:seg_end].flatten()
                center_rms_val = rms_db(c_seg)
            center_dialogue = center_rms_val > center_threshold_db

            has_dialogue = sam_dialogue or center_dialogue

            seg_start_time = start_sec + seg_start / SAMPLE_RATE
            seg_end_time = start_sec + seg_end / SAMPLE_RATE

            all_segments.append(
                {
                    "start_sec": round(seg_start_time, 3),
                    "end_sec": round(seg_end_time, 3),
                    "has_dialogue": int(has_dialogue),
                    "center_db": round(center_rms_val, 1),
                    "sam_audio_db": round(t_rms, 1),
                    "character_id": None,
                }
            )

        db.mark_progress("stage5b", chunk_key, "done")
        log.info(f"Stage 5B: chunk {ci + 1}/{num_chunks} done")

    # Move model off GPU
    model.cpu()
    del model
    torch.cuda.empty_cache()

    if all_segments:
        db.insert_dialogue_segments(all_segments)

    dialogue_count = sum(1 for s in all_segments if s["has_dialogue"])
    log.info(
        f"Stage 5B: {dialogue_count}/{len(all_segments)} seconds with dialogue "
        f"({100 * dialogue_count / max(len(all_segments), 1):.1f}%)"
    )


def _run_stage5d(db: AnalysisDB, full_audio: torch.Tensor, workspace_dir: str):
    """Voice fingerprinting: extract voice embeddings for characters with audio."""
    if db.is_done(STAGE, "voice_fingerprint"):
        log.info("Stage 5D: voice fingerprinting already done, skipping")
        return

    characters = db.get_characters()
    if not characters:
        log.info("Stage 5D: no characters to fingerprint")
        return

    # Get dialogue segments attributed to characters
    char_audio = db.get_character_audio()
    if not char_audio:
        log.info("Stage 5D: no character audio for fingerprinting yet")
        db.mark_progress(STAGE, "voice_fingerprint", "done")
        return

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        log.warning(
            "Stage 5D: resemblyzer not installed, skipping voice fingerprinting"
        )
        db.mark_progress(STAGE, "voice_fingerprint", "done")
        return

    encoder = VoiceEncoder()

    from collections import defaultdict

    char_wavs: dict[int, list[np.ndarray]] = defaultdict(list)
    for ca in char_audio:
        if ca["audio_path"] and Path(ca["audio_path"]).exists():
            wav, sr = torchaudio.load(ca["audio_path"])
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            char_wavs[ca["character_id"]].append(wav[0].numpy())

    for char_id, wavs in char_wavs.items():
        if not wavs:
            continue
        combined = np.concatenate(wavs)
        processed = preprocess_wav(combined, source_sr=16000)
        if len(processed) < 1600:  # too short
            continue
        embedding = encoder.embed_utterance(processed)
        db.update_character(char_id, voice_embedding=embedding)
        log.info(
            f"Stage 5D: voice embedding for character {char_id} ({len(combined) / 16000:.1f}s)"
        )

    db.mark_progress(STAGE, "voice_fingerprint", "done")


def _run_stage5e(db: AnalysisDB, full_audio: torch.Tensor, workspace_dir: str):
    """Voice matching: attribute unidentified dialogue to characters by voice similarity."""
    if db.is_done(STAGE, "voice_match"):
        log.info("Stage 5E: voice matching already done, skipping")
        return

    characters = db.get_characters()
    chars_with_voice = [c for c in characters if c.get("voice_embedding") is not None]
    if not chars_with_voice:
        log.info("Stage 5E: no characters with voice embeddings, skipping")
        db.mark_progress(STAGE, "voice_match", "done")
        return

    # Get dialogue segments without character attribution
    unattributed = db.conn.execute(
        "SELECT * FROM dialogue_segments WHERE has_dialogue = 1 AND character_id IS NULL"
    ).fetchall()

    if not unattributed:
        log.info("Stage 5E: all dialogue segments already attributed")
        db.mark_progress(STAGE, "voice_match", "done")
        return

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        log.warning("Stage 5E: resemblyzer not installed, skipping")
        db.mark_progress(STAGE, "voice_match", "done")
        return

    encoder = VoiceEncoder()
    voice_threshold = 0.55

    matched = 0
    for seg in unattributed:
        seg = dict(seg)
        start = int(seg["start_sec"] * SAMPLE_RATE)
        end = int(seg["end_sec"] * SAMPLE_RATE)
        chunk = full_audio[..., start:end]

        if chunk.shape[-1] < SAMPLE_RATE // 2:  # too short
            continue

        # Resample to 16kHz for Resemblyzer
        chunk_16k = torchaudio.functional.resample(chunk, SAMPLE_RATE, 16000)
        wav_np = chunk_16k[0].numpy()
        processed = preprocess_wav(wav_np, source_sr=16000)
        if len(processed) < 1600:
            continue

        seg_embed = encoder.embed_utterance(processed)

        best_char = None
        best_sim = -1.0
        for c in chars_with_voice:
            sim = float(
                np.dot(seg_embed, c["voice_embedding"])
                / (
                    np.linalg.norm(seg_embed) * np.linalg.norm(c["voice_embedding"])
                    + 1e-8
                )
            )
            if sim > best_sim:
                best_sim = sim
                best_char = c["character_id"]

        if best_char is not None and best_sim >= voice_threshold:
            db.conn.execute(
                "UPDATE dialogue_segments SET character_id = ? WHERE id = ?",
                (best_char, seg["id"]),
            )
            matched += 1

    db.conn.commit()
    log.info(f"Stage 5E: voice-matched {matched}/{len(unattributed)} dialogue segments")
    db.mark_progress(STAGE, "voice_match", "done")
