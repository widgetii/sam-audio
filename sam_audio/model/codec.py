# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import math
from abc import ABCMeta, abstractmethod
from typing import Union

import dacvae
import torch

from sam_audio.model.config import DACVAEConfig


class Encoder(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...


class Codec(Encoder):
    @abstractmethod
    def decode(self, encoded_frames: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def wav_idx_to_feature_idx(
        self, wav_idx: Union[torch.Tensor, int], sample_rate=None
    ) -> Union[torch.Tensor, int]: ...

    @abstractmethod
    def feature_idx_to_wav_idx(
        self, feature_idx: Union[torch.Tensor, int], sample_rate=None
    ) -> Union[torch.Tensor, int]: ...

    @staticmethod
    def cast_to_int(
        x: Union[int, torch.Tensor],
    ) -> Union[int, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return x.int()
        else:
            return int(x)


class DACVAEEncoder(Encoder):
    def __init__(self, config: DACVAEConfig) -> None:
        super().__init__()
        model = dacvae.DACVAE(
            encoder_dim=config.encoder_dim,
            encoder_rates=config.encoder_rates,
            latent_dim=config.latent_dim,
            decoder_dim=config.decoder_dim,
            decoder_rates=config.decoder_rates,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
            sample_rate=config.sample_rate,
        ).eval()
        self._setup_model(model)
        self.hop_length = config.hop_length
        self.sample_rate = config.sample_rate

    def _setup_model(self, model):
        self.encoder = model.encoder
        self.quantizer = model.quantizer

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            z = self.encoder(self._pad(waveform))
            mean, _ = self.quantizer.in_proj(z).chunk(2, dim=1)
            encoded_frames = mean
        return encoded_frames

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs


class DACVAE(DACVAEEncoder, Codec):
    def _setup_model(self, model):
        super()._setup_model(model)
        self.decoder = model.decoder

    def decode(self, encoded_frames: torch.Tensor) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            emb = self.quantizer.out_proj(encoded_frames)
            return self.decoder(emb)

    def decode_chunked(
        self,
        encoded_frames: torch.Tensor,
        chunk_tokens: int = 500,
        overlap_tokens: int = 16,
    ) -> torch.Tensor:
        T = encoded_frames.shape[-1]
        if T <= chunk_tokens:
            return self.decode(encoded_frames)

        overlap_samples = overlap_tokens * self.hop_length
        output_chunks = []
        start = 0

        while start < T:
            end = min(start + chunk_tokens, T)
            chunk_feat = encoded_frames[..., start:end]
            chunk_wav = self.decode(chunk_feat)

            if start == 0:
                # First chunk: keep everything (right overlap will be skipped by next chunk)
                output_chunks.append(chunk_wav)
            else:
                # Subsequent chunks: discard left overlap (already covered by previous chunk)
                output_chunks.append(chunk_wav[..., overlap_samples:])

            del chunk_wav
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start += chunk_tokens - overlap_tokens

        return torch.cat(output_chunks, dim=-1)

    def feature_idx_to_wav_idx(self, feature_idx, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        wav_chunklen = feature_idx * self.hop_length * (orig_freq / new_freq)
        return self.cast_to_int(wav_chunklen)

    def wav_idx_to_feature_idx(self, wav_idx, sample_rate=None):
        ceil = math.ceil
        if torch.is_tensor(wav_idx):
            ceil = torch.ceil
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        target_length = ceil(new_freq * wav_idx / orig_freq)
        res = ceil(target_length / self.hop_length)
        return self.cast_to_int(res)
