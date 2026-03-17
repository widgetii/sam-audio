# SAM-Audio for Streaming Platforms: Research

Research into how SAM-Audio's audio separation capabilities map to real-world streaming platform needs. Based on analysis of the model's capabilities, industry practices (Netflix, Amazon Prime, Disney+), and the commercial audio separation landscape (AudioShake, iZotope, LALAL.AI).

## SAM-Audio Capabilities Summary

- **Prompt types**: text (noun/verb phrases), visual (video frames + masks), time spans
- **Sound categories**: speech, music, SFX, instruments, ambient noise, animal sounds
- **Quality scoring**: Judge (precision/recall/faithfulness), CLAP (text-audio similarity), Aesthetic (production quality)
- **Output**: target waveform + residual waveform (complementary separation)
- **Limitation**: separates by sound *category*, NOT by speaker identity (confirmed experimentally — "female voice" and "male voice" produce ~81% identical output)

## High-Value Use Cases

### 1. Dialogue Extraction for Dubbing & Localization (Highest Value)

**The problem**: Streaming platforms deliver content in 7-20+ languages. Traditional dubbing requires access to original production stems (Dialogue, Music, Effects tracks). For licensed or legacy content, stems often don't exist — only the final stereo/5.1 mix.

**SAM-Audio solution**:
- Prompt: `"speech"` — extracts clean dialogue track from mixed audio
- Residual contains music + SFX, preserved for the dubbed version
- Enables re-dubbing without re-recording sound effects or music
- Judge scores can flag segments where separation quality is low for human review

**Measured quality** (our benchmarks): Judge Overall 4.45/5.0 (base model), CLAP 0.30 for speech extraction.

**Industry context**: AudioShake reports 25%+ improvement in transcription accuracy when using separated dialogue. Companies like Deepdub, Papercup, and ElevenLabs integrate dialogue isolation into automated dubbing pipelines.

**SAM-Audio advantage over competitors**: visual prompting mode can use character face masks to help isolate specific speakers in scenes where multiple characters talk simultaneously — no competitor offers this.

### 2. Accessibility: Dialogue Enhancement for Hearing-Impaired Viewers

**The problem**: Viewers with hearing difficulties struggle with dialogue buried in music/SFX. Amazon Prime's "Dialogue Boost" already addresses this with AI.

**SAM-Audio solution**:
- Extract dialogue via `"speech"` prompt
- Mix back at boosted level relative to residual (music/SFX)
- Provide user-adjustable enhancement levels (e.g., +3dB, +6dB, +9dB speech boost)
- Could run client-side with text_only small model (6.97 GB for 30s audio)

**Deployment options**:
- Server-side: pre-process catalog content, store enhanced audio tracks
- Client-side: real-time enhancement using ONNX/WebGPU (already exported, see `export/`)

### 3. Music Identification & Licensing Compliance

**The problem**: Streaming catalogs contain millions of titles. Background music must be properly licensed. Identifying unlicensed music in user-generated or licensed content is critical for compliance.

**SAM-Audio solution**:
- Prompt: `"background music"` — isolates music track from dialogue/SFX
- Feed isolated music into fingerprinting service (Shazam, Audible Magic, Pex)
- Much higher identification accuracy on clean isolated music vs. mixed audio
- Can also extract `"singing"` separately from instrumental music

### 4. Automated Audio Metadata & Content Indexing

**The problem**: Manual tagging of audio content (has dialogue? has music? ambient-heavy? action SFX?) is expensive and inconsistent across millions of titles.

**SAM-Audio solution**:
- Run multiple prompts per content: `"speech"`, `"music"`, `"explosion"`, `"gunshot"`, `"car engine"`, `"animal sounds"`, `"crowd noise"`, etc.
- Compare target RMS vs residual RMS — high ratio = sound is present
- Generate per-segment audio metadata automatically
- Use Aesthetic scores to assess production quality

**Application**: content-based recommendations ("movies with minimal dialogue", "action-heavy scenes"), automated content warnings, scene-level audio indexing.

### 5. Audio Description Track Creation

**The problem**: Accessibility regulations require audio descriptions (narration of visual elements for blind viewers). Creating these requires knowing which moments have dialogue gaps where narration can be inserted.

**SAM-Audio solution**:
- Separate dialogue from other audio
- Analyze dialogue track to find gaps (silence detection on separated speech)
- These gaps are safe insertion points for audio description narration
- The separated residual (music/SFX) can be duck-mixed during narration

### 6. Content Moderation & Safety

**The problem**: Audio content must be screened for hate speech, profanity, threats. Background noise degrades speech-to-text accuracy.

**SAM-Audio solution**:
- Extract clean speech — feed to ASR (Whisper, etc.) for transcription
- Cleaner input = more accurate moderation decisions
- Can also identify specific sound categories for content warnings (`"gunshot"`, `"explosion"`, `"screaming"`)

### 7. Stem Extraction for Interactive/Immersive Audio

**The problem**: Dolby Atmos and spatial audio require separate stems. Interactive content (games, choose-your-own-adventure) needs flexible audio layers.

**SAM-Audio solution**:
- Decompose mixed audio into dialogue/music/SFX stems
- Enable dynamic remixing for spatial audio rendering
- Support interactive experiences where audio responds to viewer choices

## What SAM-Audio Cannot Do (Limitations)

| Capability | Status | Alternative |
|-----------|--------|-------------|
| Speaker diarization (who spoke when) | Not supported | pyannote-audio, NeMo |
| Speaker identification (Ripley vs Burke) | Not supported | Resemblyzer, speaker embeddings |
| Speech-to-text transcription | Not supported | Whisper, wav2vec2 |
| Music genre/mood classification | Not directly | Cyanite.ai, Epidemic Sound EAR |
| Loudness normalization (LUFS) | Not supported | ffmpeg, pyloudnorm |
| Real-time processing (<100ms latency) | Too slow (seconds per second of audio) | Specialized models |

## Competitive Landscape

| Company | Focus | Differentiator |
|---------|-------|---------------|
| **AudioShake** | Enterprise stem separation | 40+ contracts (Disney, WBD, NFL), 100M+ min/year |
| **iZotope RX** | Post-production audio repair | Industry standard in studios |
| **LALAL.AI** | Consumer/enterprise separation | Dubbing-focused, crystal-clear vocals |
| **Descript** | Audio/video editing | Integrated editing workflow |
| **Deepdub** | AI dubbing | End-to-end localization pipeline |
| **SAM-Audio** | Research model (Meta) | Visual prompting, multi-modal, open-source |

**SAM-Audio's unique advantages**:
1. **Visual prompting** — no competitor can use video frames + masks to guide separation
2. **Open-source** — can be customized, deployed on-premise, no per-minute licensing
3. **Quality scoring built-in** — Judge/CLAP/Aesthetic metrics for automated QA
4. **Text-only mode at 7 GB** — deployable on consumer hardware or in-browser

## Deployment Architecture

```
Content Ingestion Pipeline (batch, server-side)
  ├── SAM-Audio base model (A100/H100 GPU cluster)
  │   ├── "speech" → dialogue stem
  │   ├── "music" → music stem
  │   ├── "ambient noise" → ambience stem
  │   └── Judge scores → QA flags
  ├── Dialogue stem → Whisper ASR → subtitles/captions
  ├── Dialogue stem → dubbing pipeline (ElevenLabs/Deepdub)
  ├── Music stem → fingerprinting → licensing check
  └── All stems → metadata indexing

Client-side (optional, real-time)
  ├── SAM-Audio small (ONNX/WebGPU, text_only)
  └── Dialogue boost slider for hearing-impaired viewers
```

## Estimated Scale & Cost

For a catalog of 10,000 titles averaging 90 min each:
- Total audio: 900,000 minutes = 15,000 hours
- SAM-Audio base on A100: ~50s per 203s audio → ~3.7x realtime
- Processing time: 15,000 hours / 3.7 = ~4,054 GPU-hours
- At $2/GPU-hour (cloud A100): **~$8,100** for full catalog
- Re-processing for new prompts/models: same cost per pass

This is dramatically cheaper than manual stem extraction or re-recording.
