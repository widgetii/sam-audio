# Auto-Framing Comparison Player — Spec

Desktop app that plays back the original 1080p video with auto-framing crop rectangles overlaid, comparing two scoring modes side-by-side (with/without TalkNet ASD speaker detection).

## Inputs

1. **Video file** — original 1080p source (e.g. `Aliens.1080p.mkv`)
2. **SQLite DB** — pipeline analysis database (e.g. `Aliens.1080p.10cab98a.analysis.db`)
3. **Audio stream index** — which audio stream to play (e.g. `6` for English)
4. **Time range** (optional) — start/end seconds to focus on a scene

## Layout

```
+-----------------------------------------------------------------------+
|  [toolbar: play/pause, seek bar, speed, time display, scene selector] |
+-----------------------------------------------------------------------+
|                                                                       |
|    +-------- original 1080p frame (scaled to fit) --------+           |
|    |                                                      |           |
|    |   [green rect]         [blue rect]                   |           |
|    |   "No ASD"             "With ASD"                    |           |
|    |   607x1080             607x1080                      |           |
|    |                                                      |           |
|    |   Person bboxes drawn as thin colored lines          |           |
|    |   Main person bbox highlighted (thick, matching      |           |
|    |   crop color). Face detection shown as small circle.  |           |
|    |                                                      |           |
|    +------------------------------------------------------+           |
|                                                                       |
+--[ no-ASD crop preview ]--+--[ ASD crop preview ]---------------------+
|                            |                                          |
|  Vertical 9:16 crop       |  Vertical 9:16 crop                      |
|  rendered live             |  rendered live                           |
|  Label: character name     |  Label: character name + ASD score       |
|                            |                                          |
+----------------------------+------------------------------------------+
```

**Top section** (~60% height): original frame with overlaid rectangles.
**Bottom section** (~40% height): two vertical crop previews side by side.

Both crop rectangles are 607x1080 on the 1920x1080 source frame. They differ only in horizontal position (crop_x).

## DB Schema Reference

All coordinates are in source video pixels (1920x1080).

### `shots` — shot boundaries
| Column | Type | Description |
|--------|------|-------------|
| shot_id | INTEGER PK | Auto-increment |
| start_frame | INTEGER | First frame number |
| end_frame | INTEGER | Last frame number |
| start_sec | REAL | Start time in seconds |
| end_sec | REAL | End time in seconds |

### `scenes` — groups of shots
| Column | Type | Description |
|--------|------|-------------|
| scene_id | INTEGER PK | Auto-increment |
| start_sec | REAL | Scene start |
| end_sec | REAL | Scene end |
| shot_ids | TEXT | JSON array of shot_id values |

### `person_tracks` — SAM3 person detections at 1fps
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| shot_id | INTEGER | Which shot |
| sam3_obj_id | INTEGER | Object ID within the shot (0-based) |
| frame_sec | REAL | Timestamp of this keyframe |
| bbox_x1, bbox_y1, bbox_x2, bbox_y2 | REAL | Person bounding box in source pixels |
| mask_rle | BLOB | RLE-encoded segmentation mask (optional, can be NULL) |

### `speaker_scores` — TalkNet ASD results at 1fps
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| shot_id | INTEGER | |
| sam3_obj_id | INTEGER | |
| frame_sec | REAL | |
| asd_score | REAL | Speaking probability. Positive = likely speaking, negative = not speaking. Range typically [-3.5, +3.0] |

### `face_detections` — InsightFace detections
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| shot_id | INTEGER | |
| sam3_obj_id | INTEGER | Which person track this face belongs to |
| frame_sec | REAL | |
| face_bbox_x1..y2 | REAL | Face bounding box in source pixels |
| face_score | REAL | Detection confidence |
| embedding | BLOB | ArcFace embedding (512-dim float32, can ignore) |

### `track_identities` — character assignments
| Column | Type | Description |
|--------|------|-------------|
| shot_id | INTEGER | |
| sam3_obj_id | INTEGER | |
| character_id | INTEGER | FK to characters table |
| confidence | REAL | |
| method | TEXT | e.g. "face_cluster" |

### `characters` — identity profiles
| Column | Type | Description |
|--------|------|-------------|
| character_id | INTEGER PK | |
| name | TEXT | Display name (may be NULL — use "Char {id}") |
| face_image | BLOB | JPEG thumbnail (can display in UI) |

### `dialogue_segments` — audio analysis
| Column | Type | Description |
|--------|------|-------------|
| start_sec | REAL | |
| end_sec | REAL | |
| has_dialogue | INTEGER | 0 or 1 |

## Scoring Algorithm (must replicate exactly)

For each person at each timestamp, compute a score. Highest score wins = main subject.

```
score = W_SPEAKING * asd_score
      + W_BLUR * (1.0 - blur_score)
      + W_CLASS * (1.0 if has_face else 0.0)
      + bbox_area * 0.01

Where:
  W_SPEAKING = 30401
  W_BLUR     = -301
  W_CLASS    = 301
  bbox_area  = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
  has_face   = True if (shot_id, sam3_obj_id) exists in face_detections
  asd_score  = from speaker_scores table (0.0 if no entry)
  blur_score = from blur_scores table (0.5 if no entry)
```

**"No ASD" mode**: same formula but force `asd_score = 0.0` for all persons.

### Crop Position

1. For each frame at playback fps, find person tracks active at that timestamp (interpolate linearly between 1fps keyframes within each track's time span; skip tracks whose bbox covers >85% of frame width AND height — those are background).
2. Pick highest-scoring person = main subject.
3. Compute horizontal center from the **mask silhouette head position**: decode `mask_rle` from `person_tracks`, find the topmost rows with mask pixels, take the horizontal center of the top 10% of the mask height. This gives the head position (~30px accuracy vs face detection ground truth). Fall back to person bbox center `(bbox_x1 + bbox_x2) / 2` only if no mask is stored. Then `crop_x = center_x - crop_w / 2`, clamped to `[0, video_width - crop_w]`. Interpolate `head_cx` linearly between 1fps keyframes.
4. Smooth: `crop_x = 0.7 * crop_x + 0.3 * prev_crop_x`.
5. `crop_y = (video_height - crop_h) / 2` (centered vertically, always 0 for 1080p).
6. `crop_w = 607, crop_h = 1080` (9:16 from 1920x1080).

## Overlay Rendering

On the original frame:

1. **All person bboxes** — thin outlines (1-2px), color-coded by character_id (use a palette). Show `sam3_obj_id` label at top-left of each bbox.
2. **Main person bbox (no-ASD)** — thick green outline (3-4px). Label: "No ASD" + character name.
3. **Main person bbox (with-ASD)** — thick blue outline (3-4px). Label: "ASD" + character name + asd_score.
4. **Green crop rectangle** — 607x1080 semi-transparent green overlay at the no-ASD crop_x position.
5. **Blue crop rectangle** — 607x1080 semi-transparent blue overlay at the ASD crop_x position.
6. **Overlap region** — where both rectangles overlap, blend to cyan.
7. **Face detection dots** — small circles at face bbox centers for persons that have face detections.
8. **Dialogue indicator** — small bar/dot in corner showing if current segment has_dialogue=1.

## Bottom Previews

Two panels, each showing the cropped 607x1080 region scaled to fit the panel:
- Left: "No ASD" crop
- Right: "With ASD" crop
- Each has a label overlay showing: character name, score breakdown, asd_score (right panel only)

## Playback Controls

- **Play/Pause** (spacebar)
- **Seek bar** with shot boundary markers (thin vertical lines)
- **Scene selector** dropdown — jump to scene by scene_id
- **Speed** — 0.25x, 0.5x, 1x, 2x
- **Frame step** — left/right arrow keys
- **Time display** — current time in MM:SS.ms format
- **Shot info** — show current shot_id, scene_id in a status bar

## Data Loading Strategy

1. On startup: load all `shots`, `scenes`, `characters` tables (small).
2. Load `person_tracks` indexed by `(shot_id, frame_sec)` — 13.9K rows, fits in memory.
3. Load `speaker_scores` indexed by `(shot_id, sam3_obj_id, frame_sec)` — 13K rows.
4. Load `face_detections` as a set of `(shot_id, sam3_obj_id)` for the `has_face` check.
5. Load `track_identities` for character name lookups.
6. Video decoding: use the platform video decoder (e.g. OpenCV, ffmpeg, or native). Seek by timestamp.

All DB data fits easily in RAM (<50MB). Precompute crop positions for both modes at 1fps, then interpolate at display time.

## Tech Stack Suggestion

Any desktop framework that can decode video frames and draw overlays:
- **Python + OpenCV + Qt/tkinter** — simplest, all deps available
- **Electron + ffmpeg.wasm** — if web-based preferred
- **Rust + egui + ffmpeg** — if performance matters

The critical requirement: frame-accurate video seeking and smooth overlay rendering at playback speed.

## Files on GPU Machine

- Video: `/data/huggingface/Aliens.1080p.mkv` (audio stream 6 = English 5.1)
- DB: `/data/huggingface/Aliens.1080p.10cab98a.analysis.db` (162MB)
- DB backup without ASD: `/data/huggingface/Aliens.1080p.10cab98a.analysis.db.bak-no-asd`

## Test Scene

Scene 7 (briefing room): `--start 1991 --end 2269` (33:11 to 37:49).
17 characters tracked, rapid dialogue cuts, lots of ASD-vs-no-ASD divergence (49% of frames pick a different person).
