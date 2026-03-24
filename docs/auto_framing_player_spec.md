# Framer — Auto-Framing Comparison Player Spec

Desktop app that plays the original 1080p video with auto-framing crop rectangles overlaid, comparing two scoring modes (with/without TalkNet ASD). Person segmentation masks rendered as colored overlays from sidecar FFV1 label-map videos.

## Inputs

1. **Video file** — original 1080p source (e.g. `Aliens.1080p.mkv`)
2. **SQLite DB** — pipeline analysis database (e.g. `Aliens.1080p.10cab98a.analysis.db`, 162MB)
3. **Masks directory** — FFV1 label-map videos (e.g. `Aliens.1080p.10cab98a.masks/`, 432MB, 1040 `.mkv` files)
4. **Audio stream index** — which audio stream to play (e.g. `6` for English)
5. **Time range** (optional) — start/end seconds to focus on a scene

## Layout

```
+-----------------------------------------------------------------------+
|  [toolbar: play/pause, seek bar, speed, time display, scene selector] |
+-----------------------------------------------------------------------+
|                                                                       |
|    +-------- original 1080p frame (scaled to fit) --------+           |
|    |                                                      |           |
|    |   Person masks as semi-transparent colored overlays  |           |
|    |   [green rect]         [blue rect]                   |           |
|    |   "No ASD"             "With ASD"                    |           |
|    |   607x1080             607x1080                      |           |
|    |                                                      |           |
|    |   Main person highlighted (thick outline matching    |           |
|    |   crop color). Head position shown as crosshair.     |           |
|    |                                                      |           |
|    +------------------------------------------------------+           |
|                                                                       |
+--[ no-ASD crop preview ]--+--[ ASD crop preview ]---------------------+
|                            |                                          |
|  Vertical 9:16 crop       |  Vertical 9:16 crop                      |
|  with mask overlays       |  with mask overlays                      |
|  Label: character name     |  Label: character name + ASD score       |
|                            |                                          |
+----------------------------+------------------------------------------+
```

**Top section** (~60% height): original frame with mask overlays + crop rectangles.
**Bottom section** (~40% height): two vertical crop previews side by side.

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

### `person_tracks` — SAM3 person detections at native fps (~24fps)
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| shot_id | INTEGER | Which shot |
| sam3_obj_id | INTEGER | Object ID within the shot |
| frame_sec | REAL | Timestamp of this detection |
| bbox_x1, bbox_y1, bbox_x2, bbox_y2 | REAL | Person bounding box in source pixels |
| head_cx | REAL | Head horizontal center (from mask silhouette top 10%), NULL if no mask |
| mask_rle | BLOB | Deprecated — always NULL. Masks are in sidecar video files now |

~200K rows at ~24fps. `head_cx` is precomputed during Stage 2 from the mask silhouette — no need to decode masks for crop positioning.

### `mask_videos` — FFV1 label-map video metadata
| Column | Type | Description |
|--------|------|-------------|
| shot_id | INTEGER PK | Which shot |
| video_path | TEXT | Filename relative to masks directory (e.g. `shot_0042.mkv`) |
| fps | REAL | Frame rate of the mask video (~24fps) |
| width | INTEGER | Frame width (1920) |
| height | INTEGER | Frame height (1080) |
| frame_count | INTEGER | Total frames in the video |

### `speaker_scores` — TalkNet ASD results
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| shot_id | INTEGER | |
| sam3_obj_id | INTEGER | |
| frame_sec | REAL | |
| asd_score | REAL | Speaking probability. Positive = likely speaking, negative = not. Range [-3.5, +3.0] |

### `face_detections` — InsightFace detections
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | |
| shot_id | INTEGER | |
| sam3_obj_id | INTEGER | Which person track this face belongs to |
| frame_sec | REAL | |
| face_bbox_x1..y2 | REAL | Face bounding box in source pixels |
| face_score | REAL | Detection confidence |
| embedding | BLOB | ArcFace embedding (can ignore) |

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

## Mask Video Format (FFV1 Label Maps)

Each shot has a sidecar `.mkv` file containing a **lossless grayscale video** where each pixel value is a person ID:

- **Codec**: FFV1 level 3, intra-only (`-g 1`), grayscale (`gray` pixel format)
- **Container**: Matroska (MKV)
- **Resolution**: 1920x1080 (same as source video)
- **FPS**: ~24fps (same as source video, stored in `mask_videos.fps`)
- **Pixel values**: `0` = background, `1..N` = person `sam3_obj_id + 1`
- **Compression**: ~8000:1 ratio for typical scenes (large background regions). 432MB total for a 2h movie.

### Decoding Mask Frames

For real-time playback, open the shot's MKV as a sequential video stream alongside the source video. Each frame is a 1920x1080 grayscale image:

```python
# Using ffmpeg pipe (simplest):
ffmpeg -i shot_0042.mkv -f rawvideo -pix_fmt gray pipe:1
# Each frame = 1920 * 1080 = 2,073,600 bytes

# Or using OpenCV:
cap = cv2.VideoCapture("shot_0042.mkv")
ret, frame = cap.read()  # frame is (1080, 1920, 1) or (1080, 1920)
# Pixel value 0 = background, N = person N
```

### Rendering Mask Overlays

For each non-zero pixel value in the label map:
1. Map `pixel_value` to `sam3_obj_id = pixel_value - 1`
2. Look up character via `track_identities` → `character_id`
3. Assign a palette color per character (consistent across shots)
4. Render as semi-transparent colored overlay on the source frame

The label map gives **pixel-perfect person silhouettes** — much better than bbox rectangles.

### Frame Synchronization

The mask video runs at the same fps as tracking (stored in `mask_videos.fps`). To sync with the source video:

```
source_frame_time = current_playback_time
mask_frame_index = int((source_frame_time - shot_start_sec) * mask_fps)
```

If the shot has no mask video (127 shots failed), fall back to bbox-only rendering.

## Scoring Algorithm (must replicate exactly)

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

1. For each frame, find person tracks at that timestamp from `person_tracks` (data is at ~24fps, so typically one row per person per frame; interpolate linearly if needed). Skip tracks whose bbox covers >85% of frame width AND height (background detections).
2. Pick highest-scoring person = main subject.
3. Use `head_cx` from `person_tracks` as crop center. Fall back to `(bbox_x1 + bbox_x2) / 2` if `head_cx` is NULL. Then `crop_x = center_x - crop_w / 2`, clamped to `[0, video_width - crop_w]`.
4. Smooth: `crop_x = 0.7 * crop_x + 0.3 * prev_crop_x`.
5. `crop_y = 0` (vertically centered, always 0 for 1080p with 9:16 crop).
6. `crop_w = 607, crop_h = 1080` (9:16 from 1920x1080).

## Overlay Rendering

On the original frame:

1. **Person mask overlays** — from the FFV1 label map. Each person gets a semi-transparent color fill based on their character_id. Alpha ~0.3 for non-selected persons, ~0.5 for the main selected person.
2. **Main person outline (no-ASD)** — thick green outline (3-4px) around the mask silhouette edge.
3. **Main person outline (with-ASD)** — thick blue outline (3-4px).
4. **Head position crosshairs** — small crosshair at `(head_cx, bbox_y1 + 0.1 * bbox_height)` for the main person.
5. **Green crop rectangle** — 607x1080 semi-transparent green border at the no-ASD crop_x position.
6. **Blue crop rectangle** — 607x1080 semi-transparent blue border at the ASD crop_x position.
7. **Overlap region** — where both crop rectangles overlap, blend to cyan.
8. **Dialogue indicator** — small bar/dot in corner showing if current segment has_dialogue=1.
9. **Person labels** — `sam3_obj_id` and character name (if known) near each person's head position.

## Bottom Previews

Two panels, each showing the cropped 607x1080 region scaled to fit:
- Left: "No ASD" crop (with mask overlays)
- Right: "With ASD" crop (with mask overlays)
- Each has a label overlay showing: character name, score, asd_score (right panel only)

## Playback Controls

- **Play/Pause** (spacebar)
- **Seek bar** with shot boundary markers (thin vertical lines)
- **Scene selector** dropdown — jump to scene by scene_id
- **Speed** — 0.25x, 0.5x, 1x, 2x
- **Frame step** — left/right arrow keys
- **Time display** — current time in MM:SS.ms format
- **Shot info** — show current shot_id, scene_id in a status bar
- **Mask toggle** — button to show/hide mask overlays (for performance)

## Data Loading Strategy

1. On startup: load `shots`, `scenes`, `characters`, `mask_videos` tables (all small).
2. Load `person_tracks` — 200K rows, ~50MB in memory. Index by `(shot_id, frame_sec)`.
3. Load `speaker_scores` indexed by `(shot_id, sam3_obj_id, frame_sec)` — 13K rows.
4. Load `face_detections` as a set of `(shot_id, sam3_obj_id)` for the `has_face` check.
5. Load `track_identities` for character name lookups.
6. **Source video**: open with platform video decoder, seek by timestamp.
7. **Mask videos**: open the current shot's MKV on shot transitions. Read frames sequentially during playback — no random access needed within a shot.

Person tracks at ~24fps mean the Framer typically has exact-frame data without interpolation. The DB data fits in RAM (~50MB for tracks + scores).

## Performance Notes

- Mask video decode is lightweight: FFV1 is fast, grayscale 1920x1080 = 2MB/frame
- At 24fps: ~48MB/s mask bandwidth — trivially handled by any modern SSD
- The mask overlay rendering (colorize + alpha blend) is a simple per-pixel operation
- If performance is tight, render masks at reduced resolution (e.g. 960x540) and upscale

## Tech Stack Suggestion

- **Python + OpenCV + Qt** — simplest, all deps available, OpenCV handles both video streams
- **Rust + egui + ffmpeg** — if performance matters for smooth 24fps with overlays

## Files

- Video: `Aliens.1080p.mkv` (audio stream 6 = English 5.1)
- DB: `Aliens.1080p.10cab98a.analysis.db` (162MB)
- Masks: `Aliens.1080p.10cab98a.masks/` (432MB, 1040 shot videos)

## Test Scene

Scene 7 (briefing room): `--start 1991 --end 2269` (33:11 to 37:49).
17 characters tracked, rapid dialogue cuts, lots of ASD-vs-no-ASD divergence (49% of frames pick a different person).
