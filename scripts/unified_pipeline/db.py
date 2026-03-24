"""SQLite database schema and access layer for the unified video analysis pipeline.

Single file: <stem>.<sha256[:8]>.analysis.db
All pipeline stages write here; consumers read from here.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# RLE codec for binary masks
# ---------------------------------------------------------------------------


def mask_to_rle(mask: np.ndarray) -> bytes:
    """Encode a 2D binary mask as run-length encoded bytes.

    Format: height(u16) + width(u16) + (value(u8) + run_length(u32)) pairs.
    """
    h, w = mask.shape
    flat = mask.ravel().astype(np.uint8)
    parts = bytearray()
    parts += h.to_bytes(2, "little")
    parts += w.to_bytes(2, "little")

    if len(flat) == 0:
        return bytes(parts)

    current = flat[0]
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == current and count < 0xFFFFFFFF:
            count += 1
        else:
            parts.append(current)
            parts += count.to_bytes(4, "little")
            current = flat[i]
            count = 1
    parts.append(current)
    parts += count.to_bytes(4, "little")
    return bytes(parts)


def rle_to_mask(data: bytes) -> np.ndarray:
    """Decode RLE bytes back to a 2D binary mask."""
    h = int.from_bytes(data[0:2], "little")
    w = int.from_bytes(data[2:4], "little")
    flat = np.empty(h * w, dtype=np.uint8)
    pos = 4
    idx = 0
    while pos < len(data):
        val = data[pos]
        run = int.from_bytes(data[pos + 1 : pos + 5], "little")
        flat[idx : idx + run] = val
        idx += run
        pos += 5
    return flat.reshape(h, w)


# ---------------------------------------------------------------------------
# Embedding serialization
# ---------------------------------------------------------------------------


def embed_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def blob_to_embed(data: bytes, dim: int = 512) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Database path from video file
# ---------------------------------------------------------------------------


def db_path_for_video(video_path: str | Path) -> Path:
    """Return <stem>.<sha256[:8]>.analysis.db next to the video."""
    p = Path(video_path)
    sha = hashlib.sha256(p.name.encode()).hexdigest()[:8]
    return p.parent / f"{p.stem}.{sha}.analysis.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS shots (
    shot_id     INTEGER PRIMARY KEY,
    start_frame INTEGER NOT NULL,
    end_frame   INTEGER NOT NULL,
    start_sec   REAL NOT NULL,
    end_sec     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id    INTEGER PRIMARY KEY,
    start_sec   REAL NOT NULL,
    end_sec     REAL NOT NULL,
    shot_ids    TEXT NOT NULL  -- JSON array of shot_id
);

CREATE TABLE IF NOT EXISTS person_tracks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id     INTEGER NOT NULL,
    sam3_obj_id INTEGER NOT NULL,
    frame_sec   REAL NOT NULL,
    bbox_x1     REAL NOT NULL,
    bbox_y1     REAL NOT NULL,
    bbox_x2     REAL NOT NULL,
    bbox_y2     REAL NOT NULL,
    head_cx     REAL,
    mask_rle    BLOB,
    FOREIGN KEY (shot_id) REFERENCES shots(shot_id)
);
CREATE INDEX IF NOT EXISTS idx_pt_shot ON person_tracks(shot_id);
CREATE INDEX IF NOT EXISTS idx_pt_sec  ON person_tracks(frame_sec);

CREATE TABLE IF NOT EXISTS mask_videos (
    shot_id     INTEGER PRIMARY KEY,
    video_path  TEXT NOT NULL,
    fps         REAL NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    FOREIGN KEY (shot_id) REFERENCES shots(shot_id)
);

CREATE TABLE IF NOT EXISTS track_identities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id         INTEGER NOT NULL,
    sam3_obj_id     INTEGER NOT NULL,
    character_id    INTEGER,
    confidence      REAL DEFAULT 0.0,
    method          TEXT,  -- 'face', 'voice', 'manual'
    FOREIGN KEY (shot_id) REFERENCES shots(shot_id),
    FOREIGN KEY (character_id) REFERENCES characters(character_id)
);
CREATE INDEX IF NOT EXISTS idx_ti_shot ON track_identities(shot_id);

CREATE TABLE IF NOT EXISTS characters (
    character_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT,
    face_embedding      BLOB,   -- 512d float32
    voice_embedding     BLOB,   -- 256d float32
    face_image          BLOB    -- JPEG thumbnail
);

CREATE TABLE IF NOT EXISTS face_detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id         INTEGER NOT NULL,
    sam3_obj_id     INTEGER NOT NULL,
    frame_sec       REAL NOT NULL,
    face_bbox_x1    REAL NOT NULL,
    face_bbox_y1    REAL NOT NULL,
    face_bbox_x2    REAL NOT NULL,
    face_bbox_y2    REAL NOT NULL,
    face_score      REAL NOT NULL,
    embedding       BLOB,   -- 512d float32
    FOREIGN KEY (shot_id) REFERENCES shots(shot_id)
);

CREATE TABLE IF NOT EXISTS speaker_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id     INTEGER NOT NULL,
    sam3_obj_id INTEGER NOT NULL,
    frame_sec   REAL NOT NULL,
    asd_score   REAL NOT NULL,  -- TalkNet active speaker probability
    FOREIGN KEY (shot_id) REFERENCES shots(shot_id)
);
CREATE INDEX IF NOT EXISTS idx_ss_sec ON speaker_scores(frame_sec);

CREATE TABLE IF NOT EXISTS blur_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_id     INTEGER NOT NULL,
    sam3_obj_id INTEGER NOT NULL,
    frame_sec   REAL NOT NULL,
    blur_score  REAL NOT NULL,  -- DDFFNet sharpness (higher = sharper)
    FOREIGN KEY (shot_id) REFERENCES shots(shot_id)
);
CREATE INDEX IF NOT EXISTS idx_bs_sec ON blur_scores(frame_sec);

CREATE TABLE IF NOT EXISTS dialogue_segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    start_sec       REAL NOT NULL,
    end_sec         REAL NOT NULL,
    has_dialogue    INTEGER NOT NULL DEFAULT 0,
    center_db       REAL,           -- center channel energy dB
    sam_audio_db    REAL,           -- SAM-Audio speech energy dB
    character_id    INTEGER,
    FOREIGN KEY (character_id) REFERENCES characters(character_id)
);

CREATE TABLE IF NOT EXISTS character_audio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id    INTEGER NOT NULL,
    start_sec       REAL NOT NULL,
    end_sec         REAL NOT NULL,
    audio_path      TEXT,       -- path to separated audio file
    method          TEXT,       -- 'visual', 'voice', 'face_direct'
    FOREIGN KEY (character_id) REFERENCES characters(character_id)
);

CREATE TABLE IF NOT EXISTS pipeline_progress (
    stage       TEXT NOT NULL,
    item_key    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending, done, error
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    error_msg   TEXT,
    PRIMARY KEY (stage, item_key)
);
"""


# ---------------------------------------------------------------------------
# Database connection wrapper
# ---------------------------------------------------------------------------


class AnalysisDB:
    """Read/write access to the unified analysis SQLite database."""

    def __init__(self, db_file: str | Path):
        self.db_file = Path(db_file)
        self.conn = sqlite3.connect(str(self.db_file), timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        self.conn.close()

    @contextmanager
    def transaction(self):
        """Context manager for explicit transactions."""
        self.conn.execute("BEGIN")
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- Shots --

    def insert_shots(self, shots: list[dict]):
        """Insert shot boundaries. Each dict: shot_id, start_frame, end_frame, start_sec, end_sec."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO shots (shot_id, start_frame, end_frame, start_sec, end_sec) "
            "VALUES (:shot_id, :start_frame, :end_frame, :start_sec, :end_sec)",
            shots,
        )
        self.conn.commit()

    def get_shots(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM shots ORDER BY start_sec").fetchall()
        return [dict(r) for r in rows]

    # -- Scenes --

    def insert_scenes(self, scenes: list[dict]):
        """Each dict: scene_id, start_sec, end_sec, shot_ids (list of int)."""
        rows = []
        for s in scenes:
            rows.append(
                {
                    "scene_id": s["scene_id"],
                    "start_sec": s["start_sec"],
                    "end_sec": s["end_sec"],
                    "shot_ids": json.dumps(s["shot_ids"]),
                }
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO scenes (scene_id, start_sec, end_sec, shot_ids) "
            "VALUES (:scene_id, :start_sec, :end_sec, :shot_ids)",
            rows,
        )
        self.conn.commit()

    def get_scenes(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM scenes ORDER BY start_sec").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["shot_ids"] = json.loads(d["shot_ids"])
            result.append(d)
        return result

    # -- Person tracks --

    def insert_person_tracks(self, tracks: list[dict]):
        """Batch insert person track rows.

        Each dict: shot_id, sam3_obj_id, frame_sec, bbox_x1..y2,
        head_cx (optional float), mask_rle (optional bytes).
        """
        self.conn.executemany(
            "INSERT INTO person_tracks "
            "(shot_id, sam3_obj_id, frame_sec, bbox_x1, bbox_y1, bbox_x2, bbox_y2, head_cx, mask_rle) "
            "VALUES (:shot_id, :sam3_obj_id, :frame_sec, :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2, :head_cx, :mask_rle)",
            tracks,
        )
        self.conn.commit()

    def insert_mask_video(self, row: dict):
        """Insert a mask video metadata row."""
        self.conn.execute(
            "INSERT OR REPLACE INTO mask_videos "
            "(shot_id, video_path, fps, width, height, frame_count) "
            "VALUES (:shot_id, :video_path, :fps, :width, :height, :frame_count)",
            row,
        )
        self.conn.commit()

    def get_person_tracks(
        self, shot_id: int | None = None, time_range: tuple[float, float] | None = None
    ) -> list[dict]:
        query = "SELECT * FROM person_tracks WHERE 1=1"
        params: list = []
        if shot_id is not None:
            query += " AND shot_id = ?"
            params.append(shot_id)
        if time_range is not None:
            query += " AND frame_sec >= ? AND frame_sec <= ?"
            params.extend(time_range)
        query += " ORDER BY frame_sec, sam3_obj_id"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def get_person_tracks_at_sec(
        self, sec: float, tolerance: float = 0.5
    ) -> list[dict]:
        """Get all person tracks near a given timestamp."""
        rows = self.conn.execute(
            "SELECT * FROM person_tracks WHERE frame_sec >= ? AND frame_sec <= ? "
            "ORDER BY sam3_obj_id",
            (sec - tolerance, sec + tolerance),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Track identities --

    def insert_track_identity(
        self,
        shot_id: int,
        sam3_obj_id: int,
        character_id: int,
        confidence: float = 0.0,
        method: str = "face",
    ):
        self.conn.execute(
            "INSERT OR REPLACE INTO track_identities "
            "(shot_id, sam3_obj_id, character_id, confidence, method) "
            "VALUES (?, ?, ?, ?, ?)",
            (shot_id, sam3_obj_id, character_id, confidence, method),
        )
        self.conn.commit()

    def get_track_identities(self, shot_id: int | None = None) -> list[dict]:
        query = "SELECT * FROM track_identities"
        params: list = []
        if shot_id is not None:
            query += " WHERE shot_id = ?"
            params.append(shot_id)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # -- Characters --

    def insert_character(
        self,
        name: str | None = None,
        face_embedding: np.ndarray | None = None,
        voice_embedding: np.ndarray | None = None,
        face_image: bytes | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO characters (name, face_embedding, voice_embedding, face_image) "
            "VALUES (?, ?, ?, ?)",
            (
                name,
                embed_to_blob(face_embedding) if face_embedding is not None else None,
                embed_to_blob(voice_embedding) if voice_embedding is not None else None,
                face_image,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_character(self, character_id: int, **kwargs):
        """Update character fields. Supports: name, face_embedding, voice_embedding, face_image."""
        for key, val in kwargs.items():
            if key in ("face_embedding", "voice_embedding") and val is not None:
                val = embed_to_blob(val)
            self.conn.execute(
                f"UPDATE characters SET {key} = ? WHERE character_id = ?",
                (val, character_id),
            )
        self.conn.commit()

    def get_characters(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM characters").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d["face_embedding"]:
                d["face_embedding"] = blob_to_embed(d["face_embedding"])
            if d["voice_embedding"]:
                d["voice_embedding"] = blob_to_embed(d["voice_embedding"])
            result.append(d)
        return result

    # -- Face detections --

    def insert_face_detections(self, detections: list[dict]):
        self.conn.executemany(
            "INSERT INTO face_detections "
            "(shot_id, sam3_obj_id, frame_sec, face_bbox_x1, face_bbox_y1, "
            "face_bbox_x2, face_bbox_y2, face_score, embedding) "
            "VALUES (:shot_id, :sam3_obj_id, :frame_sec, :face_bbox_x1, :face_bbox_y1, "
            ":face_bbox_x2, :face_bbox_y2, :face_score, :embedding)",
            detections,
        )
        self.conn.commit()

    def get_face_detections(self, shot_id: int | None = None) -> list[dict]:
        query = "SELECT * FROM face_detections"
        params: list = []
        if shot_id is not None:
            query += " WHERE shot_id = ?"
            params.append(shot_id)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # -- Speaker scores --

    def insert_speaker_scores(self, scores: list[dict]):
        self.conn.executemany(
            "INSERT INTO speaker_scores (shot_id, sam3_obj_id, frame_sec, asd_score) "
            "VALUES (:shot_id, :sam3_obj_id, :frame_sec, :asd_score)",
            scores,
        )
        self.conn.commit()

    def get_speaker_scores(
        self, time_range: tuple[float, float] | None = None
    ) -> list[dict]:
        query = "SELECT * FROM speaker_scores"
        params: list = []
        if time_range:
            query += " WHERE frame_sec >= ? AND frame_sec <= ?"
            params.extend(time_range)
        query += " ORDER BY frame_sec"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # -- Blur scores --

    def insert_blur_scores(self, scores: list[dict]):
        self.conn.executemany(
            "INSERT INTO blur_scores (shot_id, sam3_obj_id, frame_sec, blur_score) "
            "VALUES (:shot_id, :sam3_obj_id, :frame_sec, :blur_score)",
            scores,
        )
        self.conn.commit()

    def get_blur_scores(
        self, time_range: tuple[float, float] | None = None
    ) -> list[dict]:
        query = "SELECT * FROM blur_scores"
        params: list = []
        if time_range:
            query += " WHERE frame_sec >= ? AND frame_sec <= ?"
            params.extend(time_range)
        query += " ORDER BY frame_sec"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # -- Dialogue segments --

    def insert_dialogue_segments(self, segments: list[dict]):
        self.conn.executemany(
            "INSERT INTO dialogue_segments "
            "(start_sec, end_sec, has_dialogue, center_db, sam_audio_db, character_id) "
            "VALUES (:start_sec, :end_sec, :has_dialogue, :center_db, :sam_audio_db, :character_id)",
            segments,
        )
        self.conn.commit()

    def get_dialogue_segments(self, dialogue_only: bool = False) -> list[dict]:
        query = "SELECT * FROM dialogue_segments"
        if dialogue_only:
            query += " WHERE has_dialogue = 1"
        query += " ORDER BY start_sec"
        return [dict(r) for r in self.conn.execute(query).fetchall()]

    # -- Character audio --

    def insert_character_audio(self, rows: list[dict]):
        self.conn.executemany(
            "INSERT INTO character_audio (character_id, start_sec, end_sec, audio_path, method) "
            "VALUES (:character_id, :start_sec, :end_sec, :audio_path, :method)",
            rows,
        )
        self.conn.commit()

    def get_character_audio(self, character_id: int | None = None) -> list[dict]:
        query = "SELECT * FROM character_audio"
        params: list = []
        if character_id is not None:
            query += " WHERE character_id = ?"
            params.append(character_id)
        query += " ORDER BY start_sec"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # -- Pipeline progress --

    def mark_progress(
        self,
        stage: str,
        item_key: str,
        status: str = "done",
        error_msg: str | None = None,
    ):
        self.conn.execute(
            "INSERT OR REPLACE INTO pipeline_progress (stage, item_key, status, error_msg) "
            "VALUES (?, ?, ?, ?)",
            (stage, item_key, status, error_msg),
        )
        self.conn.commit()

    def get_progress(self, stage: str) -> dict[str, str]:
        """Return {item_key: status} for a stage."""
        rows = self.conn.execute(
            "SELECT item_key, status FROM pipeline_progress WHERE stage = ?",
            (stage,),
        ).fetchall()
        return {r["item_key"]: r["status"] for r in rows}

    def is_done(self, stage: str, item_key: str) -> bool:
        row = self.conn.execute(
            "SELECT status FROM pipeline_progress WHERE stage = ? AND item_key = ?",
            (stage, item_key),
        ).fetchone()
        return row is not None and row["status"] == "done"

    # -- Utility queries --

    def get_tracks_with_identity(
        self, time_range: tuple[float, float] | None = None
    ) -> list[dict]:
        """Join person_tracks with track_identities and characters."""
        query = """
            SELECT pt.*, ti.character_id, ti.confidence, ti.method as id_method,
                   c.name as character_name
            FROM person_tracks pt
            LEFT JOIN track_identities ti ON pt.shot_id = ti.shot_id AND pt.sam3_obj_id = ti.sam3_obj_id
            LEFT JOIN characters c ON ti.character_id = c.character_id
        """
        params: list = []
        if time_range:
            query += " WHERE pt.frame_sec >= ? AND pt.frame_sec <= ?"
            params.extend(time_range)
        query += " ORDER BY pt.frame_sec, pt.sam3_obj_id"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def count_rows(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
