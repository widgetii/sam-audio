"""CLI entry point for the unified video analysis pipeline.

Usage:
    # Full extraction
    python -m unified_pipeline.cli extract /path/to/video.mkv --audio-stream 6

    # Run specific stages
    python -m unified_pipeline.cli extract /path/to/video.mkv --stages 0,1,2

    # Consumer: auto-framing
    python -m unified_pipeline.cli auto-frame /path/to/video.mkv --start 120 --end 240 -o output.mp4

    # Consumer: dialogue summary
    python -m unified_pipeline.cli dialogue /path/to/video.mkv

    # Consumer: screen time
    python -m unified_pipeline.cli screen-time /path/to/video.mkv
"""

import argparse
import logging


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_extract(args):
    from unified_pipeline.pipeline import run_pipeline

    run_pipeline(
        video_path=args.video,
        audio_stream=args.audio_stream,
        device=args.device,
        stages=args.stages,
        db_file=args.db,
        store_masks=not args.no_masks,
        cluster_threshold=args.cluster_threshold,
        rms_threshold_db=args.rms_threshold,
        center_threshold_db=args.center_threshold,
        talknet_model=args.talknet_model,
        ddffnet_model=args.ddffnet_model,
    )


def cmd_auto_frame(args):
    from unified_pipeline.consumers.auto_framing import (
        compute_crop_positions,
        render_vertical_video,
    )
    from unified_pipeline.db import AnalysisDB, db_path_for_video
    from unified_pipeline.pipeline import get_video_fps

    db_file = args.db or str(db_path_for_video(args.video))
    db = AnalysisDB(db_file)

    # Get video dimensions
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            args.video,
        ],
        capture_output=True,
        text=True,
    )
    w, h = [int(x) for x in result.stdout.strip().split(",")]
    fps = get_video_fps(args.video)

    start = args.start or 0
    end = args.end or db.conn.execute("SELECT MAX(end_sec) FROM shots").fetchone()[0]

    crops = compute_crop_positions(db, start, end, w, h, fps)
    print(f"Computed {len(crops)} crop positions for {start:.1f}s - {end:.1f}s")

    if args.output:
        render_vertical_video(
            args.video,
            crops,
            args.output,
            start_sec=start,
            end_sec=end,
            audio_stream=args.audio_stream,
            target_fps=fps,
        )
        print(f"Rendered to {args.output}")

    db.close()


def cmd_dialogue(args):
    from unified_pipeline.consumers.dialogue import print_dialogue_summary
    from unified_pipeline.db import AnalysisDB, db_path_for_video

    db_file = args.db or str(db_path_for_video(args.video))
    db = AnalysisDB(db_file)
    print_dialogue_summary(db)
    db.close()


def cmd_screen_time(args):
    from unified_pipeline.consumers.screen_time import print_screen_time_summary
    from unified_pipeline.db import AnalysisDB, db_path_for_video

    db_file = args.db or str(db_path_for_video(args.video))
    db = AnalysisDB(db_file)
    print_screen_time_summary(db)
    db.close()


def cmd_stats(args):
    from unified_pipeline.db import AnalysisDB, db_path_for_video

    db_file = args.db or str(db_path_for_video(args.video))
    db = AnalysisDB(db_file)

    tables = [
        "shots",
        "scenes",
        "person_tracks",
        "track_identities",
        "characters",
        "face_detections",
        "speaker_scores",
        "blur_scores",
        "dialogue_segments",
        "character_audio",
    ]
    print(f"\nDatabase: {db_file}")
    for table in tables:
        try:
            count = db.count_rows(table)
            print(f"  {table}: {count}")
        except Exception:
            pass
    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Unified video analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser("extract", help="Run extraction pipeline")
    p_extract.add_argument("video", help="Path to video file")
    p_extract.add_argument(
        "--audio-stream", type=int, default=0, help="Audio stream index (default: 0)"
    )
    p_extract.add_argument("--device", default="cuda")
    p_extract.add_argument(
        "--stages",
        default="0,1,2,3,4,5",
        help="Comma-separated stage numbers (default: 0,1,2,3,4,5)",
    )
    p_extract.add_argument("--db", help="Override database path")
    p_extract.add_argument(
        "--no-masks", action="store_true", help="Skip storing RLE masks (saves space)"
    )
    p_extract.add_argument("--cluster-threshold", type=float, default=0.4)
    p_extract.add_argument("--rms-threshold", type=float, default=-50.0)
    p_extract.add_argument("--center-threshold", type=float, default=-40.0)
    p_extract.add_argument("--talknet-model", help="TalkNet model checkpoint path")
    p_extract.add_argument("--ddffnet-model", help="DDFFNet model checkpoint path")
    p_extract.set_defaults(func=cmd_extract)

    # auto-frame
    p_af = sub.add_parser("auto-frame", help="Generate vertical video crop positions")
    p_af.add_argument("video", help="Path to video file")
    p_af.add_argument("--db", help="Override database path")
    p_af.add_argument("--start", type=float, help="Start time in seconds")
    p_af.add_argument("--end", type=float, help="End time in seconds")
    p_af.add_argument("-o", "--output", help="Output video path")
    p_af.add_argument(
        "--audio-stream", type=int, default=0, help="Audio stream index (default: 0)"
    )
    p_af.set_defaults(func=cmd_auto_frame)

    # dialogue
    p_dl = sub.add_parser("dialogue", help="Show dialogue summary")
    p_dl.add_argument("video", help="Path to video file")
    p_dl.add_argument("--db", help="Override database path")
    p_dl.set_defaults(func=cmd_dialogue)

    # screen-time
    p_st = sub.add_parser("screen-time", help="Show screen time summary")
    p_st.add_argument("video", help="Path to video file")
    p_st.add_argument("--db", help="Override database path")
    p_st.set_defaults(func=cmd_screen_time)

    # stats
    p_stats = sub.add_parser("stats", help="Show database stats")
    p_stats.add_argument("video", help="Path to video file")
    p_stats.add_argument("--db", help="Override database path")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
