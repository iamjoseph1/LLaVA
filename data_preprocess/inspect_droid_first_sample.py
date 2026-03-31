#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

try:
    import cv2
except ImportError:
    cv2 = None


DATASET_REPO = "lerobot/droid_1.0.1"
REPO_TYPE = "dataset"
LOCAL_DIR = Path("/home/dyros/LLaVA/data_preprocess/droid_1_0_1")

INFO_PATH = "meta/info.json"
PARQUET_PATH = "data/chunk-000/file-000.parquet"
VIDEO_KEYS = [
    "observation.images.wrist_left",
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
]


def download_file(path_in_repo: str) -> Path:
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type=REPO_TYPE,
        filename=path_in_repo,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def maybe_get_video_metadata(video_path: Path) -> dict:
    if cv2 is None:
        return {"status": "opencv-python not installed"}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"status": "failed to open with OpenCV"}

    metadata = {
        "status": "ok",
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return metadata


def print_section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Downloading")
    info_file = download_file(INFO_PATH)
    parquet_file = download_file(PARQUET_PATH)
    video_files = {
        video_key: download_file(f"videos/{video_key}/chunk-000/file-000.mp4")
        for video_key in VIDEO_KEYS
    }
    print(f"Dataset repo: {DATASET_REPO}")
    print(f"Downloaded metadata: {info_file}")
    print(f"Downloaded parquet:  {parquet_file}")
    for video_key, video_file in video_files.items():
        print(f"Downloaded video:   {video_key} -> {video_file}")

    print_section("Dataset Metadata")
    info = json.loads(info_file.read_text(encoding="utf-8"))
    print(f"robot_type: {info['robot_type']}")
    print(f"fps: {info['fps']}")
    print(f"chunks_size: {info['chunks_size']}")
    print(f"data_path template:  {info['data_path']}")
    print(f"video_path template: {info['video_path']}")

    print_section("Parquet Overview")
    df = pd.read_parquet(parquet_file)
    print(f"rows in parquet file: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print("column names:")
    for column in df.columns:
        print(f"  - {column}")

    first_row = df.iloc[0]
    first_episode_index = int(first_row["episode_index"])
    unique_episode_count = df["episode_index"].nunique()
    print(f"unique episode_index count in this parquet: {unique_episode_count}")
    print(f"first episode_index: {first_episode_index}")

    print_section("First Row")
    selected_fields = [
        "episode_index",
        "frame_index",
        "timestamp",
        "is_first",
        "is_last",
        "is_terminal",
        "is_episode_successful",
        "task_category",
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
        "building",
        "collector_id",
        "date",
    ]
    for field in selected_fields:
        print(f"{field}: {first_row.get(field)}")

    print_section("Episode Slice")
    episode_df = df[df["episode_index"] == first_episode_index].copy()
    preview_columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "is_first",
        "is_last",
        "reward",
        "task_category",
        "language_instruction",
    ]
    preview_df = episode_df[preview_columns].head(10)
    print(preview_df.to_string(index=False))

    episode_csv = LOCAL_DIR / "first_episode_preview.csv"
    preview_df.to_csv(episode_csv, index=False)
    print(f"\nSaved episode preview CSV: {episode_csv}")

    print_section("Video Mapping")
    fps = float(info["fps"])
    print("Each parquet row is one timestep in the episode.")
    print("The corresponding video frame is identified by frame_index in the matching MP4 file.")
    print("The matching video files for this parquet are:")
    for video_key, video_file in video_files.items():
        print(f"  - {video_key}: {video_file}")

    print("\nFirst 10 timestep/frame mappings:")
    for _, row in episode_df.head(10).iterrows():
        frame_index = int(row["frame_index"])
        timestamp = float(row["timestamp"])
        approx_seconds_from_frame = frame_index / fps
        print(
            f"frame_index={frame_index:4d} | "
            f"timestamp={timestamp:8.3f}s | "
            f"frame_index/fps={approx_seconds_from_frame:8.3f}s"
        )

    print_section("Video Metadata")
    for video_key, video_file in video_files.items():
        metadata = maybe_get_video_metadata(video_file)
        print(f"{video_key}: {metadata}")

    print_section("Interpretation")
    print("This first parquet file appears to describe one full episode.")
    print("Rows are ordered timesteps for that episode, and frame_index/timestamp align the row with the videos.")
    print("The three MP4 files are synchronized camera views of the same episode:")
    print("  - wrist camera")
    print("  - exterior_1 camera")
    print("  - exterior_2 camera")
    print("For any parquet row, use the same frame_index against each of those three videos to inspect the same moment.")


if __name__ == "__main__":
    main()
