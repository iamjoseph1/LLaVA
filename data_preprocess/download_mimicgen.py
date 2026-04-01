from pathlib import Path
import json
import shutil

import h5py
import numpy as np
import cv2
from huggingface_hub import hf_hub_download, HfApi


# =========================
# User settings
# =========================
REPO_ID = "amandlek/mimicgen_datasets"
REMOTE_DIR = "robot"
TASK_NAME = "square_d1_panda"
OUT_DIR = Path(f"./mimicgen/robot/{TASK_NAME}")
VIDEO_OUT_DIR = OUT_DIR / "video"
STRUCTURE_SUFFIX = "_demo_structure.json"
DEFAULT_VIDEO_FPS = 20.0
VIDEO_CODEC_CANDIDATES = ("mp4v", "avc1", "H264")
MAX_DEMO_VIDEOS = 500

# Put only the files you want here
TARGET_FILES = [
    # "coffee_d0.hdf5",
    # "square_d0.hdf5",
    f"{TASK_NAME}.hdf5",
    # "pick_place_d0.hdf5",
]

# If private / gated access is needed, set your token here
HF_TOKEN = None
# HF_TOKEN = "hf_xxx"


def list_target_hdf5_files(repo_id: str, remote_dir: str):
    """
    List all .hdf5 files under the target directory in the HF dataset repo.
    """
    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [
        f for f in all_files
        if f.startswith(f"{remote_dir}/") and f.endswith(".hdf5")
    ]


def download_hdf5_file(filename: str, out_dir: str | Path, token: str | None = None):
    """
    Download one HDF5 file from:
        core/{filename}
    in dataset repo:
        amandlek/mimicgen_datasets

    and save it as:
        mimicgen/{remote_dir}/{filename}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_path = f"{REMOTE_DIR}/{filename}"
    final_path = out_dir / filename

    if final_path.exists():
        print(f"[SKIP] Already exists locally: {final_path}")
        return final_path

    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_path,
            token=token,
        )

        shutil.copy2(cached_path, final_path)
        print(f"[OK] Downloaded: {remote_path} -> {final_path}")
        return final_path

    except Exception as e:
        print(f"[ERROR] Failed to download {remote_path}")
        print(f"        {type(e).__name__}: {e}")
        return None


def build_demo_structure_summary(hdf5_path: str | Path) -> dict:
    """
    Build a serializable summary for a single demo template inside the HDF5 file.
    """
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        data_group = f.get("data")
        if data_group is None or not data_group.keys():
            raise ValueError(f"No demos found under 'data' in {hdf5_path}")

        demo_name = sorted(data_group.keys())[0]
        demo_group = data_group[demo_name]
        summary = {
            "file": str(hdf5_path.resolve()),
            "demo_template": demo_name,
            "groups": [],
            "datasets": [],
        }

        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                summary["groups"].append(name)
            elif isinstance(obj, h5py.Dataset):
                summary["datasets"].append(
                    {
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )

        demo_group.visititems(visitor)

    return summary


def save_structure_summary(hdf5_path: str | Path) -> Path:
    """
    Save the single-demo structure summary next to the downloaded file.
    """
    hdf5_path = Path(hdf5_path)
    summary_path = hdf5_path.with_name(f"{hdf5_path.stem}{STRUCTURE_SUFFIX}")
    summary = build_demo_structure_summary(hdf5_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] Saved structure summary: {summary_path}")
    return summary_path


def normalize_video_frames(frames: np.ndarray) -> np.ndarray:
    """
    Normalize image frames into uint8 HWC RGB format for video writing.
    """
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {arr.shape}")
    if arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-last frames, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr


def write_demo_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    """
    Write RGB frames to an MP4 file.
    """
    frames = normalize_video_frames(frames)
    height, width = frames.shape[1:3]

    for codec in VIDEO_CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            continue

        try:
            for frame in frames:
                bgr_frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
        finally:
            writer.release()

        if verify_video_file(output_path):
            print(f"[OK] Wrote playable video with codec {codec}: {output_path}")
            return

        print(f"[WARN] Codec {codec} produced an unreadable video: {output_path}")
        output_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Failed to write a playable MP4 for {output_path}. "
        f"Tried codecs: {', '.join(VIDEO_CODEC_CANDIDATES)}"
    )


def verify_video_file(video_path: Path) -> bool:
    """
    Verify that the written video can be opened and decoded.
    """
    if not video_path.exists() or video_path.stat().st_size == 0:
        return False

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return False

        ok, _frame = capture.read()
        return ok
    finally:
        capture.release()


def export_agentview_videos(hdf5_path: str | Path, video_out_dir: str | Path) -> list[Path]:
    """
    Export the first MAX_DEMO_VIDEOS demo videos from data/<demo>/obs/agentview_image.
    """
    hdf5_path = Path(hdf5_path)
    video_out_dir = Path(video_out_dir)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    with h5py.File(hdf5_path, "r") as f:
        data_group = f.get("data")
        if data_group is None:
            print(f"[WARN] No 'data' group found in {hdf5_path}")
            return exported_paths

        demo_names = sorted(data_group.keys())
        if not demo_names:
            print(f"[WARN] No demos found in {hdf5_path}")
            return exported_paths

        for demo_name in demo_names[:MAX_DEMO_VIDEOS]:
            dataset_path = f"data/{demo_name}/obs/agentview_image"
            if dataset_path not in f:
                print(f"[SKIP] Missing dataset: {dataset_path}")
                continue

            frames = f[dataset_path][()]
            output_path = video_out_dir / f"{demo_name}.mp4"
            write_demo_video(frames, output_path, fps=DEFAULT_VIDEO_FPS)
            exported_paths.append(output_path)
            print(f"[OK] Exported video: {dataset_path} -> {output_path}")

    return exported_paths


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: check which HDF5 files exist in remote core directory
    print(f"[INFO] Listing remote HDF5 files in {REPO_ID}/{REMOTE_DIR} ...")
    remote_hdf5_files = list_target_hdf5_files(REPO_ID, REMOTE_DIR)
    remote_names = {Path(p).name for p in remote_hdf5_files}

    print(f"[INFO] Found {len(remote_hdf5_files)} HDF5 files in remote '{REMOTE_DIR}' directory.")

    missing = [name for name in TARGET_FILES if name not in remote_names]
    if missing:
        print("[WARNING] These requested files were not found remotely:")
        for name in missing:
            print(f"  - {name}")

    downloaded_files = []
    for filename in TARGET_FILES:
        if filename not in remote_names:
            print(f"[SKIP] {filename} does not exist in remote repo.")
            continue

        local_path = download_hdf5_file(filename, OUT_DIR, token=HF_TOKEN)
        if local_path is not None:
            downloaded_files.append(local_path)

    if not downloaded_files:
        print("\n[INFO] No files downloaded.")
        return

    for file_path in downloaded_files:
        save_structure_summary(file_path)
        export_agentview_videos(file_path, VIDEO_OUT_DIR)


if __name__ == "__main__":
    main()
