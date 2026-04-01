from pathlib import Path
import shutil
import pickle
import gzip
import numpy as np

from huggingface_hub import hf_hub_download, HfApi


# =========================
# USER SETTINGS
# =========================
REPO_ID = "RoboVerseOrg/roboverse_data"
REMOTE_TASK_DIR = "trajs/rlbench/insert_onto_square_peg/v2"
LOCAL_OUT_DIR = Path("roboverse/rlbench/insert_onto_square_peg")
MAX_FILES = 1

# If None → download up to MAX_FILES pkl files in that directory
# Otherwise → specify filenames manually
ONLY_FILES = None
# Example:
# ONLY_FILES = [
#     "trajectory-franka-0_v2.pkl",
#     "trajectory-franka-10_v2.pkl",
# ]

HF_TOKEN = None  # put your token if needed


# =========================
# DOWNLOAD FUNCTION
# =========================
def download_pkl_files():
    api = HfApi()

    print(f"[INFO] Listing files in repo: {REPO_ID}")
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    # Filter only target directory
    target_files = [
        f for f in all_files
        if f.startswith(REMOTE_TASK_DIR)
        and (f.endswith(".pkl") or f.endswith(".pkl.gz"))
    ]

    if ONLY_FILES is not None:
        target_files = [
            f for f in target_files
            if Path(f).name in ONLY_FILES
        ]
    else:
        target_files = target_files[:MAX_FILES]

    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    print(f"[INFO] Preparing to download {len(target_files)} file(s)")

    for remote_path in target_files:
        filename = Path(remote_path).name
        local_path = LOCAL_OUT_DIR / filename

        try:
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_path,
                token=HF_TOKEN,
            )

            shutil.copy2(cached_path, local_path)

            print(f"[OK] Downloaded: {filename}")
            downloaded_files.append(local_path)

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    return downloaded_files


# =========================
# PKL INSPECTION UTILITIES
# =========================
def print_structure(data, prefix="", depth=0, max_depth=3):
    """Recursively print structure of pickle file"""
    if depth > max_depth:
        return

    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{prefix}{k}: {type(v)}")
            print_structure(v, prefix + "  ", depth + 1)

    elif isinstance(data, list):
        print(f"{prefix}list(len={len(data)})")
        if len(data) > 0:
            print_structure(data[0], prefix + "  ", depth + 1)

    elif isinstance(data, np.ndarray):
        print(f"{prefix}ndarray shape={data.shape}, dtype={data.dtype}")


def detect_visual_data(data, path="root"):
    """Detect image/video-like arrays"""
    results = []

    if isinstance(data, dict):
        for k, v in data.items():
            results.extend(detect_visual_data(v, f"{path}.{k}"))

    elif isinstance(data, list):
        for i, v in enumerate(data[:3]):  # check first few
            results.extend(detect_visual_data(v, f"{path}[{i}]"))

    elif isinstance(data, np.ndarray):
        shape = data.shape

        # IMAGE: (H, W) or (H, W, C)
        if len(shape) == 2 or (len(shape) == 3 and shape[-1] in [1, 3, 4]):
            results.append((path, "IMAGE", shape))

        # VIDEO: (T, H, W, C)
        elif len(shape) == 4:
            results.append((path, "VIDEO", shape))

    return results


def inspect_pkl(file_path):
    print("\n" + "=" * 50)
    print(f"[INSPECT] {file_path}")
    print("=" * 50)

    if str(file_path).endswith(".gz"):
        with gzip.open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

    print("\n[STRUCTURE]")
    print_structure(data)

    print("\n[VISUAL DATA DETECTION]")
    visuals = detect_visual_data(data)

    if len(visuals) == 0:
        print("No image/video-like arrays found.")
    else:
        for path, vtype, shape in visuals:
            print(f"{vtype} found at {path} | shape={shape}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    downloaded_files = download_pkl_files()

    if len(downloaded_files) > 0:
        # Inspect first file
        inspect_pkl(downloaded_files[0])
    else:
        print("No files downloaded.")
