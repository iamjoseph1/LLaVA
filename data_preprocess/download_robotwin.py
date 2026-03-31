from pathlib import Path
import shutil
import zipfile
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# =========================
# User settings
# =========================
REPO_ID = "TianxingChen/RoboTwin2.0"
TASK_NAMES = [
    "beat_block_hammer",
    "click_bell",
    "grab_roller", # 양팔작업
    "pick_diverse_bottles", # 없음
    "place_can_basket",
    "press_stapler",
    "stamp_seal",
    "turn_switch", # 유일하게 [0,1,0,0,0,0]인 작업
]
OUT_DIR = "./robotwin"

# If you need authentication, uncomment and set your token:
# HF_TOKEN = "hf_xxx"
HF_TOKEN = None


def unzip_downloaded_zip(zip_path: str | Path, extract_dir: str | Path | None = None):
    """
    Extract a downloaded zip file.
    If extract_dir is not provided, extract into a directory named after the zip stem.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir) if extract_dir is not None else zip_path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)

    print(f"[UNZIP] {zip_path} -> {extract_dir}")
    return extract_dir


def download_franka_clean_50(task_name: str, out_dir: str | Path, token: str | None = None):
    """
    Download:
        dataset/{task_name}/franka_clean_50.zip
    from Hugging Face dataset repo:
        TianxingChen/RoboTwin2.0

    and save it as:
        {task_name}_franka_clean_50.zip
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_path = f"dataset/{task_name}/franka_clean_50.zip"
    local_name = f"{task_name}_franka_clean_50.zip"
    final_path = out_dir / local_name

    try:
        # Downloads a single file from a dataset repo into the local HF cache
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_path,
            token=token,
        )

        # Copy from HF cache to your desired filename
        shutil.copy2(cached_path, final_path)
        unzip_downloaded_zip(final_path)

        print(f"[OK] {task_name} -> {final_path}")

    except EntryNotFoundError:
        print(f"[SKIP] File not found for task: {task_name}")
        print(f"       Expected remote file: {remote_path}")
    except Exception as e:
        print(f"[ERROR] Failed for task: {task_name}")
        print(f"        {type(e).__name__}: {e}")


if __name__ == "__main__":
    for task in TASK_NAMES:
        download_franka_clean_50(task, OUT_DIR, token=HF_TOKEN)
