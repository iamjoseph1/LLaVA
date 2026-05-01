#!/usr/bin/env python3

from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
TARGET_DIR = CURRENT_DIR / "llava" / "from_ours" / "threepiece-z"
TASK_NAME = "threepiece"
AXIS_NAME = "z"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg"}


def collect_image_paths(target_dir: Path) -> list[Path]:
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    image_paths = sorted(
        [
            path
            for path in target_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in {target_dir} "
            f"with suffixes: {sorted(SUPPORTED_SUFFIXES)}"
        )
    return image_paths


def rename_images(image_paths: list[Path]) -> None:
    temporary_paths: list[Path] = []

    # Rename to temporary names first so sequential target names never clash.
    for idx, image_path in enumerate(image_paths):
        temp_path = image_path.with_name(f"__tmp_rename_images_{idx}{image_path.suffix}")
        image_path.rename(temp_path)
        temporary_paths.append(temp_path)

    for idx, temp_path in enumerate(temporary_paths):
        new_name = f"{TASK_NAME}_ours_{AXIS_NAME}_{idx}.jpg"
        new_path = temp_path.with_name(new_name)
        temp_path.rename(new_path)
        print(f"{temp_path.name} -> {new_path.name}")


def main() -> None:
    image_paths = collect_image_paths(TARGET_DIR)
    rename_images(image_paths)
    print(f"Renamed {len(image_paths)} images in: {TARGET_DIR}")


if __name__ == "__main__":
    main()
