#!/usr/bin/env python3

import json
import re
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
FROM_OURS_ROOT = CURRENT_DIR / "llava" / "from_ours"
OUTPUT_PATH = CURRENT_DIR / "llava" / "finetune_dataset_from_ours.json"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg"}
HUMAN_PREFIX = "<image>\n"
X_AXIS_LABEL = "[1,0,0]"
Z_AXIS_LABEL = "[0,0,1]"
EXPECTED_FOLDER_COUNT = 8
EXPECTED_PAIRS_PER_FOLDER = 100


def extract_index(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Could not extract trailing index from image name: {path.name}")
    return int(match.group(1))


def collect_task_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    task_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    if len(task_dirs) != EXPECTED_FOLDER_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FOLDER_COUNT} folders in {root_dir}, found {len(task_dirs)}"
        )
    return task_dirs


def collect_image_paths(task_dir: Path) -> list[Path]:
    image_paths = sorted(
        [
            path
            for path in task_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ],
        key=extract_index,
    )
    if len(image_paths) != EXPECTED_PAIRS_PER_FOLDER:
        raise ValueError(
            f"Expected {EXPECTED_PAIRS_PER_FOLDER} images in {task_dir}, "
            f"found {len(image_paths)}"
        )
    return image_paths


def load_instruction_lines(task_dir: Path) -> list[str]:
    instruction_path = task_dir / "instruction.txt"
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction file does not exist: {instruction_path}")

    lines = instruction_path.read_text().splitlines()
    if len(lines) != EXPECTED_PAIRS_PER_FOLDER:
        raise ValueError(
            f"Expected {EXPECTED_PAIRS_PER_FOLDER} instruction lines in {instruction_path}, "
            f"found {len(lines)}"
        )
    if any(not line.strip() for line in lines):
        raise ValueError(f"Found empty instruction line in {instruction_path}")
    return lines


def get_gpt_value(image_name: str) -> str:
    lower_name = image_name.lower()
    if "_x_" in lower_name:
        return X_AXIS_LABEL
    if "_z_" in lower_name:
        return Z_AXIS_LABEL
    raise ValueError(f"Could not infer axis label from image name: {image_name}")


def build_record(task_dir: Path, image_path: Path, instruction: str) -> dict:
    image_name = image_path.name
    return {
        "id": image_path.stem,
        "image": f"from_ours/{task_dir.name}/{image_name}",
        "conversations": [
            {
                "from": "human",
                "value": f"{HUMAN_PREFIX}{instruction}",
            },
            {
                "from": "gpt",
                "value": get_gpt_value(image_name),
            },
        ],
    }


def generate_dataset() -> list[dict]:
    dataset: list[dict] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()

    for task_dir in collect_task_dirs(FROM_OURS_ROOT):
        image_paths = collect_image_paths(task_dir)
        instruction_lines = load_instruction_lines(task_dir)

        for image_path, instruction in zip(image_paths, instruction_lines, strict=True):
            record = build_record(task_dir, image_path, instruction)
            record_id = record["id"]
            image_ref = record["image"]

            if record_id in seen_ids:
                raise ValueError(f"Duplicate id detected: {record_id}")
            if image_ref in seen_images:
                raise ValueError(f"Duplicate image detected: {image_ref}")

            seen_ids.add(record_id)
            seen_images.add(image_ref)
            dataset.append(record)

    expected_total = EXPECTED_FOLDER_COUNT * EXPECTED_PAIRS_PER_FOLDER
    if len(dataset) != expected_total:
        raise ValueError(f"Expected {expected_total} records, generated {len(dataset)}")
    return dataset


def main() -> None:
    dataset = generate_dataset()
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"Saved {len(dataset)} records to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
