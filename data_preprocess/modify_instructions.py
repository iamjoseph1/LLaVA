#!/usr/bin/env python3

from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
TARGET_DIR = CURRENT_DIR / "llava" / "from_ours" / "threepiece-z"
INSTRUCTION_FILE_NAME = "instruction.txt"
LINE_TO_ADD = "Given that [1, 0, 0], [0, 1, 0], and [0, 0, 1] denote the x-, y-, and z-axes respectively, which axis of the end-effector’s local coordinate frame should its motion be constrained to in order to successfully complete the following task : "


def get_instruction_path(target_dir: Path) -> Path:
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    instruction_path = target_dir / INSTRUCTION_FILE_NAME
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction file does not exist: {instruction_path}")
    if not instruction_path.is_file():
        raise FileExistsError(f"Instruction path is not a file: {instruction_path}")
    return instruction_path


def modify_lines(lines: list[str]) -> list[str]:
    updated_lines: list[str] = []
    for line in lines:
        stripped_line = line.rstrip("\n")
        if not stripped_line.strip():
            updated_lines.append(line)
            continue

        updated_lines.append(f"{LINE_TO_ADD}{stripped_line}\n")
    return updated_lines


def main() -> None:
    instruction_path = get_instruction_path(TARGET_DIR)
    original_lines = instruction_path.read_text().splitlines(keepends=True)
    updated_lines = modify_lines(original_lines)
    instruction_path.write_text("".join(updated_lines))
    print(f"Updated instruction file: {instruction_path}")
    print(f"Modified {len(updated_lines)} lines.")


if __name__ == "__main__":
    main()
