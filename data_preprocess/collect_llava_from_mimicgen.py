import json
import re
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "opencv-python is required. Install it with `pip install opencv-python`."
    ) from exc


CURRENT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = CURRENT_DIR / "mimicgen" / "robot"
TASK_NAME = "square_d1_panda"
START_VIDEO_INDEX = 0
NUM_VIDEOS = 100
NEXT_VIDEO_DELAY_SECONDS = 1.0
PLAYBACK_SPEED = 1.0

VIDEO_DIR = DATASET_ROOT / TASK_NAME / "video"
INSTRUCTION_PATH = DATASET_ROOT / TASK_NAME / "instruction" / "instruction.txt"

LLAVA_ROOT = CURRENT_DIR / "llava"
TASK_OUTPUT_DIR = LLAVA_ROOT / TASK_NAME
ANNOTATIONS_PATH = TASK_OUTPUT_DIR / f"{TASK_NAME}_annotations.json"

WINDOW_NAME = "LLaVA Frame Collector - MimicGen"
GPT_PLACEHOLDER = "[0,0,1,0,0,0]"  # [x,y,z,wx,wy,wz]


def extract_episode_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not extract episode number from {path.name}")
    return int(match.group(1))


def load_existing_annotations() -> list[dict]:
    if not ANNOTATIONS_PATH.exists():
        return []

    data = json.loads(ANNOTATIONS_PATH.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {ANNOTATIONS_PATH}")
    return data


def load_instruction_prompts() -> list[str]:
    if not INSTRUCTION_PATH.exists():
        raise FileNotFoundError(f"Instruction file not found: {INSTRUCTION_PATH}")

    prompts: list[str] = []
    for raw_line in INSTRUCTION_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
        if cleaned:
            prompts.append(cleaned)

    if len(prompts) < NUM_VIDEOS:
        raise ValueError(
            f"Instruction file has only {len(prompts)} prompts, "
            f"but NUM_VIDEOS={NUM_VIDEOS}"
        )
    return prompts[:NUM_VIDEOS]


def save_annotations(annotations: list[dict]) -> None:
    ANNOTATIONS_PATH.write_text(json.dumps(annotations, indent=2))


def build_annotation(image_name: str, video_label: str, human_prompt: str) -> dict:
    return {
        "id": f"{TASK_NAME}_{video_label}",
        "image": image_name,
        "conversations": [
            {
                "from": "human",
                "value": human_prompt,
            },
            {
                "from": "gpt",
                "value": GPT_PLACEHOLDER,
            },
        ],
    }


def collect_video_paths() -> list[Path]:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    video_paths = sorted(
        VIDEO_DIR.glob("*.mp4"),
        key=extract_episode_number,
    )
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in {VIDEO_DIR}")

    selected = video_paths[START_VIDEO_INDEX:]
    if not selected:
        raise ValueError(
            f"No videos available from START_VIDEO_INDEX={START_VIDEO_INDEX}"
        )
    return selected


def save_capture(
    frame,
    video_label: str,
    human_prompt: str,
    annotations: list[dict],
) -> None:
    image_name = f"{TASK_NAME}_image_{video_label}.jpg"
    image_path = TASK_OUTPUT_DIR / image_name
    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write image: {image_path}")

    new_annotation = build_annotation(image_name, video_label, human_prompt)
    existing_index = next(
        (
            index
            for index, annotation in enumerate(annotations)
            if annotation.get("image") == image_name
        ),
        None,
    )
    if existing_index is None:
        annotations.append(new_annotation)
        action = "appended"
    else:
        annotations[existing_index] = new_annotation
        action = "updated"

    save_annotations(annotations)
    print(f"[OK] Saved {image_path.name} & {action} {ANNOTATIONS_PATH.name}")


def compute_frame_delay_ms(capture: cv2.VideoCapture) -> int:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    if PLAYBACK_SPEED <= 0:
        raise ValueError("PLAYBACK_SPEED must be greater than 0.")
    return max(1, int((1000.0 / fps) / PLAYBACK_SPEED))


def play_and_maybe_capture(
    video_path: Path,
    human_prompt: str,
    annotations: list[dict],
) -> str:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return "error"

    video_label = video_path.stem
    current_frame = None
    frame_index = -1
    frame_delay_ms = compute_frame_delay_ms(capture)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        current_frame = frame.copy()
        frame_index += 1

        display_frame = cv2.resize(
            frame,
            dsize=None,
            fx=6.0,
            fy=6.0,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.putText(
            display_frame,
            (
                f"{video_label} | frame {frame_index} | speed {PLAYBACK_SPEED:.2f}x | "
                "SPACE=capture, ENTER=skip, q=quit"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key == ord(" "):
            save_capture(current_frame, video_label, human_prompt, annotations)
            capture.release()
            cv2.waitKey(int(NEXT_VIDEO_DELAY_SECONDS * 1000))
            return "captured"
        if key in (10, 13):
            capture.release()
            print(f"[INFO] Skipped video: {video_label}")
            cv2.waitKey(int(NEXT_VIDEO_DELAY_SECONDS * 1000))
            return "skipped"
        if key == ord("q"):
            capture.release()
            return "quit"

    capture.release()
    print(f"[INFO] Video ended without capture, skipping: {video_label}")
    cv2.waitKey(int(NEXT_VIDEO_DELAY_SECONDS * 1000))
    return "skipped"


def main() -> None:
    TASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotations = load_existing_annotations()
    instruction_prompts = load_instruction_prompts()
    selected_videos = collect_video_paths()

    print(f"[INFO] Task: {TASK_NAME}")
    print(f"[INFO] Video dir: {VIDEO_DIR}")
    print(f"[INFO] Start index: {START_VIDEO_INDEX}")
    print(f"[INFO] Videos available from start index: {len(selected_videos)}")
    print(f"[INFO] Target captures: {NUM_VIDEOS}")
    print(f"[INFO] Playback speed: {PLAYBACK_SPEED}x")
    print(f"[INFO] Instruction file: {INSTRUCTION_PATH}")
    print(f"[INFO] Output dir: {TASK_OUTPUT_DIR}")
    print(f"[INFO] Annotation file: {ANNOTATIONS_PATH}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        captured_count = 0
        stopped_early = False
        for offset, video_path in enumerate(selected_videos):
            if captured_count >= NUM_VIDEOS:
                break
            print(
                f"[INFO] ---------- Playing video {offset + 1}/{len(selected_videos)}: "
                f"{video_path.name} | captured {captured_count}/{NUM_VIDEOS}"
            )
            result = play_and_maybe_capture(
                video_path=video_path,
                human_prompt=instruction_prompts[captured_count],
                annotations=annotations,
            )
            if result == "captured":
                captured_count += 1
            elif result in {"quit", "error"}:
                stopped_early = True
                print("[INFO] Stopped by user.")
                break
        if captured_count < NUM_VIDEOS and not stopped_early:
            print(
                f"[INFO] Finished with {captured_count}/{NUM_VIDEOS} captures. "
                "No more videos available."
            )
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
