#!/usr/bin/env python3
import argparse
import ast
import atexit
import json
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import tempfile
import threading
import time

import cv2
import numpy as np
from PIL import Image
import torch

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("udp_llava_inference")


def infer_conv_mode(model_name: str) -> str:
    model_name = model_name.lower()
    if "llama-2" in model_name:
        return "llava_llama_2"
    if "mistral" in model_name:
        return "mistral_instruct"
    if "v1.6-34b" in model_name:
        return "chatml_direct"
    if "v1" in model_name:
        return "llava_v1"
    if "mpt" in model_name:
        return "mpt"
    return "llava_v0"


def parse_action_vector(output_text: str) -> list[float]:
    cleaned = output_text.strip()
    if not cleaned:
        raise ValueError("Model output is empty")

    match = re.search(r"\[[^\[\]]+\]", cleaned)
    if match is not None:
        cleaned = match.group(0)

    parsed = ast.literal_eval(cleaned)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Model output is not a list/tuple: {output_text}")
    return [float(value) for value in parsed]


def prepare_inference_model_path(model_path: str, disable_quant_config: bool) -> str:
    if not disable_quant_config:
        return model_path

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)

    temp_dir = tempfile.mkdtemp(prefix="llava_infer_cfg_")
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "non_lora_trainables.bin",
        "README.md",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ):
        src = os.path.join(model_path, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(temp_dir, name)
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    return temp_dir


class LlavaUnmergedInferenceNode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pending_image: Image.Image | None = None
        self.pending_trigger_addr: tuple[str, int] | None = None
        self.inference_started = False
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.model_name = get_model_name_from_path(args.model_path)
        self.conv_mode = args.conv_mode or "v1"
        self.model_ready = False
        self.effective_model_path = prepare_inference_model_path(
            args.model_path,
            not args.use_checkpoint_quant_config,
        )
        self.debug_image_dir = Path(args.debug_image_dir)
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        self.saved_image_count = 0
        self._stop_event = threading.Event()
        self._image_lock = threading.Lock()

        disable_torch_init()

        # UDP socket that receives 84x84 JPEG frames from rs_udp_sender.py
        # Packet format: [4B seq_num BE][4B jpeg_len BE][JPEG bytes]
        self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_socket.bind((args.image_bind_ip, args.image_bind_port))
        self.image_socket.setblocking(False)

        # UDP socket that receives trigger signals (1-byte bool)
        self.trigger_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.trigger_socket.bind((args.trigger_bind_ip, args.trigger_bind_port))
        self.trigger_socket.setblocking(False)

        self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        logger.info(
            "UDP inference node ready. "
            f"Receiving images on {args.image_bind_ip}:{args.image_bind_port}, "
            f"listening for trigger on {args.trigger_bind_ip}:{args.trigger_bind_port}, "
            f"sending result to port {args.result_port}."
        )

    def destroy(self) -> None:
        self._stop_event.set()
        for sock in (self.image_socket, self.trigger_socket, self.result_socket):
            try:
                sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Image reception (replaces ROS2 subscription)                       #
    # ------------------------------------------------------------------ #

    def _poll_image_socket(self) -> None:
        while True:
            try:
                # Max packet: 8B header + ~65KB JPEG; 70000B is safe
                data, _ = self.image_socket.recvfrom(70000)
            except BlockingIOError:
                break
            except OSError as exc:
                logger.error(f"Image recvfrom failed: {exc}")
                break

            if len(data) < 8:
                logger.warning(f"Image packet too short ({len(data)}B), discarding")
                continue

            _seq, jpeg_len = struct.unpack_from(">II", data, 0)
            jpeg_bytes = data[8: 8 + jpeg_len]

            if len(jpeg_bytes) != jpeg_len:
                logger.warning("Image packet truncated, discarding")
                continue

            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("JPEG decode failed, discarding frame")
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            with self._image_lock:
                self.pending_image = pil_image

            logger.debug(f"Received image frame seq={_seq}")

    # ------------------------------------------------------------------ #
    #  Trigger reception (unchanged logic, logger swapped)                #
    # ------------------------------------------------------------------ #

    def poll_trigger_socket(self) -> None:
        while True:
            try:
                payload, addr = self.trigger_socket.recvfrom(512)
            except BlockingIOError:
                break
            except OSError as exc:
                logger.error(f"Trigger recvfrom failed: {exc}")
                break

            if len(payload) != 1:
                logger.warning(
                    f"Ignoring trigger from {addr[0]}:{addr[1]} with unexpected payload size {len(payload)}"
                )
                continue

            trigger_value = struct.unpack("?", payload)[0]
            if not trigger_value:
                logger.info(f"Ignoring false trigger from {addr[0]}:{addr[1]}")
                continue

            with self._image_lock:
                has_image = self.pending_image is not None

            if not has_image:
                logger.warning(
                    f"Received trigger from {addr[0]}:{addr[1]} but no image is available yet"
                )
                continue

            self.pending_trigger_addr = (addr[0], addr[1])
            logger.info(f"Trigger signal arrived from {addr[0]}:{addr[1]}; inference scheduled")

    # ------------------------------------------------------------------ #
    #  Inference (unchanged)                                              #
    # ------------------------------------------------------------------ #

    def ensure_model_loaded(self) -> None:
        if self.model_ready:
            return

        logger.info(f"Loading model '{self.model_name}' from '{self.effective_model_path}'")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.effective_model_path,
            self.args.model_base,
            self.model_name,
            load_8bit=self.args.load_8bit,
            load_4bit=self.args.load_4bit,
            device=self.args.device,
        )
        self.model_ready = True
        logger.info("Model loaded successfully")

    def build_query(self) -> str:
        qs = self.args.instruction
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                return qs.replace(IMAGE_PLACEHOLDER, image_token_se)
            return qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)

        if self.model.config.mm_use_im_start_end:
            return image_token_se + "\n" + qs
        return DEFAULT_IMAGE_TOKEN + "\n" + qs

    def run_inference(self, image: Image.Image) -> str:
        query = self.build_query()
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=self.model.dtype) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        with torch.inference_mode():
            generate_kwargs = dict(
                images=image_tensor,
                image_sizes=[image_size],
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
            )
            if self.args.temperature > 0.0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = self.args.temperature
                if self.args.top_p is not None:
                    generate_kwargs["top_p"] = self.args.top_p
            else:
                generate_kwargs["do_sample"] = False

            output_ids = self.model.generate(
                input_ids,
                **generate_kwargs,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def maybe_run_inference(self) -> None:
        if self.inference_started or self.pending_trigger_addr is None:
            return

        with self._image_lock:
            image = self.pending_image

        if image is None:
            return

        self.inference_started = True
        trigger_addr = self.pending_trigger_addr
        self.pending_trigger_addr = None

        logger.info("---------------Starting inference---------------")

        try:
            self.ensure_model_loaded()
            output_text = self.run_inference(image)
            logger.info(f"Raw model output: {output_text!r}")
            action_vector = parse_action_vector(output_text)

            if len(action_vector) != 3:
                raise ValueError(
                    f"Expected 3 values for right_constraint, got {len(action_vector)}: {action_vector}"
                )

            self.send_result(action_vector, trigger_addr[0])
        except Exception as exc:
            logger.error(f"Inference failed: {exc}")
        finally:
            self.inference_started = False

    def send_result(self, action_vector: list[float], trigger_ip: str) -> None:
        result_ip = self.args.result_ip or trigger_ip
        payload = struct.pack("3d", *action_vector)
        self.result_socket.sendto(payload, (result_ip, self.args.result_port))
        logger.info(f"Sent right_constraint {action_vector} to {result_ip}:{self.args.result_port}")

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        def image_loop():
            while not self._stop_event.is_set():
                self._poll_image_socket()
                time.sleep(0.001)

        def trigger_loop():
            while not self._stop_event.is_set():
                self.poll_trigger_socket()
                time.sleep(0.01)

        def inference_loop():
            while not self._stop_event.is_set():
                self.maybe_run_inference()
                time.sleep(0.05)

        threads = [
            threading.Thread(target=image_loop, name="image-poll", daemon=True),
            threading.Thread(target=trigger_loop, name="trigger-poll", daemon=True),
            threading.Thread(target=inference_loop, name="inference", daemon=True),
        ]
        for t in threads:
            t.start()

        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            self.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--model-base", required=True, type=str)
    parser.add_argument("--instruction", required=True, type=str)
    parser.add_argument("--conv-mode", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top-p", default=None, type=float)
    parser.add_argument("--num-beams", default=1, type=int)
    parser.add_argument("--max-new-tokens", default=64, type=int)
    parser.add_argument("--debug-image-dir", default="./debug_received_images", type=str)
    parser.add_argument("--use-checkpoint-quant-config", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-bind-ip", default="0.0.0.0", type=str)
    parser.add_argument("--image-bind-port", default=5010, type=int,
                        help="UDP port to receive 84x84 JPEG frames from rs_udp_sender.py")
    parser.add_argument("--trigger-bind-ip", default="0.0.0.0", type=str)
    parser.add_argument("--trigger-bind-port", default=5008, type=int)
    parser.add_argument("--result-ip", default="100.94.172.95", type=str)  # husky-rcu tailscale IP
    parser.add_argument("--result-port", default=5009, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node = LlavaUnmergedInferenceNode(args)
    node.run()


if __name__ == "__main__":
    main()
