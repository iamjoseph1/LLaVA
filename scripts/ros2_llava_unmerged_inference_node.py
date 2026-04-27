#!/usr/bin/env python3
import argparse
import ast
import atexit
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float64MultiArray
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


def ros_image_to_pil(msg: RosImage) -> Image.Image:
    encoding = msg.encoding.lower()
    height = msg.height
    width = msg.width
    step = msg.step
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        channels = 3
        row_bytes = width * channels
        image = data.reshape((height, step))[:, :row_bytes].reshape((height, width, channels))
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return Image.fromarray(image, mode="RGB")

    if encoding in ("rgba8", "bgra8"):
        channels = 4
        row_bytes = width * channels
        image = data.reshape((height, step))[:, :row_bytes].reshape((height, width, channels))
        if encoding == "bgra8":
            image = image[:, :, [2, 1, 0, 3]]
        return Image.fromarray(image, mode="RGBA").convert("RGB")

    if encoding in ("mono8", "8uc1"):
        row_bytes = width
        image = data.reshape((height, step))[:, :row_bytes].reshape((height, width))
        return Image.fromarray(image, mode="L").convert("RGB")

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


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

    # Only mirror the small top-level files required for unmerged LoRA inference.
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


class LlavaUnmergedInferenceNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("llava_unmerged_inference_node")
        self.args = args
        self.inference_started = False
        self.pending_image: Image.Image | None = None
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

        disable_torch_init()

        constraint_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(
            Float64MultiArray,
            "sa_right_eef_constraint",
            constraint_qos,
        )
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            RosImage,
            "sa_front_overview/image_raw",
            self.image_callback,
            image_qos,
        )
        self.inference_timer = self.create_timer(0.1, self.maybe_run_inference)

        self.get_logger().info(
            f"Inference node ready. Waiting for images on sa_front_overview/image_raw "
            f"with conv mode '{self.conv_mode}'. Model will load on first inference."
        )

    def ensure_model_loaded(self) -> None:
        if self.model_ready:
            return

        self.get_logger().info(
            f"Loading model '{self.model_name}' from '{self.effective_model_path}'"
        )
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.effective_model_path,
            self.args.model_base,
            self.model_name,
            load_8bit=self.args.load_8bit,
            load_4bit=self.args.load_4bit,
            device=self.args.device,
        )
        self.model_ready = True
        self.get_logger().info("Model loaded successfully")

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

    def image_callback(self, msg: RosImage) -> None:
        try:
            self.pending_image = ros_image_to_pil(msg)
            # image_path = self.debug_image_dir / f"received_{self.saved_image_count:06d}.png"
            # self.pending_image.save(image_path)
            # self.saved_image_count += 1
            # self.get_logger().info(f"Saved received image to {image_path}")
            self.get_logger().debug("Received image. Inference will run outside the callback.")
        except Exception as exc:
            self.get_logger().error(f"Failed to reconstruct ROS image: {exc}")

    def maybe_run_inference(self) -> None:
        if self.inference_started or self.pending_image is None:
            return

        self.inference_started = True
        image = self.pending_image
        self.pending_image = None

        try:
            self.ensure_model_loaded()
            output_text = self.run_inference(image)
            self.get_logger().info(f"Raw model output: {output_text!r}")
            action_vector = parse_action_vector(output_text)

            out_msg = Float64MultiArray()
            out_msg.data = action_vector
            self.publisher.publish(out_msg)

            self.get_logger().info(f"Published {action_vector} to sa_right_eef_constraint")
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
            self.pending_image = image
        finally:
            self.inference_started = False


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = LlavaUnmergedInferenceNode(args)
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
