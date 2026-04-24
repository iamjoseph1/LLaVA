#!/usr/bin/env python3
import argparse
import ast
from typing import Optional

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
    parsed = ast.literal_eval(output_text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Model output is not a list/tuple: {output_text}")
    return [float(value) for value in parsed]


class LlavaUnmergedInferenceNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("llava_unmerged_inference_node")
        self.args = args
        self.processed = False

        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        self.conv_mode = args.conv_mode or infer_conv_mode(model_name)
        self.get_logger().info(f"Loading model '{model_name}' with conv mode '{self.conv_mode}'")

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            device=args.device,
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/sa_right_eef_constraint",
            10,
        )
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            RosImage,
            "sa_front_overview/image_raw",
            self.image_callback,
            image_qos,
        )

        self.get_logger().info("Node ready. Waiting for first image on sa_front_overview/image_raw")

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
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=self.args.temperature > 0.0,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
            )

        generated_ids = output_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def image_callback(self, msg: RosImage) -> None:
        if self.processed:
            return
        self.processed = True

        try:
            image = ros_image_to_pil(msg)
            output_text = self.run_inference(image)
            action_vector = parse_action_vector(output_text)

            out_msg = Float64MultiArray()
            out_msg.data = action_vector
            self.publisher.publish(out_msg)

            self.get_logger().info(f"Published {action_vector} to /sa_right_eef_constraint")
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
        finally:
            self.destroy_subscription(self.subscription)
            self.create_timer(0.5, self._shutdown_once)

    def _shutdown_once(self) -> None:
        self.get_logger().info("Shutting down after one-shot inference")
        rclpy.shutdown()


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
