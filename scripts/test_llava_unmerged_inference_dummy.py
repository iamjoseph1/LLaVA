#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile

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


def build_query(model, instruction: str) -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in instruction:
        if model.config.mm_use_im_start_end:
            return instruction.replace(IMAGE_PLACEHOLDER, image_token_se)
        return instruction.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)

    if model.config.mm_use_im_start_end:
        return image_token_se + "\n" + instruction
    return DEFAULT_IMAGE_TOKEN + "\n" + instruction


def maybe_strip_quantization_config(model_path: str, disable_quant_config: bool) -> str:
    if not disable_quant_config:
        return model_path

    with open(f"{model_path}/config.json", "r") as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)

    tmp_dir = tempfile.mkdtemp(prefix="llava_cfg_")
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    # Preserve the rest of the unmerged LoRA directory structure so
    # `load_pretrained_model()` can find adapter and non-LoRA files locally.
    for name in os.listdir(model_path):
        if name == "config.json":
            continue
        src = os.path.join(model_path, name)
        dst = os.path.join(tmp_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    return tmp_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--model-base", required=True, type=str)
    parser.add_argument("--instruction", required=True, type=str)
    parser.add_argument("--image-file", default=None, type=str)
    parser.add_argument("--conv-mode", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top-p", default=None, type=float)
    parser.add_argument("--num-beams", default=1, type=int)
    parser.add_argument("--max-new-tokens", default=64, type=int)
    parser.add_argument("--min-new-tokens", default=0, type=int)
    parser.add_argument("--disable-quant-config", action="store_true")
    args = parser.parse_args()

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    conv_mode = args.conv_mode or infer_conv_mode(model_name)
    effective_model_path = maybe_strip_quantization_config(args.model_path, args.disable_quant_config)

    print(f"Loading model: {model_name}")
    print(f"Conversation mode: {conv_mode}")
    print(f"Effective model path: {effective_model_path}")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        effective_model_path if args.disable_quant_config else args.model_path,
        args.model_base,
        model_name,
        device=args.device,
    )

    query = build_query(model, args.instruction)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if args.image_file is not None:
        image = Image.open(args.image_file).convert("RGB")
    else:
        image = Image.new("RGB", (84, 84), color=(128, 128, 128))
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=model.dtype) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=model.dtype)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    if args.image_file is not None:
        print(f"Image file: {args.image_file}")
    else:
        print(f"Dummy image size: {image_size}")
    print(f"Prompt length: {input_ids.shape[1]}")

    with torch.inference_mode():
        generate_kwargs = dict(
            images=image_tensor,
            image_sizes=[image_size],
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if args.temperature > 0.0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                generate_kwargs["top_p"] = args.top_p
        else:
            generate_kwargs["do_sample"] = False

        generation_output = model.generate(
            input_ids,
            **generate_kwargs,
        )

    output_ids = generation_output.sequences
    # generated_ids = output_ids[:, input_ids.shape[1]:]
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    raw_output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

    print("Model output:")
    print(output_text)
    print("Raw decoded output:")
    print(repr(raw_output_text))
    # print("Generated token ids:")
    # print(generated_ids[0].tolist())

    if generation_output.scores:
        first_step_logits = generation_output.scores[0][0]
        topk = torch.topk(first_step_logits, k=10)
        decoded = [tokenizer.decode([idx]) for idx in topk.indices.tolist()]
        print("Top-10 first-step tokens:")
        for token_id, token_text, logit in zip(topk.indices.tolist(), decoded, topk.values.tolist()):
            print(f"  id={token_id:>6} logit={logit:>10.4f} token={token_text!r}")


if __name__ == "__main__":
    main()
