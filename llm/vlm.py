#!/usr/bin/env python3
"""
VLM Inference Module

This module provides a simple interface for loading and running inference with Vision-Language Models.
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
import logging


def load_model(model_name: str, use_cluster: bool = False, bits_and_bytes_config=None):

    logger = logging.getLogger("execution_guard")
    cache_dir = "/data/models" if not use_cluster else "/hpcwork/p0021919/models"

    if model_name == "qwen-vl":
        full_model_name = "Qwen/Qwen3.5-9B"
    else:
        full_model_name = model_name

    current_device = 0
    if "RANK" in os.environ:
        current_device = int(os.environ['RANK'])

    logger.info(f"Loading VLM {full_model_name} on device {current_device} with cache directory {cache_dir}...")

    processor = AutoProcessor.from_pretrained(full_model_name, cache_dir=cache_dir)

    if not bits_and_bytes_config:
        model = AutoModelForImageTextToText.from_pretrained(
            full_model_name,
            torch_dtype=torch.bfloat16,
            device_map={'': current_device},
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            full_model_name,
            device_map={'': current_device},
            quantization_config=bits_and_bytes_config,
            cache_dir=cache_dir,
        )

    return model, processor, full_model_name


class VLM:
    """VLM class for loading and running inference with vision-language models."""

    def __init__(self, model_name: str, use_cluster: bool = False, quantize: bool = True):
        """
        Initialize the VLM with a specific model.

        Args:
            model_name: Name of the model to load ('qwen-vl' or a full model path)
            use_cluster: Whether running on HPC cluster (affects cache directory)
            quantize: Whether to load the model in 4-bit quantization
        """
        self.model_name = model_name
        self.use_cluster = use_cluster

        print(f"Loading VLM model... This may take a moment.")

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32,
            )
        else:
            bnb_config = None

        model, processor, full_model_name = load_model(model_name, use_cluster, bits_and_bytes_config=bnb_config)
        print(model.get_memory_footprint() / 1e6)

        self.full_model_name = full_model_name
        self.processor = processor
        self.model = model

        torch.set_float32_matmul_precision('high')
        print(f"{full_model_name} VLM loaded successfully!")

    def predict(self, prompt: str, image, max_new_tokens: int = 500) -> str:
        """
        Generate a response for a single text prompt and image.

        Args:
            prompt: The input text prompt
            image: A PIL image, local file path, or URL string
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.chat(messages, max_new_tokens)

    def chat(self, messages: list, max_new_tokens: int = 500) -> str:
        """
        Generate a response for a list of chat messages (may include image content).

        Each user message may include items of the form:
            {"type": "image", "image": <PIL image | path | URL>}
            {"type": "text", "text": "..."}

        Args:
            messages: List of messages in the standard chat format
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_text = self.processor.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )
        return generated_text[0]

    @classmethod
    def from_lora(cls, model_name: str, lora_checkpoint_path: str, use_cluster: bool = False):
        """
        Load a VLM and adapt it with LoRA from a specified checkpoint.
        """
        my_vlm = cls(model_name=model_name, use_cluster=use_cluster, quantize=False)
        my_vlm.model = PeftModel.from_pretrained(my_vlm.model, lora_checkpoint_path)
        return my_vlm


if __name__ == "__main__":
    vlm = VLM(model_name="qwen-vl", use_cluster=False)

    response = vlm.predict(
        prompt="What do we see in this image?",
        image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
    )
    print(response)
