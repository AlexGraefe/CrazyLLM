#!/usr/bin/env python3
"""
LLM Inference Module

This module provides a simple interface for loading and running inference with LLMs.
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging

def load_model(model_name: str, thinking=False, use_cluster: bool = False, bits_and_bytes_config: str = None):
    
    logger = logging.getLogger("execution_guard")
    # Set cache directory based on environment
    cache_dir = "/data/models" if not use_cluster else "/hpcwork/p0021919/models"
    
    tokenizer_name = None

    # Resolve model name to full path
    if model_name == "gemma":
        full_model_name = "google/gemma-3-4b-it" if not use_cluster else "gemma-3-27b-it"
    elif model_name == "qwen":
        if thinking:
            full_model_name = "Qwen/Qwen3-4B-Thinking-2507" if not use_cluster else "Qwen/Qwen3-4B-Thinking-2507" 
        else:
            full_model_name = "Qwen/Qwen3-4B-Instruct-2507" if not use_cluster else "Qwen/Qwen3-4B-Instruct-2507"   # Qwen/Qwen3-4B-Instruct-2507-FP8
    elif model_name == "devjalx":
        full_model_name = "halxj/Devjalx-4b"
        tokenizer_name = "Qwen/Qwen3-4B-Instruct-2507"
    else:
        full_model_name = model_name
    
    if tokenizer_name is None:
        tokenizer_name = full_model_name

    
    current_device = 0
    if "RANK" in os.environ:
        current_device = int(os.environ['RANK'])

    logger.info(f"Loading model {full_model_name} on device {current_device} with cache directory {cache_dir}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, device_map={'':current_device})

    # Load model
    if not bits_and_bytes_config:
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            torch_dtype=torch.bfloat16 if model_name == "gemma" else "auto",
            device_map={'':current_device},
            attn_implementation="sdpa",
            cache_dir=cache_dir
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            full_model_name,
            device_map={'':current_device},
            quantization_config=bits_and_bytes_config,
            cache_dir=cache_dir
        )

    return model, tokenizer, full_model_name

class LLM:
    """LLM class for loading and running inference with language models"""
    
    def __init__(self, model_name: str, thinking=False, use_cluster: bool = False, quantize=True):
        """
        Initialize the LLM with a specific model.
        
        Args:
            model_name: Name of the model to load ('gemma', 'qwen', or a full model path)
            use_cluster: Whether running on HPC cluster (affects cache directory)
        """
        self.model_name = model_name
        self.thinking = thinking
        self.use_cluster = use_cluster

        print(f"Loading model... This may take a moment.")
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32
            )
        else:
            bnb_config = None
        model, tokenizer, full_model_name = load_model(model_name, thinking, use_cluster, bits_and_bytes_config=bnb_config)
        print(model.get_memory_footprint()/1e6)
        
        self.full_model_name = full_model_name
        self.tokenizer = tokenizer
        self.model = model
        
        torch.set_float32_matmul_precision('high')
        print(f"{full_model_name} model loaded successfully!")
    
    def predict(self, prompt: str, max_new_tokens: int = 5000) -> str:
        """
        Generate text based on the input prompt.
        
        Args:
            prompt: The input prompt string
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_new_tokens)
            
    def chat(self, messages: list, max_new_tokens: int = 5000) -> str:
        """
        Generate a response based on a list of chat messages.
        
        Args:
            messages: List of messages in the format [{'role': 'user'/'assistant', 'content': '...'}, ...]
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking
        )
        
        # Tokenize input
        input_ids = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            cache_implementation="static",
            #  temperature=1.0,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
        # Decode generated text
        if self.thinking:
            output_ids = outputs[0][len(input_ids.input_ids[0]):].tolist() 
            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            generated_text = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return generated_text, thinking_content
        else:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_idx = generated_text.lower().find("assistant")
            if assistant_idx != -1:
                answer = generated_text[assistant_idx + len("assistant"):].strip()
                return answer
            else:
                return ""
        
    @classmethod
    def from_lora(cls, model_name: str, lora_checkpoint_path: str, use_cluster: bool = False):
        """
        Load an LLM model and adapt it with LoRA from a specified checkpoint.
        """
        my_llm = cls(model_name=model_name, use_cluster=use_cluster, quantize=False)
        my_llm.model = PeftModel.from_pretrained(my_llm.model, lora_checkpoint_path)

        return my_llm


if __name__ == "__main__":
    # Example usage
    llm = LLM(model_name="qwen", use_cluster=False)

    message1 = [
        {"role": "user", "content": "Hallo"}
    ]

    message2 = [
        {"role": "user", "content": "Hallooo"}
    ]

    messages = [message1, message2]
        
    # Apply chat template
    texts = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print(llm.tokenizer(texts, return_tensors="pt", padding=True))


