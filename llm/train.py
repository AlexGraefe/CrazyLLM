from functools import partial
import logging
import omegaconf
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

import os

import hydra

from llm.execution_guard import execution_guard
from llm.llm import LLM


def get_lora_model(model_name: str, use_cluster: bool, cfg):
    """
    Load a LoRA-adapted language model based on the specified model name.
    
    Args:
        model_name: Name of the base model ('gemma' or 'qwen')
        use_cluster: Whether running on HPC cluster (affects cache directory)
        cfg: Configuration object containing bnb and lora parameters
    """
    logger = logging.getLogger("execution_guard")
    logger.info(f"Loading base model for LoRA adaptation...")

    llm = LLM(model_name=model_name, use_cluster=use_cluster, quantize=True)
    logger.info(f"Base model {llm.full_model_name} loaded successfully!")

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(cfg.lora.target_modules),
    )
    return llm.model, llm.tokenizer, lora_config
    

@hydra.main(config_path="parameters", config_name="main")
@partial(execution_guard, force_overwrite=True)
def main(cfg: omegaconf.DictConfig):
    import wandb
    rank = int(os.environ.get('RANK',-1))
    print(rank)
    # Load model
    model, tokenizer, lora_config = get_lora_model(cfg.model_name, cfg.use_cluster, cfg)

    # load dataset
    dataset = load_dataset("json", data_files=cfg.dataset.path, field="data")
    eval_dataset = load_dataset("json", data_files=cfg.dataset.path_eval, field="data")

    if rank <= 0:
        # Train model
        wandb.init(
                # set the wandb project where this run will be logged
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
                # track hyperparameters and run metadata
                job_type="train",
                name=cfg.experiment_name_wandb,
            )


    sft_config = SFTConfig(
        gradient_checkpointing=True,    # this saves a LOT of memory
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,  
        per_device_train_batch_size=cfg.training.per_device_train_batch_size, 
        # per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Dataset
        # packing a dataset means no padding is needed
        # packing=True,
        # packing_strategy='wrapped', # added to approximate original packing behavior
        ## GROUP 3: These are typical training parameters
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        # max_length=cfg.training.max_length,
        # Optimizer
        # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
        optim='paged_adamw_8bit',       
        ## GROUP 4: Logging parameters
        logging_steps=1,
        logging_dir='./logs',
        output_dir='./finetuned_model',
        report_to=cfg.training.report_to,
        bf16=torch.cuda.is_bf16_supported(including_emulation=False),
        load_best_model_at_end=True,
    )

    def formatting_function(examples):
        return tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)

    print(len(dataset['train']))
    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset['train'],
        eval_dataset=eval_dataset['train'],
        # formatting_func=formatting_function,
    )

    # dl = trainer.get_train_dataloader()
    # batch = next(iter(dl))
    # print(len(dl))
    # exit()

    trainer.train()
    trainer.model.save_pretrained("./finetuned_model")

    llm = LLM.from_lora(model_name="qwen", lora_checkpoint_path=f"{os.getcwd()}/finetuned_model", use_cluster=cfg.use_cluster)
    merged_model = llm.model.merge_and_unload()
    merged_model.push_to_hub(
        "Devjalx-4b", private=False, tags=["GRPO", "Reasoning-Course"]
    )

    # merged_model = trainer.model.merge_and_unload()
    # merged_model.push_to_hub(
    #     "Devjalx-4b", private=False, tags=["GRPO", "Reasoning-Course"]
    # )

if __name__ == "__main__":
    main()