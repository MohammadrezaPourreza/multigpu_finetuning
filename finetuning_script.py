import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM

# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 finetuning_script.py --config train_config.yaml

@dataclass
class ScriptArguments:
    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset, e.g. /opt/ml/input/data/train/"},
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    hub_path: str = field(
        default=None, metadata={"help": "Path to the hub"}
    )
    sample_size: int = field(
        default=None, metadata={"help": "Number of samples to use for training"}
    )


def merge_and_save_model(model_id, adapter_dir, hub_path):
    from peft import PeftModel
    print("Trying to load a Peft model. It might take a while without feedback")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = peft_model.merge_and_unload()

    print(f"Pushing the newly created merged model to {hub_path}")
    model.push_to_hub(hub_path, safe_serialization=True)
    base_model.config.push_to_hub(hub_path)


def training_function(script_args, training_args):
    ################
    # Dataset
    ################

    dataset = load_dataset(script_args.train_dataset_path)
    if script_args.sample_size:
        random_indices = random.sample(range(len(dataset['train'])), script_args.sample_size)
        dataset['train'] = dataset['train'].select(random_indices)
    dataset = dataset['train'].train_test_split(test_size=0.01, shuffle=True)
    dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})

    ################
    # Model & Tokenizer
    ################

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # template dataset
    def formatting_prompts_func(training_dataset):
        output_texts = []
        prompts = training_dataset['prompt']
        sqls = training_dataset['sql']
        for prompt, sql in zip(prompts, sqls):
            user_message = prompt
            if ";" in sql:
                sql = sql.replace(";", "").strip()
            assitant_message = f"""
```sql
{sql} ;
```
"""
            messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assitant_message},
            ]
            output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return output_texts

    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # print random sample on rank 0
    if training_args.distributed_state.is_main_process:
        print(f"Totall traning samples {len(dataset['train'])}")
        print(f"Totall validation samples {len(dataset['validation'])}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print

    # Model
    torch_dtype = torch.float32
    quant_storage_dtype = torch.float32

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        #attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        torch_dtype=quant_storage_dtype,
        use_cache=(
            False if training_args.gradient_checkpointing else True
        ),  # this is needed for gradient checkpointing
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        tokenizer=tokenizer,
        args=training_args,
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    #########################################
    # SAVE ADAPTER AND CONFIG FOR SAGEMAKER
    #########################################
    # save adapter
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        print("Training completed")

    del model
    del trainer
    torch.cuda.empty_cache()  # Clears the cache
    # load and merge
    if training_args.distributed_state.is_main_process:
        merge_and_save_model(
            script_args.model_id, training_args.output_dir, script_args.hub_path
        )
        tokenizer.push_to_hub(script_args.hub_path)
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to print


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # launch training
    training_function(script_args, training_args)