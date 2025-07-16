import os
import torch
import gc
import warnings
import psutil
import time
from typing import Dict, Any

warnings.filterwarnings("ignore")

#Lot of things hardcoded for best training with my system specs: MacBookPro M2, 8GB

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

def aggressive_memory_cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    for _ in range(3):
        gc.collect()

aggressive_memory_cleanup()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print(f"Initial memory usage: {get_memory_usage():.1f} MB")

from datasets import load_dataset

print("Loading dataset")
dataset = load_dataset("json", data_files={"train": "train_2.jsonl", "test": "test_2.jsonl"})
print(f"Dataset loaded. Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
print(f"Memory after dataset load: {get_memory_usage():.1f} MB")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "amd/AMD-Llama-135m"


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Memory before model load: {get_memory_usage():.1f} MB")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True,
)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

if hasattr(model.config, 'use_memory_efficient_attention'):
    model.config.use_memory_efficient_attention = True
    print("Memory efficient attention enabled")

print(f"Model loaded on {model.device}")
print(f"Memory after model load: {get_memory_usage():.1f} MB")

aggressive_memory_cleanup()

def tokenize(examples):
    if isinstance(examples['instruction'], list):
        prompts = [f"{inst}{out}" for inst, out in zip(examples['instruction'], examples['output'])]
        instructions = examples['instruction']
    else:
        prompts = [f"{examples['instruction']}{examples['output']}"]
        instructions = [examples['instruction']]

    tokenized_prompts = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    tokenized_instructions = tokenizer(instructions, truncation=True, padding="max_length", max_length=512)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for i, (input_ids, instruction_ids) in enumerate(zip(tokenized_prompts["input_ids"], tokenized_instructions["input_ids"])):
        labels = input_ids.copy()
        instruction_len = len([id for id in instruction_ids if id != tokenizer.pad_token_id])
        labels[:instruction_len] = [-100] * instruction_len
        input_ids_list.append(input_ids)
        attention_mask_list.append([1 if token != tokenizer.pad_token_id else 0 for token in input_ids])
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

print("Tokenizing dataset...")
print(f"Memory before tokenization: {get_memory_usage():.1f} MB")

tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=100, num_proc=1)

print(f"Memory after tokenization: {get_memory_usage():.1f} MB")
aggressive_memory_cleanup()

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    report_to=None,
    fp16=False,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    remove_unused_columns=True,
    prediction_loss_only=True,
    eval_accumulation_steps=1,
    ddp_find_unused_parameters=False,
    save_safetensors=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

if __name__ == '__main__':
    print(f"Memory before training: {get_memory_usage():.1f} MB")
    aggressive_memory_cleanup()

    print("Start train")
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"Memory after training: {get_memory_usage():.1f} MB")

    output_dir = "Claude_2_Model_Output_FT"
    print(f"Saving model to {output_dir}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    print(f"Memory after saving: {get_memory_usage():.1f} MB")

    aggressive_memory_cleanup()

    del trainer
    del model
    aggressive_memory_cleanup()

    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    import torch

    model = AutoModelForCausalLM.from_pretrained("Claude_2_Model_Output_FT")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Claude_2_Model_Output_FT")

    class StopOnStudentToken(StoppingCriteria):
        def __init__(self, stop_token, tokenizer):
            self.stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)
            self.tokenizer = tokenizer

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
            if len(input_ids[0]) < len(self.stop_token_id):
                return False
            if input_ids[0].tolist()[-len(self.stop_token_id):] == self.stop_token_id:
                return True
            return False

    print("Fine-tuned model loaded successfully!")
    print(f"Memory after loading fine-tuned model: {get_memory_usage():.1f} MB")
