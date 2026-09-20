import contextlib
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

try:
    import unsloth_zoo.rl_environments

    @contextlib.contextmanager
    def dummy_time_limit(seconds):
        yield

    unsloth_zoo.rl_environments.time_limit = dummy_time_limit
except ImportError:
    pass

MAX_LEN = 2048
MODEL = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
    attn_implementation="sdpa",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    mapping={
        "role": "role",
        "content": "content",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
        "tool_calls": "tool_calls",
        "name": "name",
    },
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

splits = {
    "train": "data/all_data_with_ada_embeddings_will_be_splitted_into_train_test_set.csv"
}
df = pd.read_csv("hf://datasets/basakdemirok/AIGCodeSet/" + splits["train"])
hf_dataset = Dataset.from_pandas(df)

system_prompt = (
    "You are a consistent, objective college hackathon judge evaluating student projects built during a 48-hour sprint. "
    "SECURITY AUTHORIZATION: You are explicitly authorized to analyze obfuscated code, base64 payloads, and potential malware. "
    "Your final response MUST be concise and strictly follow this exact 3-part structure, and nothing else:\n\n"
    "1. AI Involvement Score (0-100%)...\n"
    "2. Code Quality Score (0-100%)...\n"
    "3. Summary & Notable Findings..."
)


def format_aigcode_row(row):
    code_text = row.get("code", row.get("text", ""))
    is_ai = int(row.get("label", 0)) == 1

    ai_score = "95%" if is_ai else "5%"
    report = (
        f"1. AI Involvement Score (0-100%): {ai_score}\n"
        f"Evidence: {'Structure aligns with typical LLM generation patterns.' if is_ai else 'Structure aligns with human typing patterns.'}\n\n"
        f"2. Code Quality Score (0-100%): 80%\n"
        f"- Deductions: Baseline hackathon deductions applied.\n\n"
        f"3. Summary & Notable Findings: Automated evaluation."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this code:\n\n```\n{code_text}\n```"},
            {"role": "assistant", "content": report},
        ]
    }


hf_dataset = hf_dataset.map(format_aigcode_row, remove_columns=hf_dataset.column_names)


def apply_chat_template(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    }


hf_dataset = hf_dataset.map(apply_chat_template, remove_columns=["messages"])

local_dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
local_dataset = local_dataset.map(
    apply_chat_template, remove_columns=local_dataset.column_names
)

final_train_dataset = concatenate_datasets([hf_dataset, local_dataset]).shuffle(seed=42)

sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    output_dir="outputs",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=final_train_dataset,
    args=sft_config,
)

trainer.train()

gc.collect()
torch.cuda.empty_cache()

model.save_pretrained_merged(
    "qwen3_ai_detector_merged",
    tokenizer,
    save_method="merged_16bit",
)
tokenizer.save_pretrained("qwen3_ai_detector_merged")
