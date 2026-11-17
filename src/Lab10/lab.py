import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, pipeline
import os

from IPython.display import display_markdown, Latex, display
def print_markdown(content):
    if isinstance(content, str) and "\\" in content:
        display(Latex(content))
    else:
        display_markdown(content, raw=True)

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
OUTPUT_DIR = "/storage/slurm/home/68070701616@cpe.kmutt.ac.th/DeepLearning/src/Lab10/lora-demo"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype
).to(device)

print(f"Tokenizer pad token: {tokenizer.pad_token}")

import copy
generation_config = copy.deepcopy(model.generation_config)

generation_config.max_new_tokens = 500
generation_config.do_sample = True
generation_config.temperature = 0.5

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer
)

import pandas as pd
import datasets

dataset = pd.read_json("/storage/slurm/home/68070701616@cpe.kmutt.ac.th/DeepLearning/src/Lab10/alpaca_data_cleaned.json")

alpaca_dataset = datasets.Dataset.from_pandas(dataset)

def format_instruction(example):
    if example["input"].strip():
        text = (
            f"Instruction: {example['instruction']}\n"
            f"Input: {example['input']}\n"
            f"Response: {example['output']}"
        )
    else:
        text = (
            f"Instruction: {example['instruction']}\n"
            f"Response: {example['output']}"
        )
    return {"text": text}

formatted_dataset = alpaca_dataset.map(format_instruction)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    dataset_text_field="text",  
    max_length=512,
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=20,
    max_steps=100,
    save_strategy="epoch",
    optim="adamw_torch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    processing_class=tokenizer
)

trainer.model.print_trainable_parameters()

effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
total_steps = ((len(alpaca_dataset)+1) // effective_batch_size) * training_args.num_train_epochs

print(f"Total training steps: {total_steps}")

trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved LoRA adapter to:", OUTPUT_DIR)


#############################

from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

# Base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype
).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_generator = pipeline(
    "text-generation", 
    model=base_model, 
    tokenizer=tokenizer
)

# Base model + LoRA adapter
adapted_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
adapted_generator = pipeline(
    "text-generation",
    model=adapted_model,
    tokenizer=tokenizer,
    device=device
)

import copy
generation_config = copy.deepcopy(model.generation_config)

generation_config.max_new_tokens = 200
generation_config.do_sample = True
generation_config.temperature = 0.5

def build_prompt(instruction, input_text=""):
    if input_text.strip():
        return f"Instruction: {instruction}\nInput: {input_text}\nResponse: "
    return f"Instruction: {instruction}\nResponse: "

instruction = [
    {"instruction": "Give three tips for staying healthy.", "input": ""},
    {"instruction": "Summarize the following text.", "input": "Leptin is a hormone secreted by adipocytes..."},
    {"instruction": "What is the role of the blood-brain barrier?", "input": ""},
]

for ins in instruction:
    prompt = build_prompt(ins["instruction"], ins["input"])

    print(f"\n{"="*150}")
    print(f"Instruction: {ins['instruction']}")
    if ins["input"]:
        print(f"Input: {ins['input']}")

    base_out = base_generator(ins, generation_config=generation_config)[0]["generated_text"]
    lora_out = adapted_generator(ins, generation_config=generation_config)[0]["generated_text"]

    print("\n--- Base model ---")
    print(base_out)

    print("\n--- LoRA fine-tuned model ---")
    print(lora_out)