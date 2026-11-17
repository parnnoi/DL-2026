from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import pandas as pd
import datasets
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, pipeline

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

questions = ["Which viruses are best known to cause myocarditis?",
             "What is the role of the blood-brain barrier?",
             "From which cell type is leptin secreted?",
             "What is the causative agent of the Panama disease affecting bananas?"
            ]

for q in questions:
    print(f"\n{"="*150}")
    print(f"Question: {q}")

    base_out = base_generator(q, generation_config=generation_config)[0]["generated_text"]
    lora_out = adapted_generator(q, generation_config=generation_config)[0]["generated_text"]

    print("\n--- Base model ---")
    print(base_out)

    print("\n--- LoRA fine-tuned model ---")
    print(lora_out)