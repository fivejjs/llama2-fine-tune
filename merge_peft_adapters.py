import os

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.empty_cache()

# Update the path accordingly
adapter_dir = "./results/final_checkpoint"
output_dir = "./merged_peft"

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_dir, device_map="cpu", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)
