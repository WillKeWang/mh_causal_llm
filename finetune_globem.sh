#!/bin/bash
#SBATCH --job-name=ft-globem
#SBATCH --partition=gpu          # adapt to your cluster
#SBATCH --gres=gpu:1             # one GPU is enough for LoRA
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH -c 8
#SBATCH --output=ft_globem_%j.log

set -euo pipefail
set -x        

# ---------- 1) CUDA / Python ----------
module load cuda/12.4 || true        # pick whatever CUDA module is available

# ---------- 2) lightweight env ----------
ENV_DIR=$SCRATCH/healthllm_env
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate
pip install --upgrade pip wheel

# core deps  ↓  **SentencePiece now included**
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft bitsandbytes accelerate \
            sentencepiece tokenizers


# ---------- 3) paths ----------
DATA_JSON=datasets/GLOBEM_depression_train_all.json
MODEL_OUT=models/globem_depression_lora
export MODEL_OUT
mkdir -p $MODEL_OUT

# ---------- 4) run LoRA fine-tune ----------
python - <<'PY'
from peft import LoraConfig, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
import json, os, torch

base_model = "decapoda-research/llama-7b-hf"   # small & fast
tokenizer   = LlamaTokenizer.from_pretrained(base_model)
model       = LlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_cfg)

# -------- dataset --------
with open("datasets/GLOBEM_depression_train_all.json") as f:
    raw = json.load(f)

def format_example(ex):
    return (
        f"{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}"
    )

tok_data = tokenizer(
    [format_example(x) for x in raw],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)
class DS(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i):
        x = {k:v[i] for k,v in self.enc.items()}
        x["labels"] = x["input_ids"].clone()
        return x

ds = DS(tok_data)

# -------- training --------
args = TrainingArguments(
    output_dir=os.environ["MODEL_OUT"],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=20,
    save_total_limit=1,
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model(os.environ["MODEL_OUT"])
PY
