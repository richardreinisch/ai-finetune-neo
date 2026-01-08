
import time

from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


file_to_convert = "Pride and Prejudice"

model_name = "Llama-3.2-3B-Instruct-bnb-4bit"

model_path = f"./models/unsloth/{model_name}"

max_seq_length = 2048

print("✅ Load Dataset...")

def current_milli_time():
    return round(time.time() * 1000)

dataset_file = f"./datasets/{file_to_convert}_SFT.jsonl"
dataset = load_dataset("json", data_files = {"train" : dataset_file }, split = "train")

print("✅ Loading model and tokenizer...")

model, tokenizer = FastModel.from_pretrained(
    model_name = model_path,
    local_files_only = True,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4-bit quantization. False = 16-bit LoRA.
    load_in_8bit = False, # 8-bit quantization
    load_in_16bit = False, # [NEW!] 16-bit LoRA
    full_finetuning = False, # Use for full fine-tuning.
    # float32_mixed_precision = True, # False
    device_map = "auto",
    # low_cpu_mem_usage = True
    # token = "hf_...", # use one if using gated models
)

print("✅ Model patching...")

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print("✅ Create Trainer...")

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2, # 2
        gradient_accumulation_steps = 8, # 4
        learning_rate = 2e-5,
        num_train_epochs = 3,
        warmup_steps = 50,
        max_steps = 500,
        logging_steps = 5,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False
    ),
)
trainer.train()

print("💾 Saving the fine-tuned LoRA adapter...")

# Saves the LoRA adapter weights
folder = f"./export/{model_name}-neo-{current_milli_time()}"
print(f"Saved as: {folder}")

trainer.save_model(folder)

print("✅ Training & saving complete!")

# Go to https://docs.unsloth.ai for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM or SGLang
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Customized chat templates