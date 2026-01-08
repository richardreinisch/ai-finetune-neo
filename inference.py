
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

adapter_enabled = True

base_model = "models/unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
adapter_path = "export/Llama-3.2-3B-Instruct-bnb-4bit-neo-1767818204768"

print("🔧 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    dtype=torch.float16
)

if adapter_enabled:
    print("🔗 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model)

model.eval()

def chat(prompt: str, max_new_tokens=1024):

    input_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nDu bist ein hilfreicher Assistent.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


print("🧠 Chat started. Type your message and press Enter. Press Ctrl+C to exit.\n")

try:
    while True:
        user_input = input("You: ")
        if user_input.strip() == "exit":
            break
        response = chat(user_input)
        print("\n🧠 Assistant:", response, "\n")
except KeyboardInterrupt:
    print("\n\nChat ended by user (Ctrl+C). Goodbye!")
