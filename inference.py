
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


base_model = "models/unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
adapter_path = "export/Llama-3.2-3B--final-1767104809297"

print("🔧 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    dtype=torch.float16
)

print("🔗 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

model.eval()

def chat(prompt: str, max_new_tokens=300):
    input_text = f"""Du bist ein intelligenter Trainer namens Neo. 
    Du antwortest immer klar, wahrheitsgetreu und niemals halluzinierst du. Besonders magst du das Buch MYTH - Die Macht der Mythen. 
    Du wiederholst keinen user prompt, erfindest keine user prompts.
    
### User:
{prompt}

### Assistant:
"""

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded.split("### Assistant:")[-1].strip()


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
