
import json

file_to_convert = "MYTH-Die_Macht_der_Mythen"

def formatting_func(data):
    if len(data['input']) > 0:
        text = f"<human>: {data['instruction']} {data['input']}\\n<bot>: {data['output']}"
    else:
        raise
    return text

to_return = ""

with open("./datasets/" + file_to_convert + ".jsonl", "r") as f:
    for line in f:
        line = line.strip()
        try:
            if line:
                interpreted_line = json.loads(line)
                converted_text = formatting_func(interpreted_line)
                formatted_text = "{\"text\": \"" + converted_text + "\",\"metadata\": {\"source\": \"" + file_to_convert + "\"}}"
                print(formatted_text)
                json_text = json.loads(formatted_text)
                to_return += json.dumps(json_text, ensure_ascii=False) + "\n"
        except:
            print(f"Could not convert: {line}")

with open("./datasets/" + file_to_convert + "_SFT.jsonl", "w") as f:
    f.write(to_return)


