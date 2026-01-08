
import re
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama
import json

file_to_convert = "Pride and Prejudice"

LLM = "deepseek-r1:7b"

START_CHUNK = 0

with open("./parsed/" + file_to_convert + ".md", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_text(text)

print(chunks)


def clean_chunk(chunk):
    """Clean text chunks: remove HTML tags, page markers, normalize whitespace"""

    # 1. Remove span tags and other HTML (BeautifulSoup - most reliable)
    soup = BeautifulSoup(chunk, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)

    # 2. Regex fallback for common artifacts like <span id="page-15-0"></span>
    clean_text = re.sub(r'<span[^>]*id="page-[^"]*"[^>]*>?</span>', '', clean_text)

    # 3. Remove multiple whitespace/newlines
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # 4. Fix line breaks for natural reading
    clean_text = re.sub(r'([.!?])\s*([A-ZÄÖÜ])', r'\1\n\n\2', clean_text)

    # 5. Remove Markdown image syntax: ![alt](url) or ![](url)
    clean_text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', clean_text)  # Keeps alt text
    clean_text = re.sub(r'!\[\[[^\]]*\]\[[^\]]*\]\([^)]*\)', '', clean_text)  # Wiki-style

    # 6. Remove empty lines
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def generate_qa(chunk: str):

    prompt = f"""You are a QA-Generator for Fine-Tuning. Analyze this text chunk/chapter:

    {chunk}

    Generate EXACTLY 5 QA pairs in Alpaca format as valid JSON array in american english language. Questions fact-based, answers directly quote the text. Vary questions (What? Why? How?). No hallucinations.

    RESPONSE FORMAT - COPY EXACTLY, NO EXTRA TEXT:
    [
      {{"instruction": "Answer based on the context", "input": "Your question?", "output": "Answer with text quote."}},
      {{"instruction": "Answer based on the context", "input": "Question 2?", "output": "Answer 2 with quote."}}
    ]

    Return ONLY the JSON array:"""

    # print(prompt)

    response = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}])

    print(f"RESPONSE: {response}")

    content = response['message']['content'].strip()
    if content.startswith('```json'):
        content = content.split('```json').split('```')
    elif content.startswith('['):
        pass
    else:
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            content = content[start:end]

    try:
        qa_pairs = json.loads(content)
    except:
        print(f"NOT A VALID JSON: {content}")
        raise

    return [{"instruction": "Answer based on context", "input": pair["input"], "output": pair["output"]} for pair in
            qa_pairs]


print(f"NUMBER OF CHUNKS: {len(chunks)}")

with open("./datasets/" + file_to_convert + ".jsonl", "a") as f:
    for index, chunk in enumerate(chunks):
        if index >= START_CHUNK:

            print(f"CURRENT CHUNK: {index + 1} of {len(chunks)}\n")

            try:
                cleaned_chunk = clean_chunk(chunk)
                print(f"CHUNK: {cleaned_chunk}\n")
                prepared_chunk = generate_qa(cleaned_chunk)
                print(f"NEW QA: {prepared_chunk}\n")
                f.write(json.dumps(prepared_chunk, ensure_ascii=False) + "\n")
            except:
                print(f"COULD NOT ANALYSE CHUNK!")
