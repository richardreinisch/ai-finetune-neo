
import re
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama
import json

file_to_convert = "MYTH-Die_Macht_der_Mythen"

with open("./parsed/" + file_to_convert + ".md", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
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

def generate_qa(chunk: str, isGerman: bool = False):

    if isGerman:
        prompt = f"""The following text is in German. FIRST translate it accurately to natural English, THEN analyze for QA generation.
    
        GERMAN TEXT:
        {chunk}
    
        STEP 1: Translate to English (preserve meaning, tone, structure)
        STEP 2: Generate EXACTLY 5 QA pairs from the ENGLISH translation in Alpaca format.
    
        RESPONSE FORMAT - COPY EXACTLY, NO EXTRA TEXT:
        {{"english_translation": "Full English translation here", "qa_pairs": [
          {{"instruction": "Answer based on the context", "input": "Question 1?", "output": "Answer with quote."}},
          {{"instruction": "Answer based on the context", "input": "Question 2?", "output": "Answer with quote."}}
        ]}}
    
        Return ONLY valid JSON:"""

    else:

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

    response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])

    # print(response)

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

    return [{"instruction": "Answer based on context", "input": pair["input"], "output": pair["output"]} for pair in qa_pairs]


jsonl_data = [ ]

print(f"NUMBER OF CHUNKS: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print(f"CURRENT CHUNK: {index + 1} of {len(chunks)}\n")
    try:
        cleaned_chunk = clean_chunk(chunk)
        print(f"CHUNK: {cleaned_chunk}\n")
        prepared_chunk = generate_qa(cleaned_chunk, True)
        print(f"NEW QA: {prepared_chunk}\n")
        jsonl_data.extend(prepared_chunk)
    except:
        print(f"COULD NOT ANALYSE CHUNK!")

with open("./datasets/" + file_to_convert + ".jsonl", "w") as f:
    for item in jsonl_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
