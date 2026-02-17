
# AI Finetune Neo

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2013-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-12GB%20VRAM-success?logo=nvidia)
![Unsloth](https://img.shields.io/badge/Fine--Tuning-Unsloth-6E40C9)
![Ollama](https://img.shields.io/badge/QA%20Generation-Ollama-000000)
![License](https://img.shields.io/badge/License-Open%20Source-lightgrey)

**AI Finetune Neo** is a local LLM fine-tuning pipeline that turns PDFs into high-quality training data and fine-tunes modern language models efficiently on a **12 GB GPU** using **Unsloth**.

The project covers the full workflow:
PDF → structured text → QA generation → SFT dataset → fine-tuning → inference → optional GGUF export.

---

## Features

* 📄 Extract structured content from PDFs (Markdown)
* 🤖 Generate QA datasets automatically using an LLM (via Ollama)
* 🧠 Prepare datasets for Supervised Fine-Tuning (SFT)
* ⚡ Fine-tune modern LLMs efficiently with **Unsloth**
* 💬 Test results via a simple chat interface
* 🔄 Optional export to **GGUF** for use with Ollama / llama.cpp
* 💾 Optimized for limited VRAM (≈12 GB)

---

## Workflow Overview

1. **PDF → Markdown**
   Convert PDF documents into clean Markdown files.

2. **QA & JSONL Generation**
   Chunk, clean, and transform Markdown into QA pairs using an LLM.

3. **Dataset Preparation**
   Convert QA data into an SFT-compatible dataset format.

4. **Fine-Tuning**
   Train the model locally using Unsloth.

5. **Inference**
   Test the fine-tuned adapter using a simple chat script.

---

## Requirements

* Python 3.10+
* NVIDIA GPU with ~12 GB VRAM
* CUDA 13
* PyTorch (CUDA build)
* Ollama (for QA generation)

---

## Setup

### Install PyTorch (CUDA 13)

```bash
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

### Install Project Dependencies

```bash
pip3 install -r requirements.txt
```

---

## Models

This project uses **Unsloth** models optimized for fast and memory-efficient fine-tuning.

### Unsloth Model Collection

[https://huggingface.co/collections/unsloth/unsloth-dynamic-20-quants](https://huggingface.co/collections/unsloth/unsloth-dynamic-20-quants)

### Example Model

[https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit)

---

## PDF Extraction

PDF extraction is done using **marker**:

🔗 [https://github.com/datalab-to/marker](https://github.com/datalab-to/marker)

### Script

```bash
python3 pdf-extractor.py
```

> ⚠️ Note: Marker will download required models on first run.

---

## QA Generation (JSONL)

* QA pairs are generated using an LLM via **Ollama**
* Output is stored as JSONL
* Manual review and cleanup of generated QA data is **strongly recommended**

### Script

```bash
python3 qa-generator.py
```

---

## Dataset Preparation (SFT)

The generated QA data is converted into a format compatible with Hugging Face’s **SFT Trainer**.

### Script

```bash
python3 prepare-dataset.py
```

---

## Fine-Tuning with Unsloth

Fine-tuning is performed using Unsloth for significantly improved speed and reduced memory usage.

### Script

```bash
python3 finetune-using-unsloth.py
```

### References

* [https://unsloth.ai/](https://unsloth.ai/)
* [https://github.com/unslothai/unsloth#finetune-mistral-gemma-llama-2-5x-faster-with-70-less-memory](https://github.com/unslothai/unsloth#finetune-mistral-gemma-llama-2-5x-faster-with-70-less-memory)

---

## Inference / Testing

A simple interactive chat interface is provided to test the fine-tuned model.

### Script

```bash
python3 inference.py
```

---

## Convert to GGUF (Optional)

For usage with **Ollama** or **llama.cpp**, the fine-tuned model can be converted to **GGUF**.

### Tools

* [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
* [https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)

### Example Conversion

```bash
python3 convert_hf_to_gguf.py \
  ../ai-finetune-neo/models/unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --outfile ../ai-finetune-neo/models/unsloth/Llama-3.2-3B-Instruct-bnb-4bit/gguf \
  --outtype f16
```

---

## Example Output

Note: Many models have already been trained with Pride and Prejudice, i.e., other datasets are recommended.

```
You: Was gefällt dir besonders am Buch Pride and Prejudice?                                    

🧠 Assistant: Ein großartiges Buch! Ich bin ein KI, aber ich habe keine persönliche Vorliebe oder Vorurteile, aber ich kann dir sagen, warum "Pride and Prejudice" von Jane Austen so beliebt ist.

Das Buch ist ein Klassiker der englischen Literatur und hat viele Aspekte, die es zu einem unvergleichlichen Roman machen:

1. **Charaktere**: Die Figuren sind vielschichtig und komplex. Elizabeth Bennet, die Protagonistin, ist eine intelligente, unabhängige und humorvolle Frau, die sich langsam, aber sicher durch die Herausforderungen des Lebens wehrt. Der Anführer, Mr. Darcy, ist ein aristokratischer Ritter mit einem Hauch von Stärke und Würde, der sich langsam von Elizabeth begeistern lässt.
2. **Romantik und Komödie**: Das Buch mischt die romantische und komödiantische Elemente perfekt. Die Geschichte ist voller lustiger Momente, wie die scharfe Zunge von Mrs. Bennet und die ungeschickten Versuche von Mr. Collins, sich als guter Mann zu präsentieren.
3. **Soziale Kritik**: Austen sammelt in dem Buch scharfe Beobachtungen über die sozialen Normen und Konventionen der Zeit. Sie zeigt, wie die Gesellschaft von Frauen, die Bildung und die wirtschaftliche Stabilität bestimmen.
4. **Psychologie**: Die Charaktere sind tief und komplex, und Austen zeigt, wie sie sich entwickeln und sich verändern, wenn sie sich in Situationen begegnen, die sie herausfordern.
5. **Stil**: Austens Schreibstil ist einzigartig und einflussreich. Sie schreibt mit einer sanften, humorvollen und scharfen Sprache, die die Leser zum Lachen und zum Nachdenken bringt.

Insgesamt ist "Pride and Prejudice" ein Buch, das für seine einzigartige Mischung aus Romantik, Komödie, sozialer Kritik und psychologischer Tiefe bekannt ist. Es ist ein Buch, das Leser über 200 Jahre nach seiner Veröffentlichung immer noch begeistert und inspiriert.
```

---

## Legal Information

* This project uses open-source models and tools.
  Please refer to the respective licenses for compliance.
* Ensure compliance with applicable data protection and privacy regulations,
  especially when processing sensitive or personal data.
* The authors provide **no warranty** and assume **no liability** for any damage
  or issues arising from the use of this software.

---
