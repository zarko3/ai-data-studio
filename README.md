# ⚡ AI Data Studio

> **Modern Dataset Creator, Cleaner & Formatter for LLMs (Unsloth, ChatML, LLaMA, Qwen, Mistral).**

AI Data Studio is a lightweight, all-in-one web GUI tool built for creating, cleaning, chunking, and preparing high-quality training datasets for Large Language Models.

---

## ✨ Features

- **💬 Discord & Chat Log Cleaner**:
  - Drag & drop Discord chat exports, WhatsApp transcripts, or raw pasted logs.
  - Automatically filters out usernames, timestamps (`[12:34 PM]`), embeds, mentions (`<@12345>`), and links.
  - Automatically reconstructs multi-party chat into clean alternating `user` and `assistant` ChatML turns.

- **🤗 Hugging Face Hub Downloader**:
  - Fetch datasets directly from Hugging Face Hub (e.g. `HuggingFaceTB/everyday-conversations-llama3.1-2k`, `HuggingFaceTB/smoltalk`).
  - Auto-converts various schema styles (`messages`, `conversations`, `instruction`/`output`, `prompt`/`response`) into standard ChatML JSONL.

- **📄 Document, Book & Wiki Chunker (PDF / TXT / MD)**:
  - Ingest PDFs, Markdown documents, and raw text files.
  - Intelligently splits content into semantic chunks and wraps them into Q&A or explanatory dialogue pairs.

- **✨ Synthetic Dialogue & Chat Builder**:
  - Pre-built profiles for everyday natural banter and coding/tech Q&A pairs (no childish stories or toddler word salads).
  - Add custom single conversation turns with one click.

- **🔍 Dataset Inspector, Merger & Exporter**:
  - Merge multiple `.jsonl` files into a single unified master training dataset.
  - Automatic deduplication and token count estimation.
  - Direct 1-click export to Unsloth Studio assets and Downloads.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ai-data-studio.git
cd ai-data-studio
```

### 2. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Studio
```bash
./run.sh
# or
python3 app.py
```
Open your browser and navigate to **`http://127.0.0.1:7865`**.

---

## 🛠️ Supported Formats

AI Data Studio outputs training-ready **ChatML JSONL**:
```json
{
  "messages": [
    {"role": "user", "content": "How do I format a dataset for training?"},
    {"role": "assistant", "content": "You format it as a JSONL file containing a list of role/content dictionaries."}
  ]
}
```

Directly compatible with:
- **Unsloth Studio & Unsloth SFT**
- **LLaMA-Factory**
- **Axolotl**
- **Hugging Face TRL / SFTTrainer**

---

## 📜 License
MIT License.
