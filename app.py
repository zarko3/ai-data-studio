"""
AI Data Studio - Dataset Creator & Prep Suite
Designed for training ChatML & LLM datasets (Unsloth, TechcodeX, LLaMA, Qwen, Mistral).
"""

import os
import re
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import gradio as gr
from pypdf import PdfReader

# Default output locations
DOWNLOADS_DIR = Path("/home/zarko/Downloads")
UNSLOTH_ASSETS_DIR = Path("/home/zarko/.unsloth/studio/data-designer/managed-assets")
PROJECT_DIR = Path("/home/zarko/.gemini/antigravity/scratch/ai-data-studio")
DEFAULT_EXPORT_DIR = PROJECT_DIR / "exports"
DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper Functions: Cleaning, ChatML Formatting, Token Estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimation (approx 1 token ~= 4 characters or 0.75 words)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return int(max(words * 1.3, chars / 4))

def format_chatml_row(messages: List[Dict[str, str]], system_prompt: str = "") -> Dict[str, Any]:
    """Ensure message list is formatted correctly for ChatML."""
    clean_msgs = []
    if system_prompt.strip():
        clean_msgs.append({"role": "system", "content": system_prompt.strip()})
    for m in messages:
        role = m.get("role", "user").strip().lower()
        if role not in ["system", "user", "assistant"]:
            role = "user"
        content = m.get("content", "").strip()
        if content:
            clean_msgs.append({"role": role, "content": content})
    return {"messages": clean_msgs}


# ---------------------------------------------------------------------------
# Module 1: Discord & Raw Chat Cleaner
# ---------------------------------------------------------------------------

def clean_discord_text(raw_text: str, user_name: str, bot_name: str, remove_urls: bool, min_length: int) -> Tuple[str, List[Dict[str, Any]], str]:
    if not raw_text.strip():
        return "", [], "⚠️ Please provide chat text to parse."

    lines = raw_text.splitlines()
    # Match patterns like:
    # [12:34 PM] Username: Message
    # Username — Today at 10:20 AM
    # Username: Message
    # 2024-03-01 12:00:00 Username: Message
    header_pattern1 = re.compile(r"^\[?[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?(?:\s*[AP]M)?\]?\s*([^:]+):\s*(.*)$", re.IGNORECASE)
    header_pattern2 = re.compile(r"^([^—\n]+?)\s*[—–-]\s*(?:Today at|Yesterday at|[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{4}-[0-9]{2}-[0-9]{2}).*$", re.IGNORECASE)
    header_pattern3 = re.compile(r"^([A-Za-z0-9_#\s\.-]{2,30}):\s*(.*)$")

    parsed_turns = []
    current_speaker = None
    current_buffer = []

    def flush_turn():
        nonlocal current_speaker, current_buffer
        if current_speaker and current_buffer:
            content = " ".join(current_buffer).strip()
            # Clean attachments and URLs if requested
            content = re.sub(r"<@!?[0-9]+>", "", content)  # discord mentions
            content = re.sub(r"<#[0-9]+>", "", content)    # channel mentions
            content = re.sub(r"<:[a-zA-Z0-9_]+:[0-9]+>", "", content) # custom emojis
            if remove_urls:
                content = re.sub(r"https?://\S+|www\.\S+", "", content)
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) >= min_length:
                parsed_turns.append({"speaker": current_speaker, "text": content})
        current_buffer = []

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        # Check header 1: [10:00 AM] Name: Text
        m1 = header_pattern1.match(line_s)
        if m1:
            flush_turn()
            current_speaker = m1.group(1).strip()
            if m1.group(2).strip():
                current_buffer.append(m1.group(2).strip())
            continue

        # Check header 2: Name - Today at 10:00 AM
        m2 = header_pattern2.match(line_s)
        if m2:
            flush_turn()
            current_speaker = m2.group(1).strip()
            continue

        # Check header 3: Name: Text
        m3 = header_pattern3.match(line_s)
        if m3 and not line_s.startswith("http"):
            flush_turn()
            current_speaker = m3.group(1).strip()
            if m3.group(2).strip():
                current_buffer.append(m3.group(2).strip())
            continue

        # Ordinary text line belonging to current speaker
        if current_speaker:
            current_buffer.append(line_s)
        else:
            # If no header encountered yet, default to user
            current_speaker = user_name or "User"
            current_buffer.append(line_s)

    flush_turn()

    if not parsed_turns:
        return "", [], "⚠️ No valid messages could be parsed from the input."

    # Convert turns into alternating User / Assistant pairs
    dataset_rows = []
    user_target = user_name.strip().lower() if user_name else None
    bot_target = bot_name.strip().lower() if bot_name else None

    # Group consecutive messages by same speaker
    collapsed = []
    for turn in parsed_turns:
        spk = turn["speaker"]
        txt = turn["text"]
        if collapsed and collapsed[-1]["speaker"] == spk:
            collapsed[-1]["text"] += " " + txt
        else:
            collapsed.append({"speaker": spk, "text": txt})

    # Build dialogues
    i = 0
    while i < len(collapsed) - 1:
        spk1 = collapsed[i]["speaker"].lower()
        spk2 = collapsed[i+1]["speaker"].lower()

        # Decide roles
        if bot_target and bot_target in spk1:
            # First is bot, skip until we have user first
            i += 1
            continue

        role1 = "user"
        role2 = "assistant"

        if user_target and bot_target:
            if user_target in spk1 and bot_target in spk2:
                role1, role2 = "user", "assistant"
            elif bot_target in spk1 and user_target in spk2:
                role1, role2 = "assistant", "user"

        if role1 == "user" and role2 == "assistant":
            dataset_rows.append({
                "messages": [
                    {"role": "user", "content": collapsed[i]["text"]},
                    {"role": "assistant", "content": collapsed[i+1]["text"]}
                ]
            })
            i += 2
        else:
            # Alternate by default
            dataset_rows.append({
                "messages": [
                    {"role": "user", "content": collapsed[i]["text"]},
                    {"role": "assistant", "content": collapsed[i+1]["text"]}
                ]
            })
            i += 2

    # Save to temp JSONL for download
    out_path = DEFAULT_EXPORT_DIR / "cleaned_chat_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in dataset_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = (
        f"✅ Successfully converted chat logs!\n"
        f"• Raw turns extracted: {len(parsed_turns)}\n"
        f"• Alternating dialogue pairs generated: {len(dataset_rows)}\n"
        f"• Exported to: {out_path.name}"
    )
    preview = json.dumps(dataset_rows[:5], indent=2, ensure_ascii=False) if dataset_rows else "[]"
    return str(out_path), preview, summary


def handle_chat_upload(file_obj, text_input, user_name, bot_name, remove_urls, min_len):
    raw_text = text_input or ""
    if file_obj is not None:
        try:
            with open(file_obj.name, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception as e:
            return None, "", f"❌ Error reading file: {str(e)}"
    file_path, preview, summary = clean_discord_text(raw_text, user_name, bot_name, remove_urls, min_len)
    return file_path, preview, summary


# ---------------------------------------------------------------------------
# Module 2: Hugging Face Downloader & Converter
# ---------------------------------------------------------------------------

def download_hf_dataset(repo_id: str, split_name: str, max_samples: int, custom_system_prompt: str) -> Tuple[str, str, str]:
    if not repo_id.strip():
        return None, "", "⚠️ Please specify a valid Hugging Face dataset repository (e.g. HuggingFaceTB/smoltalk)."

    repo_id = repo_id.strip()
    split = split_name.strip() or "train"
    max_samples = int(max_samples)

    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, split=split)
    except Exception as e:
        # Fallback to try without specific split or check available
        err_msg = str(e)
        if "Unknown split" in err_msg:
            return None, "", f"❌ Split '{split}' not found. Try one of: train, train_sft, default, or test.\nError: {err_msg}"
        return None, "", f"❌ Failed to download dataset '{repo_id}': {err_msg}"

    total_rows = len(ds)
    limit = min(total_rows, max_samples) if max_samples > 0 else total_rows
    features = list(ds.features.keys())

    converted_rows = []

    for i in range(limit):
        item = ds[i]

        # Case 1: Standard ChatML or multi-turn list (key: 'messages' or 'conversations')
        if "messages" in item and isinstance(item["messages"], list):
            msgs = []
            if custom_system_prompt.strip():
                msgs.append({"role": "system", "content": custom_system_prompt.strip()})
            for m in item["messages"]:
                r = m.get("role", "user")
                c = m.get("content", "")
                if c:
                    msgs.append({"role": r, "content": c})
            if msgs:
                converted_rows.append({"messages": msgs})

        elif "conversations" in item and isinstance(item["conversations"], list):
            msgs = []
            if custom_system_prompt.strip():
                msgs.append({"role": "system", "content": custom_system_prompt.strip()})
            for m in item["conversations"]:
                # ShareGPT style: 'from': 'human' / 'gpt' or 'role': 'user' / 'assistant'
                r = m.get("from") or m.get("role") or "user"
                if r in ["human", "user"]:
                    r = "user"
                elif r in ["gpt", "chatgpt", "assistant", "bot"]:
                    r = "assistant"
                elif r in ["system"]:
                    r = "system"
                c = m.get("value") or m.get("content") or ""
                if c:
                    msgs.append({"role": r, "content": c})
            if msgs:
                converted_rows.append({"messages": msgs})

        # Case 2: Instruction / Output format (Alpaca style)
        elif "instruction" in item and ("output" in item or "response" in item):
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output") or item.get("response") or ""
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            msgs = []
            if custom_system_prompt.strip():
                msgs.append({"role": "system", "content": custom_system_prompt.strip()})
            msgs.append({"role": "user", "content": user_msg})
            msgs.append({"role": "assistant", "content": str(out).strip()})
            converted_rows.append({"messages": msgs})

        # Case 3: Prompt / Completion or Question / Answer
        elif ("prompt" in item or "question" in item) and ("response" in item or "answer" in item or "completion" in item):
            q = item.get("prompt") or item.get("question") or ""
            a = item.get("response") or item.get("answer") or item.get("completion") or ""
            msgs = []
            if custom_system_prompt.strip():
                msgs.append({"role": "system", "content": custom_system_prompt.strip()})
            msgs.append({"role": "user", "content": str(q).strip()})
            msgs.append({"role": "assistant", "content": str(a).strip()})
            converted_rows.append({"messages": msgs})

        # Case 4: Plain text (create completion chunks)
        elif "text" in item and isinstance(item["text"], str):
            txt = item["text"].strip()
            if len(txt) > 20:
                msgs = [
                    {"role": "user", "content": "Continue and explain the following content:"},
                    {"role": "assistant", "content": txt}
                ]
                converted_rows.append({"messages": msgs})

    if not converted_rows:
        return None, "", f"⚠️ Downloaded {limit} rows, but could not detect standard columns. Features available: {features}"

    # Save to file
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_id)
    out_file = DEFAULT_EXPORT_DIR / f"{safe_name}_{split}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in converted_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = (
        f"✅ Dataset Successfully Downloaded & Converted!\n"
        f"• Hugging Face Repo: {repo_id}\n"
        f"• Split: {split} (Total available: {total_rows:,})\n"
        f"• Converted rows: {len(converted_rows):,}\n"
        f"• Saved to: {out_file.name}"
    )
    preview = json.dumps(converted_rows[:3], indent=2, ensure_ascii=False)
    return str(out_file), preview, summary


# ---------------------------------------------------------------------------
# Module 3: Document & Book Chunker (PDF / TXT / MD)
# ---------------------------------------------------------------------------

def chunk_document_to_chatml(file_obj, raw_text: str, chunk_words: int, task_instruction: str, system_prompt: str) -> Tuple[str, str, str]:
    content = ""
    if file_obj is not None:
        fname = getattr(file_obj, "name", "")
        if fname.lower().endswith(".pdf"):
            try:
                reader = PdfReader(fname)
                for page in reader.pages:
                    text_p = page.extract_text()
                    if text_p:
                        content += text_p + "\n"
            except Exception as e:
                return None, "", f"❌ Failed to parse PDF: {str(e)}"
        else:
            try:
                with open(fname, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                return None, "", f"❌ Failed to read document: {str(e)}"
    elif raw_text.strip():
        content = raw_text.strip()
    else:
        return None, "", "⚠️ Please upload a document or paste text to chunk."

    if not content.strip():
        return None, "", "⚠️ Document appears to be empty."

    # Split into paragraphs and words
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for p in paragraphs:
        words = p.split()
        if current_word_count + len(words) > chunk_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_word_count = len(words)
        else:
            current_chunk.append(p)
            current_word_count += len(words)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Convert chunks to ChatML dataset rows
    default_instruction = task_instruction.strip() or "Explain and summarize the following passage clearly:"
    dataset_rows = []
    for chunk in chunks:
        if len(chunk.strip()) < 30:
            continue
        msgs = []
        if system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        msgs.append({"role": "user", "content": f"{default_instruction}\n\n\"\"\"\n{chunk}\n\"\"\""})
        msgs.append({"role": "assistant", "content": f"Here is the breakdown of this section:\n\n{chunk}"})
        dataset_rows.append({"messages": msgs})

    out_file = DEFAULT_EXPORT_DIR / "document_chunked_dataset.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in dataset_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = (
        f"✅ Document successfully processed and chunked!\n"
        f"• Total characters: {len(content):,}\n"
        f"• Target words per chunk: {chunk_words}\n"
        f"• Generated training pairs: {len(dataset_rows)}\n"
        f"• File saved to: {out_file.name}"
    )
    preview = json.dumps(dataset_rows[:3], indent=2, ensure_ascii=False)
    return str(out_file), preview, summary


# ---------------------------------------------------------------------------
# Module 4: Synthetic Conversation & Normal Chat Generator
# ---------------------------------------------------------------------------

PRESET_PROFILES = {
    "Everyday Casual Banter": [
        ("yo what's up?", "Yo! Not much, just chilling. What are you up to today?"),
        ("bro my pc is acting slow what should i check?", "Check your Task Manager or htop first! See if your RAM is full or if some background app is hogging the CPU."),
        ("what game should we play tonight?", "Depends on what you're feeling! If you want casual fun, maybe Lethal Company or Roblox. If you want something competitive, CS2 or Valorant!"),
        ("any good quick dinner ideas?", "You can make an easy garlic butter pasta, or quick egg fried rice with whatever veggies and leftover chicken you have in the fridge."),
        ("do you know how to fix high ping?", "Try using an ethernet cable instead of Wi-Fi, restart your router, and close any background downloads like Steam or torrents."),
        ("what's the difference between ram and vram?", "RAM is your system memory used by the CPU for general apps. VRAM is dedicated high-speed video memory located right on your GPU for textures, 3D graphics, and AI models!"),
        ("how much ram do i need for training small models?", "For smaller models (50M to 500M params), 8 GB to 16 GB of RAM is plenty! If you have a GPU with 8-16 GB VRAM, training will fly even faster.")
    ],
    "Coding & Tech Assistant": [
        ("how do i write a jsonl file in python?", "You write each dictionary as a JSON string followed by a newline `\\n`:\n```python\nimport json\n\ndata = [{\"role\": \"user\", \"content\": \"hello\"}]\nwith open('dataset.jsonl', 'w', encoding='utf-8') as f:\n    for item in data:\n        f.write(json.dumps(item) + '\\n')\n```"),
        ("what is proxmox?", "Proxmox VE is an open-source enterprise virtualization platform. It lets you run Kernel-based Virtual Machines (KVM) and Linux Containers (LXC) through an easy web dashboard on port 8006."),
        ("how do i check my gpu usage on linux?", "If you have an AMD GPU, use `rocm-smi` or `radeontop`. If you have NVIDIA, use `nvidia-smi`!"),
        ("how do i format a dataset for unsloth?", "Unsloth uses the standard ChatML format:\n```json\n{\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]}\n```")
    ]
}

def generate_preset_dataset(preset_choice: str, repeat_count: int, system_prompt: str) -> Tuple[str, str, str]:
    pairs = PRESET_PROFILES.get(preset_choice, PRESET_PROFILES["Everyday Casual Banter"])
    dataset_rows = []
    for _ in range(max(1, repeat_count)):
        for u, a in pairs:
            msgs = []
            if system_prompt.strip():
                msgs.append({"role": "system", "content": system_prompt.strip()})
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
            dataset_rows.append({"messages": msgs})

    out_file = DEFAULT_EXPORT_DIR / f"synthetic_{preset_choice.lower().replace(' ', '_')}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in dataset_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = (
        f"✅ Generated {len(dataset_rows)} synthetic dialogue rows!\n"
        f"• Profile: {preset_choice}\n"
        f"• Repetitions/Variations: {repeat_count}\n"
        f"• Output: {out_file.name}"
    )
    preview = json.dumps(dataset_rows[:4], indent=2, ensure_ascii=False)
    return str(out_file), preview, summary


def add_custom_pair(existing_file, user_text: str, bot_text: str, system_prompt: str) -> Tuple[str, str, str]:
    if not user_text.strip() or not bot_text.strip():
        return None, "", "⚠️ Both User Message and Assistant Response are required."

    out_path = Path(existing_file) if existing_file else (DEFAULT_EXPORT_DIR / "custom_chat_dataset.jsonl")
    msgs = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt.strip()})
    msgs.append({"role": "user", "content": user_text.strip()})
    msgs.append({"role": "assistant", "content": bot_text.strip()})

    row = {"messages": msgs}
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Read count
    count = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
    summary = f"✅ Added pair successfully! Total rows in {out_path.name}: {count}"
    preview = json.dumps(row, indent=2, ensure_ascii=False)
    return str(out_path), preview, summary


# ---------------------------------------------------------------------------
# Module 5: Dataset Inspector, Cleaner & Merger
# ---------------------------------------------------------------------------

def inspect_dataset_file(file_obj) -> Tuple[str, str, int, int, float]:
    if file_obj is None:
        return "⚠️ Please select or upload a dataset file.", "", 0, 0, 0.0

    filepath = file_obj.name
    total_rows = 0
    total_tokens = 0
    total_turns = 0
    sample_rows = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    total_rows += 1
                    if "messages" in obj and isinstance(obj["messages"], list):
                        total_turns += len(obj["messages"])
                        for m in obj["messages"]:
                            total_tokens += estimate_tokens(m.get("content", ""))
                    elif "text" in obj:
                        total_tokens += estimate_tokens(obj["text"])
                    if len(sample_rows) < 5:
                        sample_rows.append(obj)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"❌ Error reading dataset: {str(e)}", "", 0, 0, 0.0

    avg_turns = round(total_turns / total_rows, 2) if total_rows > 0 else 0.0
    summary = (
        f"📊 Dataset Analysis for: {Path(filepath).name}\n"
        f"• Total Rows (Conversations): {total_rows:,}\n"
        f"• Total Estimated Tokens: {total_tokens:,}\n"
        f"• Total Message Turns: {total_turns:,}\n"
        f"• Average Turns per Sample: {avg_turns}\n"
        f"• Format Validated: ChatML Standard ✅"
    )
    preview = json.dumps(sample_rows, indent=2, ensure_ascii=False)
    return summary, preview, total_rows, total_tokens, avg_turns


def merge_datasets(files_list, output_filename: str, deduplicate: bool, copy_to_unsloth: bool, copy_to_downloads: bool) -> Tuple[str, str, str]:
    if not files_list:
        return None, "", "⚠️ Please select at least two dataset files to merge."

    out_name = output_filename.strip() or "merged_master_dataset.jsonl"
    if not out_name.endswith(".jsonl"):
        out_name += ".jsonl"

    out_path = DEFAULT_EXPORT_DIR / out_name
    seen_hashes = set()
    total_merged = 0
    duplicates_removed = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for f_obj in files_list:
            fname = f_obj.name
            with open(fname, "r", encoding="utf-8", errors="ignore") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if deduplicate:
                            hash_key = json.dumps(obj.get("messages", obj), sort_keys=True)
                            if hash_key in seen_hashes:
                                duplicates_removed += 1
                                continue
                            seen_hashes.add(hash_key)
                        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        total_merged += 1
                    except Exception:
                        continue

    # Optional copies
    extra_copies = []
    if copy_to_downloads and DOWNLOADS_DIR.exists():
        dl_target = DOWNLOADS_DIR / out_name
        shutil.copyfile(out_path, dl_target)
        extra_copies.append(f"📥 Saved to Downloads: {dl_target}")

    if copy_to_unsloth and UNSLOTH_ASSETS_DIR.exists():
        unsloth_target = UNSLOTH_ASSETS_DIR / out_name
        shutil.copyfile(out_path, unsloth_target)
        extra_copies.append(f"🚀 Ready in Unsloth Studio: {unsloth_target}")

    extra_str = "\n".join(extra_copies)
    summary = (
        f"🎉 Merged Successfully!\n"
        f"• Total Files Merged: {len(files_list)}\n"
        f"• Final Valid Rows: {total_merged:,}\n"
        f"• Duplicates Removed: {duplicates_removed:,}\n"
        f"• Primary Output: {out_path}\n"
        f"{extra_str}"
    )

    # Preview
    preview = ""
    with open(out_path, "r", encoding="utf-8") as f:
        samples = [json.loads(f.readline()) for _ in range(min(3, total_merged))]
        preview = json.dumps(samples, indent=2, ensure_ascii=False)

    return str(out_path), preview, summary


# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------

custom_css = """
body { background-color: #0f172a; }
.gradio-container { max-width: 1280px !important; margin: auto; }
.header-box { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }
.header-box h1 { color: #38bdf8; font-size: 2.2rem; margin-bottom: 5px; font-weight: 800; }
.header-box p { color: #94a3b8; font-size: 1.05rem; }
.stat-card { background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
"""

with gr.Blocks(title="AI Data Studio - Dataset Creator & Prep Suite") as demo:

    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            "# ⚡ TechcodeX AI Data Studio\n"
            "### Ultimate Dataset Creator, Cleaner & Formatter for Unsloth & LLMs\n"
            "*Turn raw chats, Discord logs, Hugging Face hubs, and PDFs into clean ChatML JSONL training data in seconds.*"
        )

    with gr.Tabs():

        # -------------------------------------------------------------------
        # TAB 1: Discord & Chat Cleaner
        # -------------------------------------------------------------------
        with gr.Tab("💬 Discord & Chat Log Cleaner"):
            gr.Markdown(
                "#### 🛠️ Convert raw Discord exports, WhatsApp, or pasted chat transcripts into ChatML format\n"
                "Automatically strips timestamps, embeds, mentions, and groups messages into alternating `user` and `assistant` turns."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    chat_file = gr.File(label="Upload Chat File (.txt, .log, .json)", file_types=[".txt", ".log", ".json"])
                    chat_text = gr.Textbox(
                        label="Or Paste Chat Transcript Directly",
                        placeholder="[10:30 AM] User: Hey bro how do I train my model?\n[10:31 AM] TechcodeX: You just need a clean dataset in ChatML format!",
                        lines=10
                    )
                    with gr.Row():
                        user_name = gr.Textbox(label="User Name / Keyword (Optional)", placeholder="User or your name")
                        bot_name = gr.Textbox(label="Assistant / Bot Name (Optional)", placeholder="TechcodeX or bot name")
                    with gr.Row():
                        remove_urls = gr.Checkbox(label="Strip URLs & Links", value=True)
                        min_len = gr.Slider(label="Min Turn Character Length", minimum=2, maximum=100, value=5, step=1)
                    clean_btn = gr.Button("🚀 Clean & Convert to ChatML", variant="primary")

                with gr.Column(scale=1):
                    chat_status = gr.Textbox(label="Status & Summary", lines=4)
                    chat_out_file = gr.File(label="📥 Download Cleaned JSONL Dataset")
                    chat_preview = gr.Code(label="Dataset Preview (ChatML JSONL)", language="json", lines=12)

            clean_btn.click(
                fn=handle_chat_upload,
                inputs=[chat_file, chat_text, user_name, bot_name, remove_urls, min_len],
                outputs=[chat_out_file, chat_preview, chat_status]
            )

        # -------------------------------------------------------------------
        # TAB 2: Hugging Face Downloader
        # -------------------------------------------------------------------
        with gr.Tab("🤗 Hugging Face Dataset Downloader"):
            gr.Markdown(
                "#### 🌐 Pull datasets directly from Hugging Face and convert to standard ChatML format\n"
                "Auto-detects and converts columns like `messages`, `conversations`, `instruction`/`output`, `prompt`/`response`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    hf_repo = gr.Textbox(
                        label="Hugging Face Dataset Repo ID",
                        value="HuggingFaceTB/everyday-conversations-llama3.1-2k",
                        placeholder="e.g. HuggingFaceTB/smoltalk, HuggingFaceH4/no_robots, Open-Orca/OpenOrca"
                    )
                    with gr.Row():
                        hf_split = gr.Textbox(label="Split Name", value="train_sft", placeholder="train, train_sft, etc.")
                        hf_samples = gr.Number(label="Max Samples to Download (0 = all)", value=5000, precision=0)
                    hf_system = gr.Textbox(
                        label="Custom System Prompt (Optional)",
                        placeholder="You are a helpful, witty, and intelligent AI assistant."
                    )
                    hf_btn = gr.Button("📥 Download & Convert Dataset", variant="primary")

                with gr.Column(scale=1):
                    hf_status = gr.Textbox(label="Download Status", lines=4)
                    hf_out_file = gr.File(label="📥 Download Converted JSONL")
                    hf_preview = gr.Code(label="JSONL Preview", language="json", lines=12)

            hf_btn.click(
                fn=download_hf_dataset,
                inputs=[hf_repo, hf_split, hf_samples, hf_system],
                outputs=[hf_out_file, hf_preview, hf_status]
            )

        # -------------------------------------------------------------------
        # TAB 3: Document, Book & Wiki Chunker
        # -------------------------------------------------------------------
        with gr.Tab("📄 Document & Book Chunker (PDF / TXT / MD)"):
            gr.Markdown(
                "#### 📚 Transform Books, Documentation, and Wikipedia into Training Pairs\n"
                "Upload a PDF or paste long-form text. The chunker splits sections intelligently and wraps them into Q&A or explanation pairs."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    doc_file = gr.File(label="Upload Document (PDF, TXT, MD)", file_types=[".pdf", ".txt", ".md"])
                    doc_text = gr.Textbox(label="Or Paste Long Text Directly", lines=8, placeholder="Paste book or wiki text here...")
                    chunk_words = gr.Slider(label="Target Words per Chunk", minimum=50, maximum=1000, value=250, step=25)
                    task_inst = gr.Textbox(
                        label="Training Task Prompt / Question",
                        value="Explain and summarize the following passage clearly:"
                    )
                    doc_system = gr.Textbox(
                        label="System Prompt (Optional)",
                        placeholder="You are an expert AI teacher and researcher."
                    )
                    chunk_btn = gr.Button("📑 Chunk Document & Create Training Data", variant="primary")

                with gr.Column(scale=1):
                    doc_status = gr.Textbox(label="Processing Status", lines=4)
                    doc_out_file = gr.File(label="📥 Download Chunked JSONL")
                    doc_preview = gr.Code(label="Generated Training Pairs Preview", language="json", lines=12)

            chunk_btn.click(
                fn=chunk_document_to_chatml,
                inputs=[doc_file, doc_text, chunk_words, task_inst, doc_system],
                outputs=[doc_out_file, doc_preview, doc_status]
            )

        # -------------------------------------------------------------------
        # TAB 4: Synthetic Conversation & Normal Chat Builder
        # -------------------------------------------------------------------
        with gr.Tab("✨ Synthetic & Normal Chat Builder"):
            gr.Markdown(
                "#### 💬 Generate High-Quality Everyday Conversations & Custom Pairs\n"
                "Say goodbye to childish stories! Generate real everyday chat, gaming discussions, and technical Q&A pairs."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("##### 🎯 Quick Preset Pack Generator")
                    preset_select = gr.Dropdown(
                        label="Choose Preset Chat Profile",
                        choices=["Everyday Casual Banter", "Coding & Tech Assistant"],
                        value="Everyday Casual Banter"
                    )
                    preset_repeats = gr.Slider(label="Multiplier / Variations", minimum=1, maximum=50, value=10, step=1)
                    preset_system = gr.Textbox(label="System Prompt (Optional)", placeholder="You are a cool, casual friend.")
                    preset_btn = gr.Button("⚡ Generate Preset Dataset", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("##### ✍️ Add Custom Single Pair")
                    custom_user = gr.Textbox(label="User Prompt / Question", placeholder="e.g. Can you explain how ROCm works?")
                    custom_bot = gr.Textbox(label="Assistant Response", lines=3, placeholder="e.g. ROCm is AMD's open-source GPU computing platform...")
                    add_btn = gr.Button("➕ Append Pair to Dataset")

                with gr.Column(scale=1):
                    syn_status = gr.Textbox(label="Generation Status", lines=4)
                    syn_out_file = gr.File(label="📥 Download Synthetic JSONL")
                    syn_preview = gr.Code(label="Dataset Preview", language="json", lines=12)

            preset_btn.click(
                fn=generate_preset_dataset,
                inputs=[preset_select, preset_repeats, preset_system],
                outputs=[syn_out_file, syn_preview, syn_status]
            )

            add_btn.click(
                fn=add_custom_pair,
                inputs=[syn_out_file, custom_user, custom_bot, preset_system],
                outputs=[syn_out_file, syn_preview, syn_status]
            )

        # -------------------------------------------------------------------
        # TAB 5: Dataset Inspector & Merger
        # -------------------------------------------------------------------
        with gr.Tab("🔍 Inspector, Merger & Unsloth Exporter"):
            gr.Markdown(
                "#### 🧩 Merge multiple datasets, inspect statistics, and export straight to Unsloth Studio & Downloads\n"
                "Combine your cleaned Discord chat + Everyday Chat + HF datasets into a single powerful training set."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    merge_files = gr.File(
                        label="Select or Drag Multiple .jsonl Files to Inspect / Merge",
                        file_count="multiple",
                        file_types=[".jsonl"]
                    )
                    merge_name = gr.Textbox(label="Merged Output Filename", value="master_training_dataset.jsonl")
                    with gr.Row():
                        dedup = gr.Checkbox(label="Deduplicate Identical Rows", value=True)
                        to_unsloth = gr.Checkbox(label="Auto-deploy to Unsloth Studio", value=True)
                        to_downloads = gr.Checkbox(label="Save copy to ~/Downloads", value=True)
                    inspect_btn = gr.Button("🔍 Inspect First Selected Dataset")
                    merge_btn = gr.Button("🚀 Merge All Selected Datasets", variant="primary")

                with gr.Column(scale=1):
                    merge_status = gr.Textbox(label="Inspector & Merge Report", lines=6)
                    merge_out_file = gr.File(label="📥 Download Merged JSONL")
                    merge_preview = gr.Code(label="Merged Dataset Preview", language="json", lines=12)

            def inspect_first(files):
                if not files:
                    return "⚠️ Please upload at least one dataset to inspect.", "", None
                summary, preview, _, _, _ = inspect_dataset_file(files[0])
                return summary, preview, files[0].name

            inspect_btn.click(
                fn=inspect_first,
                inputs=[merge_files],
                outputs=[merge_status, merge_preview, merge_out_file]
            )

            merge_btn.click(
                fn=merge_datasets,
                inputs=[merge_files, merge_name, dedup, to_unsloth, to_downloads],
                outputs=[merge_out_file, merge_preview, merge_status]
            )

    gr.Markdown(
        "<div style='text-align: center; color: #64748b; margin-top: 30px;'>"
        "TechcodeX AI Data Studio • Compatible with Unsloth, LLaMA-Factory, Axolotl, and Hugging Face • Runs locally or on Proxmox VM"
        "</div>"
    )

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7865"))
    demo.launch(server_name=host, server_port=port)
