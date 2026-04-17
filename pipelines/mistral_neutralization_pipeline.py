import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

import requests


API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")


def read_txt_files(input_dir: Path) -> Iterable[Tuple[Path, str]]:
    for p in sorted(input_dir.glob("**/*.txt")):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="cp1251", errors="ignore")
        yield p, normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    # collapse excessive spaces while preserving newlines
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\u00A0", " ", text)
    # replace multiple spaces but not across newlines
    text = re.sub(r"[ \f\v]+", " ", text)
    # collapse >2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_sentences(text: str) -> List[str]:
    # naive sentence split for Russian
    # keeps punctuation attached to sentence
    parts = re.split(r"(?<=[.!?…])\s+", text)
    # merge very short fragments with neighbors
    merged: List[str] = []
    buf = []
    for part in parts:
        if not part:
            continue
        buf.append(part)
        if sum(len(x.split()) for x in buf) >= 20:
            merged.append(" ".join(buf))
            buf = []
    if buf:
        merged.append(" ".join(buf))
    return merged


def chunk_text_by_words(text: str, target_words: int = 512, tolerance: float = 0.25) -> List[str]:
    sentences = split_by_sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0
    lo = int(target_words * (1 - tolerance))
    hi = int(target_words * (1 + tolerance))

    for s in sentences:
        w = len(s.split())
        if cur_words + w <= hi:
            cur.append(s)
            cur_words += w
        else:
            if cur and (lo <= cur_words <= hi):
                chunks.append(" ".join(cur).strip())
                cur = [s]
                cur_words = w
            else:
                # force cut if current still too small
                if cur:
                    chunks.append(" ".join(cur).strip())
                cur = [s]
                cur_words = w
    if cur:
        chunks.append(" ".join(cur).strip())

    # drop empty or extremely short
    chunks = [c for c in chunks if len(c.split()) >= max(50, int(0.1 * target_words))]
    return chunks


def mistral_system_prompt() -> str:
    return (
        "Ты — экспертный русскоязычный редактор и сегментатор текста. "
        "Твоя задача: (1) разделить входной фрагмент на 2–4 связных смысловых блока, "
        "исключая структурный шум (например, номера глав, заголовки, оглавление, эпиграфы), "
        "(2) для каждого блока создать нейтральный, современный и деловой пересказ на русском языке, "
        "точно сохраняя факты, персонажей, хронологию и причинно-следственные связи. "
        "Ничего не выдумывай, не добавляй внешние знания и интерпретации. "
        "Нейтральный текст должен быть немного короче исходного (ориентир 0.7–0.95 от длины), "
        "убирая лишние эпитеты и пафос, но НЕ теряя смысл. "
        "Верни строго валидный JSON без Markdown и комментариев по схеме: "
        "{\"blocks\": [{\"block_id\": <int, начиная с 1>, \"original_block\": <string>, \"neutral_block\": <string>}, ...]}"
    )


def build_messages(chunk_text: str) -> List[Dict[str, str]]:
    sys = mistral_system_prompt()
    user = (
        "Раздели и нейтрализуй следующий текст. Требования: 2–4 блока, исключить структурный шум, "
        "сохранить факты, нейтральный стиль короче оригинала, ответ строго JSON.\n\n"
        f"Текст:\n{chunk_text}"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def call_mistral(api_key: str, messages: List[Dict[str, str]], model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, max_tokens: int = 2048, retries: int = 5, timeout: int = 60) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    backoff = 2.0
    for attempt in range(retries):
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                raise RuntimeError(f"Unexpected Mistral response format: {data}")
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        raise RuntimeError(
            f"Mistral API error {resp.status_code}: {resp.text}\nPayload: {json.dumps(payload)[:500]}"
        )
    raise TimeoutError("Mistral API retries exceeded")


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Remove common markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text.strip())
        text = text.strip().rstrip("`")
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first top-level JSON object
    brace_stack = []
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if not brace_stack:
                start = i
            brace_stack.append(ch)
        elif ch == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start != -1:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
    raise ValueError("Failed to parse JSON from model output")


def process_chunk(api_key: str, chunk_text: str, model: str) -> List[Dict[str, Any]]:
    messages = build_messages(chunk_text)
    raw = call_mistral(api_key=api_key, messages=messages, model=model)
    data = extract_json(raw)
    if not isinstance(data, dict) or "blocks" not in data or not isinstance(data["blocks"], list):
        raise ValueError(f"Bad JSON schema from model: {data}")

    results: List[Dict[str, Any]] = []
    for i, blk in enumerate(data["blocks"], start=1):
        original = (blk.get("original_block") or "").strip()
        neutral = (blk.get("neutral_block") or "").strip()
        if not original or not neutral:
            continue
        results.append({
            "block_id": int(blk.get("block_id", i)),
            "original_block": original,
            "neutral_block": neutral,
        })
    return results


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build parallel neutralized corpus via Mistral API")
    parser.add_argument("--input-dir", type=str, default="corpus-main", help="Directory with .txt sources")
    parser.add_argument("--output", type=str, default="outputs/parallel_corpus.jsonl", help="Output JSONL path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Mistral model name")
    parser.add_argument("--words-per-chunk", type=int, default=512, help="Target words per input chunk")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of input files (0 = no limit)")
    args = parser.parse_args()

    api_key = "FqIAZ7TBNqUomIhDd2UDEoUhUAgptHCM"

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    file_count = 0
    for file_path, text in read_txt_files(input_dir):
        if args.max_files and file_count >= args.max_files:
            break
        file_count += 1
        print(f"Processing {file_path}…")
        chunks = chunk_text_by_words(text, target_words=args.words_per_chunk)
        print(f"  → {len(chunks)} chunks")

        for ci, chunk in enumerate(chunks, start=1):
            # simple retry around chunk processing
            for attempt in range(3):
                try:
                    blocks = process_chunk(api_key=api_key, chunk_text=chunk, model=args.model)
                    rows = [{
                        "source_file": str(file_path),
                        "chunk_index": ci,
                        "block_index": b["block_id"],
                        "original_block": b["original_block"],
                        "neutral_block": b["neutral_block"],
                    } for b in blocks]
                    write_jsonl(output_path, rows)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"    Chunk {ci} failed (attempt {attempt+1}): {e} — retry in {wait}s")
                    time.sleep(wait)
            else:
                print(f"    Skipped chunk {ci} after retries")

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
