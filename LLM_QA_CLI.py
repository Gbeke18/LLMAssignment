import os
import re
import json
import sys
import time

# Try optional imports
try:
    import openai
except Exception:
    openai = None

import requests

# Basic preprocessing
def preprocess(text: str) -> dict:
    """
    Returns a dict with:
      - original: original string
      - processed: lowercased, punctuation removed
      - tokens: whitespace tokens
    """
    original = text.strip()
    lowered = original.lower()
    # remove punctuation except question mark (optionally)
    processed = re.sub(r"[^\w\s]", " ", lowered)
    # collapse whitespace
    processed = re.sub(r"\s+", " ", processed).strip()
    tokens = processed.split()
    return {"original": original, "processed": processed, "tokens": tokens}

# Build a prompt for the LLM
def build_prompt(preprocessed: dict) -> str:
    prompt = (
        "You are an assistant that answers user questions concisely and accurately.\n\n"
        f"User question (original): {preprocessed['original']}\n"
        f"User question (processed): {preprocessed['processed']}\n\n"
        "Answer the user's question directly. If the question is ambiguous, ask one brief clarifying "
        "question. Keep the answer clear and show one short paragraph. If you must give step-by-step, "
        "number them.\n\nAnswer:"
    )
    return prompt

# Send to OpenAI (ChatCompletion) if possible
def query_openai(prompt: str, max_tokens=400):
    if openai is None:
        raise RuntimeError("openai package not installed.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    openai.api_key = key
    # Use ChatCompletion if available
    try:
        # Try ChatCompletions (gpt-4/3.5) style
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list()["data"][0]["id"] else "gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        # Extract content
        if isinstance(resp, dict):
            # safe extraction
            content = resp["choices"][0]["message"]["content"]
        else:
            content = str(resp)
    except Exception as exc:
        # fallback to Completion if ChatCompletion fails
        try:
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=max_tokens, temperature=0.2)
            content = resp["choices"][0]["text"]
        except Exception as e2:
            raise RuntimeError(f"OpenAI request failed: {exc} / {e2}")
    return content.strip()

# Query Hugging Face Inference API
def query_hf(prompt: str, model="gpt2", max_length=256):
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        raise RuntimeError("HF_API_KEY not set.")
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_key}", "Accept": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_length, "temperature": 0.2}}
    res = requests.post(api_url, headers=headers, json=payload, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {res.status_code}: {res.text}")
    data = res.json()
    # Many HF models return [{'generated_text': '...'}]
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    # Other models return dict with 'generated_text'
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    # As fallback, stringify
    return json.dumps(data)[:1000]

def send_to_llm(prompt: str):
    # Try OpenAI first if openai installed and key present
    try:
        if openai is not None and os.getenv("OPENAI_API_KEY"):
            return query_openai(prompt)
    except Exception as e:
        print(f"[warning] OpenAI attempt failed: {e}", file=sys.stderr)
    # Try Hugging Face (fallback)
    try:
        # Use a reasonable HF model name; users can change to a better model in code
        return query_hf(prompt, model="facebook/opt-1.3b", max_length=400)
    except Exception as e:
        raise RuntimeError(f"No LLM available or all requests failed: {e}")

def main():
    print("=== LLM Q&A CLI ===")
    q = input("Enter your question: ").strip()
    if not q:
        print("No question entered. Exiting.")
        return
    pre = preprocess(q)
    print("\n[Processed question preview]")
    print("Processed:", pre["processed"])
    print("Tokens:", pre["tokens"])
    prompt = build_prompt(pre)
    print("\n[Sending to LLM...]\n")
    try:
        answer = send_to_llm(prompt)
    except Exception as exc:
        print("Error querying LLM:", exc)
        return
    print("=== LLM Answer ===")
    print(answer)
    print("==================")

if __name__ == "__main__":
    main()