import os
import re
import json
from flask import Flask, render_template, request, redirect, url_for, flash
import requests

# Try import openai if available
try:
    import openai
except Exception:
    openai = None

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me-in-production")

def preprocess(text: str) -> dict:
    original = text.strip()
    lowered = original.lower()
    processed = re.sub(r"[^\w\s]", " ", lowered)
    processed = re.sub(r"\s+", " ", processed).strip()
    tokens = processed.split()
    return {"original": original, "processed": processed, "tokens": tokens}

def build_prompt(preprocessed: dict) -> str:
    return (
        "You are a helpful assistant. Answer the user question clearly and concisely.\n\n"
        f"Original: {preprocessed['original']}\n"
        f"Processed: {preprocessed['processed']}\n\nAnswer:"
    )

def query_openai(prompt: str, max_tokens=400):
    if openai is None:
        raise RuntimeError("openai package not installed.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return resp["choices"][0]["message"]["content"].strip()

def query_hf(prompt: str, model="facebook/opt-1.3b", max_length=256):
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
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    return json.dumps(data)

def send_to_llm(prompt: str):
    # OpenAI preferred
    try:
        if openai is not None and os.getenv("OPENAI_API_KEY"):
            return query_openai(prompt)
    except Exception as e:
        app.logger.warning(f"OpenAI call failed: {e}")
    try:
        return query_hf(prompt)
    except Exception as e:
        raise RuntimeError(f"LLM calls failed: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    processed = None
    answer = None
    question = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            flash("Please enter a question.", "warning")
            return redirect(url_for("index"))
        processed = preprocess(question)
        prompt = build_prompt(processed)
        try:
            answer = send_to_llm(prompt)
        except Exception as e:
            flash(f"LLM request failed: {e}", "danger")
            answer = None
    return render_template("index.html", question=question, processed=processed, answer=answer)

if __name__ == "__main__":
    # for quick local testing
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)