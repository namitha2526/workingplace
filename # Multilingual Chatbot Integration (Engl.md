# Multilingual Chatbot Integration (English • Hindi • Punjabi)

Plug-and-play prototype using **Flask + Transformers** with a **translation sandwich**:

`User (hi/pa/en) → translate→ EN → small LLM (Flan‑T5) → translate→ user language`

> Zero paid APIs. Runs locally. Good for demos; *not* production-grade.

---

## 1) `requirements.txt`

```
flask==3.0.3
flask-cors==4.0.0
transformers==4.43.3
torch>=2.2.0
sentencepiece==0.2.0
accelerate>=0.33.0
```

> If running on CPU-only machine, PyTorch will default to CPU. Expect slower generation.

---

## 2) `app.py` (Flask backend)

```python
import os
from typing import Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)
CORS(app)

# --------- Config ---------
# Small instruct model for CPU-friendly demos; upgrade to flan-t5-base if you have RAM.
LLM_MODEL = os.environ.get("LLM_MODEL", "google/flan-t5-small")

# MarianMT translation models (English, Hindi, Punjabi)
MT_MODELS = {
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("pa", "en"): "Helsinki-NLP/opus-mt-pa-en",
    ("en", "pa"): "Helsinki-NLP/opus-mt-en-pa",
}

SUPPORTED_LANGS = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}

# --------- Load models once (cold start may take a minute) ---------
print("[BOOT] Loading LLM:", LLM_MODEL)
gen_pipeline = pipeline(
    "text2text-generation",
    model=LLM_MODEL,
    device_map="auto" if os.environ.get("DEVICE_AUTO", "0") == "1" else None,
)

# Cache translation pipelines per direction
_translation_pipes: Dict = {}

def get_translator(src: str, tgt: str):
    if src == tgt:
        return None
    key = (src, tgt)
    if key not in MT_MODELS:
        raise ValueError(f"Translation {src}->{tgt} not configured.")
    if key not in _translation_pipes:
        model_name = MT_MODELS[key]
        print(f"[BOOT] Loading MT: {model_name}")
        _translation_pipes[key] = pipeline("translation", model=model_name)
    return _translation_pipes[key]


def translate(text: str, src: str, tgt: str) -> str:
    if not text:
        return text
    if src == tgt:
        return text
    pipe = get_translator(src, tgt)
    out = pipe(text, max_length=512)
    return out[0]["translation_text"].strip()


def build_prompt(user_en: str) -> str:
    system = (
        "You are a safe, concise telemedicine assistant for rural clinics. "
        "Answer in plain language, bullet where helpful. "
        "Never give definitive diagnoses; recommend seeing a doctor for urgent or severe symptoms. "
        "If it looks like an emergency (chest pain, difficulty breathing, severe bleeding, loss of consciousness), "
        "advise immediate medical attention."
    )
    return f"{system}\n\nUser question: {user_en}\n\nHelpful answer:"


@app.get("/health")
def health():
    return {"status": "ok", "langs": SUPPORTED_LANGS}


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    user_text = (data.get("text") or "").strip()
    lang = (data.get("lang") or "en").lower()

    if lang not in SUPPORTED_LANGS:
        return jsonify({"error": f"Unsupported lang '{lang}'. Use one of: {list(SUPPORTED_LANGS)}"}), 400
    if not user_text:
        return jsonify({"error": "Empty text"}), 400

    try:
        # 1) Normalize to English
        text_en = translate(user_text, lang, "en") if lang != "en" else user_text

        # 2) Generate answer in English
        prompt = build_prompt(text_en)
        gen = gen_pipeline(prompt, max_new_tokens=256, temperature=0.4)
        answer_en = gen[0]["generated_text"].strip()

        # 3) Translate back to user language
        answer_out = translate(answer_en, "en", lang) if lang != "en" else answer_en

        return jsonify({
            "ok": True,
            "lang": lang,
            "answer": answer_out,
            "debug": {"normalized_en": text_en}
        })
    except Exception as e:
        # Fail soft to English if translation errors
        try:
            prompt = build_prompt(user_text if lang == "en" else translate(user_text, lang, "en"))
            gen = gen_pipeline(prompt, max_new_tokens=200)
            fallback_en = gen[0]["generated_text"].strip()
            fallback = fallback_en if lang == "en" else translate(fallback_en, "en", lang)
            return jsonify({"ok": True, "lang": lang, "answer": fallback, "warn": str(e)})
        except Exception as inner:
            return jsonify({"ok": False, "error": str(inner)}), 500


if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
```

---

## 3) Drop‑in UI widget (add to your existing `Source.html`)

Paste the **HTML** where you want the chatbot, and the **JS** once at the bottom of the page. Your page already uses Tailwind, so this matches your design language.

### 3.1 HTML block

```html
<!-- Chatbot Card -->
<div id="chatbot-card" class="bg-white dark:bg-gray-800 rounded-2xl shadow p-4 md:p-6">
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold">AI Health Assistant</h3>
    <select id="cb-lang" class="border rounded-lg px-2 py-1 text-sm">
      <option value="en">English</option>
      <option value="hi">हिन्दी</option>
      <option value="pa">ਪੰਜਾਬੀ</option>
    </select>
  </div>

  <div id="cb-log" class="h-64 overflow-y-auto space-y-3 pr-2 bg-gray-50 dark:bg-gray-900 rounded-xl p-3 text-sm"></div>

  <div class="mt-4 flex gap-2">
    <input id="cb-input" type="text" placeholder="Type your question…" class="flex-1 border rounded-xl px-3 py-2" />
    <button id="cb-send" class="bg-blue-600 hover:bg-blue-700 text-white rounded-xl px-4 py-2">Send</button>
  </div>
  <p class="text-[11px] text-gray-500 mt-2">Educational use only. Not a substitute for professional medical advice.</p>
</div>
```

### 3.2 Javascript (place before `</body>`)

```html
<script>
  const CB_API = (window.CB_API_BASE || "http://localhost:5000") + "/chat";

  const logEl = document.getElementById('cb-log');
  const inputEl = document.getElementById('cb-input');
  const sendBtn = document.getElementById('cb-send');
  const langEl = document.getElementById('cb-lang');

  function append(role, text) {
    const wrap = document.createElement('div');
    wrap.className = role === 'user' ? 'text-right' : 'text-left';
    const bubble = document.createElement('div');
    bubble.className = 'inline-block max-w-[85%] px-3 py-2 rounded-xl ' + (role==='user' ? 'bg-blue-600 text-white' : 'bg-white dark:bg-gray-800 border');
    bubble.textContent = text;
    wrap.appendChild(bubble);
    logEl.appendChild(wrap);
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    append('user', text);
    append('bot', '…thinking');

    try {
      const res = await fetch(CB_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang: langEl.value })
      });
      const data = await res.json();
      logEl.lastChild.querySelector('div').textContent = data.answer || data.error || 'Error';
    } catch (e) {
      logEl.lastChild.querySelector('div').textContent = 'Network error';
    }
  }

  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
</script>
```

---

## 4) Runbook

1. **Create venv & install deps**

   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start backend**

   ```bash
   python app.py
   # Backend at http://localhost:5000
   ```

3. **Wire the frontend**

   * Open your existing `Source.html`.
   * Paste the **HTML block** where you want the widget.
   * Paste the **JS block** before `</body>`.
   * If backend runs elsewhere, set `window.CB_API_BASE = "http://<host>:<port>";` **before** the JS block.

4. **Smoke test**

   * Load the page → ask in English/Hindi/Punjabi.
   * Watch first response take longer (model download cache).

---

## 5) Notes & knobs

* **Upgrade model**: set env `LLM_MODEL=google/flan-t5-base` (better answers, more RAM).
* **Safety**: The prompt guards against clinical overreach; still not medical advice.
* **Cold start**: first run downloads models to `~/.cache/huggingface`.
* **Offline**: works offline after first download.
* **Extensibility**: Add memory, tools (symptom DB), or doctor handoff by extending `/chat`.

---

## 6) Optional: Inline language toggles (match your site)

If you already have EN/HI/PA toggles globally, bind `#cb-lang` to the same state or set its value programmatically.
