# Charaka Sanskrit SLM — Streamlit App

An offline Sanskrit analysis tool powered by a custom-trained Small Language Model.

Upload a śloka as text, image, PDF, Word, or Excel file — get word splits, canonical English meanings, line-level glosses, and references to semantically similar ślokas in *Charaka Saṃhitā Sūtrasthāna*. Download everything as a formatted Excel.

---

## What's inside

### The SLM that powers it

| Component | Size | Role |
|---|---|---|
| `tokenizer.model` | ~330 KB | SentencePiece BPE (4,000 tokens) — sandhi-aware Sanskrit splitter |
| `trained_model.pt` | ~5 MB | 1.3M-parameter transformer (2 layers, 4 heads, 128-dim) — generates śloka embeddings |
| `embeddings.npy` | ~1 MB | 1,968 precomputed śloka embeddings — the semantic retrieval database |
| `corpus_clean.csv` | ~800 KB | Chapter & verse metadata for retrieved ślokas |
| `WHO_ayur_terminology.xlsx` | ~140 KB | 1,500+ WHO-standardized Ayurvedic terms |

Plus a built-in curated dictionary (~150 entries) providing canonical meanings for common Ayurveda terms.

### Inputs accepted

- **Text paste** (Devanagari)
- **Images**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` — OCR via Tesseract (Sanskrit + Hindi packs)
- **PDF**: text-based (direct extraction) or scanned (falls back to OCR)
- **DOCX**: Word documents
- **XLSX/XLS**: Excel (user selects the column containing ślokas)
- **TXT**: plain text

Max file size: **25 MB**.

### Output format

An Excel workbook with three sheets:

1. **Analysis** — the main deliverable, 4 columns:
   - `#` (index)
   - `Your Sloka`
   - `Splitting of Words`
   - `English Meanings of Words`
   - `Line-by-Line Meaning`
2. **Summary** — counts, dictionary coverage %, metadata
3. **About** — explains what the SLM actually did

---

## Deploy to Streamlit Community Cloud (free)

### Prerequisites

- A GitHub account
- A Streamlit Community Cloud account at [share.streamlit.io](https://share.streamlit.io) (sign in with GitHub — free)

### Step 1: Push this folder to GitHub

1. Create a new public repository on GitHub (e.g. `charaka-sanskrit-slm`)
2. Upload the entire contents of this folder. The structure must be:

   ```
   charaka-sanskrit-slm/
   ├── app.py
   ├── requirements.txt
   ├── packages.txt
   ├── README.md
   ├── .streamlit/
   │   └── config.toml
   ├── utils/
   │   ├── __init__.py
   │   ├── analyzer.py
   │   ├── ocr.py
   │   ├── extractors.py
   │   └── excel_output.py
   └── models/
       ├── tokenizer.model
       ├── tokenizer.vocab
       ├── trained_model.pt
       ├── embeddings.npy
       ├── corpus_clean.csv
       ├── corpus_train.txt
       └── WHO_ayur_terminology.xlsx
   ```

   **Tip:** GitHub's web upload supports drag-and-drop for folders. You can also use Git:
   ```bash
   cd charaka_app
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/charaka-sanskrit-slm.git
   git push -u origin main
   ```

3. All model files are well under the 100 MB per-file GitHub limit — no Git LFS needed.

### Step 2: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account
2. Click **"New app"**
3. Fill in:
   - **Repository**: `YOUR-USERNAME/charaka-sanskrit-slm`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**
5. Wait ~5 minutes for the first deploy (it installs Tesseract + Sanskrit pack + Python packages)
6. Your app will be live at `https://YOUR-USERNAME-charaka-sanskrit-slm.streamlit.app` (or a similar URL Streamlit assigns)

### Step 3: Test it

Open your app URL. Try:

1. Paste a śloka in the text tab: `वातपित्तकफा दोषाः समासेन`
2. Click **Analyze**
3. Download the Excel

---

## Run locally (optional, for development)

### On Linux/Mac

```bash
# Install system dependencies
sudo apt-get install -y tesseract-ocr tesseract-ocr-san tesseract-ocr-hin \
    poppler-utils fonts-lohit-deva  # (on Debian/Ubuntu)
# OR on macOS:
# brew install tesseract tesseract-lang poppler

# Install Python dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

### On Windows

1. Install **Tesseract** from [UB-Mannheim's Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)
   - During install, select "Additional language data" → check **Sanskrit (san)** and **Hindi (hin)**
2. Install **Poppler** for PDF rasterization: [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
3. Add Tesseract and Poppler to your PATH
4. Run:
   ```powershell
   pip install -r requirements.txt
   streamlit run app.py
   ```

---

## Updating the SLM

To swap in a better-trained model or a larger corpus, just replace the relevant files in `models/` and redeploy. The app infrastructure remains unchanged.

Typical updates:

- **Retrain longer** → replace `trained_model.pt` and `embeddings.npy`
- **Add Suśruta / Aṣṭāṅga Hṛdaya** → retrain tokenizer and model on the combined corpus, replace all files in `models/`
- **Expand the dictionary** → edit the `SUPPLEMENT` dict in `utils/analyzer.py` and redeploy

---

## Troubleshooting

**The app says "Loading the Sanskrit SLM…" forever.**
First load takes ~10 seconds. If it never finishes, check your Streamlit Cloud logs — the model files may have failed to upload correctly. Verify `trained_model.pt` is ~5 MB on GitHub.

**OCR output has many errors.**
Tesseract's Sanskrit accuracy is modest — 70–90% on clean printed text, lower on scans or handwriting. Always enable the "Review extracted text" option and fix errors before analysis.

**Excel upload says "column not found."**
Select the exact column name that contains ślokas from the dropdown before proceeding. Make sure your ślokas are in that column (not merged cells or formulas).

**PDF returns no text.**
The PDF is likely scanned (images of text, not real text). The app falls back to OCR automatically, but quality depends on scan resolution.

---

## About

Developed by **Dr. Prasanna Kulkarni** — Atharva AyurTech Healthcare.

This app is the deployment of a 4-notebook research pipeline that built a Sanskrit SLM from scratch:
1. Corpus preparation (1,968 ślokas from Charaka Sūtrasthāna)
2. Tokenizer training (SentencePiece BPE, 4,000 tokens)
3. Transformer training (MLM, 30 epochs, 1.3M parameters)
4. Inference engine (dictionary lookup + compound decomposition + retrieval)

All inference runs locally — no external API dependencies, no subscription fees.
