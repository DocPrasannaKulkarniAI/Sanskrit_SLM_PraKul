"""
app.py — Charaka Sanskrit SLM Streamlit App

A Streamlit app that uses a custom-trained Sanskrit Small Language Model
(SLM) to analyze ślokas provided as text, images, PDFs, Word, or Excel files
and returns a downloadable Excel with word splits, meanings, and line-level
glosses.

Components used:
  - Trained transformer (1.3M params) — sentence embeddings for retrieval
  - SentencePiece BPE tokenizer — sandhi-aware word splitting
  - Canonical-meaning dictionary — WHO Ayurveda + curated supplement
  - Tesseract OCR (Sanskrit + Hindi) — for image / scanned PDF inputs
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Charaka Sanskrit SLM",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports from our utils (after set_page_config)
# ---------------------------------------------------------------------------
from utils.analyzer import CharakaAnalyzer
from utils.extractors import extract_text, split_into_slokas, extract_from_excel
from utils.excel_output import build_excel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / 'models'
MAX_FILE_MB = 25
SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'xlsx', 'xls',
                         'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']


# ---------------------------------------------------------------------------
# Cached resource loader — load the SLM once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the Sanskrit SLM (first time takes ~10 seconds)...")
def get_analyzer() -> CharakaAnalyzer:
    """Load the CharakaAnalyzer once and cache it across the whole session."""
    return CharakaAnalyzer(MODELS_DIR)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def init_session() -> None:
    """Initialize session-state keys we'll use."""
    defaults = {
        'extracted_text': '',
        'extracted_filename': '',
        'review_enabled': True,
        'analysis_results': None,
        'excel_bytes': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_results() -> None:
    """Clear any stored results when user changes inputs."""
    st.session_state['analysis_results'] = None
    st.session_state['excel_bytes'] = None


# ---------------------------------------------------------------------------
# Sidebar — about + controls
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🕉️ Charaka Sanskrit SLM")
        st.markdown(
            "An offline Sanskrit analysis tool powered by a custom-trained "
            "1.3M-parameter transformer, a sandhi-aware BPE tokenizer, and a "
            "canonical-meaning Ayurveda dictionary."
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.session_state['review_enabled'] = st.checkbox(
            "Review extracted text before analysis",
            value=st.session_state['review_enabled'],
            help="Recommended: lets you correct OCR errors before the analysis runs.",
        )

        st.markdown("---")
        st.markdown("### 📘 About")
        st.markdown(
            "**How it works:**\n"
            "1. You upload or paste a śloka\n"
            "2. The SLM tokenizes with Sanskrit-aware BPE\n"
            "3. Every word is glossed using the built-in Ayurveda dictionary\n"
            "4. Compounds are split using a DP best-cover algorithm\n"
            "5. Semantically similar ślokas are retrieved from the Charaka corpus\n"
            "6. Everything lands in a downloadable Excel"
        )

        st.markdown("---")
        st.markdown(
            "<small>Developed by Dr. Prasanna Kulkarni<br>"
            "Atharva AyurTech Healthcare</small>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Input tabs
# ---------------------------------------------------------------------------
def render_text_input_tab() -> str | None:
    """Plain-text input tab. Returns entered text or None."""
    st.markdown("#### Paste one or more ślokas in Devanagari")
    st.caption(
        "You can paste multiple ślokas separated by newlines, `।`, or `॥`. "
        "Each recognized śloka becomes one row in the output Excel."
    )

    text = st.text_area(
        "Sanskrit text",
        height=200,
        placeholder="वातपित्तकफा दोषाः समासेन।\nत्रयो दोषाः वातपित्तश्लेष्माणः॥",
        label_visibility="collapsed",
        key="text_input",
    )

    if text.strip():
        return text.strip()
    return None


def render_file_upload_tab() -> str | None:
    """File upload tab for PDF / DOCX / XLSX / TXT."""
    st.markdown("#### Upload a document")
    st.caption(
        f"Supported: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.txt`. "
        f"Maximum size: {MAX_FILE_MB} MB."
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'xlsx', 'xls', 'txt'],
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded is None:
        return None

    # Size check
    size_mb = uploaded.size / 1024 / 1024
    if size_mb > MAX_FILE_MB:
        st.error(f"File is {size_mb:.1f} MB — maximum allowed is {MAX_FILE_MB} MB.")
        return None

    st.info(f"📄 **{uploaded.name}** ({size_mb:.2f} MB)")

    file_bytes = uploaded.getvalue()
    ext = Path(uploaded.name).suffix.lower().strip('.')

    # Special handling for Excel: let the user choose which column has ślokas
    if ext in ('xlsx', 'xls'):
        _, columns = extract_from_excel(file_bytes, column=None)
        if not columns:
            st.error("Could not read the Excel file.")
            return None
        chosen = st.selectbox(
            "Which column contains the Sanskrit ślokas?",
            options=columns,
            key=f"excel_col_{uploaded.name}",
        )
        lines, _ = extract_from_excel(file_bytes, column=chosen)
        return '\n'.join(lines) if lines else None

    # Everything else: extract directly
    try:
        with st.spinner(f"Extracting text from {uploaded.name}..."):
            text, meta = extract_text(file_bytes, uploaded.name)
        if not text.strip():
            st.warning("No text extracted from the file. It may be empty or scanned.")
            return None
        return text.strip()
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        return None


def render_image_upload_tab() -> str | None:
    """Image upload tab — runs OCR via Tesseract."""
    st.markdown("#### Upload an image of a śloka")
    st.caption(
        f"Supported: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`. "
        f"Maximum size: {MAX_FILE_MB} MB."
    )

    st.info(
        "🖼️ **For best OCR quality:** use clear, well-lit photos of printed text. "
        "Enable 'Review extracted text before analysis' in the sidebar — OCR is "
        "rarely 100% accurate on Sanskrit; reviewing lets you fix errors."
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=SUPPORTED_IMAGE_TYPES,
        label_visibility="collapsed",
        key="image_uploader",
    )

    if uploaded is None:
        return None

    size_mb = uploaded.size / 1024 / 1024
    if size_mb > MAX_FILE_MB:
        st.error(f"Image is {size_mb:.1f} MB — maximum is {MAX_FILE_MB} MB.")
        return None

    file_bytes = uploaded.getvalue()

    # Show a thumbnail of the image
    col1, col2 = st.columns([1, 2])
    with col1:
        img = Image.open(BytesIO(file_bytes))
        st.image(img, caption=uploaded.name, use_container_width=True)

    with col2:
        with st.spinner("Running Sanskrit OCR on the image..."):
            try:
                text, _ = extract_text(file_bytes, uploaded.name)
                if not text.strip():
                    st.warning("OCR produced no text. Try a clearer image.")
                    return None
                st.success(f"OCR extracted {sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)} Devanagari characters")
                return text.strip()
            except Exception as e:
                st.error(f"OCR failed: {e}")
                return None


# ---------------------------------------------------------------------------
# Review step — let user edit extracted text
# ---------------------------------------------------------------------------
def render_review_step(extracted: str) -> str | None:
    """Show extracted text in an editable box. Returns the (possibly edited) text."""
    st.markdown("### ✏️ Review & edit the extracted text")
    st.caption(
        "OCR is not always 100% accurate on Sanskrit. Fix any errors below "
        "before clicking **Analyze**. Tip: compare with your source to catch "
        "swapped characters like र/द or missing viramas (्)."
    )

    edited = st.text_area(
        "Extracted Sanskrit text (edit if needed)",
        value=extracted,
        height=240,
        key="review_edit",
    )
    return edited.strip() if edited.strip() else None


# ---------------------------------------------------------------------------
# Analysis + results display
# ---------------------------------------------------------------------------
def run_analysis(analyzer: CharakaAnalyzer, text: str) -> list[dict]:
    """Split text into ślokas and analyze each one."""
    slokas = split_into_slokas(text)
    if not slokas:
        # The whole text might be just one short line (no dandas)
        if sum(1 for c in text if '\u0900' <= c <= '\u097F') >= 8:
            slokas = [text.strip()]

    if not slokas:
        return []

    results = []
    progress = st.progress(0.0, text=f"Analyzing {len(slokas)} śloka(s)...")
    for i, s in enumerate(slokas, 1):
        results.append(analyzer.analyze_sloka(s, top_k_similar=3))
        progress.progress(i / len(slokas), text=f"Analyzed {i}/{len(slokas)} śloka(s)")
    progress.empty()
    return results


def render_results(results: list[dict]) -> None:
    """Show a preview of the results table and similar-sloka lookups."""
    st.markdown("### 📊 Analysis results")

    # Build a preview DataFrame
    rows = []
    for i, r in enumerate(results, 1):
        # Quick word-meaning summary for preview
        word_meanings = []
        for word, subs in r['words']:
            if not subs:
                word_meanings.append(f"{word}: ❓")
            elif len(subs) == 1:
                gloss = re.split(r'\s*[—(]\s*', subs[0][1]['english'], maxsplit=1)[0]
                word_meanings.append(f"{word} → {gloss}")
            else:
                inner = ' + '.join(
                    re.split(r'\s*[—(]\s*', s[1]['english'], maxsplit=1)[0] for s in subs
                )
                word_meanings.append(f"{word} → ({inner})")
        rows.append({
            '#': i,
            'Your Sloka': r['input'][:80] + ('…' if len(r['input']) > 80 else ''),
            'Line-by-Line Meaning': r['english_gloss'],
        })

    df_preview = pd.DataFrame(rows)
    st.dataframe(df_preview, hide_index=True, use_container_width=True)

    # Expandable detail for each śloka
    st.markdown("#### Details per śloka")
    for i, r in enumerate(results, 1):
        with st.expander(f"Śloka {i}: {r['input'][:60]}...", expanded=(i == 1)):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Word analysis**")
                for word, subs in r['words']:
                    if not subs:
                        st.markdown(f"- **{word}**: *[no dictionary match]*")
                    elif len(subs) == 1 and subs[0][0] == word:
                        st.markdown(f"- **{word}** = {subs[0][1]['english']}")
                    else:
                        st.markdown(f"- **{word}** (compound):")
                        for sub, info in subs:
                            st.markdown(f"    - `{sub}` = {info['english']}")

            with col2:
                st.markdown("**Line-level gloss**")
                st.info(r['english_gloss'])

                st.markdown("**Top similar ślokas in Charaka Sūtrasthāna**")
                for j, sim in enumerate(r['similar'][:3], 1):
                    st.markdown(
                        f"{j}. *Ch {sim['chapter']}, Śl {sim['sloka_num']}* "
                        f"(similarity {sim['similarity']:.3f})\n\n"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;{sim['text'][:100]}..."
                    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_session()
    render_sidebar()

    # Header
    st.markdown("# 🕉️ Charaka Sanskrit SLM")
    st.markdown(
        "**Analyze any Sanskrit śloka — get word splits, English meanings, "
        "and line-by-line translation as a downloadable Excel file.**"
    )
    st.markdown("---")

    # Load the analyzer (cached)
    analyzer = get_analyzer()

    # ----- Step 1: Input -----
    st.markdown("### 1️⃣ Provide your śloka(s)")
    tab1, tab2, tab3 = st.tabs([
        "📝 Paste text",
        "📄 Upload document",
        "🖼️ Upload image",
    ])

    extracted = None
    with tab1:
        extracted = render_text_input_tab() or extracted
    with tab2:
        t2 = render_file_upload_tab()
        if t2:
            extracted = t2
    with tab3:
        t3 = render_image_upload_tab()
        if t3:
            extracted = t3

    if not extracted:
        st.info("👆 Use any of the tabs above to provide a śloka.")
        return

    # Store the new extraction
    if extracted != st.session_state.get('extracted_text'):
        st.session_state['extracted_text'] = extracted
        reset_results()

    # ----- Step 2: Review (optional) -----
    final_text = extracted
    if st.session_state['review_enabled']:
        st.markdown("---")
        st.markdown("### 2️⃣ Review")
        final_text = render_review_step(extracted) or extracted

    # ----- Step 3: Analyze -----
    st.markdown("---")
    st.markdown("### 3️⃣ Analyze")

    col_a, col_b = st.columns([1, 4])
    with col_a:
        analyze_btn = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True,
            disabled=not final_text,
        )
    with col_b:
        slokas_preview = split_into_slokas(final_text) if final_text else []
        if slokas_preview:
            st.caption(f"📋 {len(slokas_preview)} śloka(s) detected in the input")
        elif final_text and sum(1 for c in final_text if '\u0900' <= c <= '\u097F') >= 8:
            st.caption("📋 1 śloka will be analyzed")

    if analyze_btn:
        results = run_analysis(analyzer, final_text)
        if not results:
            st.error(
                "No ślokas could be analyzed. The input may not contain enough "
                "Sanskrit text. Make sure you have at least one Devanagari śloka."
            )
            return
        st.session_state['analysis_results'] = results
        with st.spinner("Building your Excel..."):
            st.session_state['excel_bytes'] = build_excel(results)

    # ----- Step 4: Results + download -----
    if st.session_state['analysis_results']:
        st.markdown("---")
        render_results(st.session_state['analysis_results'])

        st.markdown("---")
        st.markdown("### 4️⃣ Download")

        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button(
                label="📥 Download Excel",
                data=st.session_state['excel_bytes'],
                file_name="charaka_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        with col_dl2:
            st.caption(
                "The Excel has three sheets: **Analysis** (your 4 requested columns), "
                "**Summary** (coverage stats), and **About** (how the SLM works)."
            )


if __name__ == '__main__':
    main()
