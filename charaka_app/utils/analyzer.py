"""
analyzer.py — Core SLM-powered Sanskrit analysis engine.

This module ships the SLM we built across 4 notebooks:
  - Loads the trained transformer (Notebook 3)
  - Loads the BPE tokenizer (Notebook 2)
  - Loads precomputed corpus embeddings (Notebook 3)
  - Uses the canonical-meaning dictionary + DP decomposition (Notebook 4)

Every call to analyze_sloka() uses ALL of the above.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dev -> IAST transliteration
# ---------------------------------------------------------------------------
DEV_TO_IAST = {
    'अ':'a','आ':'ā','इ':'i','ई':'ī','उ':'u','ऊ':'ū','ऋ':'ṛ','ॠ':'ṝ',
    'ऌ':'ḷ','ए':'e','ऐ':'ai','ओ':'o','औ':'au',
    'क':'ka','ख':'kha','ग':'ga','घ':'gha','ङ':'ṅa',
    'च':'ca','छ':'cha','ज':'ja','झ':'jha','ञ':'ña',
    'ट':'ṭa','ठ':'ṭha','ड':'ḍa','ढ':'ḍha','ण':'ṇa',
    'त':'ta','थ':'tha','द':'da','ध':'dha','न':'na',
    'प':'pa','फ':'pha','ब':'ba','भ':'bha','म':'ma',
    'य':'ya','र':'ra','ल':'la','व':'va',
    'श':'śa','ष':'ṣa','स':'sa','ह':'ha',
    'ा':'ā','ि':'i','ी':'ī','ु':'u','ू':'ū','ृ':'ṛ',
    'े':'e','ै':'ai','ो':'o','ौ':'au',
    'ं':'ṃ','ः':'ḥ','्':'','ँ':'m̐','ऽ':'a',
}


def dev_to_iast(text: str) -> str:
    """Convert Devanagari text to IAST (International Alphabet of Sanskrit Transliteration)."""
    text = unicodedata.normalize('NFC', str(text))
    result: list[str] = []
    consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
    matras = 'ािीुूृेैोौ'
    for c in text:
        if c == '्':  # virama removes previous consonant's inherent 'a'
            if result and result[-1].endswith('a'):
                result[-1] = result[-1][:-1]
        elif c in consonants:
            result.append(DEV_TO_IAST.get(c, c))
        elif c in matras:
            mapped = DEV_TO_IAST[c]
            if result and result[-1].endswith('a'):
                result[-1] = result[-1][:-1] + mapped
            else:
                result.append(mapped)
        elif c in DEV_TO_IAST:
            result.append(DEV_TO_IAST[c])
        else:
            result.append(c)
    return ''.join(result)


# ---------------------------------------------------------------------------
# Dictionary data — canonical meanings (Notebook 4 final version)
# ---------------------------------------------------------------------------
SUPPLEMENT: dict[str, str] = {
    # === Doshas ===
    "vāta": "Vāta — dosha governing movement & cognition",
    "vāyu": "Vāyu (= Vāta) — dosha of movement",
    "anila": "Anila (= Vāta) — dosha of movement",
    "māruta": "Māruta (= Vāta) — dosha of movement",
    "marut": "Marut (= Vāta) — dosha of movement",
    "pitta": "Pitta — dosha governing transformation & metabolism",
    "kapha": "Kapha — dosha governing structure & cohesion",
    "śleṣma": "Śleṣma (= Kapha) — dosha of structure",
    "śleṣman": "Śleṣma (= Kapha) — dosha of structure",
    "doṣa": "Doṣa — regulatory functional factor",
    "tridoṣa": "Tridoṣa — the three doṣas: vāta, pitta, kapha",

    # === Seven Dhātus ===
    "rasa": "Rasa — plasma (1st dhātu); also: taste, essence",
    "rasā": "Rasa — plasma (1st dhātu) [compound form]",
    "rakta": "Rakta — blood (2nd dhātu)",
    "asṛk": "Asṛk — blood (2nd dhātu; = rakta)",
    "asṛj": "Asṛj — blood (2nd dhātu; = rakta)",
    "asṛṅ": "Asṛk — blood (2nd dhātu; = rakta) [sandhi form before nasal]",
    "sṛk": "Asṛk — blood (2nd dhātu; = rakta) [compound-internal form]",
    "sṛj": "Asṛj — blood (2nd dhātu; = rakta) [compound-internal form]",
    "sṛṅ": "Asṛk — blood (2nd dhātu; = rakta) [compound-internal form]",
    "śoṇita": "Śoṇita — blood (= rakta)",
    "lohita": "Lohita — blood (= rakta)",
    "rudhira": "Rudhira — blood (= rakta)",
    "māṃsa": "Māṃsa — muscle tissue (3rd dhātu)",
    "mānsa": "Māṃsa — muscle tissue (3rd dhātu) [variant spelling]",
    "pala": "Pala — flesh (= māṃsa, in some contexts)",
    "meda": "Meda — adipose tissue (4th dhātu)",
    "medas": "Meda — adipose tissue (4th dhātu)",
    "medo": "Meda — adipose tissue (4th dhātu) [sandhi form before voiced sound]",
    "vasā": "Vasā — muscle fat (related to meda)",
    "asthi": "Asthi — bone (5th dhātu)",
    "sthi": "Asthi — bone (5th dhātu) [compound-internal form after avagraha]",
    "majjā": "Majjā — marrow (6th dhātu)",
    "majja": "Majjā — marrow (6th dhātu) [compound form]",
    "majjan": "Majjā — marrow (6th dhātu)",
    "śukra": "Śukra — reproductive tissue (7th dhātu)",
    "retas": "Retas — semen (= śukra)",
    "bīja": "Bīja — seed / reproductive element (= śukra, in some contexts)",
    "dhātu": "Dhātu — body tissue / fundamental constituent",

    # === Other body substances ===
    "ojas": "Ojas — vital essence of all dhātus",
    "tejas": "Tejas — subtle fire; vital luminosity",
    "prāṇa": "Prāṇa — vital breath / life-force",
    "agni": "Agni — digestive/metabolic fire",
    "mala": "Mala — waste product",
    "purīṣa": "Purīṣa — stool / faecal waste",
    "mūtra": "Mūtra — urine",
    "sveda": "Sveda — sweat",
    "srotas": "Srotas — channel of circulation",
    "srota": "Srotas — channel of circulation [compound form]",

    # === Indriyas (senses) ===
    "indriya": "Indriya — sensory or motor organ",
    "cakṣu": "Cakṣu — eye / organ of sight",
    "cakṣus": "Cakṣus — eye / organ of sight",
    "karṇa": "Karṇa — ear / organ of hearing",
    "nāsā": "Nāsā — nose / organ of smell",
    "jihvā": "Jihvā — tongue / organ of taste",
    "tvak": "Tvak — skin / organ of touch",

    # === Disease concepts ===
    "prakṛti": "Prakṛti — innate constitution",
    "vikṛti": "Vikṛti — current pathological state",
    "svastha": "Svastha — one established in health",
    "svāsthya": "Svāsthya — health / wellbeing",
    "roga": "Roga — disease",
    "vyādhi": "Vyādhi — disorder",
    "lakṣaṇa": "Lakṣaṇa — sign / symptom",
    "hetu": "Hetu — cause",
    "nidāna": "Nidāna — aetiology / cause",
    "pūrvarūpa": "Pūrvarūpa — prodromal sign",
    "rūpa": "Rūpa — manifest sign / form",
    "upaśaya": "Upaśaya — diagnostic trial (what alleviates)",

    # === Rasas (tastes) ===
    "madhura": "Madhura — sweet taste",
    "amla": "Amla — sour taste",
    "lavaṇa": "Lavaṇa — salty taste",
    "kaṭu": "Kaṭu — pungent taste",
    "tikta": "Tikta — bitter taste",
    "kaṣāya": "Kaṣāya — astringent taste",

    # === Guṇas (qualities) ===
    "guru": "Guru — heavy (guṇa)",
    "laghu": "Laghu — light (guṇa)",
    "śīta": "Śīta — cold (guṇa)",
    "uṣṇa": "Uṣṇa — hot (guṇa)",
    "snigdha": "Snigdha — unctuous (guṇa)",
    "rūkṣa": "Rūkṣa — dry (guṇa)",
    "mṛdu": "Mṛdu — soft (guṇa)",
    "tīkṣṇa": "Tīkṣṇa — sharp (guṇa)",
    "sthira": "Sthira — stable (guṇa)",
    "sara": "Sara — mobile (guṇa)",
    "manda": "Manda — slow (guṇa)",
    "viśada": "Viśada — clear (guṇa)",
    "picchila": "Picchila — slimy (guṇa)",
    "khara": "Khara — rough (guṇa)",
    "sūkṣma": "Sūkṣma — subtle (guṇa)",
    "sthūla": "Sthūla — gross (guṇa)",

    # === Bhūtas (elements) ===
    "pṛthvī": "Pṛthvī — earth element",
    "pṛthivī": "Pṛthivī — earth element",
    "ap": "Ap — water element",
    "jala": "Jala — water element",
    "udaka": "Udaka — water",
    "ākāśa": "Ākāśa — space / ether element",

    # === Text structure ===
    "sūtra": "Sūtra — aphorism",
    "sthāna": "Sthāna — section of a treatise",
    "adhyāya": "Adhyāya — chapter",
    "tantra": "Tantra — treatise",
    "samāsa": "Samāsa — compound / brief summary",
    "samāsena": "in brief / summarily",
    "vistara": "Vistara — detailed exposition",
    "vistareṇa": "in detail / elaborately",

    # === Verbal / idiomatic ===
    "vyākhyāsyāmaḥ": "we shall expound",
    "vyākhyā": "exposition / commentary",
    "ucyate": "is said / is called",
    "bhavati": "becomes / is",
    "syāt": "should be / may be",
    "iti": "thus / end-quote",
    "tatra": "there / in that (case)",
    "atha": "now / then",
    "athātaḥ": "now henceforth (standard text-opening)",

    # === Persons / sages ===
    "agniveśa": "Agniveśa — original author (Caraka's predecessor)",
    "ātreya": "Ātreya — the teacher of Agniveśa",
    "punarvasu": "Punarvasu Ātreya — the teacher",
    "caraka": "Caraka — the redactor",
    "bharadvāja": "Bharadvāja — sage",
    "indra": "Indra — lord of devas",
    "bhagavān": "venerable one / the lord",
    "ṛṣi": "Ṛṣi — sage / seer",
    "maharṣi": "Maharṣi — great sage",

    # === Life / āyu ===
    "jīvita": "jīvita — life",
    "jīvitam": "jīvitam — life",
    "jīva": "jīva — living being",
    "āyu": "āyu — lifespan",
    "āyus": "āyus — lifespan",
    "āyurveda": "Āyurveda — the science of life",
    "dīrgha": "dīrgha — long, prolonged",
    "hita": "hita — beneficial",
    "ahita": "ahita — harmful",
    "sukha": "sukha — happy / comfortable",
    "duḥkha": "duḥkha — unhappy / uncomfortable",

    # === Body / mind ===
    "śarīra": "Śarīra — body",
    "deha": "Deha — body (= śarīra)",
    "kāya": "Kāya — body (= śarīra)",
    "sattva": "Sattva — mind / mental faculty",
    "ātman": "Ātman — self / soul",
    "ātmā": "Ātman — self / soul",
    "manas": "Manas — mind",
    "mana": "Manas — mind",
    "buddhi": "Buddhi — intellect",
    "ahaṃkāra": "Ahaṃkāra — ego / I-sense",

    # === Treatment ===
    "cikitsā": "Cikitsā — treatment / therapy",
    "auṣadha": "Auṣadha — medicine",
    "bheṣaja": "Bheṣaja — medicine",
    "dravya": "Dravya — substance / material",
    "rasāyana": "Rasāyana — rejuvenation therapy",
    "vājīkaraṇa": "Vājīkaraṇa — aphrodisiac therapy",
    "pathya": "Pathya — wholesome",
    "apathya": "Apathya — unwholesome",
    "āhāra": "Āhāra — food / diet",
    "vihāra": "Vihāra — lifestyle / conduct",
    "pañcakarma": "Pañcakarma — the five cleansing procedures",
    "vamana": "Vamana — therapeutic emesis",
    "virecana": "Virecana — therapeutic purgation",
    "basti": "Basti — enema therapy",
    "nasya": "Nasya — nasal therapy",
    "raktamokṣaṇa": "Raktamokṣaṇa — bloodletting",

    # === Numerals ===
    "eka": "eka — one", "dvi": "dvi — two", "tri": "tri — three",
    "trayaḥ": "trayaḥ — three (masc. nom. pl.)",
    "trayas": "trayas — three",
    "trayo": "trayo — three [sandhi form]",
    "catur": "catur — four", "pañca": "pañca — five",
    "ṣaṭ": "ṣaṭ — six", "sapta": "sapta — seven",
    "aṣṭa": "aṣṭa — eight", "nava": "nava — nine", "daśa": "daśa — ten",
}

STANDALONE_ONLY: dict[str, str] = {
    "ca": "ca — and", "tu": "tu — but", "hi": "hi — indeed / for",
    "eva": "eva — just / only", "na": "na — not", "vā": "vā — or",
    "ha": "ha — indeed (emphatic)", "iva": "iva — like / as",
    "api": "api — also / even", "yathā": "yathā — as / in the manner of",
    "tathā": "tathā — so / thus", "yadi": "yadi — if",
}

INFLECTIONS: list[tuple[str, str]] = [
    ('āyaiḥ', 'ā'), ('ābhyāṃ', 'ā'), ('ābhyaḥ', 'ā'), ('ānām', 'ā'), ('āsu', 'ā'),
    ('asya', 'a'), ('asmin', 'a'), ('asmāt', 'a'), ('asyāḥ', 'ā'),
    ('ebhyaḥ', 'a'), ('aiḥ', 'a'), ('eṣu', 'a'), ('eṣām', 'a'), ('ayoḥ', 'a'),
    ('ānāṃ', 'a'), ('ānām', 'a'), ('āni', 'a'), ('āṇi', 'a'), ('āṇaḥ', 'an'),
    ('āḥ', 'a'), ('āṃ', 'a'), ('ān', 'a'), ('ām', 'a'),
    ('āya', 'a'), ('āt', 'a'), ('aiḥ', 'a'), ('au', 'a'), ('ais', 'a'), ('ena', 'a'),
    ('aḥ', 'a'), ('aṃ', 'a'), ('o', 'aḥ'),
    ('avaḥ', 'u'), ('ūn', 'u'), ('ubhiḥ', 'u'),
    ('ḥ', ''), ('ṃ', ''), ('m', ''), ('n', ''),
]


# ---------------------------------------------------------------------------
# The transformer architecture (matches Notebook 3 exactly)
# ---------------------------------------------------------------------------
class SanskritSLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, ff_dim,
                 max_seq_len, dropout, pad_id=0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos_ids = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(token_ids) + self.pos_emb(pos_ids)
        x = self.layer_norm(x)
        pad_mask = (token_ids == self.pad_id)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.mlm_head(x), x

    def get_sentence_embedding(self, token_ids):
        with torch.no_grad():
            _, x = self.forward(token_ids)
            mask = (token_ids != self.pad_id).unsqueeze(-1).float()
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# ---------------------------------------------------------------------------
# The Analyzer — loads everything once, analyzes on demand
# ---------------------------------------------------------------------------
class CharakaAnalyzer:
    """Loads SLM artifacts and provides analysis of any Sanskrit śloka."""

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self._load_all()

    def _load_all(self) -> None:
        # Tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.models_dir / 'tokenizer.model'))

        # Trained model
        ckpt = torch.load(
            self.models_dir / 'trained_model.pt',
            map_location='cpu',
            weights_only=False,
        )
        self.config = ckpt['config']
        self.pad_id = ckpt.get('pad_id', 0)
        self.bos_id = ckpt.get('bos_id', 2)
        self.eos_id = ckpt.get('eos_id', 3)
        self.max_seq_len = self.config['max_seq_len']

        self.model = SanskritSLM(**self.config, pad_id=self.pad_id)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        # Corpus embeddings (precomputed)
        self.embeddings = np.load(self.models_dir / 'embeddings.npy')
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / (norms + 1e-8)

        # Corpus lines and metadata
        with open(self.models_dir / 'corpus_train.txt', encoding='utf-8') as f:
            self.corpus_lines = [l.strip() for l in f if l.strip()]
        self.df_meta = pd.read_csv(self.models_dir / 'corpus_clean.csv')

        # WHO Ayurveda Terminology
        who = pd.read_excel(self.models_dir / 'WHO_ayur_terminology.xlsx')
        who.columns = ['idx', 'term_id', 'english', 'description', 'iast', 'sanskrit']
        who = who.dropna(subset=['iast']).reset_index(drop=True)
        self.iast_index: dict[str, dict[str, Any]] = {}
        for _, row in who.iterrows():
            terms = re.split(r'[,;]\s*|\s+or\s+', str(row['iast']).lower().strip())
            for t in terms:
                t = t.strip()
                if t and t not in self.iast_index and len(t) >= 3:
                    self.iast_index[t] = {
                        'english': str(row['english']),
                        'iast': t,
                        'source': 'WHO',
                    }

    # -----------------------------------------------------------------------
    # Lookup
    # -----------------------------------------------------------------------
    def lookup_iast(self, w: str, allow_particles: bool = True) -> dict | None:
        w = w.lower().strip()
        if not w:
            return None
        if allow_particles and w in STANDALONE_ONLY:
            return {'english': STANDALONE_ONLY[w], 'iast': w,
                    'source': 'curated', 'matched_form': w}
        if w in SUPPLEMENT:
            return {'english': SUPPLEMENT[w], 'iast': w,
                    'source': 'curated', 'matched_form': w}
        if w in self.iast_index:
            return {**self.iast_index[w], 'matched_form': w}
        for ending, repl in INFLECTIONS:
            if w.endswith(ending) and len(w) > len(ending):
                cand = w[:-len(ending)] + repl
                if cand in SUPPLEMENT:
                    return {'english': SUPPLEMENT[cand], 'iast': cand,
                            'source': 'curated', 'matched_form': cand,
                            'matched_via': f'-{ending}'}
                if cand in self.iast_index:
                    return {**self.iast_index[cand], 'matched_form': cand,
                            'matched_via': f'-{ending}'}
        return None

    def lookup_word(self, devanagari_word: str) -> dict | None:
        w = re.sub(r'[।॥,;\s]+', '', devanagari_word).strip()
        if not w:
            return None
        return self.lookup_iast(dev_to_iast(w).lower(), allow_particles=True)

    # -----------------------------------------------------------------------
    # Decomposition
    # -----------------------------------------------------------------------
    def decompose_compound(self, iast_str: str, min_len: int = 3) -> list:
        n = len(iast_str)
        if n < min_len:
            return []
        dp: list = [(0, [])] * (n + 1)
        for end in range(min_len, n + 1):
            best = (dp[end-1][0], dp[end-1][1])
            for start in range(max(0, end - 25), end - min_len + 1):
                sub = iast_str[start:end]
                info = self.lookup_iast(sub, allow_particles=False)
                if info is None:
                    continue
                prev_score, prev_matches = dp[start]
                new_score = prev_score + (end - start) + 2
                if new_score > best[0]:
                    best = (new_score, prev_matches + [(start, end, sub, info)])
            dp[end] = best
        matches = dp[n][1]
        # Dedup by canonical meaning
        seen = set()
        result = []
        for _, _, sub, info in matches:
            key = re.split(r'[—(]', info['english'], maxsplit=1)[0].strip().lower()
            if key not in seen:
                seen.add(key)
                result.append((sub, info))
        return result

    def analyze_words(self, sloka: str) -> list:
        words = re.findall(r'[\u0900-\u097F]+', sloka)
        results = []
        for w in words:
            r = self.lookup_word(w)
            if r:
                results.append((w, [(w, r)]))
                continue
            decomp = self.decompose_compound(dev_to_iast(w).lower())
            results.append((w, decomp))
        return results

    # -----------------------------------------------------------------------
    # Embedding + retrieval
    # -----------------------------------------------------------------------
    def embed_sloka(self, sloka: str) -> np.ndarray:
        ids = [self.bos_id] + self.sp.EncodeAsIds(sloka)[:self.max_seq_len - 2] + [self.eos_id]
        ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        tensor = torch.tensor([ids], dtype=torch.long)
        return self.model.get_sentence_embedding(tensor)[0].numpy()

    def find_similar(self, sloka: str, top_k: int = 5, exclude_self: bool = True) -> list:
        q = self.embed_sloka(sloka)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        sims = self.embeddings_norm @ q_norm
        if exclude_self:
            for i, line in enumerate(self.corpus_lines):
                if line == sloka:
                    sims[i] = -1
                    break
        top_idx = np.argsort(-sims)[:top_k]
        out = []
        for idx in top_idx:
            meta = self.df_meta.iloc[int(idx)]
            out.append({
                'chapter': int(meta['chapter_num']),
                'sloka_num': int(meta['source_sloka_num']),
                'text': self.corpus_lines[int(idx)],
                'similarity': float(sims[int(idx)]),
            })
        return out

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------
    def build_gloss(self, word_analyses: list) -> str:
        parts = []
        for w, subs in word_analyses:
            if not subs:
                parts.append(f"[{w}]")
            elif len(subs) == 1:
                # Use only the short canonical name (before em-dash)
                parts.append(re.split(r'\s*[—(]\s*', subs[0][1]['english'], maxsplit=1)[0])
            else:
                inner = ' + '.join(
                    re.split(r'\s*[—(]\s*', s[1]['english'], maxsplit=1)[0] for s in subs
                )
                parts.append(f"({inner})")
        return ' '.join(parts)

    def analyze_sloka(self, sloka: str, top_k_similar: int = 5) -> dict:
        sloka = sloka.strip()
        word_analyses = self.analyze_words(sloka)
        english_gloss = self.build_gloss(word_analyses)
        similar = self.find_similar(sloka, top_k=top_k_similar)
        return {
            'input': sloka,
            'words': word_analyses,
            'english_gloss': english_gloss,
            'similar': similar,
        }
