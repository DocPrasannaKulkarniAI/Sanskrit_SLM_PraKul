"""
analyzer.py — Core SLM-powered Sanskrit analysis engine (v2, expanded).

Uses:
  - Trained 1.3M-parameter transformer (Notebook 3)
  - SentencePiece BPE tokenizer (Notebook 2)
  - Precomputed embeddings of 1,968 Charaka ślokas (Notebook 3)
  - Canonical-meaning dictionary with ~580+ terms
  - Length-weighted DP compound decomposition
  - Inflection patterns with min_stem_len to prevent noise matches
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
    text = unicodedata.normalize('NFC', str(text))
    result: list[str] = []
    consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
    matras = 'ािीुूृेैोौ'
    for c in text:
        if c == '्':
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


SUPPLEMENT: dict[str, str] = {
    # Doshas
    "vāta": "Vāta — dosha governing movement & cognition",
    "vāyu": "Vāyu (= Vāta) — dosha of movement",
    "anila": "Anila (= Vāta) — dosha of movement",
    "māruta": "Māruta (= Vāta) — dosha of movement",
    "marut": "Marut (= Vāta) — dosha of movement",
    "pitta": "Pitta — dosha governing transformation & metabolism",
    "tejas": "Tejas — subtle fire; also 4th element",
    "kapha": "Kapha — dosha governing structure & cohesion",
    "śleṣma": "Śleṣma (= Kapha) — dosha of structure",
    "śleṣman": "Śleṣma (= Kapha) — dosha of structure",
    "doṣa": "Doṣa — regulatory functional factor",
    "tridoṣa": "Tridoṣa — the three doṣas",
    "sannipāta": "Sannipāta — tri-doṣic condition",

    # Dhātus + sandhi variants
    "rasa": "Rasa — plasma (1st dhātu); also taste, essence",
    "rasā": "Rasa — plasma (1st dhātu) [compound form]",
    "rakta": "Rakta — blood (2nd dhātu)",
    "asṛk": "Asṛk — blood (2nd dhātu; = rakta)",
    "asṛj": "Asṛj — blood (2nd dhātu; = rakta)",
    "asṛṅ": "Asṛk — blood (2nd dhātu; = rakta) [sandhi form]",
    "sṛk": "Asṛk — blood (2nd dhātu; = rakta) [compound form]",
    "sṛj": "Asṛj — blood (2nd dhātu; = rakta) [compound form]",
    "sṛṅ": "Asṛk — blood (2nd dhātu; = rakta) [compound form]",
    "śoṇita": "Śoṇita — blood",
    "lohita": "Lohita — blood",
    "rudhira": "Rudhira — blood",
    "māṃsa": "Māṃsa — muscle tissue (3rd dhātu)",
    "mānsa": "Māṃsa — muscle tissue",
    "pala": "Pala — flesh",
    "meda": "Meda — adipose tissue (4th dhātu)",
    "medas": "Meda — adipose tissue",
    "medo": "Meda — adipose tissue [sandhi form]",
    "vasā": "Vasā — muscle fat",
    "asthi": "Asthi — bone (5th dhātu)",
    "sthi": "Asthi — bone [compound form]",
    "majjā": "Majjā — marrow (6th dhātu)",
    "majja": "Majjā — marrow [compound form]",
    "majjan": "Majjā — marrow",
    "śukra": "Śukra — reproductive tissue (7th dhātu)",
    "retas": "Retas — semen",
    "dhātu": "Dhātu — body tissue / fundamental constituent",
    "upadhātu": "Upadhātu — secondary body tissue",

    # Ojas / Prāṇa / Agni
    "ojas": "Ojas — vital essence",
    "prāṇa": "Prāṇa — vital breath",
    "agni": "Agni — digestive/metabolic fire",
    "agnau": "Agni — digestive fire [locative]",
    "āgnau": "Agni — digestive fire [locative, sandhi form after a+a fusion]",
    "agneḥ": "Agni — digestive fire [ablative/genitive]",
    "jāṭharāgni": "Jāṭharāgni — central digestive fire",
    "bhūtāgni": "Bhūtāgni — elemental fires",
    "dhātvagni": "Dhātvagni — tissue metabolic fires",
    "pācaka": "Pācaka — digestive (agni)",
    "pacana": "Pacana — digestion / cooking",
    "pāka": "Pāka — digestion / ripening",
    "pac": "pac — to cook / digest (verbal root √pac)",
    "ādoṣa": "ādoṣa — until the doṣa (pathological factor)",

    # Malas / Srotas
    "mala": "Mala — waste product",
    "purīṣa": "Purīṣa — faeces",
    "mūtra": "Mūtra — urine",
    "sveda": "Sveda — sweat",
    "srotas": "Srotas — channel of circulation",
    "srota": "Srotas — channel [compound form]",

    # Indriyas
    "indriya": "Indriya — sensory/motor organ",
    "jñānendriya": "Jñānendriya — sense organ",
    "karmendriya": "Karmendriya — motor organ",
    "cakṣu": "Cakṣu — eye",
    "cakṣus": "Cakṣus — eye",
    "netra": "Netra — eye",
    "karṇa": "Karṇa — ear",
    "śrotra": "Śrotra — ear",
    "nāsā": "Nāsā — nose",
    "ghrāṇa": "Ghrāṇa — organ of smell",
    "jihvā": "Jihvā — tongue",
    "rasana": "Rasana — organ of taste",
    "tvak": "Tvak — skin",
    "sparśana": "Sparśana — organ of touch",

    # Mind
    "manas": "Manas — mind",
    "mana": "Manas — mind",
    "sattva": "Sattva — mind / mental faculty",
    "rajas": "Rajas — mental guṇa of activity",
    "tamas": "Tamas — mental guṇa of inertia",
    "ātman": "Ātman — self / soul",
    "ātmā": "Ātman — self / soul",
    "buddhi": "Buddhi — intellect",
    "ahaṃkāra": "Ahaṃkāra — ego",
    "citta": "Citta — consciousness",

    # Concept
    "prakṛti": "Prakṛti — innate constitution",
    "vikṛti": "Vikṛti — pathological state",
    "svastha": "Svastha — one established in health",
    "svāsthya": "Svāsthya — health",
    "puruṣa": "Puruṣa — person",
    "strī": "Strī — woman",
    "bāla": "Bāla — child",
    "vṛddha": "Vṛddha — elderly",
    "yuvā": "Yuvā — youth",

    # Diseases
    "roga": "Roga — disease",
    "vyādhi": "Vyādhi — disorder",
    "āmaya": "Āmaya — disease",
    "gada": "Gada — disease",
    "jvara": "Jvara — fever",
    "jvarita": "Jvarita — feverish / afflicted by fever",
    "kāsa": "Kāsa — cough",
    "śvāsa": "Śvāsa — dyspnoea",
    "hikkā": "Hikkā — hiccup",
    "chardi": "Chardi — vomiting",
    "atīsāra": "Atīsāra — diarrhoea",
    "grahaṇī": "Grahaṇī — malabsorption disorder",
    "arśas": "Arśas — haemorrhoids",
    "udara": "Udara — abdomen",
    "gulma": "Gulma — abdominal tumour",
    "kuṣṭha": "Kuṣṭha — skin disease",
    "prameha": "Prameha — urinary disorder",
    "madhumeha": "Madhumeha — diabetes mellitus",
    "mūtrakṛcchra": "Mūtrakṛcchra — dysuria",
    "aśmarī": "Aśmarī — urinary calculi",
    "udāvarta": "Udāvarta — upward-moving wind",
    "ādhmāna": "Ādhmāna — distension / flatulence",
    "śūla": "Śūla — colic pain",
    "śopha": "Śopha — swelling",
    "śotha": "Śotha — swelling",
    "pāṇḍu": "Pāṇḍu — anaemia",
    "kāmala": "Kāmala — jaundice",
    "unmāda": "Unmāda — insanity",
    "apasmāra": "Apasmāra — epilepsy",
    "mūrcchā": "Mūrcchā — syncope",
    "klama": "Klama — fatigue",
    "daurbalya": "Daurbalya — weakness",
    "tandrā": "Tandrā — drowsiness",
    "nidrā": "Nidrā — sleep",
    "anidrā": "Anidrā — insomnia",

    # Aetiology / signs
    "lakṣaṇa": "Lakṣaṇa — sign / symptom",
    "liṅga": "Liṅga — sign",
    "hetu": "Hetu — cause",
    "nidāna": "Nidāna — aetiology",
    "kāraṇa": "Kāraṇa — cause",
    "nimitta": "Nimitta — immediate cause",
    "pūrvarūpa": "Pūrvarūpa — prodromal sign",
    "rūpa": "Rūpa — manifest sign / form",
    "upaśaya": "Upaśaya — diagnostic trial",
    "samprāpti": "Samprāpti — pathogenesis",

    # Rasas
    "madhura": "Madhura — sweet taste",
    "amla": "Amla — sour taste",
    "lavaṇa": "Lavaṇa — salty taste",
    "kaṭu": "Kaṭu — pungent taste",
    "tikta": "Tikta — bitter taste",
    "kaṣāya": "Kaṣāya — astringent taste",

    # Guṇas
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
    "ślakṣṇa": "Ślakṣṇa — smooth (guṇa)",
    "sūkṣma": "Sūkṣma — subtle (guṇa)",
    "sthūla": "Sthūla — gross (guṇa)",
    "drava": "Drava — liquid (guṇa)",
    "sāndra": "Sāndra — dense (guṇa)",
    "vipāka": "Vipāka — post-digestive effect",
    "vīrya": "Vīrya — potency",
    "prabhāva": "Prabhāva — specific action",

    # Bhūtas
    "pṛthvī": "Pṛthvī — earth element",
    "pṛthivī": "Pṛthivī — earth element",
    "ap": "Ap — water element",
    "jala": "Jala — water element",
    "udaka": "Udaka — water",
    "toya": "Toya — water",
    "vāri": "Vāri — water",
    "ambu": "Ambu — water",
    "ākāśa": "Ākāśa — space / ether element",

    # Text structure
    "sūtra": "Sūtra — aphorism",
    "sthāna": "Sthāna — section",
    "adhyāya": "Adhyāya — chapter",
    "tantra": "Tantra — treatise",
    "saṃhitā": "Saṃhitā — compendium",
    "samāsa": "Samāsa — compound / brief",
    "samāsena": "in brief / summarily",
    "vistareṇa": "in detail",
    "iti": "iti — thus",

    # Vyākhyā verbs (for अथातो...व्याख्यास्यामः type opening)
    "vyākhyā": "Vyākhyā — exposition",
    "vyākhyāsyāmaḥ": "vyākhyāsyāmaḥ — we shall expound",
    "vyākhyāta": "Vyākhyāta — expounded",

    # Persons
    "agniveśa": "Agniveśa — Caraka's predecessor",
    "ātreya": "Ātreya — the teacher",
    "punarvasu": "Punarvasu Ātreya — the teacher",
    "caraka": "Caraka — the redactor",
    "dṛḍhabala": "Dṛḍhabala — the completer of Caraka Saṃhitā",
    "bharadvāja": "Bharadvāja — sage",
    "indra": "Indra — lord of devas",
    "brahma": "Brahma — Creator deity",
    "prajāpati": "Prajāpati — lord of beings",
    "aśvin": "Aśvin — the Aśvin twins",
    "bhagavān": "Bhagavān — the venerable",
    "ṛṣi": "Ṛṣi — sage",
    "maharṣi": "Maharṣi — great sage",
    "muni": "Muni — sage",
    "ācārya": "Ācārya — teacher",
    "vaidya": "Vaidya — physician",
    "bhiṣaj": "Bhiṣaj — physician",
    "cikitsaka": "Cikitsaka — physician",
    "rogī": "Rogī — patient",
    "ātura": "Ātura — patient",

    # Life
    "jīvita": "jīvita — life",
    "jīvitam": "jīvitam — life",
    "jīva": "jīva — living being",
    "āyu": "āyu — lifespan",
    "āyus": "āyus — lifespan",
    "āyurveda": "Āyurveda — the science of life",
    "dīrgha": "dīrgha — long, prolonged",
    "hita": "hita — beneficial",
    "ahita": "ahita — harmful",
    "sukha": "sukha — happy",
    "duḥkha": "duḥkha — sorrowful",

    # Body
    "śarīra": "Śarīra — body",
    "deha": "Deha — body",
    "kāya": "Kāya — body",
    "aṅga": "Aṅga — limb / body part",
    "avayava": "Avayava — body part",
    "hṛdaya": "Hṛdaya — heart",
    "hṛd": "Hṛd — heart",
    "śiras": "Śiras — head",
    "mukha": "Mukha — face / mouth",
    "uras": "Uras — chest",
    "pakvāśaya": "Pakvāśaya — large intestine",
    "āmāśaya": "Āmāśaya — stomach",
    "bhasma": "Bhasma — ash",
    "bhasman": "Bhasman — ash",

    # Food
    "āhāra": "Āhāra — food / diet",
    "anna": "Anna — food (esp. staple grain)",
    "anne": "Anna — food [locative: in the food]",
    "annaṃ": "Anna — food [accusative/nominative]",
    "bhojana": "Bhojana — meal / eating",
    "pāna": "Pāna — drink",
    "peya": "Peya — thin gruel",
    "yūṣa": "Yūṣa — soup",
    "kṣīra": "Kṣīra — milk",
    "dadhi": "Dadhi — curd",
    "takra": "Takra — buttermilk",
    "ghṛta": "Ghṛta — ghee",
    "sarpis": "Sarpis — ghee",
    "taila": "Taila — oil",
    "madhu": "Madhu — honey",
    "guḍa": "Guḍa — jaggery",
    "śarkarā": "Śarkarā — sugar",
    "madya": "Madya — alcoholic drink",
    "śāka": "Śāka — vegetable",
    "phala": "Phala — fruit / result",
    "dhānya": "Dhānya — grain",
    "śāli": "Śāli — rice",
    "yava": "Yava — barley",
    "godhūma": "Godhūma — wheat",
    "māṣa": "Māṣa — black gram",
    "mudga": "Mudga — green gram",

    # Lifestyle
    "vihāra": "Vihāra — lifestyle / conduct",
    "ācāra": "Ācāra — conduct",
    "sadvṛtta": "Sadvṛtta — virtuous conduct",
    "dinacaryā": "Dinacaryā — daily regimen",
    "rātricaryā": "Rātricaryā — night regimen",
    "ṛtucaryā": "Ṛtucaryā — seasonal regimen",

    # Seasons
    "ṛtu": "Ṛtu — season",
    "vasanta": "Vasanta — spring",
    "grīṣma": "Grīṣma — summer",
    "varṣā": "Varṣā — monsoon",
    "śarat": "Śarat — autumn",
    "hemanta": "Hemanta — early winter",
    "śiśira": "Śiśira — late winter",

    # Time
    "kāla": "Kāla — time",
    "dina": "Dina — day",
    "rātri": "Rātri — night",
    "māsa": "Māsa — month",
    "varṣa": "Varṣa — year",
    "prātaḥ": "Prātaḥ — morning",
    "sāyam": "Sāyam — evening",

    # Treatment
    "cikitsā": "Cikitsā — treatment / therapy",
    "cikitsita": "Cikitsita — treated",
    "auṣadha": "Auṣadha — medicine",
    "bheṣaja": "Bheṣaja — medicine",
    "kalpa": "Kalpa — formulation",
    "dravya": "Dravya — substance",
    "mātrā": "Mātrā — dose",
    "sevana": "Sevana — consumption",
    "prayoga": "Prayoga — application",
    "rasāyana": "Rasāyana — rejuvenation therapy",
    "vājīkaraṇa": "Vājīkaraṇa — aphrodisiac therapy",
    "pathya": "Pathya — wholesome",
    "apathya": "Apathya — unwholesome",
    "pañcakarma": "Pañcakarma — the five cleansing procedures",
    "śodhana": "Śodhana — cleansing therapy",
    "śamana": "Śamana — palliative therapy",
    "langhana": "Langhana — depletion therapy",
    "bṛṃhaṇa": "Bṛṃhaṇa — nourishing therapy",
    "rūkṣaṇa": "Rūkṣaṇa — drying therapy",
    "snehana": "Snehana — oleation therapy",
    "svedana": "Svedana — sudation therapy",
    "sneha": "Sneha — unction",
    "vamana": "Vamana — therapeutic emesis",
    "virecana": "Virecana — therapeutic purgation",
    "basti": "Basti — enema therapy",
    "nasya": "Nasya — nasal therapy",
    "raktamokṣaṇa": "Raktamokṣaṇa — bloodletting",
    "niruha": "Niruha — decoction enema",
    "anuvāsana": "Anuvāsana — oil enema",
    "dhūmapāna": "Dhūmapāna — therapeutic smoking",
    "lepa": "Lepa — topical paste",
    "abhyaṅga": "Abhyaṅga — oil massage",
    "udvartana": "Udvartana — dry massage",
    "upavāsa": "Upavāsa — fasting",
    "upavāsayet": "upavāsayet — should fast / cause to fast",
    "upavāsana": "Upavāsana — fasting",

    # Processes
    "vardhana": "Vardhana — increase",
    "kṣaya": "Kṣaya — decrease / wasting",
    "kopa": "Kopa — aggravation",
    "prakopa": "Prakopa — aggravation",
    "praśama": "Praśama — pacification",
    "sañcaya": "Sañcaya — accumulation",
    "vyakti": "Vyakti — manifestation",
    "bheda": "Bheda — differentiation / rupture",

    # Common verbs
    "bhavati": "bhavati — becomes / is",
    "bhavanti": "bhavanti — they become / are",
    "ucyate": "ucyate — is said / is called",
    "ucyante": "ucyante — are called",
    "syāt": "syāt — should be / may be",
    "syuḥ": "syuḥ — they should be",
    "vidyate": "vidyate — exists",
    "vidyante": "vidyante — they exist",
    "vartate": "vartate — exists / functions",
    "jāyate": "jāyate — is born / arises",
    "pravartate": "pravartate — proceeds",
    "nivartate": "nivartate — ceases",
    "labhate": "labhate — obtains",
    "prāpnoti": "prāpnoti — attains",
    "gacchati": "gacchati — goes",
    "karoti": "karoti — does",
    "kurute": "kurute — does",
    "kuryāt": "kuryāt — should do",
    "dadāti": "dadāti — gives",
    "dhārayet": "dhārayet — should bear",
    "vipac": "vipac — to digest",
    "vipacyate": "vipacyate — is digested (passive)",
    "pacyate": "pacyate — is cooked / digested",
    "janayati": "janayati — produces",
    "nāśayati": "nāśayati — destroys",
    "hanti": "hanti — destroys",
    "nihanti": "nihanti — strikes down",
    "praśamayati": "praśamayati — pacifies",
    "apaharati": "apaharati — removes",
    "vardhayati": "vardhayati — increases",
    "kṣīyate": "kṣīyate — is diminished",
    "prāha": "prāha — said",
    "āha": "āha — said",
    "uvāca": "uvāca — said",

    # Gerunds / infinitives
    "bhoktum": "bhoktum — to eat",
    "pātum": "pātum — to drink",
    "kartum": "kartum — to do",
    "bhūtvā": "bhūtvā — having become",
    "kṛtvā": "kṛtvā — having done",
    "gatvā": "gatvā — having gone",
    "dṛṣṭvā": "dṛṣṭvā — having seen",
    "śrutvā": "śrutvā — having heard",
    "tyaktvā": "tyaktvā — having abandoned",

    # Pronouns / quantifiers
    "sarva": "sarva — all",
    "anya": "anya — other",
    "sva": "sva — own / self",
    "yad": "yad — which",
    "tad": "tad — that",
    "idam": "idam — this",
    "eṣa": "eṣa — this",
    "eṣaḥ": "eṣaḥ — this one",
    "tasmāt": "tasmāt — from that / therefore",
    "tasmād": "tasmāt — therefore [sandhi: t→d before vowel]",
    "tasmin": "tasmin — in that",
    "atra": "atra — here",

    # Numerals
    "eka": "eka — one",
    "dvi": "dvi — two",
    "tri": "tri — three",
    "trayaḥ": "trayaḥ — three",
    "trayas": "trayas — three",
    "trayo": "trayo — three [sandhi form]",
    "catur": "catur — four",
    "pañca": "pañca — five",
    "ṣaṭ": "ṣaṭ — six",
    "sapta": "sapta — seven",
    "aṣṭa": "aṣṭa — eight",
    "aṣṭau": "aṣṭau — eight",
    "nava": "nava — nine",
    "daśa": "daśa — ten",
    "śata": "śata — hundred",
    "sahasra": "sahasra — thousand",
    "bahu": "bahu — many",
    "alpa": "alpa — little",

    # Adjectives
    "mahā": "mahā — great",
    "mahat": "mahat — great",
    "prathama": "prathama — first",
    "dvitīya": "dvitīya — second",
    "tṛtīya": "tṛtīya — third",
    "uttama": "uttama — best",
    "madhyama": "madhyama — middle",
    "śreṣṭha": "śreṣṭha — best",
    "nitya": "nitya — constant",
    "hrasva": "hrasva — short",
    "viṣama": "viṣama — irregular",
    "sama": "sama — equal / balanced",
    "pūrva": "pūrva — prior / eastern",
    "uttara": "uttara — later / northern",
    "apara": "apara — other / later",

    # Compound-internal stems
    "channa": "channa — covered / obscured",
    "chādita": "chādita — covered",
    "āvṛta": "āvṛta — covered",
    "samvṛta": "samvṛta — enclosed",
    "yukta": "yukta — joined / endowed with",
    "sahita": "sahita — together with",
    "hīna": "hīna — deficient",
    "rahita": "rahita — devoid of",
    "pūrṇa": "pūrṇa — full / complete",
    "vardhita": "vardhita — increased",
    "kṣīṇa": "kṣīṇa — diminished",
    "śuddha": "śuddha — pure",
    "aśuddha": "aśuddha — impure",

    # Passive participles
    "udāhṛta": "udāhṛta — stated",
    "kathita": "kathita — said",
    "ukta": "ukta — said",
    "prokta": "prokta — said / declared",
    "nirdiṣṭa": "nirdiṣṭa — indicated",
    "uddiṣṭa": "uddiṣṭa — mentioned",
    "upadiṣṭa": "upadiṣṭa — taught",
    "prāpta": "prāpta — obtained",
    "gata": "gata — gone",
    "sthita": "sthita — located",

    # Verbal roots
    "kṛ": "kṛ — to do",
    "bhū": "bhū — to be",
    "gam": "gam — to go",
    "dā": "dā — to give",
    "dhā": "dhā — to hold",
    "jan": "jan — to be born",
    "sthā": "sthā — to stand",
    "tap": "tap — to heat",
    "vid": "vid — to know",

    # Abstract terms
    "jñāna": "jñāna — knowledge",
    "vijñāna": "vijñāna — specialized knowledge",
    "smṛti": "smṛti — memory",
    "dhī": "dhī — intellect",
    "prajñā": "prajñā — wisdom",
    "prajñāparādha": "Prajñāparādha — intellectual transgression",
    "dharma": "dharma — duty / quality",
    "adharma": "adharma — unrighteousness",
    "karma": "karma — action",
    "moha": "moha — delusion",
    "lobha": "lobha — greed",
    "krodha": "krodha — anger",
    "kāma": "kāma — desire",
    "mada": "mada — intoxication",
    "śoka": "śoka — sorrow",
    "bhaya": "bhaya — fear",

    # Conjunctions / adverbs
    "yadā": "yadā — when",
    "tadā": "tadā — then",
    "yatra": "yatra — where",
    "atha": "atha — now / then",
    "athātaḥ": "now henceforth (text-opening)",
    "ataḥ": "ataḥ — hence",
    "tataḥ": "tataḥ — thereafter",
    "iha": "iha — here",
    "evam": "evam — thus",
    "punar": "punar — again",
    "saha": "saha — together",
    "vinā": "vinā — without",

    # Persons
    "mānava": "mānava — human being",
    "manuṣya": "manuṣya — human",
    "nara": "nara — man / human",
    "deva": "deva — god",
    "pitṛ": "pitṛ — father",
    "mātṛ": "mātṛ — mother",
    "putra": "putra — son",
    "śiṣya": "śiṣya — student",

    # Compound-permitted particles (appear inside compounds after sandhi)
    "iva": "iva — like / as",
    "eva": "eva — just / only (or 'like/as' after a+i sandhi)",
    "api": "api — also / even",
    "yathā": "yathā — as",
    "tathā": "tathā — so / thus",
    "yadi": "yadi — if",
}

# Particles that match ONLY as full standalone words (never inside compounds)
STANDALONE_ONLY: dict[str, str] = {
    "ca": "ca — and",
    "tu": "tu — but",
    "hi": "hi — indeed / for",
    "eva": "eva — just / only",
    "na": "na — not",
    "vā": "vā — or",
    "ha": "ha — indeed",
    "kila": "kila — indeed",
}

# INFLECTIONS: (ending, replacement, min_stem_len)
# min_stem_len prevents noise matches: single-char endings need stem ≥5 chars
INFLECTIONS: list[tuple[str, str, int]] = [
    # long endings first (5+ chars)
    ('ebhyaḥ', 'a', 3), ('ābhyāṃ', 'ā', 3), ('ābhyaḥ', 'ā', 3),
    ('ānāṃ', 'a', 3), ('ānām', 'a', 3), ('eṣām', 'a', 3),
    ('asmāt', 'a', 3), ('asmin', 'a', 3),
    # -a stem
    ('asya', 'a', 3), ('aiḥ', 'a', 3), ('ena', 'a', 3), ('eṇa', 'a', 3),
    ('eṣu', 'a', 3), ('āni', 'a', 3), ('āṇi', 'a', 3),
    ('āḥ', 'a', 3), ('āṃ', 'a', 3), ('ān', 'a', 4), ('ām', 'a', 4),
    ('āya', 'a', 3), ('āt', 'a', 4), ('au', 'a', 4),
    ('āj', 'a', 4),  # ablative sandhi: -āt + voiced consonant → -āj (e.g. pacanāt+j → pacanāj)
    ('ād', 'āt', 4),  # ablative sandhi: -āt + vowel → -ād (e.g. tasmāt+ā → tasmād)
    ('aḥ', 'a', 4), ('aṃ', 'a', 4), ('o', 'aḥ', 4),
    # -e (locative singular) — important
    ('e', 'a', 4),
    # -u stem
    ('avaḥ', 'u', 3), ('ūn', 'u', 4),
    # -i stem
    ('ayaḥ', 'i', 3), ('īnām', 'i', 3), ('iḥ', 'i', 4),
    # Verb endings
    ('ante', 'a', 3), ('anti', 'a', 3),
    ('āmi', 'a', 3), ('āmaḥ', 'a', 3),
    ('yati', 'a', 3), ('ayati', 'a', 3),
    # Gerund
    ('tvā', '', 4), ('tum', '', 4),
    # Single-char endings (STRICT — stem must be ≥5 chars)
    ('ḥ', '', 5), ('ṃ', '', 5), ('m', '', 5), ('n', '', 5),
    ('t', '', 5), ('d', '', 5),
]


class SanskritSLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, ff_dim,
                 max_seq_len, dropout, pad_id=0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        x = self.layer_norm(x)
        x = self.encoder(x, src_key_padding_mask=(token_ids == self.pad_id))
        return self.mlm_head(x), x

    def get_sentence_embedding(self, token_ids):
        with torch.no_grad():
            _, x = self.forward(token_ids)
            mask = (token_ids != self.pad_id).unsqueeze(-1).float()
            return (x * mask).sum(1) / mask.sum(1).clamp(min=1)


class CharakaAnalyzer:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self._load_all()

    def _load_all(self) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.models_dir / 'tokenizer.model'))

        ckpt = torch.load(self.models_dir / 'trained_model.pt',
                          map_location='cpu', weights_only=False)
        self.config = ckpt['config']
        self.pad_id = ckpt.get('pad_id', 0)
        self.bos_id = ckpt.get('bos_id', 2)
        self.eos_id = ckpt.get('eos_id', 3)
        self.max_seq_len = self.config['max_seq_len']

        self.model = SanskritSLM(**self.config, pad_id=self.pad_id)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.embeddings = np.load(self.models_dir / 'embeddings.npy')
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / (norms + 1e-8)

        with open(self.models_dir / 'corpus_train.txt', encoding='utf-8') as f:
            self.corpus_lines = [l.strip() for l in f if l.strip()]
        self.df_meta = pd.read_csv(self.models_dir / 'corpus_clean.csv')

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
                        'iast': t, 'source': 'WHO',
                    }

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
        for ending, repl, min_len in INFLECTIONS:
            if w.endswith(ending) and len(w) > len(ending):
                cand = w[:-len(ending)] + repl
                if len(cand) < min_len:
                    continue
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

    def decompose_compound(self, iast_str: str, min_len: int = 3) -> list:
        """DP best-cover with length-weighted scoring (length^1.5)."""
        n = len(iast_str)
        if n < min_len:
            return []
        dp: list = [(0.0, [])] * (n + 1)
        for end in range(min_len, n + 1):
            best = (dp[end - 1][0], dp[end - 1][1])
            for start in range(max(0, end - 25), end - min_len + 1):
                sub = iast_str[start:end]
                info = self.lookup_iast(sub, allow_particles=False)
                if info is None:
                    continue
                length = end - start
                match_score = length ** 1.5
                prev_score, prev_matches = dp[start]
                new_score = prev_score + match_score
                if new_score > best[0]:
                    best = (new_score, prev_matches + [(start, end, sub, info)])
            dp[end] = best
        matches = dp[n][1]
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
            # Strip trailing dandas for cleaner display
            w_clean = re.sub(r'[।॥]+\s*$', '', w).strip()
            if not w_clean:
                continue
            r = self.lookup_word(w_clean)
            if r:
                results.append((w_clean, [(w_clean, r)]))
                continue
            decomp = self.decompose_compound(dev_to_iast(w_clean).lower())
            results.append((w_clean, decomp))
        return results

    def embed_sloka(self, sloka: str) -> np.ndarray:
        ids = [self.bos_id] + self.sp.EncodeAsIds(sloka)[:self.max_seq_len - 2] + [self.eos_id]
        ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        t = torch.tensor([ids], dtype=torch.long)
        return self.model.get_sentence_embedding(t)[0].numpy()

    def find_similar(self, sloka: str, top_k: int = 5, exclude_self: bool = True) -> list:
        q = self.embed_sloka(sloka)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        sims = self.embeddings_norm @ q_norm
        if exclude_self:
            for i, line in enumerate(self.corpus_lines):
                if line == sloka:
                    sims[i] = -1
                    break
        top = np.argsort(-sims)[:top_k]
        return [{
            'chapter': int(self.df_meta.iloc[int(idx)]['chapter_num']),
            'sloka_num': int(self.df_meta.iloc[int(idx)]['source_sloka_num']),
            'text': self.corpus_lines[int(idx)],
            'similarity': float(sims[int(idx)]),
        } for idx in top]

    def build_gloss(self, word_analyses: list) -> str:
        parts = []
        for w, subs in word_analyses:
            if not subs:
                parts.append(f"[{w}]")
            elif len(subs) == 1:
                parts.append(re.split(r'\s*[—(]\s*', subs[0][1]['english'], maxsplit=1)[0])
            else:
                inner = ' + '.join(
                    re.split(r'\s*[—(]\s*', s[1]['english'], maxsplit=1)[0] for s in subs
                )
                parts.append(f"({inner})")
        return ' '.join(parts)

    def analyze_sloka(self, sloka: str, top_k_similar: int = 5) -> dict:
        sloka = sloka.strip()
        wa = self.analyze_words(sloka)
        return {
            'input': sloka,
            'words': wa,
            'english_gloss': self.build_gloss(wa),
            'similar': self.find_similar(sloka, top_k=top_k_similar),
        }
