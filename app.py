# app.py
# Streamlit app for the Phoenician 11-gon spell system.

import io
import re
import random
import math
from typing import Dict, List, Tuple
from pathlib import Path




import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm


from phoenician_11gon_system import (
    Phoenician11System,
    PHOENICIAN_NAMES_22,
    phoenician_glyph,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kodex", layout="wide")
st.title("🜂 Kodex")
st.caption("Zakódujte si své kouzlo — 11-gon rules enforced, one word per glyph.")

# ──────────────────────────────────────────────────────────────────────────────
# Engine (no lexicon mutation)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_engine():
    font_path = Path("fonts/NotoSansPhoenician-Regular.ttf").as_posix()
    g = Phoenician11System(glyph_font_path=font_path)

    # Tagged meanings for fallback only (LEXICON below is the primary source of words)
    g.set_meanings({
        1:"noun: Origin; adj: Primordial",
        2:"noun: House; verb: Shelter; adj: Domestic; adv: Safely",
        3:"verb: Guide; adv: Gently",
        4:"noun: Door; verb: Open; adj: Threshold",
        5:"adj: Merciful; noun: Mercy",
        6:"verb: Link; noun: Linkage",
        7:"noun: Weapon; adj: Keen",
        8:"adj: Steady; noun: Balance",
        9:"noun: Coil; verb: Bind",
        10:"verb: Lend; noun: Gift",
        11:"noun: Palm; adj: Open",
        12:"verb: Goad; noun: Spur",
        13:"noun: Water; adj: Flowing; verb: Wash",
        14:"verb: Nourish; noun: Sustenance",
        15:"noun: Prop; adj: Supporting",
        16:"noun: Eye; verb: Watch; adj: Watchful; adv: Watchfully",
        17:"verb: Speak; noun: Voice; adv: Clearly",
        18:"verb: Hook; noun: Anchor",
        19:"noun: Back of head; adj: Hidden",
        20:"noun: Head; adj: Leading",
        21:"noun: Tooth; adj: Biting",
        22:"noun: Mark; verb: Sign; adj: Marked; adv: Indelibly",
    })

    # Example costs + stackable multipliers
    g.set_costs({1:2.0,2:1.3,3:1.7,4:2.2,5:1.1,6:1.9,7:2.8,8:1.6,9:1.4,10:1.2,11:1.5,
                 12:2.5,13:1.8,14:2.1,15:1.3,16:2.7,17:1.6,18:1.9,19:2.2,20:1.4,21:2.0,22:1.1})
    g.set_connection_multipliers({2: 1.5, 4: 1.5, 6: 2, 8: 2, 10: 3})
    return g

g = init_engine()

# ──────────────────────────────────────────────────────────────────────────────
# Read-only LEXICON (primary word source; DO NOT install into engine)
# ──────────────────────────────────────────────────────────────────────────────
LEXICON: Dict[int, Dict[str, str]] = {
    1:  {'noun':'Origin',   'verb':'Begin',   'adj':'Primordial',  'adv':'Originally'},
    2:  {'noun':'House',    'verb':'Shelter', 'adj':'Domestic',    'adv':'Safely'},
    3:  {'noun':'Caravan',  'verb':'Convey',  'adj':'Enduring',    'adv':'Steadily'},
    4:  {'noun':'Door',     'verb':'Open',    'adj':'Liminal',     'adv':'Inwardly'},
    5:  {'noun':'Breath',   'verb':'Reveal',  'adj':'Aerial',      'adv':'Openly'},
    6:  {'noun':'Link',     'verb':'Bind',    'adj':'Joined',      'adv':'Together'},
    7:  {'noun':'Blade',    'verb':'Cleave',  'adj':'Keen',        'adv':'Sharply'},
    8:  {'noun':'Wall',     'verb':'Ward',    'adj':'Enclosed',    'adv':'Securely'},
    9:  {'noun':'Coil',     'verb':'Constrict','adj':'Coiled',     'adv':'Tightly'},
    10: {'noun':'Hand',     'verb':'Grasp',   'adj':'Dexterous',   'adv':'Skillfully'},
    11: {'noun':'Palm',     'verb':'Offer',   'adj':'Openhanded',  'adv':'Generously'},
    12: {'noun':'Lesson',   'verb':'Teach',   'adj':'Instructive', 'adv':'Wisely'},
    13: {'noun':'Water',    'verb':'Wash',    'adj':'Fluid',       'adv':'Fluidly'},
    14: {'noun':'Seed',     'verb':'Sprout',  'adj':'Fecund',      'adv':'Fruitfully'},
    15: {'noun':'Support',  'verb':'Sustain', 'adj':'Supporting',  'adv':'Steadfastly'},
    16: {'noun':'Eye',      'verb':'Watch',   'adj':'Watchful',    'adv':'Vigilantly'},
    17: {'noun':'Voice',    'verb':'Speak',   'adj':'Articulate',  'adv':'Plainly'},
    18: {'noun':'Snare',    'verb':'Catch',   'adj':'Exacting',    'adv':'Precisely'},
    19: {'noun':'Hindsight','verb':'Recall',  'adj':'Hidden',      'adv':'Obliquely'},
    20: {'noun':'Leader',   'verb':'Lead',    'adj':'Foremost',    'adv':'Chiefly'},
    21: {'noun':'Flame',    'verb':'Ignite',  'adj':'Fiery',       'adv':'Fiercely'},
    22: {'noun':'Seal',     'verb':'Sign',    'adj':'Marked',      'adv':'Indelibly'},
}

# Presets (password-locked legacy)
GOOD_SPELLS = {
    "Copperfield_Good": ["Beth","Daleth","Yod","Waw","Gimel","Kaph","Zayin","Lamed","Teth","He","Heth"],
    "Scholars_Good":    ["Lamed","Pe","Qoph","Mem","Tsade","Samekh","Taw","Ayin","Resh","Nun","Shin"],
    "Verdant_Good":     ["Mem","Ayin","Lamed","Tsade","Yod","Samekh","Qoph","Kaph","Nun","Pe","Teth"],
    "Span_Good":        ["Yod","Zayin","Lamed","Samekh","Heth","Mem","Teth","Nun","Waw","Kaph","Ayin"],
    "River_Good":       ["Mem","Waw","Lamed","Ayin","Nun","Heth","Kaph","Teth","Zayin","Yod","Samekh"],
}

EVIL_SPELLS = {
    "Copperfield_Evil": ["Beth","Daleth","Yod","Waw","Gimel","Kaph","Zayin","Aleph","Teth","He","Heth"],
    "Scholars_Evil":    ["Lamed","Pe","Qoph","Mem","Tsade","Samekh","Kaph","Ayin","Resh","Nun","Shin"],
    "Verdant_Evil":     ["Mem","Ayin","Lamed","Tsade","Yod","Samekh","Qoph","Kaph","Nun","Pe","Resh"],
    "Span_Evil":        ["Yod","Zayin","Lamed","Samekh","Heth","Mem","Teth","Nun","Pe","Kaph","Ayin"],
    "River_Evil":       ["Mem","Pe","Lamed","Ayin","Nun","Heth","Kaph","Teth","Zayin","Yod","Samekh"],
}



# ──────────────────────────────────────────────────────────────────────────────
# Meaning-driven D&D preset generator (seeded, avoids neighbors where possible)
# Level lengths: 1→2, 2→3, 3→4, 4→5, 5→6, 6→8
# L1–L5: no adjacent letters; L6: allow up to 2 adjacencies if needed.
# Always fits a single 11-letter window (so solver works).
# ──────────────────────────────────────────────────────────────────────────────

# keyword → preferred letter indices (1..22)
THEME_PREFS: Dict[str, List[int]] = {
    # fire / damage
    "fire": [21,5,14], "flame":[21,5], "ignite":[21], "ball":[21,14],
    # cold / barriers
    "cold":[13,22,8], "cone":[8,13], "wall":[8,15], "stone":[8,15],
    # defense / shield
    "shield":[8,15,6], "ward":[8,6], "barrier":[8,15],
    # healing / cleansing
    "cure":[11,13], "heal":[11,13], "wounds":[11,13], "wash":[13],
    # magic / meta / spellcraft
    "magic":[22,12,17], "spell":[22,12], "counterspell":[22,17,12], "dispel":[22,12],
    # seeing / knowledge
    "see":[16,17], "seeing":[16,17], "true":[16,22], "detect":[16,12],
    # stealth / illusion
    "invisibility":[19,16], "invisible":[19,16], "hidden":[19], "blur":[19,13],
    "mirror":[16,11,22], "image":[16,11],
    # motion / travel / control
    "misty":[13,5], "step":[10,6], "fly":[5,10],
    "dimension":[4,1,20], "door":[4], "plane":[1,20], "shift":[22,13,4],
    "telekinesis":[10,6,22], "move":[10,6],
    # control / unmake
    "disintegrate":[21,18,22], "shatter":[21,18],
    # change
    "polymorph":[14,13,22], "greater":[20,22],
    # wish / speech / seal
    "wish":[22,17,1], "speak":[17], "seal":[22],
}

DND_TITLES = [
    "Magic Missile (1)", "Shield (1)", "Cure Wounds (1)", "Detect Magic (1)",
    "Invisibility (2)", "Mirror Image (2)", "Misty Step (2)", "Blur (2)",
    "Fireball (3)", "Counterspell (3)", "Fly (3)", "Dispel Magic (3)",
    "Dimension Door (4)", "Polymorph (4)", "Greater Invisibility (4)",
    "Cone of Cold (5)", "Wall of Stone (5)", "Telekinesis (5)",
    "Disintegrate (6)", "True Seeing (6)"
]

def level_to_len(level: int) -> int:
    return {1:2, 2:3, 3:4, 4:5, 5:6, 6:8}.get(level, 8)

def intent_keywords(preset_label: str) -> List[str]:
    base = preset_label.lower()
    base = re.sub(r"\(.*?\)", "", base)
    tokens = re.findall(r"[a-z]+", base)
    extra = []
    for t in tokens:
        if t.endswith("ing"): extra.append(t[:-3])
        if t.endswith("ed"):  extra.append(t[:-2])
        if t == "fireball":   extra += ["fire","ignite","ball"]
        if t == "counterspell": extra += ["counterspell","dispel","spell"]
        if t == "dispel":     extra += ["dispel","spell"]
        if t == "true":       extra += ["true"]
        if t == "seeing":     extra += ["see","seeing"]
    return list(dict.fromkeys(tokens + extra))

def score_letter(letter: int, kws: List[str]) -> float:
    score = 0.0
    for kw in kws:
        prefs = THEME_PREFS.get(kw, [])
        if not prefs:
            continue
        if letter in prefs:
            idx = prefs.index(letter)
            score += [1.0, 0.7, 0.5, 0.3, 0.2][idx] if idx < 5 else 0.1
    return score

def is_adjacent(a: int, b: int) -> bool:
    return (a - b) % 22 in (1, 21)

def fits_in_window(indices: List[int]) -> bool:
    if not indices: return False
    xs = sorted(indices)
    xs2 = xs + [x+22 for x in xs]
    k = len(xs)
    best_span = 99
    for i in range(len(xs)):
        span = xs2[i+k-1] - xs2[i]
        if span < best_span: best_span = span
    return best_span <= 10

def softmax_sample(candidates: List[int], weights: List[float], temperature: float, rng: random.Random) -> int:
    t = max(0.05, float(temperature))
    ws = [math.exp(w / t) for w in weights]
    s = sum(ws) or 1.0
    probs = [w/s for w in ws]
    r = rng.random()
    acc = 0.0
    for idx, p in zip(candidates, probs):
        acc += p
        if r <= acc:
            return idx
    return candidates[-1]

def infer_level_from_title(title: str) -> int:
    m = re.search(r"\((\d+)\)", title)
    return int(m.group(1)) if m else 6

def generate_dnd_glyph_names(title: str, temperature: float = 0.7, seed: int = 42) -> List[str]:
    level = infer_level_from_title(title)
    target = level_to_len(level)
    kws = intent_keywords(title)
    base_scores = {j: score_letter(j, kws) + 1e-3 for j in range(1, 23)}
    adj_budget = 0 if level <= 5 else 2  # allow up to two adjacencies for level 6
    rng = random.Random(seed)

    best: List[int] = []
    for _ in range(60):
        chosen: List[int] = []
        used_adj_pairs = 0
        pool = list(range(1, 23))

        while len(chosen) < target and pool:
            cand = []
            for j in pool:
                conflict = any(is_adjacent(j, c) for c in chosen)
                if not conflict:
                    cand.append(j)
                else:
                    if used_adj_pairs < adj_budget:
                        cand.append(j)
            if not cand:
                break
            weights = [base_scores[j] for j in cand]
            pick = softmax_sample(cand, weights, temperature, rng)
            if any(is_adjacent(pick, c) for c in chosen):
                used_adj_pairs += 1
            chosen.append(pick)
            pool.remove(pick)

        if len(chosen) == target and fits_in_window(chosen):
            best = chosen
            break

    if not best:
        # fallback greedy
        chosen, used = [], 0
        for j in sorted(range(1,23), key=lambda x: -base_scores[x]):
            if len(chosen) >= target: break
            if any(is_adjacent(j, c) for c in chosen):
                if used < adj_budget:
                    used += 1
                else:
                    continue
            chosen.append(j)
        best = chosen[:target]

    return [PHOENICIAN_NAMES_22[j-1] for j in best]

# ──────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ──────────────────────────────────────────────────────────────────────────────
def ensure_png_for_spell(spell: Dict, basename: str) -> str:
    path = f"{basename}.png"
    if spell.get("view") == "Glyphs":
        g.draw_glyphs(spell["vertices"], S=spell["S"], start_shift=spell["start_shift"], savepath=path)
    elif spell.get("view") == "Names":
        g.draw_named(spell["vertices"], S=spell["S"], start_shift=spell["start_shift"], savepath=path)
    else:
        g.draw_cryptic(spell["vertices"], S=spell["S"], start_shift=spell["start_shift"], savepath=path)
    return path

def build_folio_pdf(folio: List[Dict]) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        raise RuntimeError("reportlab is not installed. Install with: pip install reportlab") from e

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    for idx, spell in enumerate(folio, start=1):
        c.setFont("Helvetica-Bold", 18)
        title = spell.get("title", "Spell")
        c.drawString(20*mm, H - 25*mm, title)

        meta = f"S={spell.get('S')}   start_shift={spell.get('start_shift')}   vertices={spell.get('vertices')}"
        c.setFont("Helvetica", 11)
        c.drawString(20*mm, H - 33*mm, meta)

        names_line = " / ".join(spell.get("names", []))
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(20*mm, H - 40*mm, names_line)

        png_path = ensure_png_for_spell(spell, f"spell_{idx}_diagram")
        try:
            img = ImageReader(png_path)
            img_w = 90*mm; img_h = 90*mm
            c.drawImage(img, 20*mm, H - 40*mm - img_h - 10*mm, width=img_w, height=img_h)
        except Exception:
            pass

        words_line = "  ·  ".join(spell.get("words", []))
        c.setFont("Helvetica", 12)
        c.drawString(20*mm, H - 40*mm - 90*mm - 14*mm, words_line)

        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()

# ──────────────────────────────────────────────────────────────────────────────
# Role picking (deterministic, context-aware) + pompous name
# ──────────────────────────────────────────────────────────────────────────────
ROLE_ORDER = ("noun", "verb", "adj", "adv")

def _parse_roles_tagged(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(raw, str) and raw:
        for part in raw.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ROLE_ORDER and v:
                    out[k] = v
    return out

def roles_for_letter(letter: int) -> Dict[str, str]:
    if letter in LEXICON:
        base = {k: v for k, v in LEXICON[letter].items() if v}
    else:
        base = {}
    if not base:
        tagged = _parse_roles_tagged(getattr(g, "meanings", {}).get(letter, ""))
        if tagged:
            base = tagged
    if not base:
        base = {"noun": PHOENICIAN_NAMES_22[letter-1]}
    return {k: base.get(k, "") for k in ROLE_ORDER}

def assign_roles_context(letter_indices: List[int]) -> List[str]:
    N = len(letter_indices)
    avail = [roles_for_letter(j) for j in letter_indices]
    roles = [""] * N

    def has(i, r): return bool(avail[i].get(r))
    def choose(i, r): roles[i] = r

    # Short spells: priority, then diversify if possible
    if N < 4:
        for i in range(N):
            for pref in ROLE_ORDER:
                if has(i, pref): choose(i, pref); break
        missing = [r for r in ROLE_ORDER if r not in roles]
        if missing:
            for i in range(N):
                for r in list(missing):
                    if has(i, r):
                        choose(i, r)
                        missing.remove(r)
                        break
        return [avail[i][roles[i]] if roles[i] else next((avail[i][k] for k in ROLE_ORDER if avail[i][k]), "") for i in range(N)]

    # N >= 4: aim for noun/verb/adj/adv presence
    used = set()

    # Verb near nouns/adjs
    verb_idx = None
    for i in range(1, N-1):
        if has(i, "verb") and (has(i-1, "noun") or has(i-1, "adj")) and (has(i+1, "noun") or has(i+1, "adj")):
            verb_idx = i; break
    if verb_idx is None:
        for i in range(N):
            if has(i, "verb"): verb_idx = i; break
    if verb_idx is not None:
        choose(verb_idx, "verb"); used.add(verb_idx)

    # Noun adjacent to verb preferred
    noun_idx = None
    if verb_idx is not None and verb_idx-1 >= 0 and has(verb_idx-1, "noun"): noun_idx = verb_idx-1
    elif verb_idx is not None and verb_idx+1 < N and has(verb_idx+1, "noun"):  noun_idx = verb_idx+1
    else:
        for i in range(N):
            if i not in used and has(i, "noun"): noun_idx = i; break
    if noun_idx is not None:
        choose(noun_idx, "noun"); used.add(noun_idx)

    # Adj beside noun
    adj_idx = None
    if noun_idx is not None:
        for j in (noun_idx-1, noun_idx+1):
            if 0 <= j < N and j not in used and has(j, "adj"):
                adj_idx = j; break
    if adj_idx is None:
        for i in range(N):
            if i not in used and has(i, "adj"): adj_idx = i; break
    if adj_idx is not None:
        choose(adj_idx, "adj"); used.add(adj_idx)

    # Adv beside verb
    adv_idx = None
    if verb_idx is not None:
        for j in (verb_idx-1, verb_idx+1):
            if 0 <= j < N and j not in used and has(j, "adv"):
                adv_idx = j; break
    if adv_idx is None:
        for i in range(N):
            if i not in used and has(i, "adv"): adv_idx = i; break
    if adv_idx is not None:
        choose(adv_idx, "adv"); used.add(adj_idx)

    # Fill remaining based on local context
    for i in range(N):
        if roles[i]: continue
        prev = roles[i-1] if i-1 >= 0 else ""
        if prev == "noun":
            for pref in ("verb","adv","noun","adj"):
                if has(i, pref): choose(i, pref); break
            continue
        if prev == "adj":
            for pref in ("noun","verb","adj","adv"):
                if has(i, pref): choose(i, pref); break
            continue
        if prev == "verb":
            for pref in ("noun","adv","verb","adj"):
                if has(i, pref): choose(i, pref); break
            continue
        for pref in ("noun","verb","adj","adv"):
            if has(i, pref) and pref != prev:
                choose(i, pref); break
        if not roles[i]:
            for pref in ROLE_ORDER:
                if has(i, pref): choose(i, pref); break

    # Ensure all roles appear if possible (minimal rebalance)
    present = set(roles)
    for needed in ROLE_ORDER:
        if needed in present: continue
        for i in range(N):
            if roles[i] and roles[i] != needed and avail[i].get(needed):
                rank = {"noun":0,"verb":1,"adj":2,"adv":3}
                if rank[roles[i]] > rank[needed]:
                    roles[i] = needed
                    present.add(needed)
                    break

    return [avail[i][roles[i]] if roles[i] else next((avail[i][k] for k in ROLE_ORDER if avail[i][k]), "") for i in range(N)]

# ──────────────────────────────────────────────────────────────────────────────
# Stronger, varied, deterministic title generator
# ──────────────────────────────────────────────────────────────────────────────
def _title_case(s: str) -> str:
    if not s: return s
    lowers = {"of","the","and","by","from","over","within","beneath","on","in","for","to","into","with"}
    parts = s.split()
    out = []
    for i,w in enumerate(parts):
        if i>0 and w.lower() in lowers:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)

def _unique_keep_order(xs):
    seen = set(); out=[]
    for x in xs:
        if not x: continue
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _gerund(v: str) -> str:
    if not v: return v
    w = v.split()
    head = w[0]
    if head.endswith("e") and len(head) > 2:
        head = head[:-1] + "ing"
    elif not head.endswith("ing"):
        head = head + "ing"
    return " ".join([head] + w[1:])

def _adv_to_adj(a: str) -> str:
    if not a: return a
    # cheap heuristic; good for your LEXICON ("Vigilantly"->"Vigilant", "Precisely"->"Precise")
    if a.endswith("ly") and len(a) > 3:
        return a[:-2] if a.lower().endswith("ily") else a[:-2]  # keep simple
    return a

def _phrase(adj: str, noun: str) -> str:
    if adj and noun: return f"{adj} {noun}"
    return noun or adj

def make_pompous_name(letter_indices: List[int], words: List[str]) -> str:
    """
    Deterministic, varied, and sensible:
      - Prefer distinct nouns; ornament with adjectives if available.
      - If nouns repeat/absent, use verb gerunds or adverbs→adjectives as fallbacks.
      - Rotate templates by a stable hash of the letter pattern.
    """
    # Build role pools (keep order, ensure uniqueness)
    roles_per = [roles_for_letter(j) for j in letter_indices]
    nouns = _unique_keep_order([r.get("noun","") for r in roles_per if r.get("noun")])
    adjs  = _unique_keep_order([r.get("adj","")  for r in roles_per if r.get("adj")])
    verbs = _unique_keep_order([r.get("verb","") for r in roles_per if r.get("verb")])
    advs  = _unique_keep_order([r.get("adv","")  for r in roles_per if r.get("adv")])

    # Primary & secondary concepts
    primary_n  = nouns[0] if nouns else ""
    secondary_n = ""
    for n in reversed(nouns):
        if n and n != primary_n:
            secondary_n = n; break

    # Fallbacks if nouns are missing or identical
    if not primary_n:
        primary_n = words[0] if words else "Working"
    if not secondary_n or secondary_n == primary_n:
        # try verb gerund
        if verbs:
            secondary_n = _gerund(verbs[-1]).title()  # e.g., "Watching"
        elif advs:
            secondary_n = _adv_to_adj(advs[-1]).title()
        elif len(words) > 1:
            secondary_n = words[-1]
        else:
            secondary_n = "Crown"

    # Pick ornaments (adjectives), prefer different ones for left/right
    left_adj  = adjs[0] if adjs else ""
    right_adj = ""
    for a in reversed(adjs):
        if a and a != left_adj:
            right_adj = a; break
    if not right_adj and len(adjs) >= 1:
        right_adj = adjs[0] if len(adjs) == 1 else ""

    # Compose left/right noun phrases
    left_phrase  = _phrase(left_adj,  primary_n)
    right_phrase = _phrase(right_adj, secondary_n)

    # If still identical, mix in verb/adverb to diversify
    if left_phrase.lower() == right_phrase.lower():
        if verbs:
            right_phrase = _phrase(right_adj, _gerund(verbs[-1]).title())
        elif advs:
            right_phrase = _phrase(right_adj, _adv_to_adj(advs[-1]).title())
        elif len(words) > 1:
            right_phrase = words[-1]

    # Deterministic template selection
    # Stable small hash from pattern + length
    h = (sum((i+1)*j for i,j in enumerate(letter_indices)) + 7*len(words)) % 11

    templates = [
        "The {R} of the {L}",
        "The {R} Over the {L}",
        "The {R} Beneath the {L}",
        "The Covenant of {R} and the {L}",
        "The Litany of {L} and {R}",
        "{R}: A Rite of {L}",
        "Treatise on {L}, Crowned by {R}",
        "Arcana of {L} and {R}",
        "The {R} Within the {L}",
        "Pact of the {L} and the {R}",
        "The {R} Wrought from the {L}",
    ]
    tmpl = templates[h]

    # Title-case with nice small-words handling
    title = tmpl.format(R=_title_case(right_phrase), L=_title_case(left_phrase))

    # Final polish: collapse double spaces if any blanks slipped through
    title = " ".join(title.split())
    return title

# ──────────────────────────────────────────────────────────────────────────────
# Solver — faithful to ring law (names → indices → (S, start_shift, vertices))
# ──────────────────────────────────────────────────────────────────────────────
def make_spell_from_names(
    names: List[str],
    align: str = "first_on_top",      # "first_on_top" | "a0_on_top"
    unique: bool = True,
    list_all_solutions: bool = False
) -> Dict[str, object]:
    N_ALPHA, N_VERT = 22, 11
    canon_map = {n.lower(): i+1 for i, n in enumerate(PHOENICIAN_NAMES_22)}
    aliases = {
        "vav":"waw","kaf":"kaph","caph":"kaph","tet":"teth","zain":"zayin",
        "het":"heth","chet":"heth","qof":"qoph","sin":"shin","tav":"taw","tau":"taw",
        "yud":"yod","samech":"samekh","tsadi":"tsade","tzadi":"tsade","ṣade":"tsade",
        "heh":"he","hey":"he","bet":"beth","dalet":"daleth","lamedh":"lamed"
    }
    for k, v in aliases.items():
        if v in canon_map:
            canon_map[k] = canon_map[v]

    idxs: List[int] = []
    norm_names: List[str] = []
    seen = set()
    for raw in names:
        key = str(raw).strip().lower().replace(" ","").replace("-","")
        j = canon_map.get(key)
        if j is None:
            return {"possible": False, "message": f"Unknown glyph name: {raw}",
                    "normalized_names":[], "letter_indices":[], "start_shift":None, "S":None,
                    "a0":None, "v0":None, "vertex_series":[], "window_letters":[], "alternatives":[]}
        proper = PHOENICIAN_NAMES_22[j-1]
        if unique and j in seen:
            continue
        seen.add(j); idxs.append(j); norm_names.append(proper)

    if not idxs:
        return {"possible": False, "message":"Empty selection.",
                "normalized_names":[], "letter_indices":[], "start_shift":None, "S":None,
                "a0":None, "v0":None, "vertex_series":[], "window_letters":[], "alternatives":[]}

    if len(set(idxs)) > 11:
        return {"possible": False, "message":"More than 11 distinct symbols cannot fit on a single ring.",
                "normalized_names":norm_names, "letter_indices":idxs, "start_shift":None, "S":None,
                "a0":None, "v0":None, "vertex_series":[], "window_letters":[], "alternatives":[]}

    def k_from_a0(j, a0): return (j - a0) % N_ALPHA

    valid_a0s: List[Tuple[int, List[int]]] = []
    for a0 in range(1, N_ALPHA+1):
        ks = [k_from_a0(j, a0) for j in idxs]
        if all(0 <= k <= 10 for k in ks):
            valid_a0s.append((a0, ks))
    if not valid_a0s:
        return {"possible": False, "message":"Selection cannot appear on a single 11-letter ring.",
                "normalized_names":norm_names, "letter_indices":idxs, "start_shift":None, "S":None,
                "a0":None, "v0":None, "vertex_series":[], "window_letters":[], "alternatives":[]}

    def wrap1(x, m): return ((int(x) - 1) % m) + 1

    def solution_for(a0, ks):
        if align == "first_on_top":
            k_first = ks[0]
            v0 = wrap1(1 - k_first, 11)  # place first chosen at vertex 1 (top)
        else:
            v0 = 1                        # alphabet start (a0) on top
        S = (v0 - 1) % 11
        vertices = [wrap1(v0 + k, 11) for k in ks]
        start_shift = (a0 - 1) % 22
        window = [wrap1(a0 + t, 22) for t in range(0, 11)]
        return start_shift, S, a0, v0, vertices, window

    valid_sorted = sorted(valid_a0s, key=lambda ak: ak[1][0]) if align == "first_on_top" else valid_a0s
    a0, ks = valid_sorted[0]
    start_shift, S, a0, v0, vertices, window = solution_for(a0, ks)

    alternatives = []
    if list_all_solutions:
        for a0_alt, ks_alt in valid_a0s:
            st_alt, S_alt, a0_alt2, v0_alt, verts_alt, window_alt = solution_for(a0_alt, ks_alt)
            alternatives.append({
                "start_shift": st_alt, "S": S_alt, "a0": a0_alt2, "v0": v0_alt,
                "vertex_series": verts_alt, "window_letters": window_alt
            })

    return {
        "possible": True, "message": "OK",
        "normalized_names": norm_names, "letter_indices": idxs,
        "start_shift": start_shift, "S": S, "a0": a0, "v0": v0,
        "vertex_series": vertices, "window_letters": window,
        "alternatives": alternatives
    }

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — selection
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔺 Sigil Configuration")
names_choice = st.sidebar.multiselect(
    "Vyberte až 11 znaků (jména)",
    PHOENICIAN_NAMES_22,
    max_selections=11
)

align_mode = st.sidebar.radio(
    "Zarovnání",
    options=["first_on_top", "a0_on_top"],
    format_func=lambda x: "První na vrcholu" if x=="first_on_top" else "Abecední start na vrcholu",
)

unique = st.sidebar.checkbox("Vynechat duplikáty", True)

# D&D presets — meaning-driven generator
st.sidebar.markdown("---")
st.sidebar.subheader("🧙 D&D Presety (významově řízené)")
dnd_key = st.sidebar.selectbox("Vyberte předlohu", ["—"] + DND_TITLES)
use_meaning = st.sidebar.checkbox("Vybírat glyfy podle významu", True)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000_000, value=1234, step=1)
temperature = st.sidebar.slider("Náhodnost (temperature)", 0.05, 2.0, 0.7, 0.05)

if dnd_key != "—":
    if use_meaning:
        names_choice = generate_dnd_glyph_names(dnd_key, temperature=float(temperature), seed=int(seed))
    else:
        # fallback: low randomness still based on meaning
        names_choice = generate_dnd_glyph_names(dnd_key, temperature=0.2, seed=int(seed))

# Locked legendary presets
st.sidebar.markdown("---")
st.sidebar.caption("Legendární kouzla (zamčeno heslem)")
pwd = st.sidebar.text_input("Heslo", type="password", placeholder="•••••")
if pwd.strip() == "alfik1236987+":
    tab_good, tab_evil = st.sidebar.tabs(["Good","Evil"])
    with tab_good:
        gkey = st.selectbox("GOOD", ["—"] + list(GOOD_SPELLS.keys()))
        if gkey != "—":
            names_choice = GOOD_SPELLS[gkey]
    with tab_evil:
        ekey = st.selectbox("EVIL", ["—"] + list(EVIL_SPELLS.keys()))
        if ekey != "—":
            names_choice = EVIL_SPELLS[ekey]
else:
    st.sidebar.info("Zadejte heslo pro legendární kouzla.")

# ──────────────────────────────────────────────────────────────────────────────
# Solve
# ──────────────────────────────────────────────────────────────────────────────
if not names_choice:
    st.info("Zvolte alespoň jeden symbol.")
    st.stop()

res = make_spell_from_names(
    names_choice,
    align=align_mode,
    unique=unique,
    list_all_solutions=True
)
if not res["possible"]:
    st.error(res["message"])
    st.stop()

alts = res["alternatives"] or [{
    "start_shift": res["start_shift"],
    "S": res["S"],
    "vertex_series": res["vertex_series"],
    "a0": res["a0"],
    "v0": res["v0"],
    "window_letters": res["window_letters"]
}]
choice_ix = 0
if len(alts) > 1:
    choice_ix = st.selectbox(
        "Možné řešení (S, start_shift):",
        options=list(range(len(alts))),
        format_func=lambda i: f"S={alts[i]['S']} | start_shift={alts[i]['start_shift']} | v0={alts[i]['v0']} | a0={alts[i]['a0']}"
    )

S = alts[choice_ix]["S"]
start_shift = alts[choice_ix]["start_shift"]
vertices = alts[choice_ix]["vertex_series"]

# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1])

with left:
    st.subheader("🜏 Inscribed 11-gon")
    view = st.radio("Zobrazení", ["Glyphs","Names","Cryptic"], horizontal=True)
    if view == "Glyphs":
        fig, _ = g.draw_glyphs(vertices, S=S, start_shift=start_shift)
    elif view == "Names":
        fig, _ = g.draw_named(vertices, S=S, start_shift=start_shift)
    else:
        fig, _ = g.draw_cryptic(vertices, S=S, start_shift=start_shift)
    st.pyplot(fig)
    st.caption(f"S={S}, start_shift={start_shift}, vrcholy={vertices}")

with right:
    df_steps_full = g.spell_cost_breakdown(vertices, S=S, start_shift=start_shift)
    steps_rows = df_steps_full.iloc[:-1].to_dict(orient="records")
    letter_indices = [int(r["LetterIndex"]) for r in steps_rows]

    words = assign_roles_context(letter_indices)
    spell_name = make_pompous_name(letter_indices, words)

    st.subheader(f"✨ {spell_name}")
    st.markdown("**Počet slov:** " + str(len(words)))
    st.markdown("**Sekvence:** " + " — ".join(words))

    st.subheader("Tabulka nákladů")
    slim_rows = []
    for r in steps_rows:
        j = int(r["LetterIndex"])
        slim_rows.append({
            "Step": int(r["Step"]),
            "Name": r["Name"],
            "Glyph": phoenician_glyph(j),
            "Cumulative": float(r["Cumulative"]),
        })
    st.dataframe(pd.DataFrame(slim_rows), hide_index=True, width="stretch")
    total_cost = int(round(float(df_steps_full.iloc[-1]["Cumulative"])))
    st.markdown(f"**TOTAL COST: {total_cost}**")

# ──────────────────────────────────────────────────────────────────────────────
# Folio (collect multiple spells, export to PDF)
# ──────────────────────────────────────────────────────────────────────────────
if "folio" not in st.session_state:
    st.session_state["folio"] = []  # list of dicts

def current_spell_record() -> Dict:
    df_now = g.spell_cost_breakdown(vertices, S=S, start_shift=start_shift)
    words_now = assign_roles_context([int(r["LetterIndex"]) for r in df_now.iloc[:-1].to_dict(orient="records")])
    title_now = make_pompous_name([int(r["LetterIndex"]) for r in df_now.iloc[:-1].to_dict(orient="records")], words_now)
    return {
        "names": res["normalized_names"],
        "S": S,
        "start_shift": start_shift,
        "vertices": vertices,
        "view": view,
        "title": title_now,
        "words": words_now,
    }

st.markdown("---")
colA, colB, colC = st.columns([1,1,1])
with colA:
    if st.button("➕ Přidat do folia"):
        st.session_state["folio"].append(current_spell_record())
        st.success("Přidáno do folia.")

with colB:
    if st.button("🗑️ Vyčistit folio"):
        st.session_state["folio"] = []
        st.warning("Folio vyčištěno.")

with colC:
    if st.button("📄 Export folia do PDF"):
        if not st.session_state["folio"]:
            st.error("Folio je prázdné.")
        else:
            try:
                pdf_bytes = build_folio_pdf(st.session_state["folio"])
            except RuntimeError as e:
                st.error(str(e))
            else:
                st.download_button("Stáhnout PDF", data=pdf_bytes, file_name="kodex_folio.pdf", mime="application/pdf")

st.subheader("📚 Folio")
if st.session_state["folio"]:
    table = []
    for i, item in enumerate(st.session_state["folio"], start=1):
        table.append({
            "Index": i,
            "Titul": item["title"],
            "Jména": " · ".join(item["names"]),
            "S": item["S"],
            "start_shift": item["start_shift"],
            "Vrcholů": len(item["vertices"]),
            "Slova": " — ".join(item["words"]),
        })
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
else:
    st.caption("Zatím žádná kouzla ve foliu.")

# ──────────────────────────────────────────────────────────────────────────────
# Reference tables (LEXICON with BaseCost; multipliers)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📖 Referenční tabulky")

st.subheader("LEXICON (22 znaků) — včetně základní ceny")
lex_rows = []
for j in range(1, 23):
    row = {
        "Name": PHOENICIAN_NAMES_22[j-1],
        "Glyph": phoenician_glyph(j),
        "Noun": LEXICON.get(j, {}).get("noun", ""),
        "Verb": LEXICON.get(j, {}).get("verb", ""),
        "Adj":  LEXICON.get(j, {}).get("adj",  ""),
        "Adv":  LEXICON.get(j, {}).get("adv",  ""),
        "BaseCost": float(g.costs.get(j, 0.0)),
    }
    lex_rows.append(row)
st.dataframe(pd.DataFrame(lex_rows), hide_index=True, width="stretch")

st.subheader("Násobiče spojení (stackable)")
thr = sorted((g.connection_thresholds or {}).items())
mult_rows = [{"After ≥ lines": int(k), "× Multiplier": float(v)} for k, v in thr] or [{"After ≥ lines": "—", "× Multiplier": "—"}]
st.dataframe(pd.DataFrame(mult_rows), hide_index=True, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# 🔭 Path Builder: click vertices *in order*, set start_shift, scroll S
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🧭 Stavitel cesty (znám spojení i začátek)")

# Session state for the ordered path
if "pb_path" not in st.session_state:
    st.session_state["pb_path"] = []        # ordered list of vertices (ints 1..11)
if "pb_start_vertex" not in st.session_state:
    st.session_state["pb_start_vertex"] = None
if "pb_S" not in st.session_state:
    st.session_state["pb_S"] = 0
if "pb_start_shift" not in st.session_state:
    st.session_state["pb_start_shift"] = 0

# Controls row: start vertex, start_shift, S slider + step buttons
ctrl1, ctrl2, ctrl3 = st.columns([1, 1.2, 2])
with ctrl1:
    # If unset, default to first in path
    if st.session_state["pb_path"] and st.session_state["pb_start_vertex"] not in st.session_state["pb_path"]:
        st.session_state["pb_start_vertex"] = st.session_state["pb_path"][0]
    st.session_state["pb_start_vertex"] = st.selectbox(
        "Start (vrchol)",
        options=[None] + list(range(1,12)),
        index=(0 if not st.session_state["pb_start_vertex"] else st.session_state["pb_start_vertex"]),
        format_func=lambda v: "—" if v is None else f"V{v}",
        key="pb_start_vertex_sel"
    )
with ctrl2:
    st.session_state["pb_start_shift"] = int(st.number_input(
        "start_shift (0–21)",
        min_value=0, max_value=21, step=1,
        value=int(st.session_state["pb_start_shift"])
    ))
with ctrl3:
    colS1, colS2, colS3 = st.columns([6,1,1])
    st.session_state["pb_S"] = int(colS1.slider(
        "S (posun vrcholu s prvním symbolem)",
        min_value=0, max_value=10, step=1,
        value=int(st.session_state["pb_S"])
    ))
    if colS2.button("◀", use_container_width=True):
        st.session_state["pb_S"] = (st.session_state["pb_S"] - 1) % 11
    if colS3.button("▶", use_container_width=True):
        st.session_state["pb_S"] = (st.session_state["pb_S"] + 1) % 11

# Vertex button grid to append to the ordered path
st.caption("Klikněte na **V1..V11** v pořadí, v jakém jsou **spojeny**. (Duplikáty budou při výpočtu automaticky odstraněny.)")
grid_row1 = st.columns(6)
grid_row2 = st.columns(5)

# Handler to append a vertex
def _append_vertex(v: int):
    st.session_state["pb_path"].append(v)
    if not st.session_state["pb_start_vertex"]:
        st.session_state["pb_start_vertex"] = v

for i, v in enumerate(range(1,7)):
    if grid_row1[i].button(f"➕ V{v}", key=f"pb_add_{v}"):
        _append_vertex(v)
for i, v in enumerate(range(7,12)):
    if grid_row2[i].button(f"➕ V{v}", key=f"pb_add_{v}"):
        _append_vertex(v)

# Path management
mcol1, mcol2, mcol3 = st.columns([1,1,4])
if mcol1.button("↩️ Zpět (odebrat poslední)"):
    if st.session_state["pb_path"]:
        st.session_state["pb_path"].pop()
        # keep start vertex consistent
        if st.session_state["pb_start_vertex"] and st.session_state["pb_start_vertex"] not in st.session_state["pb_path"]:
            st.session_state["pb_start_vertex"] = st.session_state["pb_path"][0] if st.session_state["pb_path"] else None
if mcol2.button("🗑️ Vyčistit cestu"):
    st.session_state["pb_path"] = []
    st.session_state["pb_start_vertex"] = None

# Display current path
path_str = " → ".join([f"V{v}" for v in st.session_state["pb_path"]]) if st.session_state["pb_path"] else "—"
mcol3.markdown(f"**Aktuální cesta:** {path_str}")

# Reorder the path so that it starts at the chosen start vertex (if given)
def _rotate_to_start(seq: List[int], start_v: int | None) -> List[int]:
    if not seq or start_v is None:
        return seq
    try:
        i0 = seq.index(start_v)
        return seq[i0:] + seq[:i0]
    except ValueError:
        return seq

seq_path = _rotate_to_start(st.session_state["pb_path"], st.session_state["pb_start_vertex"])

# If we have at least one vertex, render results
if seq_path:
    S_ex = int(st.session_state["pb_S"])
    shift_ex = int(st.session_state["pb_start_shift"])

    # Compute breakdown with your engine (duplicates auto-removed inside)
    df_pb = g.spell_cost_breakdown(seq_path, S=S_ex, start_shift=shift_ex)
    rows_pb = df_pb.iloc[:-1].to_dict(orient="records")
    letter_idx_pb = [int(r["LetterIndex"]) for r in rows_pb]
    words_pb = assign_roles_context(letter_idx_pb)
    title_pb = make_pompous_name(letter_idx_pb, words_pb)

    # Layout: diagram + info
    pL, pR = st.columns([1.1, 1])
    with pL:
        st.subheader(" náhled cesty")
        view_pb = st.radio("Zobrazení", ["Glyphs","Names","Cryptic"], index=1, key="pb_view", horizontal=True)
        if view_pb == "Glyphs":
            fig_pb, _ = g.draw_glyphs(seq_path, S=S_ex, start_shift=shift_ex)
        elif view_pb == "Names":
            fig_pb, _ = g.draw_named(seq_path, S=S_ex, start_shift=shift_ex)
        else:
            fig_pb, _ = g.draw_cryptic(seq_path, S=S_ex, start_shift=shift_ex)
        st.pyplot(fig_pb)
        st.caption(f"S={S_ex}, start_shift={shift_ex}, vrcholy={seq_path}")

    with pR:
        st.subheader(f"✨ {title_pb}")
        st.markdown("**Počet slov:** " + str(len(words_pb)))
        st.markdown("**Sekvence:** " + " — ".join(words_pb))

        # Slim cost table + rounded TOTAL
        slim_pb = []
        for r in rows_pb:
            j = int(r["LetterIndex"])
            slim_pb.append({
                "Step": int(r["Step"]),
                "Name": r["Name"],
                "Glyph": phoenician_glyph(j),
                "Cumulative": float(r["Cumulative"]),
            })
        st.dataframe(pd.DataFrame(slim_pb), hide_index=True, width="stretch")
        total_cost_pb = int(round(float(df_pb.iloc[-1]["Cumulative"])))
        st.markdown(f"**TOTAL COST: {total_cost_pb}**")

    # Ring letter map for current (S, start_shift)
    ring_letters = g.ring_letters(S=S_ex, start_shift=shift_ex)
    ring_df = pd.DataFrame({
        "Vertex": list(range(1,12)),
        "LetterIndex": ring_letters,
        "Name": [PHOENICIAN_NAMES_22[j-1] for j in ring_letters],
        "Glyph": [phoenician_glyph(j) for j in ring_letters],
        "In Path": ["✓" if v in seq_path else "" for v in range(1,12)]
    })
    st.subheader("Aktuální prstenec (mapování písmen na vrcholy)")
    st.dataframe(ring_df, hide_index=True, width="stretch")
else:
    st.info("Začněte klikáním na vrcholy (V1..V11) v požadovaném pořadí. Poté nastavte start_shift a projíždějte S.")


# ─────────────────────────────────────────────────────────────
# 🔮 Semantic Meaning Map (Venn-style)
# ─────────────────────────────────────────────────────────────


DND_DOMAINS = [
    "Fire",
    "Water",
    "Air",
    "Earth",
    "Life",
    "Death",
    "Light",
    "Darkness",
    "Knowledge",
    "Mind",
    "Binding",
    "Motion",
    "Time",
    "Protection",
    "Transformation",
    "Communication",
]

GLYPH_DOMAINS = {
    "Aleph":   ["Air", "Life", "Origin"],
    "Beth":    ["Earth", "Protection", "Containment"],
    "Gimel":   ["Motion", "Travel"],
    "Daleth":  ["Transition", "Transformation"],
    "He":      ["Air", "Light", "Revelation"],
    "Waw":     ["Binding", "Connection", "Time"],
    "Zayin":   ["Fire", "Conflict"],
    "Heth":    ["Protection", "Earth"],
    "Teth":    ["Binding", "Transformation"],
    "Yod":     ["Motion", "Creation"],
    "Kaph":    ["Protection", "Control"],
    "Lamed":   ["Knowledge", "Communication"],
    "Mem":     ["Water", "Life"],
    "Nun":     ["Water", "Growth", "Life"],
    "Samekh":  ["Binding", "Support"],
    "Ayin":    ["Perception", "Mind"],
    "Pe":      ["Communication", "Power"],
    "Tsade":   ["Judgment", "Constraint"],
    "Qoph":    ["Darkness", "Mind"],
    "Resh":    ["Mind", "Authority"],
    "Shin":    ["Fire", "Destruction"],
    "Taw":     ["Binding", "Fate", "Time"],
}

# ─────────────────────────────────────────────────────────────
# 🐉 D&D Semantic Domain Map (Legendary-consistent)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📜 Sémantické domény glyfů (D&D)")

rows = []

for i, name in enumerate(PHOENICIAN_NAMES_22, start=1):
    domains = GLYPH_DOMAINS.get(name, [])
    rows.append({
        "Glyf": phoenician_glyph(i),
        "Název": name,
        "Domény": ", ".join(domains) if domains else "—",
    })

df_domains = pd.DataFrame(rows)

st.dataframe(
    df_domains,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Každý glyf je přiřazen k jedné nebo více sémantickým doménám "
    "odpovídajícím D&D konceptům. Tabulka je kanonická a konzistentní "
    "s legendárními kouzly."
)

