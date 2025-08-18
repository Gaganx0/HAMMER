# HAMMERAI_async_headlines.py
# DGM4-style synthetic builder
# - Async workers + reservation
# - 70% obvious backdrop mismatch
# - Headline-style captions (no "a/an" subject; motivated by scene)
# - Text-attribute = affect flip; we compute indices deterministically
# - Text-swap = fully irrelevant headline (all tokens flagged)
# - OCR scrub to avoid legible text; no borders; 1–3 faces; 400x256 output

import os, io, json, uuid, base64, random, re, asyncio, sys
from pathlib import Path
from typing import List, Tuple, Dict, Set

from PIL import Image, ImageFilter
import imagehash
from tqdm.auto import tqdm
from openai import AsyncOpenAI
from facenet_pytorch import MTCNN

# ---------------- CONFIG -----------------
OUT_ROOT = Path("DGM4")
IMG_MODEL  = "gpt-image-1"

# Generation → we caption locally (no text model calls)
GEN_SIZE     = "1024x1024"
GEN_QUALITY  = "low"            # low|medium|high (billing tier)
POST_W, POST_H = 400, 256
JPEG_QUALITY = 85

MIN_FACES, MAX_FACES = 1, 3
PHASH_DISTANCE_MAX   = 3
CONCURRENCY          = 5

# Deliberate mismatch rate
MISPLACED_BG_RATE    = 0.70     # ← 70% mismatched backdrops

# Targets (smoke). For ~5000 use 1668/1666/1666 and raise CONCURRENCY.
TARGET_LITERAL = 2
TARGET_INV_EMO = 2
TARGET_IRRELEV = 1

# Optional spend guard on image calls (None disables)
MAX_IMAGE_CALLS = None

SEED = 7
random.seed(SEED)

# OCR: enforce no legible text/logos
STRICT_NO_TEXT = True
TEXT_BLACKLIST = ["PRESS","HOSPITAL","FINISH","CENTER","POLICE","BANK","SCHOOL","STATION","UNIVERSITY","HOSPICE","GOVERNMENT"]

# DGM4-style dirs
DIR_LITERAL = OUT_ROOT / "origin" / "generated"
DIR_INV     = OUT_ROOT / "manipulation" / "text_attribute"
DIR_IRR     = OUT_ROOT / "manipulation" / "text_swap"
for d in [DIR_LITERAL, DIR_INV, DIR_IRR]:
    d.mkdir(parents=True, exist_ok=True)

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY not set.")

aclient = AsyncOpenAI()
mtcnn = MTCNN(keep_all=True, device="cpu")

# ---------------- OCR -----------------
OCR_OK = False
if STRICT_NO_TEXT:
    try:
        import pytesseract
        cand = Path(sys.prefix) / "Library" / "bin" / "tesseract.exe"
        if cand.exists():
            pytesseract.pytesseract.tesseract_cmd = str(cand)
        _ = pytesseract.get_tesseract_version()
        OCR_OK = True
    except Exception:
        OCR_OK = False
    if not OCR_OK:
        raise SystemExit(
            "STRICT_NO_TEXT=True but Tesseract not available.\n"
            "Install in this env:  conda install -c conda-forge tesseract pytesseract"
        )

def has_readable_text(img: Image.Image) -> bool:
    import pytesseract
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        for i, txt in enumerate(data.get("text", [])):
            if not txt or len(txt) < 3: continue
            try: conf = int(float(data["conf"][i]))
            except: conf = 0
            if conf >= 60:
                return True
    except Exception:
        return False
    return False

def blur_rect(img, box, radius=6):
    x1,y1,x2,y2 = [max(0,int(v)) for v in box]
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2<=x1 or y2<=y1: return img
    region = img.crop((x1,y1,x2,y2)).filter(ImageFilter.GaussianBlur(radius))
    img.paste(region, (x1,y1))
    return img

def scrub_text_if_detected(img: Image.Image):
    import pytesseract
    dirty=False
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    for i, txt in enumerate(data.get("text", [])):
        if not txt or len(txt) < 3: continue
        try: conf = int(float(data["conf"][i]))
        except: conf = 0
        if conf >= 60:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            img = blur_rect(img, (x-2,y-2,x+w+2,y+h+2))
            dirty=True
    if dirty and has_readable_text(img):
        return img, True
    return img, False

# ---------------- Scene vocabulary -----------------
ROLES = [
    "a politician","a spokesperson","a mayor","a health official","a coach","a referee","a scientist","a CEO",
    "a weather official","an activist","a news anchor","a reporter","a union representative","a judge","a fire chief",
    "a police spokesperson","a minister","a school principal","a researcher","a charity director","a local resident",
    "a parent","a teacher","a student","a nurse","a doctor","a shop owner","a barista","a delivery courier",
    "a bus driver","a train operator","a commuter","a farmer","a fisher","a factory worker","a mechanic",
    "a construction worker","an architect","an engineer","a small business owner","a volunteer","a librarian",
    "a museum curator","an art teacher","an athlete","a runner","a cyclist","a grocer","a baker","a chef","a waiter",
    "a bartender","a security guard","a police officer","a firefighter","an EMT","a paramedic","a postal worker",
    "a taxi driver","a rideshare driver","a pilot","a flight attendant","a sailor","a park ranger","a conservationist",
    "a professor","a school counselor","a PTA member","a voter","an election worker","a lawyer","a public defender",
    "a prosecutor","a witness","a city planner","an urban planner","a sanitation worker","a recycling worker",
    "a utility worker","a power line technician","a grid operator","a station manager","a ferry operator",
    "a dock worker","a warehouse worker","a hospital administrator","a nurse practitioner","a midwife",
    "a pharmacist","a lab technician","a veterinarian","an animal shelter worker","a homeowner","a neighbor",
    "a tourist","a hotel manager","a conference attendee","a trade show exhibitor","a community organizer",
    "a disaster relief worker"
]

# “Everyday” events; we won’t force them into mismatched backdrops in captions
EVENTS = [
    "addresses a press briefing","holds a town hall","meets neighborhood residents","visits a classroom",
    "opens a new library","visits a hospital ward","observes a vaccination clinic","thanks health workers",
    "visits a laboratory","announces research findings","reviews safety protocols","visits a fire station",
    "visits a police precinct","joins an emergency drill","inspects storm damage","distributes supplies",
    "inspects road repairs","announces transit upgrades","talks to commuters","walks through a factory",
    "meets union representatives","announces a jobs initiative","launches a pilot program","signs a partnership",
    "answers questions on the sidelines","meets a sports team","visits a community garden","plants a tree",
    "visits a recycling facility","discusses clean energy plans","visits a wind farm","visits a solar field",
    "meets housing advocates","cuts a ribbon at an opening","attends a vigil","visits a cultural center",
    "tours a museum exhibit","opens a book fair","visits an animal shelter","meets farmers at a fair",
    "visits a food bank","visits a tech lab","tests a prototype device","greets graduates","meets researchers",
    "answers questions in a corridor","walks with aides to a car","boards a commuter train",
]

# Normal backdrops plus extra “obvious mismatch” locations
BACKDROPS = [
    "on courthouse steps","inside a courtroom","in a media briefing room","inside a town hall chamber",
    "in a community center hall","on a neighborhood street","in a bright classroom","inside a public library",
    "in a hospital corridor","inside a research laboratory","in a fire station garage","inside a police precinct lobby",
    "on a factory floor","beside an assembly line","in a startup co-working space","in a modern boardroom",
    "on a sports sideline","in a locker room interview area","at a ballpark dugout rail",
    # Strong mismatches / attention-catching:
    "on a football stadium sideline","on a hockey rink","at a roller-coaster platform",
    "inside an aquarium tunnel","at a carnival midway","on a ski slope base area",
    "at a volcano viewpoint","beside wind turbines","inside a space launch facility",
]

LIGHTING = ["daylight","overcast","evening light","fluorescent indoor light","front-lit","side-lit"]
SHOT_TYPES = [("wide shot", 0.25), ("medium shot", 0.65), ("close-up", 0.10)]

# Tagging to ensure match/mismatch logic
ROLE_RULES = {
    "health": ["nurse","doctor","pharmacist","paramedic","emt","hospital administrator","nurse practitioner","midwife","health official","lab technician"],
    "education": ["teacher","student","professor","school principal","school counselor","pta"],
    "transport": ["bus driver","train operator","taxi","rideshare","pilot","flight attendant","ferry operator","station manager","dock worker","commuter"],
    "industry": ["factory","mechanic","construction","engineer","architect","warehouse","grid operator","power line","utility","recycling worker"],
    "agriculture": ["farmer","fisher"],
    "public_safety": ["firefighter","fire chief","police","police spokesperson","security guard","park ranger","conservationist","disaster relief"],
    "gov": ["politician","mayor","spokesperson","judge","minister","city planner","urban planner","election worker","charity director","ceo","researcher","scientist","weather official"],
    "media": ["news anchor","reporter","conference attendee"],
    "community": ["local resident","parent","volunteer","homeowner","neighbor","tourist","community organizer","voter","witness","athlete","runner","cyclist","sailor","veterinarian","animal shelter worker","museum curator","art teacher","librarian"]
}
EVENT_TAGS = {
    "health": ["hospital","clinic","vaccination","ward","patients","nurse","medical","lab"],
    "education": ["classroom","school","university","board","campus","graduates"],
    "transport": ["station","train","bus","airport","gate","terminal","platform","rail"],
    "industry": ["factory","assembly","construction","bridge","warehouse","equipment","prototype"],
    "agriculture": ["farm","farmers market","fair"],
    "public_safety": ["fire station","police","emergency","storm","wildfire","relief","shelter"],
    "gov": ["hearing","court","courthouse","town hall","press briefing","city hall","election"],
    "community": ["community","garden","vigil","charity","memorial","volunteers","food bank","seniors","neighborhood","aid"]
}
BACKDROP_TAGS = {
    "health": ["hospital"],
    "education": ["classroom","library"],
    "transport": ["station","platform","airport","roller-coaster","launch","stadium","rink"],
    "industry": ["factory","assembly","warehouse"],
    "agriculture": ["farm"],
    "public_safety": ["fire station","police"],
    "gov": ["courthouse","town hall"],
    "community": ["community center","garden","memorial","midway","aquarium","volcano","wind"]
}

def infer_role_tags(role: str) -> Set[str]:
    role_l = role.lower()
    tags = set()
    for k, kws in ROLE_RULES.items():
        if any(kw in role_l for kw in kws):
            tags.add(k)
    if not tags: tags.add("community")
    return tags

def infer_text_tags(txt: str, table: Dict[str, List[str]]) -> Set[str]:
    t = txt.lower(); out = set()
    for k, kws in table.items():
        if any(kw in t for kw in kws): out.add(k)
    return out or {"community"}

def weighted_choice(items):
    r = random.random(); c = 0.0
    for name, w in items:
        c += w
        if r <= c: return name
    return items[-1][0]

# Build scene pool and remember structured fields (role,event,backdrop, mismatch)
def build_scene_pool(n=1200):
    seen, pool = set(), []
    while len(pool) < n:
        role = random.choice(ROLES)
        shot = weighted_choice(SHOT_TYPES)
        rt = infer_role_tags(role)

        for _ in range(60):
            event = random.choice(EVENTS)
            et = infer_text_tags(event, EVENT_TAGS)

            mismatch = (random.random() < MISPLACED_BG_RATE)
            back = random.choice(BACKDROPS)
            bt = infer_text_tags(back, BACKDROP_TAGS)

            # if matched: require overlap; if mismatched: require no overlap
            good = False
            if mismatch:
                good = not ((rt & bt) or (et & bt))
            else:
                good = (rt & et) and ((rt & bt) or (et & bt))
            if not good: continue

            props=[]
            if random.random() < 0.12: props.append("reporters nearby")
            if random.random() < 0.25: props.append("holding documents or a folder")
            if random.random() < 0.22: props.append("wearing safety gear appropriate to the site")
            prop_text = "; ".join(props) if props else "natural ambient details"

            deny = ", ".join(TEXT_BLACKLIST)
            no_text = (
                f"; avoid any readable text, letters, numbers, logos, banners, captions, scoreboards, watermarks; "
                f"signage abstract/blurred; do not render words such as {deny}"
            )
            light = random.choice(LIGHTING)
            face_hint = "; faces clearly visible, unobstructed; no sunglasses or opaque masks"

            prompt = (
                f"{role} {event} {back}; {shot}; editorial photojournalism; realistic; "
                f"1 to 3 people visible{face_hint}; {light}; {prop_text}{no_text}; "
                f"brandless; do not depict any recognizable public figure"
            )
            if prompt not in seen:
                seen.add(prompt); pool.append((prompt, shot, role, event, back, mismatch)); break
    return pool

SCENE_POOL = build_scene_pool()

# ---------------- Headlines (local, deterministic) -----------------
AFFECT_PAIRS = [
    ("smiles", "looks worried"),
    ("appears calm", "appears tense"),
    ("laughs", "looks solemn"),
    ("looks relieved", "appears anxious"),
    ("celebrates", "appears dejected"),
]

GENERIC_VERBS = [
    "appears", "stands", "speaks", "meets staff", "reviews notes",
    "listens to questions", "greets attendees", "talks with officials"
]

def strip_articles(s: str) -> str:
    # remove initial "a/an/the" but keep prepositions inside backdrop phrases
    return re.sub(r"\b(a|an)\b\s+", "", s, flags=re.I).strip()

def role_to_subject(role: str) -> str:
    # "a doctor" -> "Doctor"
    r = role.strip().lower()
    r = re.sub(r"^(a|an)\s+", "", r)
    return r.capitalize() if " " not in r else " ".join(w.capitalize() for w in r.split())

def clean_event(event: str) -> str:
    e = strip_articles(event)
    # tame a few long forms
    e = e.replace("press briefing", "press briefing").replace("town hall Q&A", "town hall")
    e = e.replace("answers questions on the sidelines", "answers questions")
    return e

def headline_from(role: str, event: str, backdrop: str, mismatch: bool, affect_pair=None, literal_positive=True) -> Tuple[str, str, List[int]]:
    subj = role_to_subject(role)                 # "Doctor"
    e = clean_event(event)                       # "addresses press briefing"
    b = strip_articles(backdrop)                 # "on courthouse steps" / "in a courtroom" -> "in courtroom"
    # for mismatch, use generic verb to avoid awkward pairings
    verb_phrase = random.choice(GENERIC_VERBS) if mismatch else e
    # choose affect pair
    pos, neg = affect_pair if affect_pair else random.choice(AFFECT_PAIRS)
    aff_lit, aff_inv = (pos, neg) if literal_positive else (neg, pos)

    # Compose headline (avoid double spaces; ensure period)
    core = f"{subj} {verb_phrase} {b}".strip()
    headline_lit = re.sub(r"\s+", " ", f"{core}, {aff_lit}.").replace(" ,", ",")
    headline_inv = re.sub(r"\s+", " ", f"{core}, {aff_inv}.").replace(" ,", ",")

    # token indices for swapped phrase in inverted headline
    def idx_phrase(text, phrase):
        toks = text.split()
        words = phrase.split()
        # strip punctuation for match
        norm = [re.sub(r"[^\w]", "", t).lower() for t in toks]
        wnorm = [re.sub(r"[^\w]", "", t).lower() for t in words]
        for i in range(0, len(norm) - len(wnorm) + 1):
            if norm[i:i+len(wnorm)] == wnorm:
                return list(range(i, i+len(wnorm)))
        return []
    fake_pos = idx_phrase(headline_inv, aff_inv)
    return headline_lit, headline_inv, fake_pos

# ---------------- Irrelevant headlines -----------------
IRREL_SUBJECTS = ["Marathon Runner","Chef","Violinist","Farmer","Spacecraft Engineer","Children",
                  "Tennis Player","Cyclist","Jazz Quartet","Tourists","Zookeeper","Surfer",
                  "Mountain Climber","Chess Grandmaster","Barista","Artist","Dancer","Sailor",
                  "Librarian","Florist"]
IRREL_ACTIONS  = ["wins city race","plates signature dish","performs at gala","displays fresh produce",
                  "unveils prototype","paints mural","serves during tournament","repairs flat tire",
                  "rehearses before show","photographs landmark","feeds animals","rides large wave",
                  "reaches rocky summit","makes decisive move","pours latte art","opens new collection",
                  "rehearses routine","navigates rough seas","hosts reading","arranges bouquets"]
IRREL_PLACES   = ["downtown","in concert hall","at weekend market","at technology expo",
                  "at community workshop","under stadium lights","on mountain trail","in small club",
                  "on windy coastline","at wildlife enclosure","on sunny beach","in gallery space",
                  "in rehearsal studio","near historic harbor","at public library","in flower shop"]

def irrelevant_headline():
    s = random.choice(IRREL_SUBJECTS)
    a = random.choice(IRREL_ACTIONS)
    p = random.choice(IRREL_PLACES)
    hl = f"{s} {a} {p}."
    toks = hl.split()
    return hl, list(range(len(toks)))  # mark all tokens swapped

# ---------------- OpenAI image -----------------
async def openai_image_async(prompt: str, counters=None) -> Image.Image:
    if MAX_IMAGE_CALLS is not None and counters is not None:
        if counters.setdefault("image_calls", 0) >= MAX_IMAGE_CALLS:
            raise RuntimeError("Reached MAX_IMAGE_CALLS")
        counters["image_calls"] += 1
    r = await aclient.images.generate(
        model=IMG_MODEL,
        prompt=prompt,
        size=GEN_SIZE,
        quality=GEN_QUALITY,
        n=1,
    )
    b64 = r.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

# ---------------- Detection/Cropping (no borders) -----------------
def detect_faces(img: Image.Image) -> List[List[int]]:
    boxes, _ = mtcnn.detect(img)
    if boxes is None: return []
    out=[]
    for b in boxes:
        x1,y1,x2,y2 = [int(round(v)) for v in b.tolist()]
        x1=max(0,x1); y1=max(0,y1)
        x2=min(img.width,x2); y2=min(img.height,y2)
        if x2>x1 and y2>y1: out.append([x1,y1,x2,y2])
    return out

def _central_crop_to_ratio(img: Image.Image, w: int, h: int):
    iw, ih = img.size
    target = w / float(h)
    cur = iw / float(ih)
    if cur > target:
        nw, nh = int(round(ih * target)), ih
    else:
        nw, nh = iw, int(round(iw / target))
    x1 = (iw - nw) // 2
    y1 = (ih - nh) // 2
    return img.crop((x1, y1, x1 + nw, y1 + nh))

def crop_union(img: Image.Image, boxes: List[List[int]], w: int, h: int, expand=2.4):
    iw, ih = img.size
    if not boxes:
        return _central_crop_to_ratio(img, w, h).resize((w, h), Image.BICUBIC), []
    x1=min(b[0] for b in boxes); y1=min(b[1] for b in boxes)
    x2=max(b[2] for b in boxes); y2=max(b[3] for b in boxes)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*expand, (y2-y1)*expand
    target = w / float(h)
    if bw/bh > target: bh = bw/target
    else:              bw = bh*target
    x1n = int(max(0, cx - bw/2)); y1n = int(max(0, cy - bh/2))
    x2n = int(min(iw, cx + bw/2)); y2n = int(min(ih, cy + bh/2))
    crop = img.crop((x1n,y1n,x2n,y2n)).resize((w,h), Image.BICUBIC)
    sx, sy = w/float(x2n-x1n), h/float(y2n-y1n)
    new_boxes=[]
    for bx1,by1,bx2,by2 in boxes:
        tbx1=int(round((bx1-x1n)*sx)); tby1=int(round((by1-y1n)*sy))
        tbx2=int(round((bx2-x1n)*sx)); tby2=int(round((by2-y1n)*sy))
        tbx1=max(0,min(w,tbx1)); tbx2=max(0,min(w,tbx2))
        tby1=max(0,min(h,tby1)); tby2=max(0,min(h,tby2))
        if tbx2>tbx1 and tby2>tby1: new_boxes.append([tbx1,tby1,tbx2,tby2])
    return crop, new_boxes

def face_center_crop(img: Image.Image, boxes: List[List[int]], w: int, h: int):
    if not boxes:
        return _central_crop_to_ratio(img, w, h).resize((w, h), Image.BICUBIC), []
    iw, ih = img.size
    cx = sum((b[0]+b[2])/2 for b in boxes)/len(boxes)
    cy = sum((b[1]+b[3])/2 for b in boxes)/len(boxes)
    target = w/float(h); cur=iw/float(ih)
    if cur>target: nw=int(round(ih*target)); nh=ih
    else:          nw=iw;                   nh=int(round(iw/target))
    x1=max(0, min(iw-nw, int(round(cx - nw/2))))
    y1=max(0, min(ih-nh, int(round(cy - nh/2))))
    x2, y2 = x1+nw, y1+nh
    crop = img.crop((x1,y1,x2,y2)).resize((w,h), Image.BICUBIC)
    sx, sy = w/float(nw), h/float(nh)
    new_boxes=[]
    for bx1,by1,bx2,by2 in boxes:
        tbx1=int(round((bx1-x1)*sx)); tby1=int(round((by1-y1)*sy))
        tbx2=int(round((bx2-x1)*sx)); tby2=int(round((by2-y1)*sy))
        tbx1=max(0,min(w,tbx1)); tbx2=max(0,min(w,tbx2))
        tby1=max(0,min(h,tby1)); tby2=max(0,min(h,tby2))
        if tbx2>tbx1 and tby2>tby1: new_boxes.append([tbx1,tby1,tbx2,tby2])
    return crop, new_boxes

def apply_shot_crop(img: Image.Image, boxes: List[List[int]], shot: str, w: int, h: int):
    if shot == "wide shot":   return crop_union(img, boxes, w, h, expand=3.0)
    if shot == "medium shot": return crop_union(img, boxes, w, h, expand=2.2)
    return face_center_crop(img, boxes, w, h)

def save_jpeg(img: Image.Image, path: Path):
    img.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)

def make_record(_id, rel_path, text, fake_cls, mtcnn_boxes, fake_image_box=None, fake_text_pos=None):
    return {
        "id": _id,
        "image": rel_path.replace("\\","/"),
        "text": text,
        "fake_cls": fake_cls,
        "fake_image_box": fake_image_box or [],
        "fake_text_pos": fake_text_pos or [],
        "mtcnn_boxes": mtcnn_boxes
    }

# ---------------- ASYNC WORKERS -----------------
def random_scene():
    return random.choice(SCENE_POOL)

async def worker(lock, counters, records, pbar):
    goals = {"lit": TARGET_LITERAL, "inv": TARGET_INV_EMO, "irr": TARGET_IRRELEV}
    while True:
        async with lock:
            total_goal = sum(goals.values())
            total_done = sum(counters["done"].values())
            total_infl = sum(counters["inflight"].values())
            if total_done + total_infl >= total_goal:
                return
            bucket = None
            for key in ("lit","inv","irr"):
                if counters["done"][key] + counters["inflight"][key] < goals[key]:
                    bucket = key; break
            if bucket is None: return
            counters["inflight"][bucket] += 1
            _id = counters["next_id"]; counters["next_id"] += 1

        prompt, shot, role, event, back, mismatch = random_scene()
        try:
            img = await openai_image_async(prompt, counters=counters)
        except Exception:
            async with lock:
                counters["inflight"][bucket] -= 1
            continue

        boxes_orig = await asyncio.to_thread(detect_faces, img)
        if not (MIN_FACES <= len(boxes_orig) <= MAX_FACES):
            async with lock:
                counters["inflight"][bucket] -= 1
            continue

        ph = imagehash.phash(img)
        async with lock:
            dupe = any(ph - h <= PHASH_DISTANCE_MAX for h in counters["phashes"])
            if not dupe: counters["phashes"].append(ph)
        if dupe:
            async with lock:
                counters["inflight"][bucket] -= 1
            continue

        # captions (local)
        if bucket == "irr":
            caption, fake_pos = irrelevant_headline()
            fake_cls = "text_swap"
        else:
            # literal headline + inverted affect headline (indices computed)
            pos_first = bool(random.getrandbits(1))
            lit, inv, inv_idx = headline_from(role, event, back, mismatch, literal_positive=pos_first)
            if bucket == "lit":
                caption, fake_pos, fake_cls = lit, [], "origin"
            else:
                caption, fake_pos, fake_cls = inv, inv_idx, "text_attribute"

        img_final, boxes_final = await asyncio.to_thread(apply_shot_crop, img, boxes_orig, shot, POST_W, POST_H)

        if STRICT_NO_TEXT and has_readable_text(img_final):
            img_final, still_bad = await asyncio.to_thread(scrub_text_if_detected, img_final)
            if still_bad:
                async with lock:
                    counters["inflight"][bucket] -= 1
                continue

        out_dir = {"lit": DIR_LITERAL, "inv": DIR_INV, "irr": DIR_IRR}[bucket]
        out_path = out_dir / f"{uuid.uuid4().hex[:10]}.jpg"
        try:
            await asyncio.to_thread(save_jpeg, img_final, out_path)
        except Exception:
            async with lock:
                counters["inflight"][bucket] -= 1
            continue

        rel = out_path.relative_to(OUT_ROOT)
        rec = make_record(_id, str(rel), caption, fake_cls, boxes_final, fake_text_pos=fake_pos)

        async with lock:
            records.append(rec)
            counters["inflight"][bucket] -= 1
            counters["done"][bucket] += 1
            pbar.update(1)

async def amain():
    records = []
    total = TARGET_LITERAL + TARGET_INV_EMO + TARGET_IRRELEV
    pbar = tqdm(total=total, desc=f"Building {total} DGM4-synthetic")
    lock = asyncio.Lock()
    counters = {
        "done": {"lit":0, "inv":0, "irr":0},
        "inflight": {"lit":0, "inv":0, "irr":0},
        "next_id": 1,
        "phashes": []
    }
    tasks = [asyncio.create_task(worker(lock, counters, records, pbar)) for _ in range(CONCURRENCY)]
    await asyncio.gather(*tasks)
    pbar.close()

    with open(OUT_ROOT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"Done: {len(records)} samples saved to {OUT_ROOT}")
    print(f"Split -> literal:{counters['done']['lit']} invert:{counters['done']['inv']} irrelevant:{counters['done']['irr']}")
    print("image_calls:", counters.get("image_calls", 0))

if __name__ == "__main__":
    asyncio.run(amain())
