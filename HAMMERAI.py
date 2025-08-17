import os, io, json, uuid, base64, random, re
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import imagehash
from tqdm import tqdm

from openai import OpenAI
from facenet_pytorch import MTCNN

# ---------------- CONFIG -----------------
OUT_ROOT = Path("DGM4")
IMG_SIZE = "1024x1024"
JPEG_QUALITY = 92
PHASH_DISTANCE_MAX = 6
MIN_FACES, MAX_FACES = 1, 3
SEED = 7
random.seed(SEED)

TARGET_LITERAL   = 1668
TARGET_INV_EMO   = 1666
TARGET_IRRELEV   = 1666

IMG_MODEL = "gpt-image-1"
TEXT_MODEL = "gpt-4o-mini"   # cheap + good

DIR_LITERAL = OUT_ROOT / "origin" / "generated"
DIR_INV     = OUT_ROOT / "manipulation" / "text_attribute"
DIR_IRR     = OUT_ROOT / "manipulation" / "text_swap"
for d in [DIR_LITERAL, DIR_INV, DIR_IRR]:
    d.mkdir(parents=True, exist_ok=True)

client = OpenAI()
mtcnn = MTCNN(keep_all=True, device="cpu")

# ---------------- SCENE + IRRELEVANT POOLS (500 each) -----------------

ROLES = ["a politician","a spokesperson","a mayor","a health official","a coach",
         "a referee","a scientist","a CEO","a weather official","an activist",
         "a news anchor","a reporter","a union representative","a judge",
         "a fire chief","a police spokesperson","a minister","a school principal",
         "a researcher","a charity director"]
ACTIONS = ["speaking at a press conference","answering questions from reporters",
           "giving a statement to the media","briefing journalists","addressing a small gathering",
           "participating in a studio interview","reading a prepared statement",
           "announcing a policy update","explaining recent developments","responding to a question"]
BACKDROPS = ["at a government podium","outside city hall","beside the field",
             "in a hospital media room","in a laboratory press area","at emergency operations center",
             "in a studio with lights","in a modern office","near a branded media wall","on a small outdoor stage"]
LIGHTING = ["daylight","overcast","evening","indoor fluorescent lighting","front-lit","side-lit"]

def _dedupe(seq): seen=set(); out=[]; [out.append(s) for s in seq if not (s in seen or seen.add(s))]; return out

scene_candidates = []
for r in ROLES:
    for a in ACTIONS:
        for b in BACKDROPS:
            desc = f"{r} {a} {b}; editorial news photo; realistic; 1 to 3 people visible; {random.choice(LIGHTING)}; do not depict any recognizable public figure"
            scene_candidates.append(desc)
SCENES = _dedupe(scene_candidates)[:500]

IRREL_SUBJECTS = ["a marathon runner","a chef","a violinist","a farmer","a spacecraft engineer",
                  "children","a tennis player","a cyclist","a jazz quartet","tourists",
                  "a zookeeper","a surfer","a mountain climber","a chess grandmaster",
                  "a barista","an artist","a dancer","a sailor","a librarian","a florist"]
IRREL_ACTIONS = ["celebrates at the finish line","plates a signature dish","performs on stage","displays fresh produce",
                 "unveils a prototype","paint a mural","serves during a tournament","repairs a flat tire","rehearses before a show",
                 "photograph a landmark","feeds animals","rides a large wave","reaches a rocky summit","makes a decisive move",
                 "pours latte art","unveils a new collection","rehearses a routine","navigates rough seas","hosts a reading","arranges bouquets"]
IRREL_CONTEXTS = ["downtown","in a busy kitchen","at a concert hall","at a weekend market","at a technology expo","at a community workshop",
                  "under bright stadium lights","on a mountain trail","in a small club","on a windy coastline","at a wildlife enclosure","on a sunny beach",
                  "on a snowy ridge","in a quiet venue","in a bustling cafe","in a gallery space","in a rehearsal studio","near a historic harbor",
                  "at a public library","in a flower shop"]

irrel_candidates = []
for s in IRREL_SUBJECTS:
    for a in IRREL_ACTIONS:
        for c in IRREL_CONTEXTS:
            irrel_candidates.append(f"{s} {a} {c}.")
IRRELEVANT_POOL = _dedupe(irrel_candidates)[:500]

def random_scene(): return random.choice(SCENES)
def irrelevant_caption() -> Tuple[str, List[int]]:
    txt = random.choice(IRRELEVANT_POOL)
    return txt, list(range(min(5, len(txt.split()))))

# ---------------- HELPERS -----------------

def openai_image(prompt:str) -> Image.Image:
    r = client.images.generate(model=IMG_MODEL, prompt=prompt, size=IMG_SIZE, quality="high", n=1)
    b64 = r.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def detect_faces(img:Image.Image) -> List[List[int]]:
    boxes, _ = mtcnn.detect(img)
    if boxes is None: return []
    out = []
    for b in boxes:
        x1,y1,x2,y2 = [int(round(v)) for v in b.tolist()]
        out.append([x1,y1,x2,y2])
    return out

def literal_caption(prompt:str) -> str:
    r = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role":"system","content":"Write one neutral news-style photo caption (5–30 words). No names."},
                  {"role":"user","content":prompt}],
        temperature=0.6,max_tokens=60)
    return r.choices[0].message.content.strip()

def invert_emotion(text:str) -> Tuple[str, List[int]]:
    r = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role":"system","content":"Rewrite the caption with opposite emotion (calm↔angry, happy↔upset, confident↔nervous). Mark changed words in [brackets]."},
                  {"role":"user","content":text}],
        temperature=0.7,max_tokens=80)
    s = r.choices[0].message.content.strip()
    clean = s.replace("[","").replace("]","")
    tokens = clean.split()
    idxs = []
    for m in re.finditer(r"\[([^\]]+)\]", s):
        for w in m.group(1).split():
            for i,t in enumerate(tokens):
                if i not in idxs and t.lower().strip(".,") == w.lower().strip(".,"):
                    idxs.append(i); break
    return clean, sorted(set(idxs))

def save_jpeg(img, path): img.save(path,"JPEG",quality=JPEG_QUALITY,optimize=True)

def make_record(_id, rel_path, text, fake_cls, mtcnn_boxes, fake_image_box=None, fake_text_pos=None):
    return {"id":_id,"image":rel_path.replace("\\","/"),"text":text,
            "fake_cls":fake_cls,"fake_image_box":fake_image_box or [],
            "fake_text_pos":fake_text_pos or [],"mtcnn_boxes":mtcnn_boxes}

# ---------------- MAIN -----------------

def main():
    records, phashes = [], []
    next_id = 1
    c_lit=c_inv=c_irr=0
    total = TARGET_LITERAL+TARGET_INV_EMO+TARGET_IRRELEV
    pbar = tqdm(total=total, desc="Building 5000 DGM4-synthetic")

    while (c_lit+c_inv+c_irr)<total:
        prompt = random_scene()
        try: img=openai_image(prompt)
        except: continue
        boxes=detect_faces(img)
        if not (MIN_FACES<=len(boxes)<=MAX_FACES): continue
        ph=imagehash.phash(img)
        if any(ph-h<=PHASH_DISTANCE_MAX for h in phashes): continue
        phashes.append(ph)

        if c_lit<TARGET_LITERAL:
            out_path=DIR_LITERAL/f"{uuid.uuid4().hex[:10]}.jpg"
            caption=literal_caption(prompt); fake_cls="origin"; fake_text_pos=[]
            c_lit+=1
        elif c_inv<TARGET_INV_EMO:
            out_path=DIR_INV/f"{uuid.uuid4().hex[:10]}.jpg"
            basecap=literal_caption(prompt); caption,fake_text_pos=invert_emotion(basecap)
            fake_cls="text_attribute"; c_inv+=1
        else:
            out_path=DIR_IRR/f"{uuid.uuid4().hex[:10]}.jpg"
            caption,fake_text_pos=irrelevant_caption()
            fake_cls="text_swap"; c_irr+=1

        try: save_jpeg(img,out_path)
        except: continue

        rel=out_path.relative_to(OUT_ROOT)
        rec=make_record(next_id,str(rel),caption,fake_cls,boxes,fake_text_pos=fake_text_pos)
        records.append(rec); next_id+=1; pbar.update(1)

    pbar.close()
    with open(OUT_ROOT/"metadata.json","w",encoding="utf-8") as f: json.dump(records,f,ensure_ascii=False,indent=4)
    print(f"Done: {len(records)} samples saved to {OUT_ROOT}")

if __name__=="__main__":
    main()
