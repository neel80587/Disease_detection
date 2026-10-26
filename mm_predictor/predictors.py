"""Single entry point for the three trained models. Each function returns plain dicts."""
import sys, pathlib, numpy as np, pandas as pd, joblib
from PIL import Image
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "train"))
import train_symptoms as TS, train_heart as TH

_cache = {}
def _m(name):
    if name not in _cache: _cache[name] = joblib.load(ROOT / "models" / f"{name}.joblib")
    return _cache[name]

SYMPTOM_LABELS = {"fever":"Fever","cough":"Cough","sore":"Sore throat","runny":"Runny nose","fatigue":"Fatigue","ache":"Body aches",
 "sob":"Shortness of breath","chest":"Chest pain","head":"Headache","nausea":"Nausea","vomit":"Vomiting","diar":"Diarrhea",
 "photo":"Light sensitivity","smell":"Loss of smell/taste","thirst":"Excessive thirst","urine":"Frequent urination","blur":"Blurred vision","dizzy":"Dizziness"}

def predict_symptoms(symptoms=(), note="", temp=None, hr=None, spo2=None, sbp=None, glu=None, top=5):
    """Synthetic-data model (8 common conditions). Demo only."""
    m = _m("symptoms"); row = {k: int(k in symptoms) for k in TS.SYM}
    row.update(dict(zip(TS.VIT, [temp, hr, spo2, sbp, glu])), note=note)
    df = pd.DataFrame([row]).astype({c: float for c in TS.VIT})
    p = m["fuse"].predict_proba(np.hstack([m["tab"].predict_proba(TS.tab(df)), m["txt"].predict_proba(df.note)]))[0]
    return sorted(((str(c), float(x)) for c, x in zip(m["classes"], p)), key=lambda t: -t[1])[:top]

def predict_heart(age, sex, cp, trestbps=None, chol=None, fbs=None, restecg=None, thalach=None, exang=None,
                  oldpeak=None, slope=None, ca=None, thal=None, use=("clinical", "exercise_ecg", "imaging_scan")):
    """UCI Heart Disease model. `use` lists the modalities that were actually measured."""
    d = pd.DataFrame([dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
                           exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)]).astype(float)
    M = _m("heart"); per = {k: float(M["models"][k].predict_proba(d[TH.MODS[k]])[0, 1]) for k in use}
    return dict(probability=float(TH.proba(M, d, list(use))[0]), per_modality=per)

_net = None
def _embed(img):
    global _net
    import torch, timm
    if _net is None:
        _net = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0,
                                 pretrained_cfg_overlay=dict(file=str(ROOT / "models" / "mobilenetv3_large_100.pth"))).eval()
    mu, sd = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    t = (torch.from_numpy(img[None]).permute(0, 3, 1, 2).float() / 255 - mu) / sd
    with torch.no_grad(): return ((_net(t) + _net(t.flip(3)) + _net(t.flip(2))) / 3).numpy()

SKIN_SITES = ["abdomen","acral","back","chest","ear","face","foot","genital","hand","lower extremity","neck","scalp","trunk","unknown","upper extremity"]

def skin_info():
    """Which skin model is installed? models/skin_ft.pt (fine-tuned, from train_skin_ft.py) is preferred over the legacy one."""
    return dict(fine_tuned=(ROOT / "models" / "skin_ft.pt").exists())

def predict_skin(image, age=None, sex=None, site=None):
    """Benign/malignant estimate. age/sex/site are only used by a fine-tuned image+metadata model."""
    if not isinstance(image, Image.Image): image = Image.open(image)
    ft = ROOT / "models" / "skin_ft.pt"
    if ft.exists():
        import train_skin_ft as FT
        if "skin_ft" not in _cache: _cache["skin_ft"] = FT.load_ft(ft)
        net, ck = _cache["skin_ft"]; r = FT.predict_image(net, ck, image, age, sex, site); r["model"] = "fine-tuned"; return r
    return _predict_skin_legacy(image)

def _predict_skin_legacy(image):
    """ISIC 2016 benign/malignant model. `image` = PIL image or path."""
    import train_skin as TK  # imported here so heart/symptom tabs work without torch/timm
    img = np.asarray(image.convert("RGB").resize((224, 224), Image.LANCZOS))
    M = _m("skin"); pd_ = float(M["deep"].predict_proba(_embed(img))[0, 1])
    ph = float(M["hand"].predict_proba(np.nan_to_num(np.array([TK.hand(img)])))[0, 1])
    pf = float(M["fuse"].predict_proba(np.c_[TK.logit(np.array([pd_])), TK.logit(np.array([ph]))])[0, 1])
    return dict(probability=pf, deep=pd_, handcrafted=ph, flagged=None, model="legacy")
