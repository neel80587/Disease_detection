"""Multimodal disease predictor: trains tabular (symptoms+vitals) and text (notes)
models, then fuses them with a stacked logistic regression.

DATA: synthetic, generated from a hand-written symptom/vital prior (see gen_data).
The model learns that prior; it is NOT validated on real patients.
Swap gen_data() for a real dataset loader to train on real records.
"""
import numpy as np, pandas as pd, joblib, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)
SYM = ["fever","cough","sore","runny","fatigue","ache","sob","chest","head","nausea",
       "vomit","diar","photo","smell","thirst","urine","blur","dizzy"]
PHRASE = {"fever":["fever","chills","running a temperature"],"cough":["dry cough","persistent cough","coughing a lot"],
 "sore":["sore throat","scratchy throat"],"runny":["runny nose","stuffy nose","sneezing"],
 "fatigue":["very tired","exhausted","fatigue"],"ache":["body aches","achy muscles"],
 "sob":["short of breath","hard to breathe","breathless"],"chest":["chest pain","chest tightness"],
 "head":["headache","pounding head"],"nausea":["nausea","feel sick"],"vomit":["vomiting","throwing up"],
 "diar":["diarrhea","loose stools"],"photo":["light sensitivity","bright light hurts"],
 "smell":["lost sense of smell","can't taste food"],"thirst":["constant thirst","very thirsty"],
 "urine":["urinating often","peeing a lot at night"],"blur":["blurred vision","vision is blurry"],
 "dizzy":["dizziness","lightheaded"]}
FILLER = ["for a few days","since yesterday","getting worse","started this week","mild but annoying",
          "no travel recently","at work lately","not sleeping well","took paracetamol","feeling off"]
# P(symptom | disease)
P = {
 "Common cold":{"runny":.85,"sore":.6,"cough":.5,"fatigue":.25,"head":.2,"fever":.1},
 "Influenza":{"fever":.85,"ache":.8,"fatigue":.8,"cough":.7,"head":.5,"sore":.35},
 "COVID-19":{"fever":.6,"cough":.75,"smell":.5,"fatigue":.65,"ache":.45,"sore":.35,"sob":.3,"head":.35},
 "Pneumonia":{"fever":.75,"cough":.9,"sob":.8,"chest":.6,"fatigue":.6},
 "Diabetes (high glucose)":{"thirst":.85,"urine":.85,"blur":.5,"fatigue":.55,"dizzy":.15},
 "Hypertension":{"head":.4,"dizzy":.4,"blur":.2,"chest":.15,"fatigue":.15},
 "Migraine":{"head":.98,"photo":.8,"nausea":.6,"vomit":.3,"dizzy":.25},
 "Gastroenteritis":{"diar":.9,"vomit":.75,"nausea":.85,"fever":.35,"ache":.2,"fatigue":.35}}
# vitals mean/sd per disease: temp, hr, spo2, sbp, glucose
V = {"Common cold":(37.2,.4,76,9,98,1,118,12,95,12),"Influenza":(38.6,.5,92,12,97,1.2,120,12,96,13),
 "COVID-19":(38.0,.7,88,13,94.5,2.5,119,12,97,14),"Pneumonia":(38.5,.6,108,14,91,3,121,13,98,14),
 "Diabetes (high glucose)":(36.8,.3,80,10,97.5,1,128,14,230,55),"Hypertension":(36.7,.3,78,10,97.5,1,152,13,102,18),
 "Migraine":(36.8,.3,76,10,98,.8,122,12,94,11),"Gastroenteritis":(37.6,.6,92,14,98,1,112,12,95,12)}
CLASSES = list(P)

def gen_data(n_per=1500):
    rows = []
    for c in CLASSES:
        for _ in range(n_per):
            s = {k: int(rng.random() < max(P[c].get(k, 0), 0.04) * (0.85 if rng.random() < .2 else 1)) for k in SYM}
            v = V[c]; vit = [rng.normal(v[i], v[i+1]) for i in range(0, 10, 2)]
            # missing vitals: each measurement absent 35% of the time
            vit = [np.nan if rng.random() < .35 else x for x in vit]
            # notes: mention ~60% of present symptoms in free text (+ occasional unrelated symptom noise)
            ph = [rng.choice(PHRASE[k]) for k in SYM if s[k] and rng.random() < .6]
            if rng.random() < .1: ph.append(rng.choice(PHRASE[rng.choice(SYM)]))
            rng.shuffle(ph)
            note = ("Patient reports " + ", ".join(ph) + " " + rng.choice(FILLER)) if ph else rng.choice(FILLER)
            y = c if rng.random() > .04 else rng.choice(CLASSES)   # 4% label noise
            rows.append({**s, **dict(zip(["temp","hr","spo2","sbp","glu"], vit)), "note": note, "y": y})
    return pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)

VIT = ["temp","hr","spo2","sbp","glu"]
def tab(df):
    X = df[SYM + VIT].copy()
    for c in VIT: X[c + "_missing"] = X[c].isna().astype(int)
    return X

if __name__ == "__main__":
    df = gen_data()
    tr, te = train_test_split(df, test_size=.2, stratify=df.y, random_state=0)
    ytr, yte = tr.y.values, te.y.values
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    tab_m = HistGradientBoostingClassifier(max_iter=150, learning_rate=.08, random_state=0)
    txt_m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(C=5, max_iter=1000))
    # out-of-fold probabilities train the fusion layer without leakage
    oof_tab = cross_val_predict(tab_m, tab(tr), ytr, cv=skf, method="predict_proba")
    oof_txt = cross_val_predict(txt_m, tr.note, ytr, cv=skf, method="predict_proba")
    tab_m.fit(tab(tr), ytr); txt_m.fit(tr.note, ytr)
    fuse = LogisticRegression(C=2, max_iter=2000).fit(np.hstack([oof_tab, oof_txt]), ytr)

    pt, px = tab_m.predict_proba(tab(te)), txt_m.predict_proba(te.note)
    pf = fuse.predict_proba(np.hstack([pt, px]))
    cl = tab_m.classes_
    res = {}
    for name, p in [("tabular (symptoms+vitals)", pt), ("text (notes)", px), ("fused", pf)]:
        yh = cl[p.argmax(1)]
        res[name] = dict(accuracy=round(accuracy_score(yte, yh), 4), macro_f1=round(f1_score(yte, yh, average="macro"), 4),
                         top3=round(top_k_accuracy_score(yte, p, k=3, labels=cl), 4))
    print(pd.DataFrame(res).T.to_string())
    print("\nFused model, per class:\n", classification_report(yte, cl[pf.argmax(1)], digits=3))
    joblib.dump(dict(tab=tab_m, txt=txt_m, fuse=fuse, classes=list(cl)), "model.joblib")
    json.dump(res, open("metrics.json", "w"), indent=2)
