"""Multimodal heart-disease predictor on the UCI Heart Disease data (4 hospitals, 920 patients).
Target: any angiographic narrowing (num>0). Three modalities, late fusion by stacking."""
import glob, numpy as np, pandas as pd, joblib, json, warnings
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
warnings.filterwarnings("ignore")
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
SITES = {"cleveland":"Cleveland","hungarian":"Hungary","switzerland":"Switzerland","va":"Long Beach VA"}
MODS = {"clinical":["age","sex","cp","trestbps","chol","fbs"],
        "exercise_ecg":["restecg","thalach","exang","oldpeak","slope"],
        "imaging_scan":["ca","thal"]}   # fluoroscopy vessel count + thallium scan

def load(d="/mnt/user-data/uploads"):
    fr = []
    for k, v in SITES.items():
        f = glob.glob(f"{d}/*processed_{k}.data")[0]
        x = pd.read_csv(f, header=None, names=COLS, na_values="?"); x["site"] = v; fr.append(x)
    df = pd.concat(fr, ignore_index=True); df["y"] = (df.num > 0).astype(int); return df

def mk(): return HGB(max_depth=3, max_iter=80, learning_rate=.06, min_samples_leaf=15, l2_regularization=1.0, random_state=0)
def logit(p): p = np.clip(p, 1e-4, 1-1e-4); return np.log(p/(1-p))

def fit(tr):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); ms, Z = {}, []
    for m, c in MODS.items():
        Z.append(logit(cross_val_predict(mk(), tr[c], tr.y, cv=skf, method="predict_proba")[:,1]))
        ms[m] = mk().fit(tr[c], tr.y)
    fuse = LogisticRegression(C=1.0).fit(np.column_stack(Z), tr.y)
    return dict(models=ms, fuse=fuse)

def proba(M, d, use=None):
    use = use or list(MODS)
    Z = np.column_stack([logit(M["models"][m].predict_proba(d[MODS[m]])[:,1]) if m in use else np.zeros(len(d)) for m in MODS])
    return M["fuse"].predict_proba(Z)[:,1]

def score(y, p): return dict(auc=round(roc_auc_score(y,p),3), acc=round(accuracy_score(y,p>.5),3), f1=round(f1_score(y,p>.5),3))

if __name__ == "__main__":
    df = load(); print("patients:", len(df), " disease prevalence:", round(df.y.mean(),3))
    print(df.groupby("site").agg(n=("y","size"), disease=("y","mean")).round(2).to_string())
    print("\nmissing % by feature:\n", (df[COLS[:-1]].isna().mean()*100).round(0).astype(int).to_dict())
    tr, te = train_test_split(df, test_size=.25, stratify=df.y, random_state=3)
    M = fit(tr); res = {}
    for m in MODS: res[m+" only"] = score(te.y, mk().fit(tr[MODS[m]], tr.y).predict_proba(te[MODS[m]])[:,1])
    allc = sum(MODS.values(), []); res["all features, single GBM"] = score(te.y, mk().fit(tr[allc], tr.y).predict_proba(te[allc])[:,1])
    res["FUSED (3 modalities)"] = score(te.y, proba(M, te))
    print(f"\nHold-out (25 pct, n={len(te)}):"); print(pd.DataFrame(res).T.to_string())
    # external validation: train on 3 hospitals, test on the 4th
    ext = {}
    for s in SITES.values():
        a, b = df[df.site != s], df[df.site == s]
        if b.y.nunique() < 2: continue
        ext[s] = dict(n=len(b), **score(b.y, proba(fit(a), b)))
    print("\nLeave-one-hospital-out (fused):"); print(pd.DataFrame(ext).T.to_string())
    # robustness: what happens when a modality is unavailable
    print("\nFused AUC with modalities removed at test time:")
    for use in [["clinical"],["clinical","exercise_ecg"],["clinical","imaging_scan"],list(MODS)]:
        print(" ", "+".join(use).ljust(36), round(roc_auc_score(te.y, proba(M, te, use)),3))
    joblib.dump(M, "heart_model.joblib"); json.dump(dict(holdout=res, external=ext), open("heart_metrics.json","w"), indent=1)
