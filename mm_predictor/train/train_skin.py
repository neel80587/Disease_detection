"""Skin-lesion classifier (ISIC 2016, benign vs malignant) with two image views fused:
 (1) deep features: ImageNet-pretrained MobileNetV3 embeddings (+flip TTA) -> logistic regression
 (2) dermoscopy-style handcrafted features (asymmetry, border, colour, contrast) -> gradient boosting
Fusion: stacked logistic regression on out-of-fold probabilities."""
import os, numpy as np, pandas as pd, torch, timm, joblib, json, warnings
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, balanced_accuracy_score
warnings.filterwarnings("ignore"); torch.set_num_threads(os.cpu_count() or 1)
IMG, LAB = "isic", "/mnt/user-data/uploads/ISBI2016_ISIC_Part3_Training_GroundTruth.csv"

def load():
    l = pd.read_csv(LAB, header=None, names=["id","lab"]); l = l[l.id.map(lambda i: os.path.exists(f"{IMG}/{i}.jpg"))].reset_index(drop=True)
    l["y"] = (l.lab == "malignant").astype(int)
    return l, np.stack([np.asarray(Image.open(f"{IMG}/{i}.jpg").convert("RGB")) for i in l.id])

def deep(X):
    m = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file="mnv3.pth")).eval()
    mu, sd = torch.tensor([.485,.456,.406]).view(1,3,1,1), torch.tensor([.229,.224,.225]).view(1,3,1,1); out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            t = (torch.from_numpy(X[i:i+32]).permute(0,3,1,2).float()/255 - mu)/sd
            f = (m(t) + m(t.flip(3)) + m(t.flip(2)))/3; out.append(f.numpy())
    return np.concatenate(out)

def hand(img):
    g = np.asarray(Image.fromarray(img).convert("L"), float); hsv = np.asarray(Image.fromarray(img).convert("HSV"), float)
    m = g < threshold_otsu(g)
    if m.mean() > .6: m = ~m
    m = ndi.binary_opening(ndi.binary_fill_holes(m), iterations=2); lab, n = ndi.label(m)
    if n: m = lab == (np.argmax(ndi.sum(m, lab, range(1, n+1))) + 1)
    a = m.sum() + 1; per = (m & ~ndi.binary_erosion(m)).sum() + 1
    ys, xs = np.nonzero(m) if m.any() else (np.array([0]), np.array([0]))
    asym = [1 - (m & m[::-1]).sum()/a, 1 - (m & m[:, ::-1]).sum()/a]
    L, S = img[m].astype(float) if m.any() else np.zeros((1,3)), img[~m].astype(float)
    ring = ndi.binary_dilation(m, iterations=6) & ~m; gx, gy = np.gradient(ndi.gaussian_filter(g, 1.5)); gm = np.hypot(gx, gy)
    q = (img[m] // 64) if m.any() else np.zeros((1,3)); ncol = len(np.unique(q[:,0]*16 + q[:,1]*4 + q[:,2]))
    return [a/g.size, per**2/(4*np.pi*a), *asym, (np.ptp(ys)+1)/(np.ptp(xs)+1), *L.mean(0), *L.std(0), *S.mean(0), *(S.mean(0)-L.mean(0)),
            hsv[...,1][m].mean() if m.any() else 0, hsv[...,1][m].std() if m.any() else 0, ncol, gm[ring].mean() if ring.any() else 0, g[m].std() if m.any() else 0]

def lr(): return make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=.05, class_weight="balanced", max_iter=2000))
def gb(): return HGB(max_depth=3, max_iter=100, learning_rate=.05, min_samples_leaf=12, l2_regularization=2., class_weight="balanced", random_state=0)
def logit(p): p = np.clip(p, 1e-4, 1-1e-4); return np.log(p/(1-p))
def rep(y, p, t=.5):
    return dict(auc=round(roc_auc_score(y,p),3), ap=round(average_precision_score(y,p),3), sens=round(recall_score(y,p>t),3),
                spec=round(recall_score(y,p>t,pos_label=0),3), bal_acc=round(balanced_accuracy_score(y,p>t),3))

if __name__ == "__main__":
    l, X = load(); print(len(l), "images; malignant:", l.y.sum(), flush=True)
    if not os.path.exists("feat.npz"): np.savez("feat.npz", D=deep(X), H=np.array([hand(i) for i in X]))
    z = np.load("feat.npz"); D, H = z["D"], np.nan_to_num(z["H"]); y = l.y.values; print("features", D.shape, H.shape, flush=True)
    idx = np.arange(len(y)); tr, te = train_test_split(idx, test_size=.25, stratify=y, random_state=7)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    o1 = cross_val_predict(lr(), D[tr], y[tr], cv=skf, method="predict_proba")[:,1]
    o2 = cross_val_predict(gb(), H[tr], y[tr], cv=skf, method="predict_proba")[:,1]
    m1, m2 = lr().fit(D[tr], y[tr]), gb().fit(H[tr], y[tr]); fu = LogisticRegression(class_weight="balanced").fit(np.c_[logit(o1), logit(o2)], y[tr])
    p1, p2 = m1.predict_proba(D[te])[:,1], m2.predict_proba(H[te])[:,1]; pf = fu.predict_proba(np.c_[logit(p1), logit(p2)])[:,1]
    res = {"deep CNN features": rep(y[te], p1), "handcrafted ABCD-style": rep(y[te], p2), "FUSED": rep(y[te], pf)}
    print(f"\nHold-out n={len(te)} (malignant {y[te].sum()}):"); print(pd.DataFrame(res).T.to_string())
    # 5-fold CV over all data for a steadier estimate (fusion weights learned inside each fold's training part)
    cv = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(D, y):
        s = StratifiedKFold(5, shuffle=True, random_state=0)
        q1 = cross_val_predict(lr(), D[a], y[a], cv=s, method="predict_proba")[:,1]; q2 = cross_val_predict(gb(), H[a], y[a], cv=s, method="predict_proba")[:,1]
        f = LogisticRegression(class_weight="balanced").fit(np.c_[logit(q1), logit(q2)], y[a])
        r1, r2 = lr().fit(D[a], y[a]).predict_proba(D[b])[:,1], gb().fit(H[a], y[a]).predict_proba(H[b])[:,1]
        cv.append([roc_auc_score(y[b], r1), roc_auc_score(y[b], r2), roc_auc_score(y[b], f.predict_proba(np.c_[logit(r1), logit(r2)])[:,1])])
    cv = np.array(cv); print("\n5-fold CV AUC (mean ± sd): deep %.3f±%.3f | hand %.3f±%.3f | fused %.3f±%.3f" % tuple(v for i in range(3) for v in (cv[:,i].mean(), cv[:,i].std())))
    json.dump(dict(holdout=res, cv_auc=dict(zip(["deep","hand","fused"], cv.mean(0).round(3).tolist()))), open("skin_metrics.json","w"), indent=1)
    s = StratifiedKFold(5, shuffle=True, random_state=0)
    q1 = cross_val_predict(lr(), D, y, cv=s, method="predict_proba")[:,1]; q2 = cross_val_predict(gb(), H, y, cv=s, method="predict_proba")[:,1]
    fuse = LogisticRegression(class_weight="balanced").fit(np.c_[logit(q1), logit(q2)], y)
    joblib.dump(dict(deep=lr().fit(D, y), hand=gb().fit(H, y), fuse=fuse), "skin_model.joblib")
