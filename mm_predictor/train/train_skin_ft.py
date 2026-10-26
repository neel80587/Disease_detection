"""Fine-tune a CNN on HAM10000 and fuse it with patient metadata (age, sex, body site).
Run on a GPU (Google Colab). Trains two models, image-only and image+metadata, so you can see whether metadata helps,
and saves whichever has the better validation AUC as out/skin_ft.pt.
Label: malignant = melanoma, basal cell carcinoma, actinic keratosis/intraepithelial carcinoma (akiec).
Splits are grouped by lesion_id so the same lesion never appears in both train and test."""
import os, glob, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression

DATA, OUT = os.environ.get("HAM_DIR", "ham10000"), os.environ.get("OUT_DIR", "out")
EPOCHS, BS, SIZE, PRE = int(os.environ.get("EPOCHS", 12)), 64, 224, 256
BACKBONE, PRETRAINED = os.environ.get("BACKBONE", "efficientnet_b0"), os.environ.get("PRETRAINED", "1") == "1"
MALIGNANT = {"mel", "bcc", "akiec"}
SEXES = ["male", "female", "unknown"]
SITES = ["abdomen","acral","back","chest","ear","face","foot","genital","hand","lower extremity","neck","scalp","trunk","unknown","upper extremity"]
META_DIM = 2 + len(SEXES) + len(SITES)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def meta_vec(age=None, sex=None, site=None):
    a = np.nan if age is None else float(age)
    sex = sex if sex in SEXES[:2] else "unknown"; site = site if site in SITES else "unknown"
    v = [0.5 if np.isnan(a) else a / 100.0, float(np.isnan(a))]
    return np.array(v + [float(sex == s) for s in SEXES] + [float(site == s) for s in SITES], np.float32)

class Net(nn.Module):
    def __init__(self, backbone, use_tab, pretrained):
        super().__init__(); self.use_tab = use_tab
        self.b = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        with torch.no_grad(): d = self.b.eval()(torch.zeros(1, 3, SIZE, SIZE)).shape[1]   # true pooled feature size
        self.b.train()
        self.t = nn.Sequential(nn.Linear(META_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()) if use_tab else None
        self.h = nn.Sequential(nn.Dropout(.3), nn.Linear(d + (64 if use_tab else 0), 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))
    def forward(self, x, m=None):
        f = self.b(x)
        if self.use_tab: f = torch.cat([f, self.t(m)], 1)
        return self.h(f).squeeze(1)

def _norm(x, dev):
    mu = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1); sd = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)
    return (x - mu) / sd

def prep(x, dev):  # uint8 NHWC -> normalised NCHW, centre crop
    o = (PRE - SIZE) // 2
    return _norm(x.permute(0, 3, 1, 2).float()[:, :, o:o + SIZE, o:o + SIZE] / 255, dev)

def aug(x, dev):
    n = x.shape[0]; x = x.permute(0, 3, 1, 2).float() / 255
    ij = np.random.randint(0, PRE - SIZE + 1, (n, 2))
    x = torch.stack([x[k, :, i:i + SIZE, j:j + SIZE] for k, (i, j) in enumerate(ij)])
    for d in (3, 2):
        f = torch.rand(n, device=dev) < .5; x[f] = x[f].flip(d)
    x = x * (1 + (torch.rand(n, 1, 1, 1, device=dev) - .5) * .4)
    mn = x.mean((1, 2, 3), keepdim=True); x = ((x - mn) * (1 + (torch.rand(n, 1, 1, 1, device=dev) - .5) * .4) + mn).clamp(0, 1)
    return _norm(x, dev)

@torch.no_grad()
def predict(net, X, M, dev=DEV, bs=128):
    net.eval().to(dev); out = []
    for k in range(0, len(X), bs):
        x = prep(torch.from_numpy(X[k:k + bs]).to(dev), dev); m = torch.from_numpy(M[k:k + bs]).to(dev)
        with torch.autocast(dev, enabled=dev == "cuda"):
            z = (net(x, m) + net(x.flip(3), m) + net(x.flip(2), m)) / 3
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)

def load_ft(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(ck["backbone"], ck["use_tab"], False); net.load_state_dict(ck["state"]); return net.eval(), ck

def predict_image(net, ck, pil, age=None, sex=None, site=None):
    X = np.asarray(pil.convert("RGB").resize((PRE, PRE), Image.BILINEAR))[None]
    z = float(predict(net, X, meta_vec(age, sex, site)[None], dev="cpu")[0])
    p = 1 / (1 + np.exp(-(ck["calib"][0] * z + ck["calib"][1])))
    return dict(probability=float(p), flagged=bool(z >= ck["thr"]), uses_metadata=ck["use_tab"])

def load_data():
    meta = pd.read_csv(f"{DATA}/HAM10000_metadata.csv")
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{DATA}/**/*.jpg", recursive=True)}
    meta = meta[meta.image_id.isin(paths)].reset_index(drop=True); print(len(meta), "images found", flush=True)
    rd = lambda i: np.asarray(Image.open(paths[i]).convert("RGB").resize((PRE, PRE), Image.BILINEAR))
    with ThreadPoolExecutor(8) as ex: X = np.stack(list(ex.map(rd, meta.image_id)))
    M = np.stack([meta_vec(None if pd.isna(a) else a, s, l) for a, s, l in zip(meta.age, meta.sex, meta.localization)])
    return meta, X, M, meta.dx.isin(MALIGNANT).values.astype(np.float32)

def split(meta, y):
    g = meta.lesion_id.values; idx = np.arange(len(y))
    rest, te = next(StratifiedGroupKFold(7, shuffle=True, random_state=0).split(idx, y, g))
    tr_, va_ = next(StratifiedGroupKFold(6, shuffle=True, random_state=0).split(rest, y[rest], g[rest]))
    return rest[tr_], rest[va_], te

def train(use_tab, X, M, Y, tr, va):
    torch.manual_seed(0); np.random.seed(0)
    net = Net(BACKBONE, use_tab, PRETRAINED).to(DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - Y[tr].mean()) / Y[tr].mean(), device=DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=EPOCHS * int(np.ceil(len(tr) / BS)), pct_start=.15)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda"); best = (-1, None)
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(tr); t0 = time.time()
        for k in range(0, len(perm), BS):
            idx = np.sort(perm[k:k + BS])
            x = aug(torch.from_numpy(X[idx]).to(DEV), DEV); m = torch.from_numpy(M[idx]).to(DEV); y = torch.from_numpy(Y[idx]).to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"): loss = lossf(net(x, m), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        auc = roc_auc_score(Y[va], predict(net, X[va], M[va]))
        print(f"  [{'image+meta' if use_tab else 'image only'}] epoch {ep+1}/{EPOCHS}  loss {loss.item():.3f}  val AUC {auc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if auc > best[0]: best = (auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); return net, best[0]

def evaluate(zv, yv, zt, yt):
    fpr, tpr, th = roc_curve(yv, zv); thr = float(th[np.argmax(tpr >= .9)])   # val threshold for >=90% sensitivity
    cal = LogisticRegression(C=1e6).fit(zv[:, None], yv); pr = zt >= thr
    rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() != yt[i].max(): bs.append(roc_auc_score(yt[i], zt[i]))
    return dict(auc=round(roc_auc_score(yt, zt), 3), auc_ci95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                ap=round(average_precision_score(yt, zt), 3), sens_at_val_thr=round(float((pr & (yt == 1)).sum() / (yt == 1).sum()), 3),
                spec_at_val_thr=round(float((~pr & (yt == 0)).sum() / (yt == 0).sum()), 3)), thr, (float(cal.coef_[0, 0]), float(cal.intercept_[0]))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); meta, X, M, Y = load_data(); tr, va, te = split(meta, Y)
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}; malignant rate {Y.mean():.2f}; device {DEV}", flush=True)
    res, models = {}, {}
    for use_tab in (False, True):
        net, vauc = train(use_tab, X, M, Y, tr, va); name = "image+metadata" if use_tab else "image only"
        r, thr, cal = evaluate(predict(net, X[va], M[va]), Y[va], predict(net, X[te], M[te]), Y[te]); r["val_auc"] = round(vauc, 3)
        res[name] = r; models[name] = (vauc, net, thr, cal, use_tab); print(name, r, flush=True)
    vauc, net, thr, cal, use_tab = max(models.values(), key=lambda t: t[0])
    torch.save(dict(state=net.cpu().state_dict(), backbone=BACKBONE, use_tab=use_tab, calib=cal, thr=thr), f"{OUT}/skin_ft.pt")
    json.dump(res, open(f"{OUT}/skin_ft_metrics.json", "w"), indent=1); print("\nSaved", f"{OUT}/skin_ft.pt", "| metadata used:", use_tab)
