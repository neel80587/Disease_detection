# Multimodal Human Disease Predictor

Three trained models behind one Streamlit app.

| Module | Data | Modalities fused | Held-out result |
|---|---|---|---|
| Heart disease | UCI Heart Disease (920 patients, 4 hospitals) | clinical + exercise/ECG + scan results | AUC 0.89 (leave-one-hospital-out 0.68-0.89) |
| Skin lesion | ISIC 2016 (865 dermoscopy images) | deep CNN features + handcrafted ABCD-style features | 5-fold CV AUC 0.81 |
| Common symptoms | **Synthetic** (hand-written prior) | symptoms + vitals + free-text notes | demo of the pipeline only |

**Research demo, not medical advice.** Not clinically validated.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.10-3.12. All trained models and the pretrained MobileNetV3 weights are already in `models/`, so no training is needed.

## Use from code
```python
import predictors as P
P.predict_heart(60, 1, 4, 145, 250, 1, use=["clinical"])
P.predict_skin("lesion.jpg")
P.predict_symptoms(["fever", "cough"], "chills", temp=38.7)
```

## Retrain
Scripts are in `train/`. Each expects its data (paths are at the top of each file):
- `train_heart.py`: the four `processed_*.data` files from the UCI Heart Disease download.
- `train_skin.py`: the ISIC 2016 images in `isic/` plus `ISBI2016_ISIC_Part3_Training_GroundTruth.csv`, and `mnv3.pth` (timm `mobilenetv3_large_100_ra-f55367f5.pth`). `shrink_isic.py` resizes the images to 224 px.
- `train_symptoms.py`: generates its own synthetic data.
Copy the resulting `.joblib` files into `models/` (as `heart`, `skin`, `symptoms`).

## Upgrade the skin model (optional, needs a GPU)
`train/train_skin_colab.ipynb` fine-tunes an EfficientNet on HAM10000 (10,015 images) together with age, sex and body site,
and reports image-only vs image+metadata results. Copy the resulting `skin_ft.pt` into `models/`. The app then switches to it
automatically, adds age/sex/site inputs, and flags lesions above a threshold tuned for about 90% sensitivity.

## Layout
```
app.py            Streamlit UI (3 tabs)
predictors.py     predict_heart / predict_skin / predict_symptoms
train/            training scripts
models/           trained models + pretrained weights
```
