"""Shrink the ISIC 2016 training images so they fit in a chat upload.
Usage:  python shrink_isic.py  path/to/unzipped/images  out.zip
Resizes every ISIC_*.jpg to 224x224 (JPEG quality 90) and zips them (~15-25 MB for 900 images).
Needs: pip install pillow"""
import sys, zipfile, io, glob, os
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(src, "**", "ISIC_*.jpg"), recursive=True))
print(len(files), "images found")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        b = io.BytesIO(); Image.open(f).convert("RGB").resize((224, 224), Image.LANCZOS).save(b, "JPEG", quality=90)
        z.writestr(os.path.basename(f), b.getvalue())
print("wrote", out, round(os.path.getsize(out) / 1e6, 1), "MB")
