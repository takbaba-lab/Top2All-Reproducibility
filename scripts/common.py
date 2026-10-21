"""Shared model input contract and provenance helpers."""
from pathlib import Path
import hashlib
from preprocessing.preprocessing import preprocess
BUNDLE = Path(__file__).resolve().parent / "preprocessing"
PIPELINE = "Top2All"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
