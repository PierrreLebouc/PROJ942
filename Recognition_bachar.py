#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recognition_bachar.py — PCA / Eigenfaces trainer & simple recognizer

Usage examples:
  # Train on a dataset organised as data_dir/person_id/*.jpg
  python Recognition_bachar.py --train --data_dir ./data/orl_faces --k 50 --model_out ./eigenfaces_model.npz

  # Predict (after training) for a single image
  python Recognition_bachar.py --predict --model ./eigenfaces_model.npz --image ./some_face.jpg

Dataset layout expected:
  data_dir/
    personA/ img1.jpg img2.jpg ...
    personB/ ...

Dependencies: Pillow (PIL), numpy
(Optionally you can later switch to OpenCV for CLAHE/alignments.)
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps

# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_image(path: Path, size: Tuple[int, int] = (112, 92)) -> np.ndarray:
    """Load → grayscale → resize → histogram equalization → float32 [0,1] → flatten (m,).
    Returns a 1D vector of length m = size[0]*size[1].
    """
    img = Image.open(path).convert("L")            # grayscale
    if img.size != (size[1], size[0]):             # PIL uses (width, height)
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    img = ImageOps.equalize(img)                   # illumination normalization (simple HE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)                         # flatten to (m,)


def load_dataset(data_dir: Path, size: Tuple[int, int] = (112, 92)) -> Tuple[np.ndarray, List[str], Dict[int, str], Tuple[int, int]]:
    """Recursively load all images from subfolders; one subfolder = one class/person.
    Returns X ∈ R^{m×n} with columns=images, labels (list of class names per image), id2label mapping, and image shape (h,w).
    """
    exts = {".jpg", ".jpeg", ".png", ".pgm", ".bmp", ".tif", ".tiff"}
    persons = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not persons:
        raise RuntimeError(f"No person subfolders found in {data_dir}.")

    vectors: List[np.ndarray] = []
    labels: List[str] = []

    for person in persons:
        images = sorted([p for p in person.rglob("*") if p.suffix.lower() in exts])
        if not images:
            continue
        for img_path in images:
            vec = preprocess_image(img_path, size)
            vectors.append(vec)
            labels.append(person.name)

    if not vectors:
        raise RuntimeError(f"No images found under {data_dir}.")

    X = np.column_stack(vectors).astype(np.float32)  # (m, n) columns = images
    unique_labels = sorted(set(labels))
    id2label = {i: name for i, name in enumerate(unique_labels)}
    return X, labels, id2label, (size[0], size[1])

# -----------------------------
# PCA via compact SVD
# -----------------------------

def pca_svd(X: np.ndarray, k: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA using economical SVD on centered data.
    Args:
        X: (m, n) matrix, columns are samples, rows are pixels/features.
        k: number of components to keep; if 0 → choose by 95% cumulative variance.
    Returns:
        mu: (m,1) mean vector
        W:  (m,K) eigenfaces (orthonormal columns)
        Y:  (K,n) projections of centered training data
    """
    mu = X.mean(axis=1, keepdims=True)        # (m,1)
    Xc = X - mu                                # center

    # economical SVD (works well when m >> n)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # U:(m,n) s:(n,) Vt:(n,n)
    # eigenvalues of covariance = s^2/(n-1)
    lam = (s ** 2) / max(1, (X.shape[1] - 1))

    # choose k by 95% variance if not provided
    if k is None or k <= 0 or k > len(s):
        cum = np.cumsum(lam) / np.sum(lam)
        k = int(np.searchsorted(cum, 0.95) + 1)

    W = U[:, :k]                                # (m,k)
    Y = W.T @ Xc                                # (k,n)
    return mu, W.astype(np.float32), Y.astype(np.float32)

# -----------------------------
# Distances & prediction
# -----------------------------

def l2_distance_matrix(q: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute L2 distances between query vector(s) q (K,1 or K,Q) and gallery Y (K,N).
    Returns distances (Q,N)."""
    if q.ndim == 1:
        q = q[:, None]
    # ‖q - y‖² = ‖q‖² + ‖y‖² − 2 qᵀy
    q2 = np.sum(q * q, axis=0, keepdims=True)          # (1,Q)
    y2 = np.sum(Y * Y, axis=0, keepdims=True)          # (1,N)
    cross = q.T @ Y                                    # (Q,N)
    d2 = q2.T + y2 - 2.0 * cross
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def predict_one(x: np.ndarray, mu: np.ndarray, W: np.ndarray, Y: np.ndarray, labels: List[str], image_shape: Tuple[int,int], tau: float = None) -> Tuple[str, float, int]:
    """Project single image vector x (flattened, (m,)) and find nearest neighbor in Y.
    Returns (label, distance, index). If tau is set and min_dist>tau, label="unknown".
    """
    if x.ndim == 1:
        x = x[:, None]
    y = W.T @ (x - mu)          # (K,1)
    D = l2_distance_matrix(y, Y) # (1,N)
    idx = int(np.argmin(D))
    dist = float(D[0, idx])
    label = labels[idx]
    if tau is not None and dist > tau:
        label = "unknown"
    return label, dist, idx

# -----------------------------
# I/O helpers
# -----------------------------

def save_model(path: Path, mu: np.ndarray, W: np.ndarray, Y: np.ndarray, labels: List[str], id2label: Dict[int, str], img_shape: Tuple[int,int]):
    np.savez_compressed(
        path,
        mu=mu,
        W=W,
        Y=Y,
        labels=np.array(labels, dtype=object),
        id2label=np.array([f"{k}:::${v}" for k, v in id2label.items()], dtype=object),
        img_shape=np.array(img_shape, dtype=np.int32),
    )


def load_model(path: Path):
    d = np.load(path, allow_pickle=True)
    mu = d["mu"]
    W = d["W"]
    Y = d["Y"]
    labels = list(d["labels"].tolist())
    raw = d["id2label"].tolist()
    id2label = {}
    for item in raw:
        k, v = str(item).split(":::$")
        id2label[int(k)] = v
    img_shape = tuple(d["img_shape"].tolist())
    return mu, W, Y, labels, id2label, img_shape

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="PCA / Eigenfaces trainer & recognizer")
    parser.add_argument("--data_dir", type=str, default=None, help="Root folder with person subfolders (for training)")
    parser.add_argument("--k", type=int, default=0, help="Number of components; 0 = auto (95% variance)")
    parser.add_argument("--model_out", type=str, default="./eigenfaces_model.npz", help="Output .npz model path (train)")
    parser.add_argument("--model", type=str, default=None, help="Model path (.npz) for prediction")
    parser.add_argument("--image", type=str, default=None, help="Single image to predict")
    parser.add_argument("--height", type=int, default=112, help="Image height")
    parser.add_argument("--width", type=int, default=92, help="Image width")
    parser.add_argument("--train", action="store_true", help="Train a model from --data_dir")
    parser.add_argument("--predict", action="store_true", help="Predict for --image using --model")
    parser.add_argument("--tau", type=float, default=None, help="Optional rejection threshold on L2 distance")

    args = parser.parse_args()

    size = (args.height, args.width)

    if args.train:
        if not args.data_dir:
            raise SystemExit("--data_dir is required for training")
        data_dir = Path(args.data_dir)
        X, labels, id2label, img_shape = load_dataset(data_dir, size)
        print(f"Loaded dataset: X shape={X.shape} (m×n), #images={X.shape[1]}, #classes={len(set(labels))}")
        mu, W, Y = pca_svd(X, k=args.k)
        print(f"PCA done: K={W.shape[1]} components")
        out = Path(args.model_out)
        save_model(out, mu, W, Y, labels, id2label, img_shape)
        print(f"Model saved to: {out}")
        return

    if args.predict:
        if not args.model or not args.image:
            raise SystemExit("--model and --image are required for prediction")
        mu, W, Y, labels, id2label, img_shape = load_model(Path(args.model))
        x = preprocess_image(Path(args.image), size=img_shape)
        lab, dist, idx = predict_one(x, mu, W, Y, labels, img_shape, tau=args.tau)
        print(f"Prediction: {lab}  (distance={dist:.4f})  [matched index={idx}]")
        return

    # If neither flag is provided, show basic help
    parser.print_help()


if __name__ == "__main__":
    main()