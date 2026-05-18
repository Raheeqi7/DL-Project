"""Deep learning helpers for brain tumor MRI classification."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
CLASS_LABELS = ("No Tumor", "Tumor")


def load_image_rgb(file_or_bytes) -> np.ndarray:
    if hasattr(file_or_bytes, "read"):
        file_or_bytes.seek(0)
        img = Image.open(file_or_bytes).convert("RGB")
    else:
        img = Image.open(io.BytesIO(file_or_bytes)).convert("RGB")
    return np.array(img)


def resize_rgb(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def to_batch(img: np.ndarray) -> np.ndarray:
    return np.expand_dims(img, axis=0)


def preprocess(
    file_or_bytes,
    size: int = IMG_SIZE,
    augment: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (display_rgb, model_batch)."""
    raw = load_image_rgb(file_or_bytes)
    display = resize_rgb(raw, size)

    work = display.copy()
    if augment == "flip_h":
        work = np.fliplr(work)
    elif augment == "flip_v":
        work = np.flipud(work)
    elif augment == "rotate_90":
        work = np.rot90(work)
    elif augment == "brighten":
        work = np.clip(work.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    elif augment == "contrast":
        work = np.clip((work.astype(np.float32) - 128) * 1.2 + 128, 0, 255).astype(np.uint8)

    batch = to_batch(normalize(work))
    return display, batch


def predict_proba(model: tf.keras.Model, batch: np.ndarray) -> float:
    out = model.predict(batch, verbose=0)
    return float(np.squeeze(out))


def predict_with_tta(
    model: tf.keras.Model,
    display_rgb: np.ndarray,
    use_tta: bool = True,
) -> tuple[float, dict[str, float]]:
    """Average probability across flips / light transforms."""
    variants: dict[str, np.ndarray | None] = {
        "original": None,
        "flip_h": "flip_h",
        "flip_v": "flip_v",
        "rotate_90": "rotate_90",
    }
    if not use_tta:
        variants = {"original": None}

    scores: dict[str, float] = {}
    for name, aug in variants.items():
        if aug is None:
            batch = to_batch(normalize(display_rgb))
        else:
            _, batch = preprocess_from_array(display_rgb, augment=aug)
        scores[name] = predict_proba(model, batch)

    mean_prob = float(np.mean(list(scores.values())))
    return mean_prob, scores


def preprocess_from_array(
    display_rgb: np.ndarray,
    augment: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    work = display_rgb.copy()
    if augment == "flip_h":
        work = np.fliplr(work)
    elif augment == "flip_v":
        work = np.flipud(work)
    elif augment == "rotate_90":
        work = np.rot90(work)
    batch = to_batch(normalize(work))
    return display_rgb, batch


def classify(prob: float, threshold: float) -> tuple[str, str, float]:
    is_tumor = prob >= threshold
    label = CLASS_LABELS[1] if is_tumor else CLASS_LABELS[0]
    confidence = prob if is_tumor else (1.0 - prob)
    risk = "High" if confidence >= 0.85 else "Medium" if confidence >= 0.65 else "Low"
    return label, risk, confidence


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer | None:
    last_conv: tf.keras.layers.Layer | None = None

    def visit(submodel: tf.keras.Model) -> None:
        nonlocal last_conv
        for layer in submodel.layers:
            if isinstance(
                layer,
                (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D),
            ):
                last_conv = layer
            elif isinstance(layer, tf.keras.Model):
                visit(layer)

    visit(model)
    return last_conv


def make_gradcam_heatmap(
    model: tf.keras.Model,
    batch: np.ndarray,
    display_rgb: np.ndarray,
) -> np.ndarray | None:
    conv_layer = find_last_conv_layer(model)
    if conv_layer is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(batch)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (display_rgb.shape[1], display_rgb.shape[0]))
        return heatmap
    except Exception:
        return None


def occlusion_sensitivity_map(
    model: tf.keras.Model,
    display_rgb: np.ndarray,
    grid: int = 8,
) -> np.ndarray:
    """Fallback spatial importance when Grad-CAM is unavailable."""
    h, w = display_rgb.shape[:2]
    cell_h, cell_w = h // grid, w // grid
    base = predict_proba(model, to_batch(normalize(display_rgb)))
    importance = np.zeros((grid, grid), dtype=np.float32)

    for i in range(grid):
        for j in range(grid):
            masked = display_rgb.copy()
            y0, y1 = i * cell_h, min((i + 1) * cell_h, h)
            x0, x1 = j * cell_w, min((j + 1) * cell_w, w)
            masked[y0:y1, x0:x1] = 0
            prob = predict_proba(model, to_batch(normalize(masked)))
            importance[i, j] = max(0.0, base - prob)

    if importance.max() > 0:
        importance /= importance.max()
    heatmap = cv2.resize(importance, (w, h), interpolation=cv2.INTER_CUBIC)
    return heatmap


def overlay_heatmap(display_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (alpha * colored + (1 - alpha) * display_rgb).astype(np.uint8)
    return blended


def model_summary_dict(model: tf.keras.Model) -> dict[str, Any]:
    trainable = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights]))
    conv_layers = sum(
        1
        for layer in model.layers
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D))
    )
    dense_layers = sum(1 for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))
    return {
        "name": model.name,
        "layers": len(model.layers),
        "conv_layers": conv_layers,
        "dense_layers": dense_layers,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "total_params": trainable + non_trainable,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
    }


def layer_table(model: tf.keras.Model, max_rows: int = 25) -> list[dict[str, str]]:
    rows = []
    for layer in model.layers[:max_rows]:
        rows.append(
            {
                "Layer": layer.name,
                "Type": layer.__class__.__name__,
                "Output shape": str(getattr(layer, "output_shape", "—")),
            }
        )
    return rows
