"""Digital degradation pipeline for synthetic chart generation.

The main benchmark now focuses on digital rendering conditions instead of
camera realism. Supported degradation families are:
  - tone
  - sensor_noise
  - blur
  - compression
  - resample
  - geometry

Geometric transforms update chart and tick-label annotations in the GT.
"""

import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _scale_range_around_one(rng: List[float], scale: float) -> List[float]:
    lo, hi = float(rng[0]), float(rng[1])
    return [1.0 + (lo - 1.0) * scale, 1.0 + (hi - 1.0) * scale]


def _scale_signed_range(rng: List[float], scale: float) -> List[float]:
    lo, hi = float(rng[0]), float(rng[1])
    return [lo * scale, hi * scale]


def _scale_positive_range(rng: List[float], scale: float) -> List[float]:
    lo, hi = float(rng[0]), float(rng[1])
    return [max(0.0, lo * scale), max(0.0, hi * scale)]


def _estimate_ocr_fragility(gt: Dict[str, Any]) -> float:
    """Estimate how vulnerable a chart is to aggressive synthetic degradation."""
    funcs = gt.get("functions", [])
    if not funcs:
        return 0.0

    fn = funcs[0]
    if fn.get("type") != "cubic":
        return 0.0

    q = gt.get("question", {}).get("ground_truth", {})
    extrema = q.get("extrema") or []
    if len(extrema) != 2:
        return 0.0

    try:
        ext_x = [float(e["x"]) for e in extrema]
        ext_y = [float(e["y"]) for e in extrema]
    except (KeyError, TypeError, ValueError):
        return 0.0

    score = 0.0
    min_abs_x = min(abs(x) for x in ext_x)

    if min_abs_x < 1.2:
        score += (1.2 - min_abs_x) / 1.2 * 0.45
    if all(abs(x) < 2.2 for x in ext_x):
        score += 0.15
    if all(x > -0.25 for x in ext_x) or all(x < 0.25 for x in ext_x):
        score += 0.15

    if all(x > -0.25 for x in ext_x) and max(ext_y) < 0.0:
        score += 0.35
    if all(x < 0.25 for x in ext_x) and min(ext_y) > 0.0:
        score += 0.35

    if max(ext_x) < 0.25 and min(ext_y) < 0.0 < max(ext_y):
        score += 0.20
    if min(ext_x) > -0.25 and min(ext_y) < 0.0 < max(ext_y):
        score += 0.20

    return float(min(1.0, max(0.0, score)))


def _transform_point(px: float, py: float, M: np.ndarray) -> Tuple[float, float]:
    p = np.array([[px, py]], dtype=np.float32)
    out = cv2.perspectiveTransform(p.reshape(1, 1, 2), M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def _merge_family_cfg(cfg: Dict[str, Any], name: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    family = deepcopy(defaults)
    override = cfg.get(name, {})
    if isinstance(override, dict):
        family.update(override)
    return family


def apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    out = img.astype(np.float32)
    out = out * brightness
    out = (out - 128.0) * contrast + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_rotation(
    img: np.ndarray,
    angle_deg: float,
    center: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (rotated_image, 3x3 affine matrix for point transform)."""
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(w * cos + h * sin)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - center[0]
    M[1, 2] += nh / 2 - center[1]
    out = cv2.warpAffine(
        img,
        M,
        (nw, nh),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    M3 = np.eye(3, dtype=np.float32)
    M3[:2, :] = M
    return out, M3


def apply_perspective(img: np.ndarray, strength: float) -> Tuple[np.ndarray, np.ndarray]:
    """Random small perspective warp. strength is measured in pixels."""
    h, w = img.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    r = strength
    dst = np.array(
        [
            [random.uniform(-r, r), random.uniform(-r, r)],
            [w + random.uniform(-r, r), random.uniform(-r, r)],
            [w + random.uniform(-r, r), h + random.uniform(-r, r)],
            [random.uniform(-r, r), h + random.uniform(-r, r)],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return out, M


def apply_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 1:
        return img
    ksize = ksize | 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _update_gt_after_geometric(
    gt: Dict[str, Any],
    page_w: int,
    page_h: int,
    M: np.ndarray,
    new_size: Tuple[int, int] | None,
) -> None:
    """Update chart_bbox and tick label positions in gt using 3x3 transform M."""
    if new_size is not None:
        gt["page_size"] = list(new_size)
    [cx, cy, cw, ch] = gt["chart_bbox"]
    corners = [
        (cx, cy),
        (cx + cw, cy),
        (cx + cw, cy + ch),
        (cx, cy + ch),
    ]
    transformed = [_transform_point(x, y, M) for x, y in corners]
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    gt["chart_bbox"] = [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs) - min(xs))),
        int(round(max(ys) - min(ys))),
    ]
    for label in gt.get("x_tick_labels", []) + gt.get("y_tick_labels", []):
        cx_, cy_ = label["center_x"], label["center_y"]
        nx, ny = _transform_point(cx_, cy_, M)
        label["center_x"] = nx
        label["center_y"] = ny
        l, t, r, b = label["left"], label["top"], label["right"], label["bottom"]
        nl, nt = _transform_point(l, t, M)
        nr, nb = _transform_point(r, b, M)
        label["left"] = min(nl, nr)
        label["right"] = max(nl, nr)
        label["top"] = min(nt, nb)
        label["bottom"] = max(nt, nb)


def _center_crop_to_size(
    img: np.ndarray,
    gt: Dict[str, Any],
    target_w: int,
    target_h: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop from center to restore original canvas dimensions after rotation."""
    cur_h, cur_w = img.shape[:2]
    x_off = max(0, (cur_w - target_w) // 2)
    y_off = max(0, (cur_h - target_h) // 2)

    img = img[y_off:y_off + target_h, x_off:x_off + target_w].copy()

    [cx, cy, cw, ch] = gt["chart_bbox"]
    gt["chart_bbox"] = [cx - x_off, cy - y_off, cw, ch]
    gt["page_size"] = [target_w, target_h]

    for label in gt.get("x_tick_labels", []) + gt.get("y_tick_labels", []):
        label["center_x"] -= x_off
        label["center_y"] -= y_off
        label["left"] -= x_off
        label["right"] -= x_off
        label["top"] -= y_off
        label["bottom"] -= y_off

    return img, gt


def apply_render_degradation(
    img: np.ndarray,
    gt: Dict[str, Any],
    config: dict,
    seed: int | None = None,
    fragility: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply configurable digital degradations and update GT for geometry."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cfg = config.get("noise", {})
    if not cfg.get("enabled", True):
        gt["augmentation"] = {"enabled": False, "families": {}}
        return img, gt

    page_w, page_h = gt["page_size"][0], gt["page_size"][1]

    if cfg.get("ocr_safe_adaptation", False):
        if fragility is None:
            fragility = _estimate_ocr_fragility(gt)
        fragility = float(min(1.0, max(0.0, fragility)))
    else:
        fragility = 0.0

    tone_scale = 1.0 - fragility * (1.0 - float(cfg.get("fragile_tone_scale", 0.75)))
    noise_scale = 1.0 - fragility * (1.0 - float(cfg.get("fragile_noise_scale", 0.75)))
    blur_scale = 1.0 - fragility * (1.0 - float(cfg.get("fragile_blur_scale", 0.80)))
    geometry_scale = 1.0 - fragility * (1.0 - float(cfg.get("fragile_geometry_scale", 0.60)))

    tone_cfg = _merge_family_cfg(
        cfg,
        "tone",
        {"enabled": True, "brightness_range": [0.98, 1.02], "contrast_range": [0.99, 1.01]},
    )
    sensor_noise_cfg = _merge_family_cfg(
        cfg,
        "sensor_noise",
        {"enabled": False, "gaussian_sigma": 2.0},
    )
    blur_cfg = _merge_family_cfg(
        cfg,
        "blur",
        {"enabled": False, "ksize_range": [0, 3]},
    )
    compression_cfg = _merge_family_cfg(
        cfg,
        "compression",
        {"enabled": False, "jpeg_quality_range": [80, 95]},
    )
    resample_cfg = _merge_family_cfg(
        cfg,
        "resample",
        {"enabled": False, "scale_range": [0.9, 1.0]},
    )
    geometry_cfg = _merge_family_cfg(
        cfg,
        "geometry",
        {
            "enabled": False,
            "rotation_angle_range": [-0.5, 0.5],
            "perspective_strength_range": [0.0, 0.0],
        },
    )

    families: Dict[str, Dict[str, Any]] = {}

    if tone_cfg.get("enabled", True):
        b_range = _scale_range_around_one(tone_cfg.get("brightness_range", [0.98, 1.02]), tone_scale)
        c_range = _scale_range_around_one(tone_cfg.get("contrast_range", [0.99, 1.01]), tone_scale)
        brightness = random.uniform(b_range[0], b_range[1])
        contrast = random.uniform(c_range[0], c_range[1])
        img = apply_brightness_contrast(img, brightness, contrast)
        families["tone"] = {
            "brightness": round(brightness, 5),
            "contrast": round(contrast, 5),
        }

    if sensor_noise_cfg.get("enabled", False):
        sigma = float(sensor_noise_cfg.get("gaussian_sigma", 2.0)) * noise_scale
        if sigma > 0:
            img = apply_gaussian_noise(img, sigma)
            families["sensor_noise"] = {"gaussian_sigma": round(float(sigma), 5)}

    if blur_cfg.get("enabled", False):
        k_range = blur_cfg.get("ksize_range", [0, 3])
        ksize = random.randint(int(k_range[0]), int(k_range[1])) if int(k_range[1]) >= int(k_range[0]) else 0
        ksize = int(round(ksize * blur_scale))
        if ksize > 0:
            img = apply_blur(img, ksize)
            families["blur"] = {"ksize": int(ksize | 1)}

    if compression_cfg.get("enabled", False):
        q_range = compression_cfg.get("jpeg_quality_range", [80, 95])
        quality = random.randint(int(q_range[0]), int(q_range[1]))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        families["compression"] = {"jpeg_quality": int(quality)}

    if resample_cfg.get("enabled", False):
        scale_range = resample_cfg.get("scale_range", [0.9, 1.0])
        ratio = random.uniform(float(scale_range[0]), float(scale_range[1]))
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        families["resample"] = {"scale_ratio": round(float(ratio), 5)}

    orig_w, orig_h = page_w, page_h

    if geometry_cfg.get("enabled", False):
        rot_range = _scale_signed_range(geometry_cfg.get("rotation_angle_range", [-0.5, 0.5]), geometry_scale)
        angle = random.uniform(float(rot_range[0]), float(rot_range[1]))
        if abs(angle) > 1e-6:
            img, M_rot = apply_rotation(img, angle)
            nh, nw = img.shape[:2]
            _update_gt_after_geometric(gt, page_w, page_h, M_rot, (nw, nh))
            page_w, page_h = nw, nh
            families.setdefault("geometry", {})["rotation_angle_deg"] = round(float(angle), 5)

        persp_range = _scale_positive_range(
            geometry_cfg.get("perspective_strength_range", [0.0, 0.0]),
            geometry_scale,
        )
        max_strength = max(float(persp_range[0]), float(persp_range[1]))
        if max_strength > 0:
            strength = random.uniform(float(persp_range[0]), float(persp_range[1]))
            if strength > 0:
                img, M_per = apply_perspective(img, strength)
                _update_gt_after_geometric(gt, page_w, page_h, M_per, None)
                families.setdefault("geometry", {})["perspective_strength_px"] = round(float(strength), 5)

    cur_h, cur_w = img.shape[:2]
    if cur_w != orig_w or cur_h != orig_h:
        img, gt = _center_crop_to_size(img, gt, orig_w, orig_h)

    gt["augmentation"] = {
        "enabled": True,
        "fragility_score": round(float(fragility), 5),
        "families": families,
    }
    return img, gt
