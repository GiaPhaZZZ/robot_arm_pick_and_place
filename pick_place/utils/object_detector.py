#!/usr/bin/env python3
"""
Vision Pipeline — integrates DepthAnythingV2 + YOLOv8 for world-coordinate
object detection, per Chen et al. (2023) §3 + §5.3.

Produces:
  - Per-object 64×64×1 depth image crop (SAC state, paper §4.1.1)
  - World (x,y,z) position for IK (paper §5.3 coordinate transform)
  - Label string for object classification

Camera: configurable height / tilt (defaults: 0.9 m above table, 45° tilt).
        Matches dep_any.py parameter defaults exactly.
"""

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import pipeline as hf_pipeline
from typing import List, Tuple, Dict

try:
    from ultralytics import YOLO as _UltraYOLO
    _HAVE_YOLO = True
except ImportError:
    _HAVE_YOLO = False
    print('[VisionPipeline] WARNING: ultralytics not installed — using depth-only fallback')


# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics (match Gazebo SDF camera plugin)
# ─────────────────────────────────────────────────────────────────────────────
FX, FY = 554.26, 554.26
CX, CY = 320.0, 240.0

_MAX_LIFT_M = 0.30   # objects up to 30 cm tall

# Background labels to ignore
_SURFACE_LABELS = frozenset({
    'dining table', 'table', 'floor', 'wall', 'ceiling',
    'desk', 'counter', 'shelf', 'bench', 'cabinet',
    'carpet', 'rug', 'curtain', 'blanket', 'pillow',
})
_ROBOT_LABELS = frozenset({'arm', 'robot', 'gripper', 'hand', 'person'})


# ─────────────────────────────────────────────────────────────────────────────
def extract_depth_crop(
    depth_map: np.ndarray,   # (H, W) float32, normalized [0,1]
    x1: int, y1: int,
    x2: int, y2: int,
    out_size: int = 64,
) -> np.ndarray:
    """
    Crop + resize depth ROI to 64×64×1 for SAC state input (paper §4.1.1).
    """
    roi  = depth_map[y1:y2, x1:x2]
    crop = cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop.astype(np.float32)[np.newaxis]   # (1, 64, 64)


# ─────────────────────────────────────────────────────────────────────────────
class VisionPipeline:
    """
    Combines YOLO + DepthAnythingV2 to produce SAC-ready states.
    Mirrors dep_any.py / vision_node.py logic exactly, as a standalone
    (non-ROS) class for use in the RL training loop.
    """

    def __init__(
        self,
        camera_height_m:   float = 0.9,     # matches dep_any.py default
        camera_tilt_deg:   float = 45.0,    # matches dep_any.py default
        table_z_world:     float = 0.50,
        gripper_max_m:     float = 0.25,
        min_area_px:       int   = 150,
        max_area_fraction: float = 0.045,
        yolo_model:        str   = 'yolov8n.pt',
        yolo_conf:         float = 0.15,
        depth_device:      int   = -1,
        use_hybrid:        bool  = True,
    ):
        self.table_z_world  = table_z_world
        self.gripper_max_m  = gripper_max_m
        self.min_area_px    = min_area_px
        self.max_area_frac  = max_area_fraction
        self.yolo_conf      = yolo_conf
        self.use_hybrid     = use_hybrid

        # ── Camera geometry — identical to dep_any.py ────────────────────────
        tilt_rad = np.deg2rad(camera_tilt_deg)
        ct, st   = np.cos(tilt_rad), np.sin(tilt_rad)

        self.cam_pos_world = np.array([0.15, 0.0, camera_height_m], dtype=np.float64)

        self.R_cam_to_world = np.array([
            [1,  0,   0 ],
            [0,  ct, -st],
            [0,  st,  ct],
        ], dtype=np.float64)

        # ── Load YOLO ────────────────────────────────────────────────────────
        self._yolo = None
        if _HAVE_YOLO:
            self._yolo = _UltraYOLO(yolo_model)
            print(f'[Vision] Loaded YOLO: {yolo_model}')
        else:
            print('[Vision] Running depth-only detection (no YOLO)')

        # ── Load DepthAnythingV2-Small ───────────────────────────────────────
        dev = depth_device if torch.cuda.is_available() else -1
        print('[Vision] Loading Depth-Anything-V2-Small-hf …')
        self._depth_pipe = hf_pipeline(
            task='depth-estimation',
            model='depth-anything/Depth-Anything-V2-Small-hf',
            device=dev,
        )
        print('[Vision] Depth model ready.')

    # ─────────────────────────────────────────────────────────────────────────
    #  Black-shadow rejection — identical to dep_any.py
    # ─────────────────────────────────────────────────────────────────────────

    def _is_robot_or_shadow(self, bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Rejects robot body parts and shadows that are not graspable objects.

        Two rejection criteria (either triggers rejection):

        1. HSV darkness + neutrality: catches pure-black and near-black surfaces
           (original _is_object_black logic, thresholds unchanged).

        2. Dark + low-saturation with broader V threshold: catches dark-red/brown
           robot arms that appear reddish in raw BGR but are too dark to be
           tabletop objects (V < 80, S < 60 at the 35th percentile).
        """
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_p35 = float(np.percentile(hsv[:, :, 2], 35))
        s_p35 = float(np.percentile(hsv[:, :, 1], 35))

        # Gate 1: original black/shadow check
        if v_p35 < 45.0 and s_p35 < 50.0:
            return True

        # Gate 2: dark muted tones — robot arm / gripper body
        if v_p35 < 80.0 and s_p35 < 60.0:
            return True

        return False

    # Alias for backward compatibility
    def _is_object_black(self, bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        return self._is_robot_or_shadow(bgr, x1, y1, x2, y2)

    # ─────────────────────────────────────────────────────────────────────────
    #  Geometry helpers — identical to dep_any.py / vision_node.py
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_table_depth(self, norm_depth: np.ndarray, h: int, w: int) -> Tuple[float, float]:
        ref = norm_depth[int(h * 0.35):int(h * 0.75), int(w * 0.15):int(w * 0.85)]
        med = float(np.median(ref))
        std = max(float(ref.std()), 0.015)
        return med, std

    def _pixel_to_world(self, u: float, v: float, norm_d: float) -> np.ndarray:
        """Project image pixel + normalized depth → world (x,y,z)."""
        ray_cam = np.array([(u - CX) / FX, (v - CY) / FY, 1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)

        ray_world = self.R_cam_to_world @ ray_cam
        dz = ray_world[2]
        t  = ((self.table_z_world - self.cam_pos_world[2]) / dz) if abs(dz) > 1e-6 else 1.0
        t  = float(np.clip(t, 0.05, 5.0))

        base_pt    = self.cam_pos_world + t * ray_world
        lift       = (1.0 - float(norm_d)) * _MAX_LIFT_M
        world_pt   = base_pt.copy()
        world_pt[2] = self.table_z_world + lift
        return world_pt.astype(np.float32)

    def _roi_to_mask_local(
        self, norm_depth: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        obj_depth: float,
    ) -> np.ndarray:
        roi     = norm_depth[y1:y2, x1:x2]
        roi_std = max(float(roi.std()), 0.02)
        thr     = max(obj_depth - 1.8 * roi_std, 0.0)
        mask    = ((roi > thr) * 255).astype(np.uint8)
        k       = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _best_obb(self, mask: np.ndarray, ox: int, oy: int):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            h, w = mask.shape
            return ((ox + w // 2, oy + h // 2), (w, h), 0)
        biggest = max(cnts, key=cv2.contourArea)
        (cx, cy), dims, angle = cv2.minAreaRect(biggest)
        return ((cx + ox, cy + oy), dims, angle)

    # ─────────────────────────────────────────────────────────────────────────
    #  Detection dict factory — matches dep_any.py _make_det exactly
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_det(label, conf, u_c, v_c, bbox, obb, world_pos, depth_val, size_m, area_px,
                  depth_crop=None) -> Dict:
        """
        bbox stored as (x1, y1, bw, bh) — matches dep_any.py convention.
        depth_crop (1, 64, 64) is additional for SAC state use.
        """
        return {
            'label':      label,
            'conf':       conf,
            'centroid':   (int(u_c), int(v_c)),
            'bbox':       bbox,          # (x1, y1, bw, bh)
            'obb':        obb,
            'world_pos':  world_pos,
            'depth_val':  depth_val,
            'size_m':     size_m,
            'area_px':    area_px,
            'depth_crop': depth_crop,    # (1,64,64) SAC state; None if not extracted
        }

    @staticmethod
    def _nms_centroid(detections: list, min_dist_px: int = 20) -> list:
        """Identical to dep_any.py _nms_centroid."""
        kept = []
        for det in detections:
            cx, cy = det['centroid']
            too_close = False
            for k in kept:
                kx, ky = k['centroid']
                if abs(cx - kx) < min_dist_px and abs(cy - ky) < min_dist_px:
                    too_close = True
                    break
            if not too_close:
                kept.append(det)
        return kept

    # ─────────────────────────────────────────────────────────────────────────
    #  YOLO detection pass — mirrors dep_any.py _detect_yolo
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_yolo(
        self, bgr: np.ndarray, norm_depth: np.ndarray,
        table_norm_d: float, table_std: float,
    ) -> list:
        results = self._yolo(bgr, conf=self.yolo_conf, verbose=False)[0]
        h, w    = bgr.shape[:2]
        total_pixels = h * w
        detections   = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = self._yolo.names[cls_id].lower().strip()
            conf   = float(box.conf[0])

            if label in _SURFACE_LABELS:
                continue
            if any(r in label for r in ('hand', 'gripper', 'robot', 'arm')):
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw, bh  = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                continue

            if self._is_object_black(bgr, x1, y1, x2, y2):
                continue

            # ── Spatial extent: reject arm-sized boxes ──
            if bw > (w * 0.30) and bh > (h * 0.30):
                continue

            box_area = float(bw * bh)
            if box_area > (total_pixels * self.max_area_frac):
                continue

            u_c = float(x1 + bw / 2)
            v_c = float(y1 + bh / 2)

            obj_depth = float(np.median(norm_depth[y1:y2, x1:x2]))

            world_pos = self._pixel_to_world(u_c, v_c, obj_depth)
            dist_m    = max(float(np.linalg.norm(world_pos - self.cam_pos_world)), 0.05)

            f_avg  = (FX + FY) / 2.0
            size_m = max(bw, bh) * dist_m / f_avg
            if size_m > self.gripper_max_m:
                continue

            roi_mask   = self._roi_to_mask_local(norm_depth, x1, y1, x2, y2, obj_depth)
            obb        = self._best_obb(roi_mask, x1, y1)
            depth_crop = extract_depth_crop(norm_depth, x1, y1, x2, y2)

            detections.append(self._make_det(
                label, conf, u_c, v_c,
                (x1, y1, bw, bh), obb, world_pos, obj_depth, size_m, box_area,
                depth_crop))

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    #  Depth-only fallback — mirrors dep_any.py _detect_depth_fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_depth_fallback(
        self, bgr: np.ndarray, norm_depth: np.ndarray, h: int, w: int,
        table_norm_d: float, table_std: float,
    ) -> list:
        depth_gray = (norm_depth * 255).astype(np.uint8)
        blurred    = cv2.GaussianBlur(depth_gray, (5, 5), 0)
        edges      = cv2.Canny(blurred, 15, 50)
        k          = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg         = cv2.dilate(edges, k, iterations=1)
        cnts, _    = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in cnts:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area_px or area > (h * w * self.max_area_frac):
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            u_c = M['m10'] / M['m00']
            v_c = M['m01'] / M['m00']

            if v_c < h * 0.15:
                continue

            bx, by, bw2, bh2 = cv2.boundingRect(cnt)

            # Allow bottom edge to be clipped (matches dep_any.py)
            if bx <= 0 or by <= 0 or (bx + bw2) >= w:
                continue

            # Reject objects taking up almost the whole horizontal field
            if bw2 > (w * 0.85):
                continue

            if self._is_object_black(bgr, bx, by, bx + bw2, by + bh2):
                continue

            obj_depth = float(np.median(norm_depth[by:by + bh2, bx:bx + bw2]))
            if abs(obj_depth - table_norm_d) < (0.3 * table_std):
                continue

            obb = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = obb
            side_long  = max(rw, rh)
            side_short = min(rw, rh)
            if side_short > 0 and (side_long / side_short) > 6.0:
                continue

            # ── Robot-arm rejection: solidity check ──────────────────────────
            hull   = cv2.convexHull(cnt)
            h_area = float(cv2.contourArea(hull))
            if h_area > 0 and (area / h_area) < 0.45:
                continue

            # ── Robot-arm rejection: spatial extent check ────────────────────
            if bw2 > (w * 0.30) and bh2 > (h * 0.30):
                continue

            world_pos = self._pixel_to_world(u_c, v_c, obj_depth)
            dist_m    = max(float(np.linalg.norm(world_pos - self.cam_pos_world)), 0.05)
            f_avg     = (FX + FY) / 2.0
            size_m    = side_long * dist_m / f_avg
            if size_m > self.gripper_max_m:
                continue

            depth_crop = extract_depth_crop(norm_depth, bx, by, bx + bw2, by + bh2)

            detections.append(self._make_det(
                'object', 0.85, u_c, v_c,
                (bx, by, bw2, bh2), obb, world_pos, obj_depth, size_m, area,
                depth_crop))

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    #  Hybrid sweep — mirrors dep_any.py _hybrid_depth_sweep
    # ─────────────────────────────────────────────────────────────────────────

    def _hybrid_depth_sweep(
        self, bgr: np.ndarray, norm_depth: np.ndarray, h: int, w: int,
        table_norm_d: float, table_std: float, existing: list,
    ) -> list:
        extra = self._detect_depth_fallback(bgr, norm_depth, h, w, table_norm_d, table_std)

        def _iou_1d(a0, a1, b0, b1):
            inter = max(0, min(a1, b1) - max(a0, b0))
            union = (a1 - a0) + (b1 - b0) - inter
            return inter / union if union > 0 else 0.0

        result = list(existing)
        for cand in extra:
            cx1, cy1, cw, ch = cand['bbox']
            cx2, cy2 = cx1 + cw, cy1 + ch
            overlap = False
            for ex in existing:
                ex1, ey1, ew, eh = ex['bbox']
                ex2, ey2 = ex1 + ew, ey1 + eh
                if _iou_1d(cx1, cx2, ex1, ex2) > 0.20 and _iou_1d(cy1, cy2, ey1, ey2) > 0.20:
                    overlap = True
                    break
            if not overlap:
                result.append(cand)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, bgr_img: np.ndarray) -> List[Dict]:
        """
        Main detection entry point. Returns list of detection dicts, each with:
          'label'      : str
          'conf'       : float
          'centroid'   : (u, v) int pixels
          'bbox'       : (x1, y1, bw, bh)   ← dep_any.py convention
          'obb'        : cv2.minAreaRect tuple
          'world_pos'  : np.ndarray (3,) — world x,y,z
          'depth_val'  : float — normalized depth at centroid
          'size_m'     : float — estimated longest dimension in metres
          'area_px'    : float — bounding-box pixel area
          'depth_crop' : np.ndarray (1,64,64) — SAC state crop
        """
        h, w  = bgr_img.shape[:2]
        pil   = PILImage.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

        with torch.inference_mode():
            raw_depth = np.array(self._depth_pipe(pil)['depth'], dtype=np.float32)

        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min < 1e-6:
            return []
        norm_depth = (raw_depth - d_min) / (d_max - d_min)

        table_norm_d, table_std = self._estimate_table_depth(norm_depth, h, w)

        # Primary pass
        if self._yolo is not None:
            with torch.inference_mode():
                detections = self._detect_yolo(bgr_img, norm_depth, table_norm_d, table_std)
        else:
            detections = self._detect_depth_fallback(bgr_img, norm_depth, h, w, table_norm_d, table_std)

        # Hybrid contour pass
        if self._yolo is not None and self.use_hybrid:
            detections = self._hybrid_depth_sweep(
                bgr_img, norm_depth, h, w, table_norm_d, table_std, detections)

        # Post-processing — identical ordering to dep_any.py
        detections = self._nms_centroid(detections, min_dist_px=20)
        detections.sort(key=lambda d: -d['depth_val'])

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    #  Debug visualisation — matches dep_any.py _draw_debug
    # ─────────────────────────────────────────────────────────────────────────

    def draw_debug(
        self, bgr: np.ndarray, norm_depth: np.ndarray,
        detections: List[Dict], table_norm_d: float = 0.0,
    ) -> np.ndarray:
        """Overlay detections on a depth-blended image for visualisation."""
        depth_color = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(bgr, 0.45, depth_color, 0.55, 0)

        cv2.putText(out, f'table_d={table_norm_d:.2f}', (6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        for idx, det in enumerate(detections):
            u, v           = det['centroid']
            bx, by, bw, bh = det['bbox']
            wp             = det['world_pos']

            box_pts = cv2.boxPoints(det['obb']).astype(np.int32)
            cv2.drawContours(out, [box_pts], -1, (0, 255, 0), 2)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 180, 255), 1)
            cv2.circle(out, (u, v), 5, (0, 0, 255), -1)

            lbl = (f"#{idx} {det['label']} {det['size_m']*100:.1f}cm "
                   f"[{int(det['area_px'])}px] "
                   f"W=({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) c={det['conf']:.2f}")
            cv2.putText(out, lbl, (bx, max(by - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(out, f'{len(detections)} graspable | YOLO={_HAVE_YOLO}',
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2, cv2.LINE_AA)
        return out