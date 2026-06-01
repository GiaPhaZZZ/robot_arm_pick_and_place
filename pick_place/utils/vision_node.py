#!/usr/bin/env python3
"""
Vision ROS2 Node — YOLO + DepthAnythingV2 perception pipeline.

Subscribes to:
    /camera/image_raw          (sensor_msgs/Image, BGR)

Publishes to:
    /detected_objects/poses    (geometry_msgs/PoseArray) — one pose per detection
    /detected_objects/classes  (std_msgs/String)         — comma-separated metadata
    /depth_anything/debug_image (sensor_msgs/Image)       — annotated frame

Usage (standalone):
    ros2 run rl_pick_place vision_node --yolo-model yolov26n.pt
"""

import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from transformers import pipeline as hf_pipeline

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String

# ── Camera intrinsics (match Gazebo SDF / real calibration) ─────────────────
FX = 554.26
FY = 554.26
CX = 320.0
CY = 240.0

# ── Maximum object footprint the gripper can handle ──────────────────────────
_DEFAULT_GRIPPER_MAX_M = 0.25
_MAX_LIFT_M = 0.30   # objects up to 30 cm tall

# ── Background-only YOLO/COCO labels (whitelist approach) ───────────────────
_SURFACE_LABELS = frozenset({
    'dining table', 'table', 'floor', 'wall', 'ceiling',
    'desk', 'counter', 'shelf', 'bench', 'cabinet',
    'carpet', 'rug', 'curtain', 'blanket', 'pillow',
})

try:
    from ultralytics import YOLO as _YOLO
    _HAVE_YOLO = True
except ImportError:
    _HAVE_YOLO = False


class DepthAnythingDetectorNode(Node):

    def __init__(
        self,
        yolo_model_init: str = 'yolov26n.pt',
        yolo_conf_init: float = 0.15,
        publish_debug_init: bool = True,
        use_hybrid_init: bool = True,
        depth_device_init: int = -1
    ):
        super().__init__('depth_anything_detector')

        # ── ROS Parameters ───────────────────────────────────────────────────
        self.declare_parameter('camera_height_m',      0.9)
        self.declare_parameter('camera_tilt_deg',      45.0)
        self.declare_parameter('publish_debug_image',  publish_debug_init)
        self.declare_parameter('fg_k_sigma',           1.0) 
        self.declare_parameter('min_area_px',          150)
        self.declare_parameter('max_area_fraction',    0.045) 
        self.declare_parameter('table_z_world',        0.50)
        self.declare_parameter('gripper_max_m',        _DEFAULT_GRIPPER_MAX_M)
        self.declare_parameter('yolo_conf',            yolo_conf_init) 
        self.declare_parameter('hybrid_depth_pass',    use_hybrid_init)
        self.declare_parameter('yolo_model',           yolo_model_init) 

        cam_h               = self.get_parameter('camera_height_m').value
        tilt_deg            = self.get_parameter('camera_tilt_deg').value
        self.publish_debug  = self.get_parameter('publish_debug_image').value
        self.fg_k           = self.get_parameter('fg_k_sigma').value
        self.min_area_px    = self.get_parameter('min_area_px').value
        self.max_area_frac  = self.get_parameter('max_area_fraction').value
        self.table_z_world  = self.get_parameter('table_z_world').value
        self.gripper_max_m  = self.get_parameter('gripper_max_m').value
        self.yolo_conf      = self.get_parameter('yolo_conf').value
        self.hybrid_pass    = self.get_parameter('hybrid_depth_pass').value
        yolo_model_name     = self.get_parameter('yolo_model').value

        # ── Camera geometry ───────────────────────────────────────────────────
        tilt_rad = np.deg2rad(tilt_deg)
        ct, st   = np.cos(tilt_rad), np.sin(tilt_rad)

        self.cam_pos_world = np.array([0.15, 0.0, cam_h], dtype=np.float64)

        self.R_cam_to_world = np.array([
            [1,   0,   0 ],
            [0,   ct, -st],
            [0,   st,   ct],
        ], dtype=np.float64)

        # ── Load YOLO Model ──────────────────────────────────────────────────
        if _HAVE_YOLO:
            self.get_logger().info(f'Loading {yolo_model_name}…')
            self._yolo = _YOLO(yolo_model_name)
            self.get_logger().info(f'{yolo_model_name} loaded successfully.')
        else:
            self.get_logger().warn('ultralytics not installed – using fallback pipeline.')
            self._yolo = None

        # ── Load Depth Anything V2 Small ─────────────────────────────────────
        self.get_logger().info('Loading Depth-Anything-V2-Small…')
        
        # Decide device allocation map based on user CLI vs available hardware
        if depth_device_init >= 0 and torch.cuda.is_available():
            self.device = depth_device_init
        else:
            self.device = 0 if torch.cuda.is_available() else -1

        self._depth_pipe = hf_pipeline(
            task='depth-estimation',
            model='depth-anything/Depth-Anything-V2-Small-hf',
            device=self.device,
        )
        self.get_logger().info(f'Depth model loaded on device target: {self.device}')

        # ── ROS I/O ───────────────────────────────────────────────────────────
        try:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
        except ImportError:
            self.get_logger().error('cv_bridge is missing! Python dependency broken.')
            sys.exit(1)

        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, qos_profile_sensor_data)
            
        self.pub_poses   = self.create_publisher(PoseArray, '/detected_objects/poses',   10)
        self.pub_classes = self.create_publisher(String,    '/detected_objects/classes', 10)
        
        if self.publish_debug:
            self.pub_debug = self.create_publisher(Image, '/depth_anything/debug_image', 10)

    # ═════════════════════════════════════════════════════════════════════════
    #  Black Object Verification Helper
    # ═════════════════════════════════════════════════════════════════════════

    def _is_robot_or_shadow(self, cv_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Rejects robot body parts and shadows that are not graspable objects.

        Two rejection criteria (either triggers rejection):

        1. HSV darkness + neutrality: catches pure-black and near-black surfaces
           (original _is_object_black logic, thresholds unchanged).

        2. Dark + low-saturation with broader V threshold: catches dark-red/brown
           robot arms that appear reddish in the raw BGR image but are still far
           too dark to be tabletop objects (V < 80, S < 60 at the 35th percentile).
           The robot arm in the depth overlay is dark reddish — this gate catches it.
        """
        roi = cv_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv_roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_p35    = float(np.percentile(hsv_roi[:, :, 2], 35))
        s_p35    = float(np.percentile(hsv_roi[:, :, 1], 35))

        # Gate 1: original black/shadow check
        if v_p35 < 45.0 and s_p35 < 50.0:
            return True

        # Gate 2: dark muted tones — robot arm / gripper body
        # (dark enough that no graspable object should be this dim,
        #  and low enough saturation that it is not a coloured target object)
        if v_p35 < 80.0 and s_p35 < 60.0:
            return True

        return False

    # Keep the old name as an alias so any external callers still work
    def _is_object_black(self, cv_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        return self._is_robot_or_shadow(cv_bgr, x1, y1, x2, y2)

    # ═════════════════════════════════════════════════════════════════════════
    #  Main callback
    # ═════════════════════════════════════════════════════════════════════════

    def image_callback(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        pil_rgb = PILImage.fromarray(cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB))
        h, w    = cv_bgr.shape[:2]

        # Depth map generation (with Inference Mode context for high optimization)
        with torch.inference_mode():
            raw_depth = np.array(self._depth_pipe(pil_rgb)['depth'], dtype=np.float32)
            
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min < 1e-6:
            return
        norm_depth = (raw_depth - d_min) / (d_max - d_min)

        # Estimate ground plane properties
        table_norm_d, table_std = self._estimate_table_depth(norm_depth, h, w)

        # Primary pass (YOLO)
        if self._yolo is not None:
            with torch.inference_mode():
                detections = self._detect_yolo(cv_bgr, norm_depth, table_norm_d, table_std)
        else:
            detections = self._detect_depth_fallback(cv_bgr, norm_depth, h, w, table_norm_d, table_std)

        # Secondary Hybrid contour pass (Crucial for missed items)
        if self._yolo is not None and self.hybrid_pass:
            detections = self._hybrid_depth_sweep(
                cv_bgr, norm_depth, h, w,
                table_norm_d, table_std, detections)

        # Final cleanup processing
        detections = self._nms_centroid(detections, min_dist_px=20)
        detections.sort(key=lambda d: -d['depth_val'])

        # Print out information terminal log summary
        if len(detections) > 0:
            self.get_logger().info(f"--- Frame Detections (Max Allowed Area: {int(h * w * self.max_area_frac)} px) ---")
            for idx, det in enumerate(detections):
                self.get_logger().info(f" -> Obj #{idx} [{det['label']}] - Diện tích (Area): {int(det['area_px'])} px")

        self._publish_detections(detections, msg.header)

        if self.publish_debug and hasattr(self, 'pub_debug'):
            debug = self._draw_debug(cv_bgr, norm_depth, detections, table_norm_d)
            dbg_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            dbg_msg.header = msg.header
            self.pub_debug.publish(dbg_msg)

    # ═════════════════════════════════════════════════════════════════════════
    #  Detection Methods
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_yolo(self, cv_bgr: np.ndarray, norm_depth: np.ndarray, table_norm_d: float, table_std: float) -> list:
        results = self._yolo(cv_bgr, conf=self.yolo_conf, verbose=False)[0]
        h, w    = cv_bgr.shape[:2]
        total_pixels = h * w
        detections = []

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            label   = self._yolo.names[cls_id].lower().strip()
            conf    = float(box.conf[0])

            if label in _SURFACE_LABELS:
                continue

            if 'hand' in label or 'gripper' in label or 'robot' in label or 'arm' in label:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw, bh  = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                continue

            # ── Black / robot-body colour check ──
            if self._is_object_black(cv_bgr, x1, y1, x2, y2):
                continue

            # ── Spatial extent: reject arm-sized boxes ──
            if bw > (w * 0.30) and bh > (h * 0.30):
                continue

            # Calculate area footprint 
            box_area = float(bw * bh)

            if box_area > (total_pixels * self.max_area_frac):
                continue

            u_c = float(x1 + bw / 2)
            v_c = float(y1 + bh / 2)

            roi       = norm_depth[y1:y2, x1:x2]
            obj_depth = float(np.median(roi))

            world_pos = self._pixel_to_world_depth(u_c, v_c, obj_depth)
            dist_m    = float(np.linalg.norm(world_pos - self.cam_pos_world))
            dist_m    = max(dist_m, 0.05)

            f_avg   = (FX + FY) / 2.0
            long_px = max(bw, bh)
            size_m  = long_px * dist_m / f_avg

            if size_m > self.gripper_max_m:
                continue

            roi_mask = self._roi_to_mask_local(norm_depth, x1, y1, x2, y2, obj_depth)
            obb      = self._best_obb(roi_mask, x1, y1)

            detections.append(self._make_det(
                label, conf, u_c, v_c,
                (x1, y1, bw, bh), obb, world_pos, obj_depth, size_m, box_area))

        return detections

    def _hybrid_depth_sweep(self, cv_bgr: np.ndarray, norm_depth: np.ndarray, h: int, w: int,
                            table_norm_d: float, table_std: float, existing: list) -> list:
        extra = self._detect_depth_fallback(cv_bgr, norm_depth, h, w, table_norm_d, table_std)

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
                iou_x = _iou_1d(cx1, cx2, ex1, ex2)
                iou_y = _iou_1d(cy1, cy2, ey1, ey2)
                if iou_x > 0.20 and iou_y > 0.20: 
                    overlap = True
                    break
            if not overlap:
                result.append(cand)
        return result

    def _detect_depth_fallback(self, cv_bgr: np.ndarray, norm_depth: np.ndarray, h: int, w: int,
                               table_norm_d: float, table_std: float) -> list:
        depth_gray = (norm_depth * 255).astype(np.uint8)
        
        blurred = cv2.GaussianBlur(depth_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 15, 50)

        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg = cv2.dilate(edges, k_dilate, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area_px or area > (h * w * self.max_area_frac):
                continue
                
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            u_c = M['m10'] / M['m00']
            v_c = M['m01'] / M['m00']

            # ── Ignore elements near the top horizon banner ──
            if v_c < (h * 0.15):
                continue

            bx, by, bw2, bh2 = cv2.boundingRect(cnt)
            
            # ── MODIFIED: Allow objects to touch the bottom boundary (y + h) ──
            # This ensures items close to the camera/gripper aren't discarded if clipped.
            if bx <= 0 or by <= 0 or (bx + bw2) >= w:
                continue

            # ── Black Object Check ──
            if self._is_object_black(cv_bgr, bx, by, bx + bw2, by + bh2):
                continue

            obj_depth = float(np.median(norm_depth[by:by + bh2, bx:bx + bw2]))

            if abs(obj_depth - table_norm_d) < (0.3 * table_std):
                continue

            # Get Oriented Bounding Box geometry
            obb = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = obb
            
            # ── FIXED: Increased max aspect ratio ceiling to allow long slender peg cylinders ──
            side_long = max(rw, rh)
            side_short = min(rw, rh)
            if side_short > 0:
                aspect_ratio = side_long / side_short
                if aspect_ratio > 6.0:  # Loosened from 3.5
                    continue

            # ── Robot-arm rejection: solidity check ──────────────────────────
            # A graspable tabletop object is convex or nearly so (box, cylinder,
            # ellipsoid). The robot arm is a highly irregular, branching shape
            # whose filled contour area is much smaller than its convex hull.
            # Solidity = contour_area / convex_hull_area; arms score < 0.45.
            hull   = cv2.convexHull(cnt)
            h_area = float(cv2.contourArea(hull))
            if h_area > 0 and (area / h_area) < 0.45:
                continue

            # ── Robot-arm rejection: spatial extent check ────────────────────
            # The arm spans a large portion of the frame in BOTH dimensions.
            # Any single graspable object should fit in at most ~30% of frame
            # width and ~40% of frame height simultaneously.
            if bw2 > (w * 0.30) and bh2 > (h * 0.30):
                continue

            world_pos = self._pixel_to_world_depth(u_c, v_c, obj_depth)
            dist_m    = max(float(np.linalg.norm(world_pos - self.cam_pos_world)), 0.05)
            f_avg     = (FX + FY) / 2.0
            
            size_m = side_long * dist_m / f_avg

            if size_m > self.gripper_max_m:
                continue

            detections.append(self._make_det(
                'object', 0.85, u_c, v_c,
                (bx, by, bw2, bh2), obb, world_pos, obj_depth, size_m, area))

        return detections

    # ═════════════════════════════════════════════════════════════════════════
    #  Geometry Processing Helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _estimate_table_depth(self, norm_depth: np.ndarray, h: int, w: int) -> tuple:
        ref = norm_depth[int(h * 0.35):int(h * 0.75), int(w * 0.15):int(w * 0.85)]
        med = float(np.median(ref))
        std = max(float(ref.std()), 0.015)
        return med, std

    def _pixel_to_world_depth(self, u: float, v: float, norm_d: float) -> np.ndarray:
        ray_cam = np.array([
            (u - CX) / FX,
            (v - CY) / FY,
            1.0,
        ], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)

        ray_world = self.R_cam_to_world @ ray_cam

        dz = ray_world[2]
        if abs(dz) < 1e-6:
            t = 1.0
        else:
            t = (self.table_z_world - self.cam_pos_world[2]) / dz
        t = float(np.clip(t, 0.05, 5.0))

        base_pt = self.cam_pos_world + t * ray_world

        lift        = (1.0 - float(norm_d)) * _MAX_LIFT_M
        world_pt    = base_pt.copy()
        world_pt[2] = self.table_z_world + lift
        return world_pt.astype(np.float32)

    def _roi_to_mask_local(self, norm_depth: np.ndarray, x1: int, y1: int, x2: int, y2: int, obj_depth: float) -> np.ndarray:
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

    @staticmethod
    def _make_det(label, conf, u_c, v_c, bbox, obb, world_pos, depth_val, size_m, area_px) -> dict:
        return {
            'label':     label,
            'conf':      conf,
            'centroid':  (int(u_c), int(v_c)),
            'bbox':      bbox,
            'obb':       obb,
            'world_pos': world_pos,
            'depth_val': depth_val,
            'size_m':    size_m,
            'area_px':   area_px,
        }

    @staticmethod
    def _nms_centroid(detections: list, min_dist_px: int = 20) -> list:
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

    # ═════════════════════════════════════════════════════════════════════════
    #  Publishing & Debug Presentation
    # ═════════════════════════════════════════════════════════════════════════

    def _publish_detections(self, detections: list, header):
        pa             = PoseArray()
        pa.header      = header
        pa.header.frame_id = 'world'
        labels = []
        for det in detections:
            p = Pose()
            p.position.x = float(det['world_pos'][0])
            p.position.y = float(det['world_pos'][1])
            p.position.z = float(det['world_pos'][2])
            p.orientation.w = 1.0
            pa.poses.append(p)
            labels.append(f"{det['label']}:{det['size_m']*100:.1f}cm@{det['conf']:.2f}")
        self.pub_poses.publish(pa)
        self.pub_classes.publish(String(data=','.join(labels)))

    def _draw_debug(self, bgr: np.ndarray, norm_depth: np.ndarray, detections: list, table_norm_d: float) -> np.ndarray:
        depth_color = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(bgr, 0.45, depth_color, 0.55, 0)

        cv2.putText(out, f'table_d={table_norm_d:.2f}', (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        for idx, det in enumerate(detections):
            u, v           = det['centroid']
            bx, by, bw, bh = det['bbox']

            box_pts = cv2.boxPoints(det['obb']).astype(np.int32)
            cv2.drawContours(out, [box_pts], -1, (0, 255, 0), 2)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 180, 255), 1)
            cv2.circle(out, (u, v), 5, (0, 0, 255), -1)

            label = f"#{idx} {det['label']} {det['size_m']*100:.1f}cm [{int(det['area_px'])}px] c={det['conf']:.2f}"
            cv2.putText(out, label, (bx, max(by - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(out, f'{len(detections)} graspable | YOLO={_HAVE_YOLO}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2, cv2.LINE_AA)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    p = argparse.ArgumentParser(description='YOLOv26 + DepthAnythingV2 ROS2 Vision Node')
    p.add_argument('--yolo-model',   default='yolov26n.pt',
                   help='YOLO model name or checkpoint path (default: yolov26n.pt)')
    p.add_argument('--yolo-conf',    type=float, default=0.15,
                   help='YOLO detection threshold confidence level')
    p.add_argument('--depth-device', type=int,   default=-1,
                   help='Execution device for depth transformers pipeline (-1=CPU, 0=GPU)')
    p.add_argument('--no-debug',     action='store_true',
                   help='Disable frame output debug publishing')
    p.add_argument('--no-hybrid',    action='store_true',
                   help='Disable dynamic contour execution passes')

    parsed, ros_args = p.parse_known_args(args)

    rclpy.init(args=ros_args)

    node = DepthAnythingDetectorNode(
        yolo_model_init    = parsed.yolo_model,
        yolo_conf_init     = parsed.yolo_conf,
        depth_device_init  = parsed.depth_device,
        publish_debug_init = not parsed.no_debug,
        use_hybrid_init    = not parsed.no_hybrid,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])