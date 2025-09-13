#!/usr/bin/env python
"""
Unified YOLO Runner for multiple tasks in this workspace.

Supported tasks (pre-wired to local weights in this workspace):
- accident
- fight_nano
- fight_small
- weapon
- fire_n
- fire_s
- nsfw_cls
- nsfw_seg
- objects  (uses Ultralytics built-in 'yolov8n.pt')

Also supports high-level categories:
- --category accident | fight | nsfw | fire | objects
  (maps to a default task internally; you can still use --task directly)

Usage examples:
  python run.py --task accident --source 0                 # webcam
  python run.py --category fight --source path/to/video.mp4
  python run.py --task weapon --source path/to/image.jpg --conf 0.35
  python run.py --task fire_s --source path/to/folder/ --device cpu
  python run.py --category nsfw --nsfw_mode seg --source path/to/image.jpg

You can also override weights path explicitly:
  python run.py --task custom --weights d:/akhil/your_model.pt --source 0

Outputs are saved under d:/akhil/integrated_runs/<task>/predict*/
Additionally, a JSON summary of detections/classifications is saved as results.json in the same folder.
"""
import argparse
import os
import shutil
import sys
import json
import os
import math
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    print("Error: ultralytics is not installed or failed to import.")
    print("Install with: pip install ultralytics==8.3.58")
    raise


WORKSPACE = Path("d:/akhil").resolve()
DEFAULT_OUTPUT_ROOT = WORKSPACE / "integrated_runs"

# Registry of known local weights
MODEL_REGISTRY = {
    # Accident detection
    "accident": WORKSPACE / "models/accident/weights/yolov8s.pt",
    
    # Fight detection
    "fight_nano": WORKSPACE / "models/fight/weights/nano_weights.pt",
    "fight_small": WORKSPACE / "models/fight/weights/small_weights.pt",
    
    # Weapon detection
    "weapon": WORKSPACE / "models/weapon/weights/weapon_detection.pt",
    
    # Fire detection
    "fire_n": WORKSPACE / "models/fire/weights/yolov8n.pt",
    "fire_s": WORKSPACE / "models/fire/weights/yolov8s.pt",
    
    # NSFW detection
    "nsfw_cls": WORKSPACE / "models/nsfw/weights/classification_model.pt",
    "nsfw_seg": WORKSPACE / "models/nsfw/weights/segmentation_model.pt",
    
    # Generic object detection (auto-download)
    "objects": Path("yolov8n.pt"),
}

# Optional label alias map to normalize model-specific class names to canonical names
LABEL_ALIASES = {
    # Weapon model sometimes emits a dataset string; map it to 'weapon'
    "weapon dataset - v4 2024-10-02 7-18am": "weapon",
    # Add any other known odd labels here as needed
}

# Category to default task mapping
CATEGORY_DEFAULT_TASK = {
    "accident": "accident",
    "fight": "fight_small",
    "nsfw": "nsfw_seg",  # default to segmentation; can override with --nsfw_mode
    "fire": "fire_s",
    "objects": "objects",
    "weapon": "weapon",
}


def validate_weights_path(path: Path) -> Path:
    """Validate weights path or allow known Ultralytics model names which auto-download."""
    if path.exists():
        if path.suffix.lower() != ".pt":
            raise ValueError(f"Weights file must be a .pt file, got: {path}")
        return path
    # Allow common Ultralytics model names which trigger auto-download
    allowed_names = {
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
    }
    if path.name in allowed_names:
        return path
    raise FileNotFoundError(f"Weights not found: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified YOLO Runner")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help=(
            "Task key in registry (accident, fight_nano, fight_small, weapon, fire_n, fire_s, nsfw_cls, nsfw_seg, objects) "
            "or 'custom' when providing --weights"
        ),
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["accident", "fight", "nsfw", "fire", "objects", "weapon"],
        help="High-level category alias; maps to a default task if --task is not provided",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to custom .pt weights (required if task=custom)")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source for prediction: 0 for webcam, a path to image/video/folder, or RTSP/HTTP stream URL",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default for all unless overridden)")
    parser.add_argument("--fire_conf", type=float, default=0.5, help="Confidence threshold override for fire detections (fire_n/fire_s)")
    parser.add_argument("--fire_hard", type=float, default=0.95, help="Hard confidence required to accept fire after filtering (false-positive mitigation)")
    parser.add_argument("--fire_min_area", type=float, default=0.01, help="Minimum bbox area ratio (bbox_area / image_area) to accept fire")
    parser.add_argument("--objects_conf", type=float, default=0.9, help="Confidence threshold override for objects (COCO)")
    # Fight category controls
    parser.add_argument("--fight_conf", type=float, default=0.5, help="Confidence threshold override for fight detections (fight_nano/fight_small)")
    parser.add_argument("--fight_hard", type=float, default=0.75, help="Hard confidence required to accept 'violence' detections in fight category")
    parser.add_argument("--fight_min_area", type=float, default=0.003, help="Minimum bbox area ratio (bbox_area / image_area) to accept 'violence'")
    parser.add_argument("--fight_min_ratio", type=float, default=0.3, help="For videos, min ratio of frames that must show 'violence' to accept fight category")
    # Weapon category controls
    parser.add_argument("--weapon_conf", type=float, default=0.5, help="Confidence threshold override for weapon detections")
    parser.add_argument("--weapon_hard", type=float, default=0.8, help="Hard confidence required to accept weapon detections")
    parser.add_argument("--weapon_min_area", type=float, default=0.002, help="Minimum bbox area ratio to accept weapon")
    parser.add_argument("--weapon_min_ratio", type=float, default=0.3, help="For videos, min ratio of frames that must show weapon to accept category")
    parser.add_argument("--objects_list_conf", type=float, default=0.6, help="Minimum confidence to list objects in top3_objects")
    parser.add_argument("--accident_hard", type=float, default=0.9, help="Hard confidence required to accept accident after filtering")
    parser.add_argument("--accident_require_vehicle", action="store_true", help="Require a vehicle to be present to accept accident")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--sample_secs", type=float, default=3.0, help="For videos, sample approximately one frame every N seconds")
    parser.add_argument("--progress", action="store_true", help="Show a simple progress indicator during processing")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', '0', '0,1', etc.")
    parser.add_argument("--save", action="store_true", help="Force save visual outputs (default True)")
    parser.add_argument("--nosave", action="store_true", help="Do not save visual outputs")
    parser.add_argument("--clean_outputs", action="store_true", help="Delete previous outputs for this run before writing new ones")
    parser.add_argument("--show", action="store_true", help="Show windows during inference (may be slow)")
    parser.add_argument("--project", type=str, default=None, help="Override output root directory")
    parser.add_argument("--name", type=str, default="predict", help="Run name (subfolder)")
    parser.add_argument("--nsfw_mode", type=str, choices=["cls", "seg"], default="seg", help="For --category nsfw, choose classification or segmentation")
    # Multi-run controls
    parser.add_argument("--run_all", action="store_true", help="Run all categories sequentially (weapon,fight,nsfw,fire,accident,objects)")
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to run sequentially (e.g., 'weapon,fight,nsfw,fire,accident,objects')",
    )
    # Filtered output controls
    parser.add_argument("--filter_output", action="store_true", help="Also write filtered JSON with only high-confidence results")
    parser.add_argument("--prob_thresh", type=float, default=0.5, help="Min probability for classification entries in filtered JSON")
    parser.add_argument("--preset", type=str, choices=["image_relaxed", "video_strict"], default=None, help="Apply tuned thresholds for images or videos")
    # Keyword customization (comma-separated, case-insensitive)
    parser.add_argument("--fire_keywords", type=str, default="fire,smoke,flame,blaze,burn", help="Comma-separated keywords for fire category")
    parser.add_argument("--accident_keywords", type=str, default="accident,crash,collision,wreck,smash", help="Comma-separated keywords for accident category")
    parser.add_argument("--weapon_keywords", type=str, default="weapon,gun,pistol,revolver,rifle,knife,blade,shotgun,firearm", help="Comma-separated keywords for weapon category")
    parser.add_argument("--fight_positive_keywords", type=str, default="violence", help="Comma-separated positive keywords for fight category")
    parser.add_argument("--fight_negative_keywords", type=str, default="non_violence", help="Comma-separated negative keywords to ignore for fight category")
    # Label alias mapping like "raw1->canonical1;raw2->canonical2"
    parser.add_argument("--label_aliases", type=str, default="", help="Semicolon-separated alias mappings raw->canonical, applied case-insensitively")
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply presets to override relevant thresholds
    if args.preset == "image_relaxed":
        # Moderately relaxed thresholds for single images
        args.objects_conf = max(0.6, args.objects_conf)
        args.weapon_conf = max(0.6, getattr(args, 'weapon_conf', 0.6))
        args.weapon_hard = min(0.85, getattr(args, 'weapon_hard', 0.85))
        args.weapon_min_area = min(0.005, getattr(args, 'weapon_min_area', 0.005))
        args.fight_conf = max(0.6, getattr(args, 'fight_conf', 0.6))
        args.fight_hard = min(0.8, getattr(args, 'fight_hard', 0.8))
        args.fight_min_area = min(0.005, getattr(args, 'fight_min_area', 0.005))
        args.fire_conf = max(0.7, getattr(args, 'fire_conf', 0.7))
        args.fire_hard = min(0.95, getattr(args, 'fire_hard', 0.95))
        args.fire_min_area = min(0.01, getattr(args, 'fire_min_area', 0.01))
        # Do not require vehicle for accident by default in image preset
        args.accident_require_vehicle = False
    elif args.preset == "video_strict":
        # Stricter thresholds and require temporal consistency
        args.objects_conf = max(0.9, args.objects_conf)
        args.weapon_conf = max(0.7, getattr(args, 'weapon_conf', 0.7))
        args.weapon_hard = max(0.92, getattr(args, 'weapon_hard', 0.92))
        args.weapon_min_area = max(0.01, getattr(args, 'weapon_min_area', 0.01))
        args.weapon_min_ratio = max(0.5, getattr(args, 'weapon_min_ratio', 0.5))
        args.fight_conf = max(0.6, getattr(args, 'fight_conf', 0.6))
        args.fight_hard = max(0.9, getattr(args, 'fight_hard', 0.9))
        args.fight_min_area = max(0.01, getattr(args, 'fight_min_area', 0.01))
        args.fight_min_ratio = max(0.5, getattr(args, 'fight_min_ratio', 0.5))
        args.fire_conf = max(0.7, getattr(args, 'fire_conf', 0.7))
        args.fire_hard = max(0.98, getattr(args, 'fire_hard', 0.98))
        args.fire_min_area = max(0.02, getattr(args, 'fire_min_area', 0.02))
        # Accident: require vehicle evidence in video mode
        args.accident_require_vehicle = True

    # Build keyword sets (lowercased) from CLI
    def _csv_to_set(s: str):
        return {x.strip().lower() for x in s.split(',') if x.strip()} if s else set()

    FIRE_KWS = _csv_to_set(args.fire_keywords) or {"fire", "smoke", "flame", "blaze", "burn"}
    ACCIDENT_KWS = _csv_to_set(args.accident_keywords) or {"accident", "crash", "collision", "wreck", "smash"}
    WEAPON_KWS = _csv_to_set(args.weapon_keywords) or {"weapon", "gun", "pistol", "revolver", "rifle", "knife", "blade", "shotgun", "firearm"}
    FIGHT_POS_KWS = _csv_to_set(args.fight_positive_keywords) or {"violence"}
    FIGHT_NEG_KWS = _csv_to_set(args.fight_negative_keywords) or {"non_violence"}

    # Merge user-provided label aliases
    if args.label_aliases:
        try:
            for pair in args.label_aliases.split(';'):
                if '->' in pair:
                    raw, canon = pair.split('->', 1)
                    raw = raw.strip().lower()
                    canon = canon.strip().lower()
                    if raw and canon:
                        LABEL_ALIASES[raw] = canon
        except Exception:
            pass

    def resolve_task_from_category(cat: str) -> str:
        if cat == "nsfw":
            return "nsfw_cls" if args.nsfw_mode == "cls" else "nsfw_seg"
        return CATEGORY_DEFAULT_TASK[cat]

    def run_single(task_key: str, weights_path: Path):
        # Prepare output dir per task
        project_dir = Path(args.project) if args.project else (DEFAULT_OUTPUT_ROOT / task_key)
        project_dir.mkdir(parents=True, exist_ok=True)

        # If cleaning is requested, remove previous run folder for this task/name
        if args.clean_outputs:
            try:
                out_dir_candidate = project_dir / args.name
                if out_dir_candidate.exists() and out_dir_candidate.is_dir():
                    shutil.rmtree(out_dir_candidate)
                    print(f"Cleaned previous outputs: {out_dir_candidate}")
            except Exception as e:
                print(f"Warning: failed to clean outputs for {task_key}: {e}")

        print(f"Loading model weights: {weights_path}")
        model = YOLO(str(weights_path))

        # Normalize source (allow numeric webcam index)
        source = args.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        save_flag = not args.nosave or args.save
        # Per-category conf override
        conf_thr = args.conf
        if task_key in ("fire_n", "fire_s"):
            conf_thr = args.fire_conf
        elif task_key == "objects":
            conf_thr = args.objects_conf
        elif task_key in ("fight_nano", "fight_small"):
            conf_thr = args.fight_conf
        elif task_key == "weapon":
            conf_thr = args.weapon_conf
        # Determine if source is a video file to compute stride
        vid_stride = None
        total_frames = None
        expected_processed = None
        is_video_file = False
        if isinstance(source, str):
            ext = os.path.splitext(source)[1].lower()
            if ext in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"} and os.path.exists(source):
                is_video_file = True
        if is_video_file and args.sample_secs and args.sample_secs > 0:
            try:
                import cv2
                cap = cv2.VideoCapture(source)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if fps > 0:
                    vid_stride = max(1, int(round(fps * args.sample_secs)))
                    if total_frames:
                        expected_processed = max(1, math.ceil(total_frames / vid_stride))
            except Exception:
                pass

        print(
            f"Running predict: task={task_key}, source={args.source}, conf={conf_thr}, imgsz={args.imgsz}, device={args.device}, save={save_flag}"
            + (f", vid_stride={vid_stride}" if vid_stride else "")
        )

        # Use streaming to enable progress display if desired
        if args.progress and (is_video_file or (isinstance(source, int))):
            results_list = []
            count = 0
            start_t = time.time()
            for r in model.predict(
                source=source,
                conf=conf_thr,
                imgsz=args.imgsz,
                device=args.device,
                save=save_flag,
                show=args.show,
                project=str(project_dir),
                name=args.name,
                verbose=True,
                stream=True,
                vid_stride=vid_stride,
            ):
                results_list.append(r)
                count += 1
                if expected_processed:
                    pct = int((count / expected_processed) * 100)
                    pct = max(0, min(100, pct))
                    print(f"Progress: {count}/{expected_processed} (~{pct}%)", end="\r", flush=True)
                else:
                    print(f"Processed frames: {count}", end="\r", flush=True)
            print("")
            results = results_list
        else:
            results = model.predict(
                source=source,
                conf=conf_thr,
                imgsz=args.imgsz,
                device=args.device,
                save=save_flag,
                show=args.show,
                project=str(project_dir),
                name=args.name,
                verbose=True,
                vid_stride=vid_stride,
            )

        # Prepare JSON results
        out_dir = project_dir / args.name
        summary = {
            "task": task_key,
            "weights": str(weights_path),
            "source": args.source,
            "conf": conf_thr,
            "imgsz": args.imgsz,
            "device": args.device,
        }

        items = []
        by_class = {}
        total_dets = 0

        for idx, r in enumerate(results):
            item = {"image": getattr(r, "path", f"frame_{idx}"), "type": None}
            try:
                if hasattr(r, "orig_img") and r.orig_img is not None:
                    ih, iw = r.orig_img.shape[:2]
                    item["image_wh"] = (iw, ih)
            except Exception:
                pass
            names = r.names if hasattr(r, "names") else None

            if getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes) > 0:
                # Detection
                item["type"] = "detection"
                dets = []
                for b in r.boxes:
                    cls_id = int(b.cls.item()) if hasattr(b, "cls") and b.cls is not None else -1
                    conf = float(b.conf.item()) if hasattr(b, "conf") and b.conf is not None else None
                    xyxy = b.xyxy.squeeze().tolist() if hasattr(b, "xyxy") else None
                    cls_name = (names.get(cls_id) if isinstance(names, dict) else names[cls_id]) if names and cls_id >= 0 else None
                    dets.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox_xyxy": xyxy,
                    })
                    if cls_name:
                        by_class[cls_name] = by_class.get(cls_name, 0) + 1
                    total_dets += 1
                item["detections"] = dets

            if getattr(r, "probs", None) is not None and r.probs is not None:
                # Classification
                item["type"] = "classification" if item["type"] is None else item["type"] + "+classification"
                classes = []
                try:
                    probs = r.probs.data.tolist()
                    if names:
                        for i, p in enumerate(probs):
                            classes.append({"class_id": i, "class_name": names[i] if not isinstance(names, dict) else names.get(i), "prob": float(p)})
                    else:
                        for i, p in enumerate(probs):
                            classes.append({"class_id": i, "prob": float(p)})
                except Exception:
                    pass
                item["classifications"] = classes

            items.append(item)

        result_json = {
            "summary": {**summary, "frames": len(items), "detections": total_dets, "by_class": by_class},
            "items": items,
        }

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2)
            print(f"\nSaved JSON results to: {out_dir / 'results.json'}")
        except Exception as e:
            print(f"Failed to write results.json: {e}")

        print(f"\nDone. Outputs (images/video and labels if enabled) should be in: {out_dir}")
        
        # Optionally write a filtered/minimal JSON that only includes high-confidence info
        filtered = {"category": task_key, "detected": False, "objects": [], "classes": [], "category_confidence": 0.0}
        # Detection objects are already thresholded by --conf in Ultralytics, include them directly
        for it in items:
            for det in it.get("detections", []) or []:
                # For objects task, enforce objects_conf explicitly (guard in case Ultralytics returned lower)
                if task_key == "objects" and det.get("confidence") is not None and float(det.get("confidence")) < args.objects_conf:
                    continue
                filtered["objects"].append({
                    "class_name": det.get("class_name"),
                    "confidence": det.get("confidence"),
                    "bbox_xyxy": det.get("bbox_xyxy"),
                    "image": it.get("image"),
                })
                try:
                    if det.get("confidence") is not None:
                        filtered["category_confidence"] = max(filtered["category_confidence"], float(det.get("confidence")))
                except Exception:
                    pass
        # Classification: include only classes with prob >= prob_thresh
        for it in items:
            for cl in it.get("classifications", []) or []:
                if cl.get("prob") is not None and cl.get("prob") >= args.prob_thresh:
                    filtered["classes"].append({
                        "class_name": cl.get("class_name"),
                        "prob": cl.get("prob"),
                        "image": it.get("image"),
                    })
                    try:
                        filtered["category_confidence"] = max(filtered["category_confidence"], float(cl.get("prob")))
                    except Exception:
                        pass
        filtered["detected"] = len(filtered["objects"]) > 0 or len(filtered["classes"]) > 0

        # Fight false positive mitigation: only consider 'violence' boxes (ignore 'non_violence'),
        # require hard confidence + area, and for video require presence across sufficient frames.
        if task_key in ("fight_nano", "fight_small") and filtered["detected"]:
            violence_frames = 0
            total_frames = max(1, len(items))
            for it in items:
                iw, ih = it.get("image_wh", (None, None))
                found_in_frame = False
                for det in it.get("detections", []) or []:
                    name = (det.get("class_name") or "").lower()
                    name = LABEL_ALIASES.get(name, name)
                    # Ignore explicit negatives
                    if any(neg in name for neg in FIGHT_NEG_KWS):
                        continue
                    if not any(pos in name for pos in FIGHT_POS_KWS):
                        continue
                    conf = float(det.get("confidence") or 0.0)
                    area_ok = True
                    if iw and ih and det.get("bbox_xyxy"):
                        x1, y1, x2, y2 = det["bbox_xyxy"]
                        box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                        img_area = float(iw) * float(ih)
                        area_ratio = (box_area / img_area) if img_area > 0 else 0.0
                        area_ok = area_ratio >= args.fight_min_area
                    if conf >= args.fight_hard and area_ok:
                        filtered["category_confidence"] = max(filtered["category_confidence"], conf)
                        found_in_frame = True
                        break
                if found_in_frame:
                    violence_frames += 1
            ratio = violence_frames / total_frames
            # Accept only if at least one frame satisfied criteria AND ratio passes for videos
            # For single images (total_frames == 1), require that one frame passes which is covered by the above.
            if violence_frames == 0 or (total_frames > 1 and ratio < args.fight_min_ratio):
                filtered["detected"] = False
                filtered["objects"] = []
                filtered["classes"] = []
                filtered["category_confidence"] = 0.0

        # Weapon false positive mitigation: require keyword AND hard conf AND area; for videos require ratio
        if task_key == "weapon" and filtered["detected"]:
            weapon_keywords = WEAPON_KWS
            weapon_frames = 0
            total_frames = max(1, len(items))
            for it in items:
                iw, ih = it.get("image_wh", (None, None))
                found_in_frame = False
                for det in it.get("detections", []) or []:
                    name = (det.get("class_name") or "").lower()
                    name = LABEL_ALIASES.get(name, name)
                    if not any(k in name for k in weapon_keywords):
                        continue
                    conf = float(det.get("confidence") or 0.0)
                    area_ok = True
                    if iw and ih and det.get("bbox_xyxy"):
                        x1, y1, x2, y2 = det["bbox_xyxy"]
                        box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                        img_area = float(iw) * float(ih)
                        area_ratio = (box_area / img_area) if img_area > 0 else 0.0
                        area_ok = area_ratio >= args.weapon_min_area
                    if conf >= args.weapon_hard and area_ok:
                        filtered["category_confidence"] = max(filtered["category_confidence"], conf)
                        found_in_frame = True
                        break
                if found_in_frame:
                    weapon_frames += 1
            ratio = weapon_frames / total_frames
            if weapon_frames == 0 or (total_frames > 1 and ratio < args.weapon_min_ratio):
                filtered["detected"] = False
                filtered["objects"] = []
                filtered["classes"] = []
                filtered["category_confidence"] = 0.0

        # Fire false positive mitigation: require keyword AND high conf AND sufficient area
        if task_key in ("fire_n", "fire_s") and filtered["detected"]:
            fire_keywords = FIRE_KWS
            ok = False
            for it in items:
                iw, ih = it.get("image_wh", (None, None))
                for det in it.get("detections", []) or []:
                    name = (det.get("class_name") or "").lower()
                    conf = float(det.get("confidence") or 0.0)
                    if not any(k in name for k in fire_keywords):
                        continue
                    area_ok = True
                    if iw and ih and det.get("bbox_xyxy"):
                        x1, y1, x2, y2 = det["bbox_xyxy"]
                        box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                        img_area = float(iw) * float(ih)
                        area_ratio = (box_area / img_area) if img_area > 0 else 0.0
                        area_ok = area_ratio >= args.fire_min_area
                    if conf >= args.fire_hard and area_ok:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                filtered["detected"] = False
                filtered["objects"] = []
                filtered["classes"] = []
                filtered["category_confidence"] = 0.0

        # Accident false positive mitigation: require accident keyword and, if requested, vehicle evidence with high conf
        if task_key == "accident" and filtered["detected"]:
            accident_keywords = ACCIDENT_KWS
            vehicle_keywords = ("car", "truck", "bus", "bike", "bicycle", "motorbike", "motorcycle", "vehicle", "auto", "scooter", "van")
            has_accident_label = False
            has_vehicle = False
            high_vehicle = False
            for it in items:
                for det in it.get("detections", []) or []:
                    name = (det.get("class_name") or "").lower()
                    name = LABEL_ALIASES.get(name, name)
                    if any(a in name for a in accident_keywords):
                        has_accident_label = True
                    conf = float(det.get("confidence") or 0.0)
                    if any(v in name for v in vehicle_keywords):
                        has_vehicle = True
                        if conf >= args.accident_hard:
                            high_vehicle = True
                            break
                if high_vehicle:
                    break
            # Always require an accident keyword present to consider the category
            if not has_accident_label:
                filtered["detected"] = False
                filtered["objects"] = []
                filtered["classes"] = []
                filtered["category_confidence"] = 0.0
            elif args.accident_require_vehicle and (not has_vehicle or not high_vehicle):
                filtered["detected"] = False
                filtered["objects"] = []
                filtered["classes"] = []
                filtered["category_confidence"] = 0.0

        # Add a simple safety tag for single run (useful when not in batch)
        unsafe_categories = {"weapon", "fight_nano", "fight_small", "nsfw_cls", "nsfw_seg", "fire_n", "fire_s", "accident"}
        # Objects should not influence unsafe tag; keep 'safe' for objects category
        if task_key == "objects":
            filtered["safety_tag"] = "safe"
        else:
            filtered["safety_tag"] = "unsafe" if (filtered["detected"] and task_key in unsafe_categories) else "safe"

        if args.filter_output:
            try:
                with open(out_dir / "filtered_results.json", "w", encoding="utf-8") as f:
                    json.dump(filtered, f, indent=2)
                print(f"Saved filtered JSON to: {out_dir / 'filtered_results.json'}")
            except Exception as e:
                print(f"Failed to write filtered_results.json: {e}")

        return result_json, filtered

    # Multi-run sequence if requested
    if args.run_all or (args.categories and len(args.categories.strip()) > 0):
        if args.categories:
            cats = [c.strip() for c in args.categories.split(',') if c.strip()]
        else:
            cats = ["weapon", "fight", "nsfw", "fire", "accident", "objects"]

        # Resolve each category to a task
        task_list = []
        for cat in cats:
            if cat not in CATEGORY_DEFAULT_TASK and cat != "nsfw":
                print(f"Warning: unknown category '{cat}', skipping")
                continue
            task_key = resolve_task_from_category(cat) if cat in ("nsfw", "accident", "fight", "fire", "objects", "weapon") else None
            if not task_key:
                continue
            if task_key not in MODEL_REGISTRY:
                print(f"Warning: task '{task_key}' not in registry, skipping")
                continue
            weights_path = validate_weights_path(MODEL_REGISTRY[task_key])
            task_list.append((task_key, weights_path))

        combined = {"sequence": cats, "runs": []}
        combined_filtered = {"sequence": cats, "detected_categories": [], "objects_detected": []}
        batch_root = (DEFAULT_OUTPUT_ROOT / "batch" / args.name)

        # Clean previous batch outputs if requested
        if args.clean_outputs and batch_root.exists():
            try:
                shutil.rmtree(batch_root)
                print(f"Cleaned previous batch outputs: {batch_root}")
            except Exception as e:
                print(f"Warning: failed to clean batch outputs: {e}")
        batch_root.mkdir(parents=True, exist_ok=True)

        for task_key, weights_path in task_list:
            try:
                rj, fj = run_single(task_key, weights_path)
                combined["runs"].append(rj)
                if fj.get("detected"):
                    if task_key == "objects":
                        combined_filtered["objects_detected"].extend(fj.get("objects", []))
                    else:
                        combined_filtered["detected_categories"].append({
                            "category": task_key,
                            "objects": fj.get("objects", []),
                            "classes": fj.get("classes", []),
                            "category_confidence": fj.get("category_confidence", 0.0),
                        })
            except Exception as e:
                err = {"summary": {"task": task_key, "error": str(e)}}
                combined["runs"].append(err)

        # Cross-category accident mitigation using objects_detected
        if args.accident_require_vehicle:
            # Find accident entry
            acc_idx = None
            for i, entry in enumerate(combined_filtered["detected_categories"]):
                if entry.get("category") == "accident":
                    acc_idx = i
                    break
            if acc_idx is not None:
                vehicle_keywords = ("car", "truck", "bus", "bike", "bicycle", "motorbike", "motorcycle", "vehicle", "auto", "scooter", "van")
                has_vehicle = False
                # Check within accident objects first
                for o in combined_filtered["detected_categories"][acc_idx].get("objects", []):
                    name = (o.get("class_name") or "").lower()
                    conf = float(o.get("confidence") or 0.0)
                    if any(v in name for v in vehicle_keywords) and conf >= args.accident_hard:
                        has_vehicle = True
                        break
                # If not, check objects_detected (COCO) for vehicles >= objects_conf
                if not has_vehicle:
                    for o in combined_filtered.get("objects_detected", []):
                        name = (o.get("class_name") or "").lower()
                        conf = float(o.get("confidence") or 0.0)
                        if any(v in name for v in vehicle_keywords) and conf >= args.objects_conf:
                            has_vehicle = True
                            break
                if not has_vehicle:
                    # Remove accident as insufficient evidence
                    combined_filtered["detected_categories"].pop(acc_idx)

        # Compute final conclusion (top-2 categories and safety tag)
        scored = sorted(
            combined_filtered.get("detected_categories", []),
            key=lambda x: x.get("category_confidence", 0.0),
            reverse=True,
        )
        top2 = scored[:2]
        unsafe_set = {"weapon", "fight_nano", "fight_small", "nsfw_cls", "nsfw_seg", "fire_n", "fire_s", "accident"}
        safety_tag = "unsafe" if any(x.get("category") in unsafe_set for x in top2) else (
            "unsafe" if any(x.get("category") in unsafe_set for x in scored) else "safe"
        )
        # Compute top-5 objects using ONLY the objects YOLO model outputs
        obj_sorted = sorted(
            combined_filtered.get("objects_detected", []),
            key=lambda o: float(o.get("confidence") or 0.0),
            reverse=True,
        )
        top5_objects = [
            {
                "class_name": o.get("class_name"),
                "confidence": o.get("confidence"),
                "bbox_xyxy": o.get("bbox_xyxy"),
                "image": o.get("image"),
            }
            for o in obj_sorted[:5]
        ]

        combined_filtered["final_conclusion"] = {
            "top2_categories": [
                {"category": x.get("category"), "confidence": x.get("category_confidence", 0.0)} for x in top2
            ],
            "top5_objects": top5_objects,
            "safety_tag": safety_tag,
        }

        # Write combined JSON
        with open(batch_root / "combined_results.json", "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved combined batch results to: {batch_root / 'combined_results.json'}")
        # Write combined filtered JSON (only detected categories)
        if args.filter_output:
            with open(batch_root / "combined_filtered.json", "w", encoding="utf-8") as f:
                json.dump(combined_filtered, f, indent=2)
            print(f"Saved combined filtered results to: {batch_root / 'combined_filtered.json'}")
        return

    # Single-run flow (backwards compatible)
    # Determine task from args
    task_arg = args.task
    if not task_arg and args.category:
        task_arg = resolve_task_from_category(args.category)

    # Resolve weights
    if task_arg == "custom":
        if not args.weights:
            print("Error: --weights is required when --task custom")
            sys.exit(2)
        weights_path = validate_weights_path(Path(args.weights))
        task_key = "custom"
    else:
        if not task_arg:
            print("Error: provide either --task or --category")
            sys.exit(2)
        if task_arg not in MODEL_REGISTRY:
            print(f"Error: Unknown task '{task_arg}'. Valid: {', '.join(MODEL_REGISTRY.keys())} or 'custom'")
            sys.exit(2)
        weights_path = validate_weights_path(MODEL_REGISTRY[task_arg])
        task_key = task_arg

    # Execute single task
    run_single(task_key, weights_path)


if __name__ == "__main__":
    main()
