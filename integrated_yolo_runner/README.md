# Integrated YOLO Runner (Windows)

This folder provides a single, unified CLI to run all YOLO models that exist across your workspace:

- `Accident_detection_model/Accident-Detection-Model/yolov8s.pt`
- `Fight_model/Fight-Violence-detection-yolov8/Yolo_nano_weights.pt`
- `Fight_model/Fight-Violence-detection-yolov8/yolo_small_weights.pt`
- `Weapon-Detection-YOLO/best (3).pt`
- `fire_detection_model/yolov8n.pt`
- `fire_detection_model/yolov8s (1).pt`
- `NSFW_Detectio/nsfw_detector_annotator/models/classification_model.pt`
- `NSFW_Detectio/nsfw_detector_annotator/models/segmentation_model.pt`

The runner is `run.py`. Outputs are saved under `d:/akhil/integrated_runs/<task>/predict*/`.

## 1) Create a virtual environment (recommended)

Open PowerShell in this folder `d:/akhil/integrated_yolo_runner/` and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked, allow it for the current user (once):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

Install Ultralytics, OpenCV, etc. from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Install PyTorch for your platform (choose one):

- CPU only:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
- CUDA 12.1 (adjust if needed):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```powershell
python -c "import torch, ultralytics, cv2; print('torch', torch.__version__, 'ultralytics', ultralytics.__version__, 'cv2', cv2.__version__)"
```

## 3) Run detections

General pattern:

```powershell
python run.py --task <task_name> --source <source> [--conf 0.25] [--imgsz 640] [--device cpu|0] [--show]
```

Supported `--task` values mapped to local weights:

- `accident` -> `d:/akhil/Accident_detection_model/Accident-Detection-Model/yolov8s.pt`
- `fight_nano` -> `d:/akhil/Fight_model/Fight-Violence-detection-yolov8/Yolo_nano_weights.pt`
- `fight_small` -> `d:/akhil/Fight_model/Fight-Violence-detection-yolov8/yolo_small_weights.pt`
- `weapon` -> `d:/akhil/Weapon-Detection-YOLO/best (3).pt`
- `fire_n` -> `d:/akhil/fire_detection_model/yolov8n.pt`
- `fire_s` -> `d:/akhil/fire_detection_model/yolov8s (1).pt`
- `nsfw_cls` -> `d:/akhil/NSFW_Detectio/nsfw_detector_annotator/models/classification_model.pt`
- `nsfw_seg` -> `d:/akhil/NSFW_Detectio/nsfw_detector_annotator/models/segmentation_model.pt`

You can also use your own weights via `--task custom --weights path/to/model.pt`.

### Examples

- Webcam (index 0) with accident model:
```powershell
python run.py --task accident --source 0 --show
```

- Video file with fight (nano):
```powershell
python run.py --task fight_nano --source d:/path/to/video.mp4 --conf 0.35
```

- Image with weapon model (save enabled by default):
```powershell
python run.py --task weapon --source d:/path/to/image.jpg
```

- Folder of images with fire (small) on CPU:
```powershell
python run.py --task fire_s --source d:/path/to/folder/ --device cpu
```

- Custom weights:
```powershell
python run.py --task custom --weights d:/akhil/your_model.pt --source 0
```

## 4) Outputs

Annotated media and labels (if enabled) are saved to:

```
d:/akhil/integrated_runs/<task>/predict*/
```

## 5) Troubleshooting

- Ensure weights exist at the paths above. If not, supply `--weights` or move weights accordingly.
- If `cv2.imshow` windows do not appear, try removing `--show` or run without remote/SSH sessions.
- For GPU usage, confirm `torch.cuda.is_available()`:
```powershell
python - << 'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
```
- If OpenCV fails to open webcam, try `--source 1` or update camera drivers.
- If you see dependency conflicts, install the pinned versions in `requirements.txt` and ensure only one environment is active.

## 6) Notes

- This runner uses Ultralytics' `model.predict(...)`. It supports images, videos, folders, and streams.
- Output root can be changed with `--project d:/some/other/output` and name via `--name myrun`.
- NSFW tasks here simply run the provided YOLO models. The full Streamlit workflow in `NSFW_Detectio/nsfw_detector_annotator/` is still available separately if you want the web UI.
