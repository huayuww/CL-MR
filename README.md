This repository contains the official implementation of "Detector-in-the-Loop Tracking: Active Memory Rectification for Stable Glottic Opening Localization," accepted to MIDL 2026. The paper is available at [Detector-in-the-Loop Tracking (MIDL 2026)](https://openreview.net/forum?id=dwYFYk4aeZ).

Codebase
- This work builds on [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524) and [SAM2](https://openreview.net/forum?id=Ha6RTeWMd0)
- The implementation in this repository is built on the [Ultralytics](https://docs.ultralytics.com/) codebase. We thank the Ultralytics team for their excellent work and for providing a strong foundation for real-time object detection research.

Data
- Part of the data used in this work comes from the Laryngoscope8 Datasets: [Laryngoscope8 Datasets](https://www.sciencedirect.com/science/article/pii/S0167865521002646?via%3Dihub).
- Update (Mar 5 2026): We are in communication with the dataset owners to release our processed bounding box annotations and additional data.
- YOLO detection weights are available at `ckpt/yolo_weight.pt`.

Usage

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Prepare your data**

Place your input `.mp4` video files in a `videos/` folder:

```
videos/
├── video1.mp4
├── video2.mp4
└── ...
```

**3. Configure paths**

Open `CL-MR.py` and edit the paths at the top of `main()` if needed:

```python
yolo_path      = "ckpt/yolo_weight.pt"   # YOLO detection weights
sam_model_name = "sam2.1_l.pt"           # SAM2 model (auto-downloaded by ultralytics)
video_folder   = "videos/"               # folder containing input .mp4 files
save_dir       = "output/"               # folder for output tracking results
```

The SAM2 model will be downloaded automatically by `ultralytics` on first run.

**4. Run CL-MR tracking**

```bash
python CL-MR.py
```

Results are saved as `.txt` files in `output/`, one per video, with per-frame bounding boxes in normalized `x_center y_center width height confidence` format.

**5. (Optional) Run baseline trackers**

BotSORT and ByteTrack baselines are provided as utility modules. You can invoke them from a Python script or interactive session:

```python
from botsort import load_yolo_model, run_videos

model = load_yolo_model("ckpt/yolo_weight.pt")
run_videos(folder_dir="videos/", model=model, save_dir="output/")
```

Replace `botsort` with `bytetrack` to use ByteTrack instead.

---

Acknowledgements
- We gratefully acknowledge the authors of YOLOv12 and SAM2 for their inspiring research.
- Thanks to the Ultralytics project and community for their code and tooling that made this implementation possible.

Citation
If you use this work, please cite:

```bibtex
@inproceedings{
	wang2026detectorintheloop,
	title={Detector-in-the-Loop Tracking: Active Memory Rectification for Stable Glottic Opening Localization},
	author={Huayu Wang and Bahaa Alattar and Cheng-Yen Yang and Hsiang-Wei Huang and Jung Heon Kim and Linda Shapiro and Nathan J White and Jenq-Neng Hwang},
	booktitle={Medical Imaging with Deep Learning},
	year={2026},
	url={https://openreview.net/forum?id=dwYFYk4aeZ}
}
```

Contact
- For questions or issues, please open an issue in this repository or contact the authors.