
# ⚽ Football Analysis System

A computer vision pipeline that extracts tactical and physical data from football broadcast video — player/ball detection, tracking, team assignment, possession analysis, camera motion compensation, and real-world speed/distance estimation.

## 🎥 Demo

Watch the full output video here: [Football Analysis — Output Demo](https://drive.google.com/file/d/1yhowl8jN1KoC4iBsT_9dvH_eje3nd0RQ/view?usp=sharing)

The video shows:

- Player, referee, and ball detection with persistent tracking IDs
- Team assignment via jersey color clustering (K-Means)
- Live ball possession tracking with running team control percentage
- Camera movement compensation overlay (X/Y pan per frame)
- Per-player speed (km/h) and cumulative distance covered (meters)

## 🧠 What It Does

- **Object Detection** — Custom-trained YOLOv8 model detecting 4 classes: ball, player, goalkeeper, referee
- **Multi-Object Tracking** — ByteTrack assigns persistent IDs across frames
- **Team Assignment** — K-Means clustering on jersey shirt color (unsupervised, no labels needed)
- **Ball Possession** — Distance-based assignment with sliding-window smoothing and a majority-lock mechanism to eliminate flickering, plus a ball-speed check to correctly withhold possession when the ball is in the air (mid-pass/shot)
- **Camera Movement Estimation** — Lucas-Kanade optical flow on static background zones (crowd/benches) to isolate and compensate for camera panning
- **Perspective Transformation** — Homography-based conversion from pixel coordinates to real-world pitch coordinates (meters)
- **Speed & Distance Estimation** — Per-player km/h and total meters covered, computed from real-world transformed positions

## 🏗️ Architecture

Broadcast Video
↓
YOLOv8 Detection (ball / player / goalkeeper / referee)
↓
ByteTrack (persistent IDs)
↓
┌──────────────┼──────────────────┐
↓ ↓ ↓
Team Assigner Player-Ball Camera Movement
(K-Means) Assigner Estimator (Optical Flow)
↓ ↓ ↓
Ball Possession Position Adjustment
Interpolation Smoothing (subtract camera mvt)
└──────────────┴──────────────────┘
↓
View Transformer
(perspective → real meters)
↓
Speed & Distance Estimator
↓
Draw Annotations
↓
Output Video

## 🛠️ Tech Stack

- **Language:** Python
- **Detection:** YOLOv8 (Ultralytics), custom-trained on a Roboflow football dataset
- **Tracking:** ByteTrack via the `supervision` library
- **Vision/Math:** OpenCV (optical flow, perspective transform, drawing), NumPy
- **Data wrangling:** pandas (ball interpolation)
- **Clustering:** scikit-learn (K-Means for team color assignment)
- **Caching:** pickle (avoids re-running expensive detection/optical-flow on every test)

## 📁 Project Structure

Football Analysis/
├── trackers/ # YOLO + ByteTrack, drawing, interpolation
├── team_assigner/ # K-Means shirt color clustering
├── player_ball_assigner/ # Distance-based possession logic
├── camera_movement_estimator/ # Optical flow camera compensation
├── view_transformer/ # Perspective transform (pixels → meters)
├── speed_and_distance_estimator/ # km/h and distance calculations
├── utils/ # bbox helpers, distance measurement
├── stubs/ # cached pickle results (gitignored)
└── main.py # orchestrates the full pipeline

## 🚀 Setup

```bash
# Clone the repo
git clone https://github.com/Rais-19/Football-Analysis-.git
cd Football-Analysis-

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python main.py
```

Place your input video in `input_videos/` and your trained YOLO weights in `model/` — update the paths in `main.py` accordingly.

## ⚠️ Known Limitations

- Goalkeeper team assignment is occasionally unreliable (different kit color breaks the 2-cluster assumption)
- Ball possession uses proximity + speed heuristics, not true ball-touch detection
- Perspective transform uses a single fixed calibration rectangle — assumes minimal camera zoom/pan
- Speed/distance only computed for players inside the calibrated pitch region
- Single broadcast camera cannot see the full pitch simultaneously

## 🙏 Credits

Built following the [Football Analysis tutorial](https://www.youtube.com/watch?v=neBZ6huolkg), extended with original debugging, stability improvements (ball possession smoothing, camera compensation, perspective calibration), and architectural analysis.
