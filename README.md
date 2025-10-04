# Football Tracker with YOLO

This project uses **Ultralytics YOLO** to detect soccer players and the ball in a video, track the nearest player to the ball, classify jersey colors (white/green), and optionally announce which team is attacking.

## Features
- Detect players and classify their jersey color using KMeans on HSV
- Detect the ball with YOLO or fallback on small object detection
- Track the nearest player to the ball
- Show ball direction and player statistics
- Optional audio alerts for attacking team (local only)

## Requirements

Python 3.11+ recommended. Install dependencies:

```bash
pip install -r requirements.txt

Usage

    Download or place your YOLO model as best.pt.

    Place a video file in videos/ or update the VIDEO_PATH in football_tracker.py.

    Run locally:

python football_tracker.py