# Football Tracker with YOLO

This project uses **Ultralytics YOLO** to detect soccer players and the ball in a video, track the nearest player to the ball, classify jersey colors (white/green), and optionally announce which team is attacking.

## Soccer Player & Ball Tracker

### Model
This project uses a YOLO model `best.pt` trained for soccer.  

**Download the model**: [best.pt](https://drive.google.com/file/d/1iiiIrUxjAkvR_kWWkO-UC9uOPQVqcUi-/view?usp=sharing) and place it in the project folder.

### Video
By default, the script uses the video located at:  

videos/example.mp4


If you want to use a different video, you can replace this file or update the path in `football_tracker.py`.

---

## Features
- Detect players and classify their jersey color using KMeans on HSV
- Detect the ball with YOLO or fallback on small object detection
- Track the nearest player to the ball
- Show ball direction and player statistics
- Optional audio alerts for attacking team (local only)

---

## Requirements

Python 3.11+ recommended. Install dependencies:

```bash
pip install -r requirements.txt

Usage

    Download the YOLO model best.pt and place it in the project folder.

    Place your video file as videos/example.mp4.

    Run the tracker locally:

python football_tracker.py
