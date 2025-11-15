# 🧪 Real-Time Social Distancing & Violence Proximity Detection (Jetson + poseNet)

> **Tagline:** _From raw camera feed ➜ to live pose estimation ➜ to distance-based “violence” alerts_  
> Built on **NVIDIA Jetson** + **jetson-inference** + **poseNet**

---

## 🎬 What This Project Does

This project turns your **Jetson board** into a **real-time social distancing / proximity monitor** using human **pose estimation**.

It supports:

1. ✅ **Single Image Analysis**  
   - Load an image  
   - Run pose estimation (poseNet)  
   - Compute distances between people  
   - Decide whether it’s **SAFE** or **TOO CLOSE / “VIOLENCE”**  
   - Save the **annotated image** to a `result/` folder

2. ✅ **Realtime Camera / Video Monitoring**  
   - Use `/dev/video0`, `csi://0`, RTSP, or a video file  
   - Run poseNet **per frame**  
   - Compute distances & flag violations live  
   - Display annotated video in a window  
   - **Record the full annotated stream** to `result/violence_realtime.mp4`

Everything is powered by:

- **poseNet (ResNet18-Body)** from `jetson-inference`
- **Hip-based person center + height estimation**
- **Absolute & relative distance thresholds** to decide “violence” / too close.

---

## 🧱 Core Idea

Each detected person is represented by **pose keypoints** (joints like shoulders, hips, knees, etc).

We:

1. Use the **midpoint of left & right hip** as a person’s **center**  
2. Approximate **person height** as the vertical span of keypoints  
3. For every pair of people:
   - Compute **absolute distance** in pixels  
   - Compute **relative distance** = distance / average_height  
4. If either of these is below thresholds, we flag it:

```text
if distance < ABS_THRESHOLD or distance / avg_height < REL_THRESHOLD:
    => VIOLENCE / TOO CLOSE
else:
    => SAFE
```

You can tune both thresholds:

- `--distance` (absolute pixels, e.g. 150)  
- `--rel_threshold` (normalized, e.g. 0.7)

---

## 📂 Project Structure

```bash
your-project/
├── social_distance_posenet.py   # main script (image + realtime)
├── result/                       # output folder (auto-created)
│   ├── <image_name>_result.jpg   # annotated single-image result
│   └── violence_realtime.mp4     # recorded realtime video
└── image/
    ├── violence/                 # violence / too-close example images
    └── no_violence/              # safe / socially distant images
```

> ✨ The script automatically creates `result/` in the **current working directory**.

---

## 🚀 Getting Started

### 1️⃣ Requirements

- NVIDIA **Jetson** (Nano / Orin / Xavier)
- **JetPack** (with CUDA, cuDNN, TensorRT)
- `jetson-inference` library installed
- Python 3
- A camera: USB (`/dev/video0`) or CSI (`csi://0`)

If you already have Dusty’s `jetson-inference` repo cloned, this script fits perfectly into that environment.

---

### 2️⃣ Clone This Repo

You can name the repo something like:

> **`jetson-social-distance-posenet`**

Example:

```bash
git clone https://github.com/Anjali9815/jetson-social-distance-posenet.git
cd jetson-social-distance-posenet
```

Copy `social_distance_posenet.py` into a suitable place inside your `jetson-inference` data folder, or keep it in this repo and just run it from here.

---

### 3️⃣ Single Image Mode 🖼️

Run the script on a single image:

```bash
python3 social_distance_posenet.py \
    --image ../image/violence/violence_lab_mp4-0000_jpg.rf.2f1d50c0434949b56e64c10114c509bd.jpg \
    --distance 150 \
    --rel_threshold 0.7
```

What happens:

- poseNet runs on the image
- Person centers & heights are printed:
  ```text
  [INFO] Person centers & heights (image coordinates):
    Person 0: center = (654.98, 313.59), height ~ 820.12 px
    Person 1: center = (548.65, 276.34), height ~ 795.44 px
  ```
- Pairwise distances are printed:
  ```text
  [INFO] Pairwise distances:
    Person 0 - Person 1: abs = 112.66 px, avg_height = 807.78 px, norm = 0.14
    => abs_violation=1, rel_violation=1
  ```
- Final verdict:
  ```text
  => VIOLENCE / TOO CLOSE DETECTED (at least one rule violated)
  ```
- Annotated result image is saved to:
  ```bash
  result/<original_name>_result.jpg
  ```

---

### 4️⃣ Realtime Camera Mode 🎥

For a **live USB webcam** on `/dev/video0`:

```bash
python3 social_distance_posenet.py \
    /dev/video0 \
    display://0 \
    --distance 150 \
    --rel_threshold 0.7
```

What happens:

- Video frames are captured in real time
- poseNet runs on each frame (`resnet18-body`)
- Distances are computed for every pair of people
- Status per frame is logged:

  ```text
  [FRAME] SAFE, people=1
  [FRAME] VIOLENCE, people=2
  ...
  ```

- A display window opens (on the Jetson desktop) showing:
  - Pose skeletons
  - People moving in real time
  - Window title includes status & FPS:
    ```text
    VIOLENCE | Network 22 FPS
    ```

- A **full annotated recording** is saved to:

  ```bash
  result/violence_realtime.mp4
  ```

You can also use a video file or RTSP stream:

```bash
python3 social_distance_posenet.py \
    my_video.mp4 \
    display://0 \
    --distance 150
```

or:

```bash
python3 social_distance_posenet.py \
    rtsp://user:pass@ip:port/stream \
    display://0
```

---

## 🧪 Suggested Experiments

Try the following to explore the behavior:

1. 🔄 **Change absolute threshold**  
   - `--distance 100`, `--distance 250`  
   - See how sensitive it becomes.

2. 📏 **Change relative threshold**  
   - `--rel_threshold 0.5` vs `0.9`  
   - Normalized distance is more robust to scaling / zoom.

3. 🎥 **Test different scenarios**  
   - People far apart → should be **SAFE**  
   - People very close → should trigger **VIOLENCE**  
   - Crowd scenes → many pairs, check logs.

4. 📊 **Log to CSV** (extension idea)  
   - You can easily modify the script to write:
     - frame index
     - timestamp
     - person centers
     - violation flags
   - Perfect for later analysis.

---

## 🧩 Typical Commands Recap

### ✅ Single image:

```bash
python3 social_distance_posenet.py \
    --image ../image/violence/violence_lab_mp4-0000_jpg.rf.2f1d50c0434949b56e64c10114c509bd.jpg \
    --distance 150 \
    --rel_threshold 0.7
```

### ✅ Live camera:

```bash
python3 social_distance_posenet.py \
    /dev/video0 \
    display://0 \
    --distance 150 \
    --rel_threshold 0.7
```

Resulting video:

```bash
result/violence_realtime.mp4
```

---

## 🧾 Credits & Inspiration

- NVIDIA **jetson-inference** project  
- **poseNet (ResNet18-Body)** model  
- Classroom / lab idea: turning distancing rules into a **computer vision experiment**.

If you use this in a lab or demo, consider showing:

- A **SAFE** scenario (people apart)
- A **VIOLENT / TOO CLOSE** scenario (people close)
- The **console logs** + annotated **video output** side by side.


