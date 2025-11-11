#!/usr/bin/env python3
#
# Social-distance / "violence" detection using poseNet
# - Single image mode:
#       python3 social_distance_posenet5.py --image path/to/img.jpg
#   (saves annotated image to ./result)
#
# - Video/camera realtime mode:
#       python3 social_distance_posenet5.py /dev/video0 display://0 --distance 150
#   (shows live window AND records ./result/violence_realtime.mp4)
#

import sys
import os
import math
import argparse

from jetson_inference import poseNet
from jetson_utils import (
    loadImage,
    saveImage,
    videoSource,
    videoOutput,
)

# ----------------- arguments -----------------
parser = argparse.ArgumentParser(
    description="Measure distance between people using poseNet"
)

# optional single-image mode
parser.add_argument(
    "--image",
    type=str,
    default="",
    help="path to the input image (enables single-image mode)"
)

# streaming mode (camera / video file / RTSP)
parser.add_argument(
    "input",
    type=str,
    nargs="?",
    default="",
    help="URI of the input stream (e.g. /dev/video0, csi://0, file.mp4, rtsp://...)"
)

parser.add_argument(
    "output",
    type=str,
    nargs="?",
    default="display://0",
    help="URI of the primary output (e.g. display://0, file://output.mp4)"
)

parser.add_argument(
    "--network",
    type=str,
    default="resnet18-body",
    help="poseNet model to load (default: resnet18-body)"
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.15,
    help="minimum pose detection threshold (default: 0.15)"
)

parser.add_argument(
    "--distance",
    type=float,
    default=150.0,
    help="ABSOLUTE violence/social-distance threshold in pixels (default: 150)"
)

parser.add_argument(
    "--rel_threshold",
    type=float,
    default=0.7,
    help="RELATIVE threshold: distance / avg_person_height (default: 0.7)"
)

args = parser.parse_args()

# ----------------- load poseNet -----------------
net = poseNet(args.network, sys.argv, args.threshold)

# ----------------- helper functions -----------------
LEFT_HIP_ID = 11
RIGHT_HIP_ID = 12


def find_keypoint(pose, kp_id):
    """Return keypoint with given ID, or None if not found."""
    for kp in pose.Keypoints:
        if kp.ID == kp_id:
            return kp
    return None


def person_center(pose):
    """
    Approximate person's center:
      - primary: midpoint of left/right hip
      - fallback: average of all keypoints
    """
    lh = find_keypoint(pose, LEFT_HIP_ID)
    rh = find_keypoint(pose, RIGHT_HIP_ID)

    if lh is not None and rh is not None:
        cx = (lh.x + rh.x) / 2.0
        cy = (lh.y + rh.y) / 2.0
        return (cx, cy)

    # fallback if hips not found
    xs = [kp.x for kp in pose.Keypoints]
    ys = [kp.y for kp in pose.Keypoints]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return (cx, cy)


def person_height(pose):
    """Estimate person height in pixels as the vertical span of keypoints."""
    ys = [kp.y for kp in pose.Keypoints]
    return max(ys) - min(ys)


def distance(c1, c2):
    """Euclidean distance between two centers (x1,y1) and (x2,y2)."""
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def analyze_poses(poses):
    """
    Compute centers, heights, distances and return:
      any_violence (bool), centers, heights
    Also prints per-pair stats to the console.
    """
    centers = []
    heights = []

    if len(poses) == 0:
        print("[INFO] No people detected.")
        return False, centers, heights

    print("\n[INFO] Person centers & heights (image coordinates):")
    for i, pose in enumerate(poses):
        c = person_center(pose)
        h = person_height(pose)
        centers.append(c)
        heights.append(h)
        print(f"  Person {i}: center = ({c[0]:.2f}, {c[1]:.2f}), height ~ {h:.2f} px")

    any_violence = False

    print("\n[INFO] Pairwise distances:")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d_abs = distance(centers[i], centers[j])
            avg_h = (heights[i] + heights[j]) / 2.0 if heights[i] > 0 and heights[j] > 0 else 1.0
            d_rel = d_abs / avg_h

            abs_flag = d_abs < args.distance
            rel_flag = d_rel < args.rel_threshold

            print(
                f"  Person {i} - Person {j}: "
                f"abs = {d_abs:.2f} px, "
                f"avg_height = {avg_h:.2f} px, "
                f"norm = {d_rel:.2f} "
                f"=> abs_violation={int(abs_flag)}, rel_violation={int(rel_flag)}"
            )

            if abs_flag or rel_flag:
                any_violence = True

    print(
        f"\n[INFO] Absolute threshold = {args.distance:.2f} px, "
        f"relative threshold = {args.rel_threshold:.2f} (distance/height)"
    )

    if any_violence:
        print("=> VIOLENCE / TOO CLOSE DETECTED (at least one rule violated)\n")
    else:
        print("=> NO VIOLENCE (no pair closer than thresholds)\n")

    return any_violence, centers, heights


# ============================================================
# MODE 1: SINGLE-IMAGE
# ============================================================
if args.image:
    print(f"[INFO] Loading image: {args.image}")
    img = loadImage(args.image)

    poses = net.Process(img, overlay="links,keypoints")
    print(f"\n[INFO] Detected {len(poses)} person(s) in image.")

    any_violence, centers, heights = analyze_poses(poses)

    # create result folder in the current directory
    result_dir = os.path.join(os.getcwd(), "result")
    os.makedirs(result_dir, exist_ok=True)

    base = os.path.basename(args.image)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(result_dir, f"{name}_result{ext}")

    saveImage(out_path, img)
    print(f"[INFO] Saved annotated image to: {out_path}")
    sys.exit(0)

# ============================================================
# MODE 2: VIDEO / CAMERA (REALTIME + RECORDING)
# ============================================================
if not args.input:
    print("error: please specify either --image or an input stream URI (e.g. /dev/video0)")
    sys.exit(1)

print(f"[INFO] Realtime mode from input='{args.input}' to output='{args.output}'")

# input stream (camera / video)
input_stream = videoSource(args.input, argv=sys.argv)

# primary output (usually display://0)
display_out = videoOutput(args.output, argv=sys.argv)

# secondary output: recorded MP4 in ./result
result_dir = os.path.join(os.getcwd(), "result")
os.makedirs(result_dir, exist_ok=True)
record_path = os.path.join(result_dir, "violence_realtime.mp4")
record_uri = "file://" + record_path
record_out = videoOutput(record_uri, argv=sys.argv)

print(f"[INFO] Recording realtime video to: {record_path}")

while True:
    img = input_stream.Capture()
    if img is None:   # timeout
        continue

    # pose estimation with overlay
    poses = net.Process(img, overlay="links,keypoints")

    # analyze distances for this frame
    any_violence, centers, heights = analyze_poses(poses)
    status = "VIOLENCE" if any_violence else "SAFE"

    print(f"[FRAME] {status}, people={len(poses)}")

    # render to display and to recording
    display_out.Render(img)
    record_out.Render(img)

    display_out.SetStatus(f"{status} | Network {net.GetNetworkFPS():.0f} FPS")

    # stop if any stream closes
    if (not input_stream.IsStreaming() or
        not display_out.IsStreaming() or
        not record_out.IsStreaming()):
        break

print(f"[INFO] Finished. Recorded video saved to: {record_path}")
