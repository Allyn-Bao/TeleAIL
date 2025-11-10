import os
import sys
import cv2
import json
from tqdm import tqdm
import mediapipe as mp

def detect_red_cube(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = (0, 120, 70)
    upper1 = (10, 255, 255)
    lower2 = (170, 120, 70)
    upper2 = (180, 255, 255)

    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cont:
        return None

    x, y, w, h = cv2.boundingRect(max(cont, key=cv2.contourArea))
    return [int(x), int(y), int(w), int(h)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python 2_extract_keypoints.py <folder/video.mp4>")
        sys.exit(1)

    rel_path = sys.argv[1]      # e.g. push_box/push-box-1.mp4
    folder, video_file = rel_path.split("/")

    video_base_name = os.path.basename(video_file).split('.')[0]
    print(f"video_base_name:", video_base_name)

    # Frames dir from script 1
    root = os.path.dirname(__file__)
    frames_dir = os.path.join("..", "data", "frames_cache", folder, video_base_name + "_frames", "full")

    if not os.path.isdir(frames_dir):
        print("Frames not found:", frames_dir)
        sys.exit(1)

    out_dir = os.path.join(root, "..", "data", "keypoints_cache", folder, video_base_name + "_keypoints")
    print(f"outdir:", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mp_hands = mp.solutions.hands.Hands(static_image_mode=True)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True)

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

    for f in tqdm(frames, desc="Extracting keypoints"):
        img_path = os.path.join(frames_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_res = mp_hands.process(rgb)
        pose_res = mp_pose.process(rgb)

        hand_pts = None
        if hand_res.multi_hand_landmarks:
            hand = hand_res.multi_hand_landmarks[0]
            hand_pts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand.landmark]

        pose_pts = None
        if pose_res.pose_landmarks:
            pose_pts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in pose_res.pose_landmarks.landmark]

        cube = detect_red_cube(img)

        out = {
            "frame": f,
            "hand": hand_pts,
            "pose": pose_pts,
            "cube": cube
        }

        with open(os.path.join(out_dir, f.replace(".jpg", ".json")), "w") as fp:
            json.dump(out, fp, indent=2)

    print("Done. Keypoints saved to:", out_dir)


if __name__ == "__main__":
    main()