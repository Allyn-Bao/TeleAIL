import os
import sys
import cv2
import json
import mediapipe as mp

def main():
    if len(sys.argv) < 2:
        print("Usage: python 1_preprocess_video.py <folder/video.mp4>")
        sys.exit(1)

    rel_path = sys.argv[1]                # e.g. push_box/push-box-1.mp4
    folder, video_file = rel_path.split("/")

    video_path = os.path.join("..", "data", folder, video_file)

    if not os.path.isfile(video_path):
        print("Video not found:", video_path)
        sys.exit(1)

    # output directory
    video_base_name = os.path.basename(video_file).split(".")[0]
    out_dir = os.path.join("..", "data", "frames_cache", folder, video_base_name + "_frames")
    full_dir = os.path.join(out_dir, "full")
    hand_dir = os.path.join(out_dir, "hand")

    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(hand_dir, exist_ok=True)

    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open:", video_path)
        sys.exit(1)

    meta = {}
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)

        bbox = None
        if res.multi_hand_landmarks:
            xs, ys = [], []
            for lm in res.multi_hand_landmarks[0].landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            x0, y0 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
            x1, y1 = min(max(xs) + 20, w-1), min(max(ys) + 20, h-1)

            bbox = [x0, y0, x1, y1]
            crop = frame[y0:y1, x0:x1]
            cv2.imwrite(os.path.join(hand_dir, f"{idx:06d}.jpg"), crop)

        cv2.imwrite(os.path.join(full_dir, f"{idx:06d}.jpg"), frame)

        meta[idx] = {"bbox": bbox, "w": w, "h": h}
        idx += 1

    cap.release()

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Frames saved to:", out_dir)


if __name__ == "__main__":
    main()