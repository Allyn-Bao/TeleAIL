import os
import json
import argparse

import numpy as np
from PIL import Image


def load_keypoints(json_path):
    # load the json file for a frame
    with open(json_path, "r") as f:
        return json.load(f)


def build_pixel_coords(hand_norm, W, H):
    # convert normalized hand coords into pixel coords
    # MediaPipe sometimes gives values <0 or >1 so no clamping here
    x = hand_norm["x"] * W
    y = hand_norm["y"] * H
    z = hand_norm["z"]  # z is depth-ish, relative. keep as float
    return np.array([x, y, z], dtype=np.float32)


def compute_hand_local_frame(hand_xyz):
    # Assume hand_xyz is (21,3)
    # Three key landmarks:
    wrist = hand_xyz[0]
    index_mcp = hand_xyz[5]
    middle_mcp = hand_xyz[9]

    # v1: wrist -> index knuckle
    v1 = index_mcp - wrist
    # v2: wrist -> middle knuckle
    v2 = middle_mcp - wrist

    # y axis = v1 normalized
    y_axis = v1 / (np.linalg.norm(v1) + 1e-9)

    # z axis = v1 x v2 (normal of palm)
    z_axis = np.cross(v1, v2)
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-9:
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = z_axis / norm_z

    # x axis (right-hand rule)
    x_axis = np.cross(y_axis, z_axis)

    return wrist, x_axis, y_axis, z_axis


def compute_gripper_open_angle(hand_xyz):
    # use thumb tip (4) and index tip (8)
    thumb = hand_xyz[4]
    index_tip = hand_xyz[8]
    dist = np.linalg.norm(index_tip - thumb)

    # crude angle-like feature: larger dist = more open
    # normalized-ish but kept raw since we'll later normalize data anyway
    return dist


def compute_cube_vector(hand_origin, cube_xy, W, H):
    # cube_xy in normalized coords
    x, y, w, h = cube_xy
    cx = x + w / 2.0
    cy = y + h / 2.0
    cz = 0.0
    cube = np.array([cx, cy, cz], dtype=np.float32)

    return cube - hand_origin


def build_shared_states(args):
    # parse input name
    rel_video = args.relative_path
    folder_name, file_name = rel_video.split("/")
    vid_name = file_name.replace(".mp4", "")
    print(f"[INFO] Processing {folder_name}/{vid_name}")

    # paths
    full_frames_dir = os.path.join("..", "data", "frames_cache", folder_name, f"{vid_name}_frames", "full")
    kps_dir = os.path.join("..", "data", "keypoints_cache", folder_name, f"{vid_name}_keypoints")
    out_dir = os.path.join("..", "data", "shared_states", folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # list json files
    json_files = sorted([f for f in os.listdir(kps_dir) if f.endswith(".json")])

    # construct output path
    out_path = os.path.join(out_dir, f"{vid_name}_states.ndjson")

    # determine image size using first frame
    first_frame_path = os.path.join(full_frames_dir, "000000.jpg")
    print(f"[DEBUG] first frame path: ", first_frame_path)
    if not os.path.isfile(first_frame_path):
        raise FileNotFoundError("No frames found in frames_cache.")
    img = Image.open(first_frame_path)
    W, H = img.size

    # process frame by frame
    fout = open(out_path, "w")
    print(f"[INFO] Saving states to {out_path}")

    for jf in json_files:
        data = load_keypoints(os.path.join(kps_dir, jf))

        # hand landmarks: list of dicts
        hand_raw = data.get("hand", None)
        cube_raw = data.get("cube", None)

        if hand_raw is None or len(hand_raw) != 21:
            # keep consistent structure
            state = {
                "frame": jf,
                "valid": False,
                "reason": "missing hand landmarks"
            }
            fout.write(json.dumps(state) + "\n")
            continue

        # convert hand to numpy xyz array in pixel coords
        hand_xyz = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_raw):
            hand_xyz[i] = build_pixel_coords(lm, W, H)

        # compute hand frame
        hand_origin, x_axis, y_axis, z_axis = compute_hand_local_frame(hand_xyz)

        # open angle
        open_angle = compute_gripper_open_angle(hand_xyz)

        # cube vector in pixel coords
        if cube_raw is None:
            cube_vec = None
        else:
            cube_vec = compute_cube_vector(hand_origin, cube_raw, W, H).tolist()

        # pack state
        state = {
            "frame": jf,
            "valid": True,
            "hand_origin": hand_origin.tolist(),
            "hand_axes": {
                "x": x_axis.tolist(),
                "y": y_axis.tolist(),
                "z": z_axis.tolist()
            },
            "open_angle": float(open_angle),
            "cube_vec": cube_vec
        }

        fout.write(json.dumps(state) + "\n")

    fout.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("relative_path", type=str,
                        help="format: <folder>/<video_name>.mp4  (e.g. push_box/push-box-1.mp4)")
    args = parser.parse_args()
    build_shared_states(args)