import numpy as np, cv2, argparse

def ratio_missing(arr):        # T × 33 × 3
    missing = np.isnan(arr[..., 0]).sum()
    return missing / arr.size

def overlay(img, pose, color=(0,255,0)):
    for x,y,_ in pose:
        if np.isnan(x) or np.isnan(y): continue
        cv2.circle(img,(int(x),int(y)),2,color, -1)

ap = argparse.ArgumentParser()
ap.add_argument("--npy1"); ap.add_argument("--npy2")
args = ap.parse_args()

a1, a2 = np.load(args.npy1), np.load(args.npy2)
print("missing mediapipe:", ratio_missing(a1))
print("missing crop_mp  :", ratio_missing(a2))