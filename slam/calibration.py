# calibration.py - Camera calibration using a checkerboard pattern
import cv2
import numpy as np
import argparse
import glob
import sys
import os


def collect_from_webcam(cap, rows, cols, n_required=20):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern = (cols, rows)
    obj_pts, img_pts = [], []

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    captured = 0
    print(f"[Calibration] Need {n_required} frames. SPACE=capture, q=done.")

    while captured < n_required:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern, corners, found)
            cv2.putText(display, f"Found! SPACE to capture ({captured}/{n_required})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"No board detected ({captured}/{n_required})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners_refined)
            captured += 1
            print(f"  Captured frame {captured}/{n_required}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return obj_pts, img_pts, gray.shape[::-1]


def collect_from_images(image_paths, rows, cols):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern = (cols, rows)
    obj_pts, img_pts = [], []
    img_size = None

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners_refined)
            print(f"  OK: {os.path.basename(path)}")
        else:
            print(f"  SKIP (no board): {os.path.basename(path)}")

    return obj_pts, img_pts, img_size


def calibrate(obj_pts, img_pts, img_size):
    if len(obj_pts) < 5:
        print("[ERROR] Need at least 5 valid calibration frames.")
        sys.exit(1)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None
    )
    return K, dist, ret


def print_config(K, dist, rms):
    print("\n" + "="*60)
    print("Calibration complete!")
    print(f"RMS reprojection error: {rms:.4f} px  (< 1.0 is good)")
    print("="*60)
    print("\nPaste this into slam/config.py → CAMERA_PARAMS:\n")
    print("CAMERA_PARAMS = {")
    print(f'    "fx": {K[0,0]:.4f},')
    print(f'    "fy": {K[1,1]:.4f},')
    print(f'    "cx": {K[0,2]:.4f},')
    print(f'    "cy": {K[1,2]:.4f},')
    print("}")
    print(f"\nDistortion coefficients (k1,k2,p1,p2,k3):")
    print(f"  {dist.ravel().tolist()}")
    print("="*60)


def save_calibration(K, dist, path="slam/calibration_result.npz"):
    np.savez(path, K=K, dist=dist)
    print(f"[Calibration] Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument("--source", type=str, default=None,
                        help="Webcam index (e.g. 0) for live capture")
    parser.add_argument("--images", type=str, default=None,
                        help="Glob pattern for calibration images (e.g. 'calib/*.jpg')")
    parser.add_argument("--rows", type=int, default=9,
                        help="Inner corner rows on checkerboard (default: 9)")
    parser.add_argument("--cols", type=int, default=6,
                        help="Inner corner cols on checkerboard (default: 6)")
    parser.add_argument("--size", type=float, default=25.0,
                        help="Square size in mm (default: 25)")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of frames to capture (webcam mode)")
    args = parser.parse_args()

    if args.source is not None:
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {args.source}")
            sys.exit(1)
        obj_pts, img_pts, img_size = collect_from_webcam(
            cap, args.rows, args.cols, args.n
        )
        cap.release()

    elif args.images is not None:
        paths = sorted(glob.glob(args.images))
        if not paths:
            print(f"[ERROR] No images found: {args.images}")
            sys.exit(1)
        print(f"[Calibration] Found {len(paths)} images.")
        obj_pts, img_pts, img_size = collect_from_images(paths, args.rows, args.cols)

    else:
        print("[ERROR] Provide --source (webcam) or --images (files).")
        parser.print_help()
        sys.exit(1)

    K, dist, rms = calibrate(obj_pts, img_pts, img_size)
    print_config(K, dist, rms)
    save_calibration(K, dist)


if __name__ == "__main__":
    main()
