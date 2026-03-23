#!/usr/bin/env python3
"""
OAK-D Lite Stereo Visual Odometry (SLAM)
=========================================
Estimates robot pose (position + orientation) in real-time using the OAK-D
Lite's stereo camera pair. Runs entirely on the host PC — no IMU required.

Algorithm overview:
  1. Stream left + right grayscale frames from OAK-D Lite (DepthAI v3 API)
  2. Build a disparity/depth map from each stereo pair (StereoSGBM)
  3. Detect + describe features in the left frame (SIFT)
  4. Match features between consecutive left frames (FLANN)
  5. Lift 2D matches to 3D using the depth map
  6. Estimate inter-frame motion via PnP + RANSAC (solvePnPRansac)
  7. Accumulate the 4×4 pose matrix → position (x, y, z) + Euler angles
  8. Draw a live top-down trajectory map and feature-match overlay

Install dependencies:
  pip install depthai opencv-python numpy

Camera intrinsics (K) and baseline are read from the device at runtime via
the DepthAI calibration API — no hard-coding required.

Run:
  python3 oak_d_lite_stereo_vo.py
"""

import time
import math
import numpy as np
import cv2
import depthai as dai


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

FRAME_WIDTH   = 640
FRAME_HEIGHT  = 400
TARGET_FPS    = 20

# StereoSGBM depth parameters
SGBM_NUM_DISPARITIES = 128   # must be divisible by 16
SGBM_BLOCK_SIZE      = 5

# SIFT feature detector (best accuracy for VO)
MAX_FEATURES     = 1500
MATCH_RATIO_TEST = 0.65   # Lowe's ratio — lower = stricter

# RANSAC threshold for PnP (pixels)
RANSAC_REPROJECTION_ERR = 2.0

# Depth clip: ignore points outside this range (metres)
MIN_DEPTH = 0.2
MAX_DEPTH = 10.0

# Trajectory canvas size (pixels)
TRAJ_SIZE = 800
TRAJ_SCALE = 50  # pixels per metre


# ──────────────────────────────────────────────────────────────────────────────
# DepthAI v3 pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    """Create a DepthAI v3 pipeline that streams left and right mono frames.

    DepthAI v3 Camera node API:
      - pipeline.create(dai.node.Camera).build(socket)  sets the board socket
      - .requestOutput((w, h), type=...)                returns the output stream
      - No XLinkOut nodes, no .setBoardSocket() or .setSize() on the node itself
    """
    pipeline = dai.Pipeline()

    # Left mono camera (CAM_B = left stereo on OAK-D Lite)
    # v3: FPS is passed into requestOutput(), not set on the node
    cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    out_left  = cam_left.requestOutput(
        (FRAME_WIDTH, FRAME_HEIGHT),
        type=dai.ImgFrame.Type.GRAY8,
        fps=TARGET_FPS
    )

    # Right mono camera (CAM_C = right stereo on OAK-D Lite)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    out_right = cam_right.requestOutput(
        (FRAME_WIDTH, FRAME_HEIGHT),
        type=dai.ImgFrame.Type.GRAY8,
        fps=TARGET_FPS
    )

    # Output queues — created directly on the output stream (no XLinkOut)
    q_left  = out_left.createOutputQueue(maxSize=4, blocking=False)
    q_right = out_right.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_left, q_right


def get_calibration(device):
    """
    Read camera intrinsics and stereo baseline from device calibration data.
    Returns:
        K      : 3×3 left camera intrinsic matrix (for the configured resolution)
        dist   : distortion coefficients for left camera
        baseline_m : stereo baseline in metres
    """
    calib   = device.readCalibration()
    # getCameraIntrinsics returns a 3×3 matrix as a list of lists:
    # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    M       = calib.getCameraIntrinsics(
                  dai.CameraBoardSocket.CAM_B,
                  FRAME_WIDTH, FRAME_HEIGHT
              )
    K       = np.array(M, dtype=np.float64)   # shape (3, 3)
    fx      = K[0, 0]
    fy      = K[1, 1]
    cx      = K[0, 2]
    cy      = K[1, 2]

    dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B),
                    dtype=np.float64)

    # Baseline: translation between right and left camera optical centres
    T_right = calib.getCameraTranslationVector(
        dai.CameraBoardSocket.CAM_B,
        dai.CameraBoardSocket.CAM_C
    )
    baseline_m = abs(T_right[0]) / 100.0   # cm → m

    return K, dist, baseline_m


# ──────────────────────────────────────────────────────────────────────────────
# Depth map helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_sgbm(num_disparities, block_size):
    P1 = 8  * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2
    return cv2.StereoSGBM_create(
        minDisparity    = 0,
        numDisparities  = num_disparities,
        blockSize       = block_size,
        P1              = P1,
        P2              = P2,
        disp12MaxDiff   = 1,
        uniquenessRatio = 10,
        speckleWindowSize  = 100,
        speckleRange       = 32,
        mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


def disparity_to_depth(disparity, fx, baseline_m):
    """Convert disparity map (float32) to depth map in metres."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disparity > 0,
                         fx * baseline_m / disparity,
                         0.0).astype(np.float32)
    return depth


# ──────────────────────────────────────────────────────────────────────────────
# Pose estimation
# ──────────────────────────────────────────────────────────────────────────────

class StereoVO:
    """Incremental stereo visual odometry."""

    def __init__(self, K, dist, baseline_m):
        self.K          = K
        self.dist       = dist
        self.baseline   = baseline_m
        self.fx         = K[0, 0]
        self.fy         = K[1, 1]
        self.cx         = K[0, 2]
        self.cy         = K[1, 2]

        self.sgbm       = build_sgbm(SGBM_NUM_DISPARITIES, SGBM_BLOCK_SIZE)

        # SIFT detector
        self.detector   = cv2.SIFT_create(nfeatures=MAX_FEATURES)

        # FLANN matcher for SIFT descriptors
        index_params    = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
        search_params   = {"checks": 50}
        self.matcher    = cv2.FlannBasedMatcher(index_params, search_params)

        # Accumulated world pose (4×4 homogeneous)
        self.pose       = np.eye(4, dtype=np.float64)

        # Previous frame state
        self.prev_gray  = None
        self.prev_kps   = None
        self.prev_descs = None
        self.prev_depth = None

        # History for trajectory visualisation
        self.trajectory = []   # list of (x, z) world-plane points

    # ── Core update ─────────────────────────────────────────────────────────

    def process(self, left_gray: np.ndarray, right_gray: np.ndarray):
        """
        Feed one stereo frame pair. Returns (pose_4x4, match_vis, depth_vis).
        pose_4x4 is None on the first frame.
        """
        # 1. Depth map
        disp   = self.sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
        depth  = disparity_to_depth(disp, self.fx, self.baseline)
        depth  = np.where((depth > MIN_DEPTH) & (depth < MAX_DEPTH), depth, 0.0)

        depth_vis = self._vis_depth(depth)

        # 2. Feature detection
        kps, descs = self.detector.detectAndCompute(left_gray, None)

        if self.prev_gray is None:
            # First frame — just store and return
            self._store(left_gray, kps, descs, depth)
            return None, None, depth_vis

        # 3. Feature matching (Lowe ratio test)
        matches, pts2d_prev, pts2d_curr = self._match(
            self.prev_descs, descs, self.prev_kps, kps
        )

        if len(pts2d_curr) < 8:
            self._store(left_gray, kps, descs, depth)
            return self.pose.copy(), None, depth_vis

        # 4. Lift previous 2D points → 3D using previous depth map
        pts3d = self._lift_to_3d(pts2d_prev, self.prev_depth)

        # Filter: keep only points with valid depth
        valid = pts3d[:, 2] > 0
        pts3d     = pts3d[valid]
        pts2d_cur = pts2d_curr[valid]

        if len(pts3d) < 6:
            self._store(left_gray, kps, descs, depth)
            return self.pose.copy(), None, depth_vis

        # 5. PnP + RANSAC → relative rotation + translation
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64),
            pts2d_cur.astype(np.float64),
            self.K, self.dist,
            iterationsCount    = 200,
            reprojectionError  = RANSAC_REPROJECTION_ERR,
            confidence         = 0.999,
            flags              = cv2.SOLVEPNP_EPNP
        )

        match_vis = self._vis_matches(
            self.prev_gray, left_gray,
            self.prev_kps, kps, matches,
            inliers, valid
        )

        if not success or inliers is None or len(inliers) < 6:
            self._store(left_gray, kps, descs, depth)
            return self.pose.copy(), match_vis, depth_vis

        # 6. Convert rvec/tvec → 4×4 transform and accumulate
        R, _   = cv2.Rodrigues(rvec)
        T_rel  = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3,  3] = tvec.ravel()

        self.pose = self.pose @ np.linalg.inv(T_rel)

        # Record x, z for top-down trajectory
        x = self.pose[0, 3]
        z = self.pose[2, 3]
        self.trajectory.append((x, z))

        self._store(left_gray, kps, descs, depth)
        return self.pose.copy(), match_vis, depth_vis

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _store(self, gray, kps, descs, depth):
        self.prev_gray  = gray
        self.prev_kps   = kps
        self.prev_descs = descs
        self.prev_depth = depth

    def _match(self, desc1, desc2, kps1, kps2):
        """FLANN k-NN match with Lowe ratio test. Returns (matches, pts1, pts2)."""
        if desc1 is None or desc2 is None:
            return [], np.array([]), np.array([])

        raw = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m_n in raw:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < MATCH_RATIO_TEST * n.distance:
                    good.append(m)

        pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in good])
        return good, pts1, pts2

    def _lift_to_3d(self, pts2d: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Back-project 2D pixel coordinates to 3D camera-frame points."""
        pts3d = np.zeros((len(pts2d), 3), dtype=np.float32)
        for i, (u, v) in enumerate(pts2d):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < depth_map.shape[0] and 0 <= ui < depth_map.shape[1]:
                z = depth_map[vi, ui]
                if z > 0:
                    pts3d[i] = [
                        (u - self.cx) * z / self.fx,
                        (v - self.cy) * z / self.fy,
                        z
                    ]
        return pts3d

    @staticmethod
    def _vis_depth(depth):
        """Colourised depth map for display."""
        d_vis = np.clip(depth / MAX_DEPTH, 0, 1)
        d_8   = (d_vis * 255).astype(np.uint8)
        return cv2.applyColorMap(d_8, cv2.COLORMAP_TURBO)

    @staticmethod
    def _vis_matches(img1, img2, kps1, kps2, matches, inliers, valid_mask):
        """Draw inlier matches side-by-side."""
        if matches is None or inliers is None:
            return None
        inlier_set = set(inliers.ravel())
        draw_matches = [m for i, m in enumerate(matches) if i in inlier_set]
        vis = cv2.drawMatches(
            img1, kps1, img2, kps2, draw_matches[:80],
            None,
            matchColor     = (0, 255, 0),
            singlePointColor = (200, 200, 200),
            flags          = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return vis

    # ── Pose helpers ────────────────────────────────────────────────────────

    def get_position(self):
        """Return (x, y, z) world position in metres."""
        return tuple(self.pose[:3, 3])

    def get_euler_deg(self):
        """Return (roll, pitch, yaw) in degrees from accumulated pose."""
        R = self.pose[:3, :3]
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy > 1e-6:
            roll  = math.atan2( R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2( R[1, 0], R[0, 0])
        else:
            roll  = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = 0.0
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory visualisation
# ──────────────────────────────────────────────────────────────────────────────

def draw_trajectory(vo: StereoVO) -> np.ndarray:
    """Draw a top-down (x–z plane) trajectory map."""
    canvas = np.zeros((TRAJ_SIZE, TRAJ_SIZE, 3), dtype=np.uint8)

    # Grid lines
    for g in range(0, TRAJ_SIZE, TRAJ_SIZE // 10):
        cv2.line(canvas, (g, 0), (g, TRAJ_SIZE), (30, 30, 30), 1)
        cv2.line(canvas, (0, g), (TRAJ_SIZE, g), (30, 30, 30), 1)

    origin = (TRAJ_SIZE // 2, TRAJ_SIZE // 2)
    cv2.drawMarker(canvas, origin, (0, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)
    cv2.putText(canvas, "Start", (origin[0] + 5, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    pts = vo.trajectory
    for i in range(1, len(pts)):
        x0 = int(origin[0] + pts[i-1][0] * TRAJ_SCALE)
        y0 = int(origin[1] - pts[i-1][1] * TRAJ_SCALE)   # z → up on screen
        x1 = int(origin[0] + pts[i][0]   * TRAJ_SCALE)
        y1 = int(origin[1] - pts[i][1]   * TRAJ_SCALE)
        # Colour: gradient blue → green → red with distance
        t  = min(i / max(len(pts), 1), 1.0)
        colour = (
            int(255 * (1 - t)),       # B
            int(255 * min(2*t, 1)),   # G
            int(255 * max(2*t-1, 0))  # R
        )
        cv2.line(canvas, (x0, y0), (x1, y1), colour, 2)

    # Current position marker
    if pts:
        cx_ = int(origin[0] + pts[-1][0] * TRAJ_SCALE)
        cy_ = int(origin[1] - pts[-1][1] * TRAJ_SCALE)
        cv2.circle(canvas, (cx_, cy_), 6, (0, 0, 255), -1)

    # Legend
    x, y, z = vo.get_position()
    roll, pitch, yaw = vo.get_euler_deg()
    cv2.putText(canvas, f"Pos  x={x:+.3f}  y={y:+.3f}  z={z:+.3f} m",
                (8, TRAJ_SIZE - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, f"Rot  r={roll:+.1f}  p={pitch:+.1f}  yaw={yaw:+.1f} deg",
                (8, TRAJ_SIZE - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, f"Frames: {len(pts)}",
                (8, TRAJ_SIZE - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (120, 120, 120), 1)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    pipeline, q_left, q_right = build_pipeline()
    pipeline.start()

    print("OAK-D Lite connected — reading calibration …")

    # Grab calibration from the device
    # NOTE: pipeline.getDefaultDevice() gives us a handle without closing the pipeline
    calib   = pipeline.getDefaultDevice().readCalibration()
    M       = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B,
                                        FRAME_WIDTH, FRAME_HEIGHT)
    K       = np.array(M, dtype=np.float64)  # 3×3 matrix
    fx      = K[0, 0]
    fy      = K[1, 1]
    cx      = K[0, 2]
    cy      = K[1, 2]
    dist    = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B),
                       dtype=np.float64)
    T_lr    = calib.getCameraTranslationVector(dai.CameraBoardSocket.CAM_B,
                                               dai.CameraBoardSocket.CAM_C)
    baseline = abs(T_lr[0]) / 100.0   # cm → m

    print(f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"  Baseline = {baseline*100:.1f} cm")
    print()

    vo  = StereoVO(K, dist, baseline)
    fps_timer = time.monotonic()
    frame_count = 0

    print("Running — press 'q' to quit, 'r' to reset pose.\n")

    try:
        while pipeline.isRunning():
            l_frame = q_left.get()
            r_frame = q_right.get()

            if l_frame is None or r_frame is None:
                continue

            left_gray  = l_frame.getCvFrame()
            right_gray = r_frame.getCvFrame()

            # Ensure grayscale
            if left_gray.ndim == 3:
                left_gray  = cv2.cvtColor(left_gray,  cv2.COLOR_BGR2GRAY)
            if right_gray.ndim == 3:
                right_gray = cv2.cvtColor(right_gray, cv2.COLOR_BGR2GRAY)

            # ── Visual odometry step ──────────────────────────────────────
            pose, match_vis, depth_vis = vo.process(left_gray, right_gray)

            frame_count += 1
            now = time.monotonic()
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now
                if pose is not None:
                    x, y, z     = vo.get_position()
                    roll, pitch, yaw = vo.get_euler_deg()
                    print(
                        f"[{fps:4.1f} fps] "
                        f"pos=({x:+7.3f}, {y:+7.3f}, {z:+7.3f}) m  "
                        f"roll={roll:+6.1f}°  pitch={pitch:+6.1f}°  yaw={yaw:+6.1f}°  "
                        f"inliers≈{len(vo.trajectory)}"
                    )

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        x, y, z = vo.get_position()
        print(f"\nFinal position: x={x:.4f} m, y={y:.4f} m, z={z:.4f} m")
        print(f"Final heading : yaw={vo.get_euler_deg()[2]:.2f}°")


if __name__ == "__main__":
    main()
