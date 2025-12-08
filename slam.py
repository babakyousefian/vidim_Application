# slam.py
"""
Lightweight monocular Visual SLAM module for integration with your GUI.

Features:
- Keyframe-based sparse monocular SLAM.
- Uses provided feature detector & matcher API from your project.
- For frame pairs: estimate essential matrix -> recoverPose -> triangulate points.
- Maintains:
    - list of keyframes (image, keypoints, descriptors, pose)
    - sparse 3D map (list of points with positions and observed descriptors)
    - camera trajectory (list of poses)
- Provides:
    - VisualSLAM class with .process_frame(frame) to feed frames sequentially.
    - .draw_overlay(frame) to draw trajectory + sparse map on a frame for preview.
    - Configurable params (min_matches, keyframe_interval, reproj_thresh, etc).
Notes / Limitations:
- Monocular scale is arbitrary (we keep relative scale from triangulation).
- Robustness depends on feature quality and scene baseline.
- Not intended as production ORB-SLAM replacement; it's an educational yet practical core.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import time
from dataclasses import dataclass, field

# Try to reuse user's detector/matcher factories if available; else fall back to ORB
try:
    from feature_detection import create_detector, detect_and_compute
    from feature_matching import create_matcher, match_descriptors
except Exception:
    # fallback simple wrappers using OpenCV ORB / BF
    def create_detector(name='ORB'):
        if name.upper() == 'SIFT':
            return cv2.SIFT_create()
        else:
            return cv2.ORB_create(nfeatures=1500)

    def detect_and_compute(img, detector):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        return kp, desc

    def create_matcher(method, detector_name):
        if method.upper() == 'FLANN':
            # For ORB we must set appropriate FLANN params (Lsh)
            if detector_name.upper() == 'ORB':
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING if detector_name.upper()=='ORB' else cv2.NORM_L2, crossCheck=False)

    def match_descriptors(desc1, desc2, matcher, ratio_thresh=0.75):
        if desc1 is None or desc2 is None:
            return []
        try:
            raw = matcher.knnMatch(desc1, desc2, k=2)
        except Exception:
            # fallback: brute single matches
            return []
        good = []
        for m_n in raw:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        return good

@dataclass
class Keyframe:
    img: np.ndarray
    kps: List[cv2.KeyPoint]
    desc: np.ndarray
    pose: np.ndarray  # 4x4 pose (world <- camera) homogeneous transform
    tstamp: float = field(default_factory=time.time)

@dataclass
class MapPoint:
    pos: np.ndarray  # 3D position (world)
    descriptor: np.ndarray  # descriptor vector (from origin keyframe)
    observations: List[Tuple[int,int]] = field(default_factory=list)  # list of (kf_idx, kp_idx)

class VisualSLAM:
    def __init__(self,
                 detector_name: str = 'ORB',
                 matcher_name: str = 'BF',
                 min_init_matches: int = 100,
                 min_inliers_pose: int = 30,
                 keyframe_interval: int = 10,
                 reproj_threshold: float = 4.0,
                 max_map_points: int = 2000,
                 focal: Optional[float] = None,
                 pp: Optional[Tuple[float,float]] = None,
                 use_gpu: bool = False):
        """
        Params:
          detector_name: 'SIFT'|'ORB' etc. (used to create detector)
          matcher_name: 'BF'|'FLANN'
          min_init_matches: minimum matches to attempt initial motion estimation
          min_inliers_pose: minimum inliers to accept relative pose
          keyframe_interval: default distance (#frames) between keyframes
          reproj_threshold: px threshold for considering triangulated point valid
          focal, pp: camera intrinsics. If None, focal is set to 0.9*max(dim) and principal point = center.
        """
        self.det_name = detector_name
        self.matcher_name = matcher_name
        self.detector = create_detector(detector_name)
        self.matcher = create_matcher(matcher_name, detector_name)
        self.min_init_matches = min_init_matches
        self.min_inliers_pose = min_inliers_pose
        self.keyframe_interval = keyframe_interval
        self.reproj_threshold = reproj_threshold
        self.max_map_points = max_map_points

        self.frames_processed = 0
        self.keyframes: List[Keyframe] = []
        self.map_points: List[MapPoint] = []
        self.trajectory: List[np.ndarray] = []  # list of 4x4 poses
        self.last_kp = None
        self.last_desc = None
        self.last_img = None
        self.last_pose = np.eye(4, dtype=float)  # current estimated camera pose (world<-cam)
        self.initialized = False

        # intrinsics
        self.focal = focal
        self.pp = pp
        self._intrinsic_set = (focal is not None and pp is not None)
        self.use_gpu = use_gpu

    def _ensure_intrinsics(self, img):
        h,w = img.shape[:2]
        if not self._intrinsic_set:
            f = 0.9 * max(w,h)
            self.focal = f
            self.pp = (w/2.0, h/2.0)
            self._intrinsic_set = True
        K = np.array([[self.focal, 0.0, self.pp[0]],
                      [0.0, self.focal, self.pp[1]],
                      [0.0, 0.0, 1.0]], dtype=float)
        return K

    def reset(self):
        self.keyframes.clear()
        self.map_points.clear()
        self.trajectory.clear()
        self.frames_processed = 0
        self.last_kp = None
        self.last_desc = None
        self.last_img = None
        self.last_pose = np.eye(4, dtype=float)
        self.initialized = False

    def process_frame(self, img: np.ndarray) -> Dict:
        """
        Feed a new frame to SLAM. Returns dictionary with:
         - 'pose' : 4x4 numpy array (world <- cam)
         - 'inliers' : number of inliers in last motion estimate
         - 'map_points' : number of map points
         - 'status' : 'init'|'tracking'|'lost'
        Side effect: may add keyframes, triangulate new map points, update trajectory.
        """
        self.frames_processed += 1
        K = self._ensure_intrinsics(img)
        # detect
        kps, desc = detect_and_compute(img, self.detector)
        if desc is None or len(kps) < 20:
            # not enough features — tracking likely lost
            status = 'lost'
            return {'pose': self.last_pose.copy(), 'inliers': 0, 'map_points': len(self.map_points), 'status': status}

        # First frame: create initial keyframe
        if not self.initialized:
            kf = Keyframe(img.copy(), kps, desc, pose=np.eye(4, dtype=float))
            self.keyframes.append(kf)
            self.trajectory.append(np.eye(4, dtype=float))
            self.last_kp, self.last_desc, self.last_img = kps, desc, img.copy()
            self.initialized = True
            return {'pose': self.last_pose.copy(), 'inliers': 0, 'map_points': len(self.map_points), 'status': 'init'}

        # match with last frame/keyframe (we choose last keyframe for motion estimation)
        matches = match_descriptors(self.last_desc, desc, self.matcher, ratio_thresh=0.75)
        # build point correspondences
        if not matches or len(matches) < 8:
            # not enough matches
            status = 'lost'
            return {'pose': self.last_pose.copy(), 'inliers': 0, 'map_points': len(self.map_points), 'status': status}

        pts1 = np.array([ self.last_kp[m.queryIdx].pt for m in matches ], dtype=float)
        pts2 = np.array([ kps[m.trainIdx].pt for m in matches ], dtype=float)

        # find essential matrix + recover pose (assuming calibrated camera)
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            status='lost'
            return {'pose': self.last_pose.copy(), 'inliers': 0, 'map_points': len(self.map_points), 'status': 'lost'}

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=self.focal, pp=self.pp)
        # mask_pose is Nx1 of inliers used to recover pose
        inliers = int(np.sum(mask_pose)) if mask_pose is not None else 0

        # accept pose if enough inliers
        if inliers < max(self.min_inliers_pose, 8):
            status='lost'
            return {'pose': self.last_pose.copy(), 'inliers': inliers, 'map_points': len(self.map_points), 'status': 'lost'}

        # compose new pose: last_pose * [R|t] (camera-to-world)
        # careful with conventions: recoverPose returns R,t that map from pts1->pts2:
        # We maintain pose as world <- cam (T_wc). If last_pose is T_w_c_prev, and relative transform from prev to curr is T_prev_curr,
        # then T_w_c_curr = T_w_c_prev * T_prev_curr
        T_prev = self.last_pose.copy()
        T_rel = np.eye(4, dtype=float)
        T_rel[:3,:3] = R
        T_rel[:3,3] = t.ravel()
        T_curr = T_prev.dot(T_rel)   # world <- cam_curr
        self.last_pose = T_curr
        self.trajectory.append(T_curr)

        # Triangulate inlier correspondences to extend map (use only inlier indices from mask_pose)
        inlier_idx = np.where(mask_pose.ravel() > 0)[0] if mask_pose is not None else np.arange(len(pts1))
        if len(inlier_idx) >= 8:
            # Build projection matrices (camera coords)
            P0 = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
            # For curr frame, compose relative pose from prev frame
            P1 = K.dot(np.hstack((R, t)))
            # triangulate points in homogeneous coords
            pts1_in = pts1[inlier_idx].T
            pts2_in = pts2[inlier_idx].T
            pts4 = cv2.triangulatePoints(P0, P1, pts1_in, pts2_in)  # 4xN
            pts3 = (pts4[:3] / (pts4[3:]+1e-12)).T  # Nx3

            # Filter triangulated points by reprojection error and positive depth
            valid_mask = []
            valid_pts = []
            for i, pt3 in enumerate(pts3):
                # reproj to both cameras
                X = pt3.reshape(3,1)
                # check depth in front of both cameras
                z0 = X[2,0]
                # transform to curr camera:
                X_cam1 = (R.dot(X) + t.reshape(3,1))
                z1 = X_cam1[2,0]
                if z0 <= 0 or z1 <= 0:
                    valid_mask.append(False)
                    continue
                # reproject
                x0 = (P0.dot(np.vstack((X,1.0))))
                x1 = (P1.dot(np.vstack((X,1.0))))
                x0 = (x0[:2] / x0[2]).ravel()
                x1 = (x1[:2] / x1[2]).ravel()
                e0 = np.linalg.norm(x0 - pts1_in[:,i])
                e1 = np.linalg.norm(x1 - pts2_in[:,i])
                if e0 <= self.reproj_threshold and e1 <= self.reproj_threshold:
                    valid_mask.append(True)
                    valid_pts.append(pt3)
                else:
                    valid_mask.append(False)

            # add valid triangulated points to map
            for idx_local, ok in enumerate(valid_mask):
                if not ok:
                    continue
                gp = valid_pts[idx_local]
                # choose descriptor from the reference (last frame) for the map point
                mtch_idx = inlier_idx[idx_local]
                desc_vec = self.last_desc[matches[mtch_idx].queryIdx] if (self.last_desc is not None and len(self.last_desc)>matches[mtch_idx].queryIdx) else None
                mp = MapPoint(pos=gp.copy(), descriptor=(desc_vec.copy() if desc_vec is not None else None),
                              observations=[(len(self.keyframes)-1, matches[mtch_idx].queryIdx),
                                            (len(self.keyframes), matches[mtch_idx].trainIdx)])
                self.map_points.append(mp)

            # cap map size
            if len(self.map_points) > self.max_map_points:
                # simple pruning: keep most recently added points
                self.map_points = self.map_points[-self.max_map_points:]

        # Keyframe decision: add every keyframe_interval frames or if motion is large
        add_kf = False
        if (len(self.keyframes) == 0) or (self.frames_processed % self.keyframe_interval == 0):
            add_kf = True
        else:
            # measure baseline magnitude (translation length) to decide if new keyframe needed
            prev_pos = self.keyframes[-1].pose[:3,3]
            curr_pos = T_curr[:3,3]
            baseline = np.linalg.norm(curr_pos - prev_pos)
            if baseline > 0.05 * (self.focal or 1.0):  # heuristic relative threshold
                add_kf = True

        if add_kf:
            kf = Keyframe(img.copy(), kps, desc, pose=T_curr.copy())
            self.keyframes.append(kf)

        # update last frame data
        self.last_kp = kps
        self.last_desc = desc
        self.last_img = img.copy()

        status = 'tracking'
        return {'pose': self.last_pose.copy(), 'inliers': inliers, 'map_points': len(self.map_points), 'status': status}

    def draw_overlay(self, frame: np.ndarray, size: Tuple[int,int]=(300,200)) -> np.ndarray:
        """
        Draw a compact overlay showing:
         - sparse map points projected into a small 3D canvas (top-down or simple XY)
         - camera trajectory as a polyline
        Returns annotated frame (copy).
        """
        out = frame.copy()
        h,w = frame.shape[:2]
        # Draw sparse 2D reprojection of recent map points onto frame (quick)
        # Project world map points into current camera to highlight visible points
        K = self._ensure_intrinsics(frame)
        R_wc = self.last_pose[:3,:3]
        t_wc = self.last_pose[:3,3].reshape(3,1)
        # compute camera-to-world inverse: world <- cam is self.last_pose, cam<-world is inverse
        Twc = self.last_pose
        try:
            Twc_inv = np.linalg.inv(Twc)
        except Exception:
            Twc_inv = np.eye(4)
        # project map points
        drawn = 0
        for mp in self.map_points[-500:]:
            Xw = mp.pos.reshape(3,1)
            # transform to camera coords
            Xcam = Twc_inv[:3,:3].dot(Xw) + Twc_inv[:3,3].reshape(3,1)
            if Xcam[2,0] <= 0:
                continue
            x = K.dot(Xcam)
            x = (x[:2] / x[2]).ravel().astype(int)
            if 0 <= x[0] < w and 0 <= x[1] < h:
                cv2.circle(out, (int(x[0]), int(x[1])), 2, (0,255,0), -1)
                drawn += 1
        # draw camera trajectory on a small corner
        box_w, box_h = size
        overlay = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        # compute projected 2D trajectory: use X,Z or X,Y axis
        traj_pts = []
        for T in self.trajectory[-200:]:
            pos = T[:3,3]
            traj_pts.append(pos)
        if len(traj_pts) >= 2:
            pts = np.array(traj_pts)
            # normalize into overlay coordinates using X and Z axes
            xs = pts[:,0]
            zs = pts[:,2]
            # flip sign to have forward as up on mini map
            xs_n = (xs - xs.min()) if xs.max()!=xs.min() else xs-xs.min()
            zs_n = (zs - zs.min()) if zs.max()!=zs.min() else zs-zs.min()
            if xs_n.max() > 0:
                xs_n = (xs_n / xs_n.max() * (box_w-8))
            if zs_n.max() > 0:
                zs_n = (zs_n / zs_n.max() * (box_h-8))
            for i in range(len(xs_n)-1):
                p0 = (int(xs_n[i]) + 4, box_h - int(zs_n[i]) - 4)
                p1 = (int(xs_n[i+1]) + 4, box_h - int(zs_n[i+1]) - 4)
                cv2.line(overlay, p0, p1, (255,255,255), 1)
        # place overlay on out at top-left
        oh, ow = overlay.shape[:2]
        out[4:4+oh, 4:4+ow] = overlay
        # put info text
        cv2.putText(out, f"MPs:{len(self.map_points)} KF:{len(self.keyframes)}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,0), 1)
        return out

# Minimal test:
if __name__ == '__main__':
    print("slam.py loaded. VisualSLAM class available.")
