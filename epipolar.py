# epipolar.py
"""
Epipolar geometry helper module.

Provides:
  - EpipolarProcessor: compute F/E from two images (or self-matches), find inliers, triangulate optional.
  - draw_epilines: draw epipolar lines for points on the other image.
  - utilities to return annotated images and diagnostic numbers.

Design goals:
  - Reuse user's detector/matcher if available (create_detector, detect_and_compute, create_matcher, match_descriptors).
  - Robust defaults, uses cv2.findFundamentalMat with RANSAC.
  - Minimal external dependencies (only numpy & cv2).
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import cv2

# try to reuse project helpers; fallback to simple implementations
try:
    from feature_detection import create_detector, detect_and_compute
    from feature_matching import create_matcher, match_descriptors
except Exception:
    def create_detector(name='ORB'):
        if name.upper() == 'SIFT' and hasattr(cv2, 'SIFT_create'):
            return cv2.SIFT_create()
        return cv2.ORB_create(nfeatures=1500)
    def detect_and_compute(img, detector):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        return kp, desc
    def create_matcher(method, detector_name):
        if method.upper() == 'FLANN':
            return cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=50))
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING if detector_name.upper() == 'ORB' else cv2.NORM_L2, crossCheck=False)
    def match_descriptors(d1, d2, matcher, ratio_thresh=0.75):
        if d1 is None or d2 is None:
            return []
        try:
            knn = matcher.knnMatch(d1, d2, k=2)
        except Exception:
            return []
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        return good

def _kp_to_pts(kps, idxs):
    """Return Nx2 pts array given keypoints and list of indices (or keypoint objects)."""
    pts = []
    for i in idxs:
        if isinstance(i, int):
            kp = kps[i]
        else:
            kp = i
        pts.append(kp.pt)
    return np.array(pts, dtype=float)

def draw_epilines(img, lines, pts, color=(0,255,0), thickness=1):
    """
    Draw epilines on image.
    lines: Nx3 epiline parameters (a,b,c) for ax + by + c = 0
    pts: Nx2 corresponding points (for marking)
    """
    img_out = img.copy()
    h,w = img.shape[:2]
    for r, p in zip(lines, pts):
        a,b,c = r.ravel()
        # line endpoints
        x0,y0 = 0, int(-c/b) if b!=0 else 0
        x1,y1 = w, int((-c - a*w)/b) if b!=0 else h
        cv2.line(img_out, (x0,y0), (x1,y1), color, thickness)
        cv2.circle(img_out, (int(p[0]), int(p[1])), 4, color, -1)
    return img_out

class EpipolarProcessor:
    def __init__(self, detector_name='ORB', matcher_name='BF', ratio_thresh=0.75, ransac_thresh=1.0, ransac_conf=0.99):
        self.detector_name = detector_name
        self.matcher_name = matcher_name
        self.detector = create_detector(detector_name)
        self.matcher = create_matcher(matcher_name, detector_name)
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.ransac_conf = ransac_conf

    def compute_matches(self, img1, img2):
        """
        Detect and match features between img1 and img2.
        Returns kps1, kps2, matches (list of cv2.DMatch).
        """
        kps1, desc1 = detect_and_compute(img1, self.detector)
        kps2, desc2 = detect_and_compute(img2, self.detector)
        if desc1 is None or desc2 is None or len(kps1) < 4 or len(kps2) < 4:
            return kps1, kps2, []
        matches = match_descriptors(desc1, desc2, self.matcher, ratio_thresh=self.ratio_thresh)
        return kps1, kps2, matches

    def estimate_fundamental(self, kps1, kps2, matches) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
        """
        Estimate fundamental matrix using matched keypoints + RANSAC.
        Returns (F, mask, inlier_idx_list)
        """
        if not matches or len(matches) < 8:
            return None, None, []
        pts1 = np.array([kps1[m.queryIdx].pt for m in matches], dtype=float)
        pts2 = np.array([kps2[m.trainIdx].pt for m in matches], dtype=float)
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=self.ransac_thresh, confidence=self.ransac_conf)
        if F is None or mask is None:
            return None, None, []
        inlier_idx = [i for i,m in enumerate(mask.ravel()) if m==1]
        return F, mask, inlier_idx

    def estimate_essential(self, F, K) -> Optional[np.ndarray]:
        """
        E = K^T * F * K
        """
        if F is None or K is None:
            return None
        E = K.T.dot(F).dot(K)
        # enforce rank-2 constraint (optional)
        U,S,Vt = np.linalg.svd(E)
        S[2] = 0
        E2 = U.dot(np.diag(S)).dot(Vt)
        return E2

    def compute_epilines_and_annotate(self, img1, img2, kps1, kps2, matches, F, inlier_idx):
        """
        Compute epilines for inlier matches and return annotated images:
          - img1_lines: img1 with epilines computed from pts2 (lines for pts1)
          - img2_lines: img2 with epilines computed from pts1 (lines for pts2)
        Also returns diagnostic dict.
        """
        out1 = img1.copy()
        out2 = img2.copy()
        if F is None or len(inlier_idx)==0:
            return out1, out2, {}
        pts1 = np.array([kps1[matches[i].queryIdx].pt for i in inlier_idx], dtype=float)
        pts2 = np.array([kps2[matches[i].trainIdx].pt for i in inlier_idx], dtype=float)
        # compute epilines in image2 for points in image1: lines2 = F * [x1]
        # OpenCV computeCorrespondEpilines expects points as Nx1x2
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
        out2 = draw_epilines(out2, lines2, pts2, color=(0,200,255), thickness=1)
        out1 = draw_epilines(out1, lines1, pts1, color=(0,200,255), thickness=1)
        diag = {'n_matches': len(matches), 'n_inliers': len(inlier_idx)}
        return out1, out2, diag

    def process_pair(self, img1, img2, K: Optional[np.ndarray] = None) -> Dict:
        """
        High-level: runs detection, matching, F/E estimation, epiline drawing.
        Returns:
          {
            'img1_annotated': BGR image,
            'img2_annotated': BGR image,
            'F': F matrix or None,
            'E': E matrix or None,
            'matches': matches,
            'inlier_mask': mask or None,
            'inlier_idx': list of inlier indices,
            'diagnostics': dict
          }
        """
        kps1, kps2, matches = self.compute_matches(img1, img2)
        if not matches:
            return {'img1_annotated': img1.copy(), 'img2_annotated': img2.copy(),
                    'F': None, 'E': None, 'matches': matches, 'inlier_mask': None, 'inlier_idx': [],
                    'diagnostics': {'n_matches': 0}}
        F, mask, inlier_idx = self.estimate_fundamental(kps1, kps2, matches)
        E = None
        if F is not None and K is not None:
            try:
                E = self.estimate_essential(F, K)
            except Exception:
                E = None
        img1a, img2a, diag = self.compute_epilines_and_annotate(img1, img2, kps1, kps2, matches, F, inlier_idx)
        diag.update({'n_matches': len(matches), 'n_inliers': len(inlier_idx)})
        return {
            'img1_annotated': img1a,
            'img2_annotated': img2a,
            'F': F,
            'E': E,
            'matches': matches,
            'inlier_mask': mask,
            'inlier_idx': inlier_idx,
            'diagnostics': diag,
            'kps1': kps1,
            'kps2': kps2
        }

# Quick internal test (runs only if executed directly)
if __name__ == '__main__':
    print("epipolar.py loaded. Use EpipolarProcessor to compute F/E and epilines.")
