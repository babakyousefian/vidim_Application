# feature_detection.py
import cv2
from typing import Tuple, List

def create_detector(method='SIFT'):
    """
    Create keypoint detector+descriptor.
    method: 'SIFT' or 'ORB'
    """
    method = method.upper()
    if method == 'SIFT':
        # OpenCV 4.12 SIFT_create is available
        try:
            return cv2.SIFT_create()
        except Exception:
            # Fallback to xfeatures if not present
            return cv2.xfeatures2d.SIFT_create()
    elif method == 'ORB':
        return cv2.ORB_create(nfeatures=2000)
    else:
        raise ValueError(f'Unknown detector: {method}')

def detect_and_compute(img, detector) -> Tuple[List[cv2.KeyPoint], 'np.ndarray']:
    """
    Detect keypoints and compute descriptors on BGR image.
    Returns (keypoints, descriptors)
    """
    if img is None:
        return [], None
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, desc = detector.detectAndCompute(gray, None)
    return kps, desc
