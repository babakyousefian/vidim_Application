# feature_matching.py
import cv2
import numpy as np
from typing import List

def create_matcher(method='BF', descriptor_type='SIFT'):
    """
    Create a matcher: 'BF' (Brute-Force) or 'FLANN'
    descriptor_type: 'SIFT' or 'ORB' used to choose norm
    """
    method = method.upper()
    descriptor_type = descriptor_type.upper()
    if method == 'BF':
        norm = cv2.NORM_L2 if descriptor_type in ('SIFT', 'SURF') else cv2.NORM_HAMMING
        return cv2.BFMatcher(norm, crossCheck=False)
    elif method == 'FLANN':
        if descriptor_type in ('SIFT', 'SURF'):
            index_params = dict(algorithm=1, trees=5)  # KDTree
            search_params = dict(checks=50)
        else:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict()
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f'Unknown matcher: {method}')

def ratio_test_matches(knn_matches, ratio=0.75):
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def match_descriptors(desc1, desc2, matcher, ratio_thresh=0.75):
    """
    desc1: descriptors of image1
    desc2: descriptors of image2
    returns list of cv2.DMatch (filtered by ratio test)
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    # For BFMatcher with crossCheck=False, we use knnMatch and ratio test
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = ratio_test_matches(knn, ratio_thresh)
    return good
