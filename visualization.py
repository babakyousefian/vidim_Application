# visualization.py
import cv2
import numpy as np
from utils import COLOR_INLIER, COLOR_OUTLIER, COLOR_MATCH_LINE, draw_text

def draw_keypoints(img, kps, color=(0,255,0), radius=2, max_kp=None):
    out = img.copy()
    if max_kp is None:
        max_kp = len(kps)
    for i, kp in enumerate(kps[:max_kp]):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(out, (x,y), radius, color, -1)
    return out

def draw_matches_lines(img, pts1, pts2, inlier_mask=None, max_draw=500):
    """
    Draw lines from pts1 to pts2 on the same image (pts2 coordinates must be in same image coords).
    If pts2 is in other image, use drawMatches in combined canvas.
    """
    out = img.copy()
    N = min(len(pts1), max_draw)
    for i in range(N):
        x1, y1 = int(round(pts1[i][0])), int(round(pts1[i][1]))
        x2, y2 = int(round(pts2[i][0])), int(round(pts2[i][1]))
        color_line = COLOR_MATCH_LINE
        color_pt = COLOR_INLIER if (inlier_mask is not None and inlier_mask[i]) else COLOR_OUTLIER
        cv2.circle(out, (x1,y1), 3, color_pt, -1)
        cv2.line(out, (x1,y1), (x2,y2), color_line, 1)
    return out

def draw_inliers_only(img, pts, inlier_mask, radius=3):
    out = img.copy()
    for (x,y), inl in zip(pts, inlier_mask):
        c = COLOR_INLIER if inl else COLOR_OUTLIER
        cv2.circle(out, (int(round(x)), int(round(y))), radius, c, -1)
    return out

def draw_bounding_boxes(img, detections, class_names=None):
    """
    detections: list of tuples (box, classID, conf) where box = [x,y,w,h]
    """
    out = img.copy()
    for box, cls_id, conf in detections:
        x,y,w,h = box
        x1, y1 = int(x), int(y)
        x2, y2 = int(x+w), int(y+h)
        color = (255, 165, 0)  # orange
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        label = f"{class_names[cls_id] if class_names is not None else cls_id}: {conf:.2f}"
        cv2.putText(out, label, (x1, max(y1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out
