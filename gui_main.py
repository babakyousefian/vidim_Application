import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# در قسمت imports (با دیگر importها):
from epipolar import EpipolarProcessor


# near other imports
from slam import VisualSLAM

# gui_main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox , Canvas , PhotoImage
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
import os
import time
import json
import hashlib
import pygame
from datetime import datetime

# Import processing modules (assumed present in your project)
from feature_detection import create_detector, detect_and_compute
from feature_matching import create_matcher, match_descriptors
from model_estimation import estimate_homography, estimate_fundamental, reprojection_error_homography, sampson_error
from ransac import ransac_generic
from rransac import rransac_main
from visualization import draw_keypoints, draw_inliers_only, draw_matches_lines, draw_bounding_boxes
from object_detection import YOLODetector, load_yolov4_tiny, yolo_v4_tiny_detect, load_coco_names, ULTRALYTICS_AVAILABLE

# utils
from utils import draw_text, COLOR_INLIER, COLOR_OUTLIER, COLOR_MATCH_LINE

# Default params
DEFAULT_DETECTOR = 'SIFT'
DEFAULT_MATCHER = 'BF'
DEFAULT_MODEL = 'homography'
DEFAULT_PRETEST = 'Td'  # 'Td' or 'SPRT' or None

# output & user directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
USERS_FILE = os.path.join(BASE_DIR, 'users.json')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -------------------------
# Simple user utilities
# -------------------------
SALT = "vidim_salt_v1"

def hash_password(password: str) -> str:
    return hashlib.sha256((SALT + password).encode('utf-8')).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def ensure_user_output_dir(username):
    d = os.path.join(OUTPUTS_DIR, username)
    os.makedirs(d, exist_ok=True)
    return d

def user_index_file(username):
    return os.path.join(ensure_user_output_dir(username), 'index.json')

def append_user_index(username, entry):
    idxf = user_index_file(username)
    arr = []
    if os.path.exists(idxf):
        try:
            with open(idxf, 'r', encoding='utf-8') as f:
                arr = json.load(f)
        except Exception:
            arr = []
    arr.append(entry)
    with open(idxf, 'w', encoding='utf-8') as f:
        json.dump(arr, f, indent=2, ensure_ascii=False)

def read_user_index(username):
    idxf = user_index_file(username)
    if not os.path.exists(idxf):
        return []
    try:
        with open(idxf, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

from tkinter import Button, PhotoImage , ttk

class MusicToggle:
    def __init__(self, root, music_file="assets/sounds/background.ogg"):
        self.root = root
        self.music_file = music_file
        self.is_playing = True

        pygame.mixer.init()
        pygame.mixer.music.load(self.music_file)
        pygame.mixer.music.play(-1)

        from tkinter import Button, PhotoImage
        import platform

        # رنگ پس‌زمینه متناسب با سیستم
        bg_color = "#f0f0f0" if platform.system() == "Windows" else "#d9d9d9"

        self.btn_img = PhotoImage(width=40, height=20)  # تصویر دلخواه
        self.button = Button(root, image=self.btn_img, text="Mute", compound="center",
                             fg="#111111", font=("Arial", 10, "bold"),
                             bd=2, relief="raised",  # گوشه کمی برجسته
                             bg=bg_color, activebackground="#e0e0e0",  # رنگ وقتی کلیک می‌کنه
                             command=self.toggle_music)
        self.button.place(relx=1.0, rely=1.0, x=-70, y=-70, anchor="se")

    def toggle_music(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.button.config(text="Unmute")
        else:
            pygame.mixer.music.unpause()
            self.button.config(text="Mute")
        self.is_playing = not self.is_playing


# -------------------------
# App class
# -------------------------
class App:
    def __init__(self, root):
        # user state
        self.logged_user = None           # when logged in -> username
        self.logged_fullname = None       # full name for header display

        # anonymous usage counters (per app session)
        self.anon_images_processed = 0
        self.anon_videos_processed = 0
        self.ANON_IMAGE_LIMIT = 1
        self.ANON_VIDEO_LIMIT = 1

        self.root = root
        root.title("Vidim Processor — R-RANSAC + SPRT (CPU)")
        root.geometry("1200x820")
        # placeholders
        self.img = None
        self.orig_img = None
        self.kps1 = None
        self.desc1 = None
        self.detector = create_detector(DEFAULT_DETECTOR)
        self.matcher = create_matcher(DEFAULT_MATCHER, DEFAULT_DETECTOR)
        # YOLO (lazy init)
        self.yolo_detector = None
        # video placeholders
        self.video_path = None
        self.video_cap = None
        self.video_playing = False
        self.video_preview_thread = None
        self.video_processing_thread = None
        self.last_frame = None
        self.warned_image_limit = False
        self.warned_video_limit = False
        # ad / promo state
        self.ad_playing = False          # whether ad thread is playing
        self.ad_dismissed = False        # user closed ad (remain dismissed until login)
        self.ad_video_path = os.path.join(BASE_DIR, 'assets', 'ad', 'ad.mp4')  # change path if needed
        self._ad_thread = None
        self._ad_stop_event = None
        self._ad_frame_image = None      # keep reference to Tk image to avoid GC
        self._ad_close_btn = None        # reference to close button widget

        # coco names (if needed)
        coco_path = os.path.join(BASE_DIR, 'assets', 'coco.names')
        try:
            self.coco_names = load_coco_names(coco_path)
        except Exception:
            self.coco_names = None

        # build UI
        self.build_ui()

        # load and wire sound files (uses pygame exclusively)
        self._install_sound_files()

        # پس از load sounds:
        # start background music (autoplay loop). Will silently no-op if pygame or background file missing.
        try:
            self._start_background_music(os.path.join(BASE_DIR, 'assets', 'sounds', 'background.ogg'))
        except Exception:
            pass
        # apply initial anonymous mode (buttons state)
        self.update_access_controls()

    # ---------------------------
    # Sound system (pygame-based)
    # Replaces any other backends (winsound/playsound/simpleaudio/etc.)
    # ---------------------------
    def _install_sound_files(self, sounds_dir=None):
        """
        Uses pygame to load sounds. Exposes:
          - self.play_sound(kind)  where kind in {'click','notify','error','done','login','dashboard'}
          - convenience lambdas: self._play_click(), _play_notify(), etc.
        Requires pygame installed (you said you already have it). If pygame import fails,
        functions become no-ops (silently).
        This implementation suppresses pygame start-up prompt and limits terminal noise.
        """
        import os, warnings, threading

        # suppress pygame support prompt
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

        # suppress warnings if desired (keeps terminal cleaner)
        warnings.filterwarnings("ignore", category=Warning)

        # expected keys -> filenames
        expected = {
            'click': 'click.wav',
            'notify': 'notify.wav',
            'error': 'error.wav',
            'done': 'done.wav',
            'login': 'login.wav',
            'dashboard': 'dashboard.wav',
            'background': 'background.wav',  # optional background track
        }

        if sounds_dir is None:
            sounds_dir = os.path.join(BASE_DIR, 'assets', 'sounds')

        # build map of available files (full paths)
        sound_paths = {}
        for k, fname in expected.items():
            p = os.path.join(sounds_dir, fname)
            if os.path.exists(p):
                sound_paths[k] = p
            else:
                # accept mp3 alternative
                p_mp3 = os.path.join(sounds_dir, os.path.splitext(fname)[0] + '.mp3')
                if os.path.exists(p_mp3):
                    sound_paths[k] = p_mp3

        # try import pygame and init mixer only
        try:
            import pygame
            # Initialize pygame mixer only (no video) and mute its internal prints
            try:
                pygame.mixer.init()
            except Exception:
                # some systems may require specific settings; try fallback with small parameters
                try:
                    pygame.mixer.quit()
                    pygame.mixer.init(frequency=44100, size=-16, channels=2)
                except Exception:
                    # if mixer fails, mark pygame unavailable
                    raise
            self._pygame = pygame
        except Exception:
            # pygame not available or mixer failed: provide no-op play_sound
            self._pygame = None

        # storage for loaded pygame.mixer.Sound objects
        self._sound_objs = {}

        if self._pygame is not None:
            pygame = self._pygame
            # try to load short sounds as Sound objects
            for k, path in sound_paths.items():
                if k == 'background':
                    # background track handled via music module, don't load as Sound
                    continue
                try:
                    snd = pygame.mixer.Sound(path)
                    self._sound_objs[k] = snd
                except Exception:
                    # skip load failures silently
                    pass

            # background path if exists
            self._background_path = sound_paths.get('background', None)
        else:
            self._background_path = sound_paths.get('background', None)

        # public play_sound(kind) function
        def play_sound(kind):
            # run in background thread so UI doesn't block (short sounds)
            def _play():
                try:
                    if self._pygame is None:
                        # fallback: no-op
                        return
                    pygame = self._pygame
                    # if short sound loaded as Sound, play non-blocking
                    snd = self._sound_objs.get(kind)
                    if snd is not None:
                        try:
                            snd.play()
                            return
                        except Exception:
                            pass
                    # fallback: try music (for longer files) - note music is single-channel
                    path = None
                    if kind in sound_paths:
                        path = sound_paths.get(kind)
                    if path is not None:
                        try:
                            # music.play is non-blocking; use play once for notification sounds
                            pygame.mixer.music.load(path)
                            pygame.mixer.music.play(0)
                            return
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                t = threading.Thread(target=_play, daemon=True)
                t.start()
            except Exception:
                pass

        # attach to instance
        self.play_sound = play_sound

        # convenience helpers
        self._play_click = lambda: self.play_sound('click')
        self._play_notify = lambda: self.play_sound('notify')
        self._play_error = lambda: self.play_sound('error')
        self._play_done = lambda: self.play_sound('done')
        self._play_login = lambda: self.play_sound('login')
        self._play_dashboard = lambda: self.play_sound('dashboard')

        # wrap messagebox functions to auto-play sounds for notify/error
        self._orig_showinfo = messagebox.showinfo
        self._orig_showerror = messagebox.showerror

        def _wrapped_showinfo(title, message, *args, **kwargs):
            try:
                self._play_notify()
            except Exception:
                pass
            return self._orig_showinfo(title, message, *args, **kwargs)

        def _wrapped_showerror(title, message, *args, **kwargs):
            try:
                self._play_error()
            except Exception:
                pass
            return self._orig_showerror(title, message, *args, **kwargs)

        messagebox.showinfo = _wrapped_showinfo
        messagebox.showerror = _wrapped_showerror

        # bind click sound to main UI buttons if present (non-blocking)
        btn_names = [
            'btn_load_img', 'btn_run_img', 'btn_load_vid', 'btn_run_vid',
            'btn_preview_play', 'btn_preview_stop', 'dashboard_btn', 'login_btn'
        ]
        for name in btn_names:
            try:
                btn = getattr(self, name, None)
                if btn is not None:
                    # use <Button-1> so sound plays on press
                    btn.bind('<Button-1>', lambda ev, f=self._play_click: f())
            except Exception:
                pass

    def _start_background_music(self, filename=None):
        """
        Start autoplay-loop background music using pygame.mixer.music.
        If pygame is unavailable or no background file is present, this is a no-op.
        """
        import threading, os

        if getattr(self, '_bg_music_running', False):
            return

        # pick file
        if filename is None:
            filename = getattr(self, '_background_path', None)
        if not filename:
            return

        if self._pygame is None:
            # pygame not initialized; no-op
            return

        pygame = self._pygame

        # ensure music stop flag
        self._bg_stop_event = threading.Event()

        def _bg_loop():
            try:
                while not self._bg_stop_event.is_set():
                    try:
                        # load and play loop once; we then wait until it finishes,
                        # but pygame.mixer.music.play(-1) loops forever; we prefer that,
                        # and just wait on event.
                        pygame.mixer.music.load(filename)
                        pygame.mixer.music.play(-1)  # loop indefinitely
                        # wait until stop requested
                        while not self._bg_stop_event.is_set():
                            # sleep in chunk so we respond to stop quickly
                            self._bg_stop_event.wait(0.2)
                            if self._bg_stop_event.is_set():
                                break
                        # stop playback
                        try:
                            pygame.mixer.music.stop()
                        except Exception:
                            pass
                        break
                    except Exception:
                        # if load/play failed, break out
                        break
            finally:
                self._bg_music_running = False

        t = threading.Thread(target=_bg_loop, daemon=True)
        self._bg_music_thread = t
        self._bg_music_running = True
        t.start()

    def _stop_background_music(self, wait_timeout=1.0):
        """
        Signal background music to stop and cleanup pygame mixer.
        """
        try:
            if hasattr(self, '_bg_stop_event') and self._bg_stop_event is not None:
                self._bg_stop_event.set()
            # stop music immediately if possible
            if self._pygame is not None:
                try:
                    self._pygame.mixer.music.stop()
                except Exception:
                    pass
                try:
                    # unload mixer resources
                    self._pygame.mixer.quit()
                except Exception:
                    pass
            # join thread briefly
            thr = getattr(self, '_bg_music_thread', None)
            if thr is not None and thr.is_alive():
                try:
                    thr.join(timeout=wait_timeout)
                except Exception:
                    pass
        except Exception:
            pass

    # ---------------------------
    # Ad / promo playback (in input preview area)
    # ---------------------------

    def _start_ad(self, filename=None):
        """
        Robust ad player for input preview area. Tries multiple OpenCV backends.
        Loops the video until stopped. If cannot open video, shows poster or message.
        """
        import cv2, threading, time

        if self.logged_user:
            return
        if self.ad_dismissed:
            return
        if self.ad_playing:
            return

        if filename is None:
            filename = getattr(self, 'ad_video_path', os.path.join(BASE_DIR, 'assets', 'ad', 'ad.mp4'))

        # if file missing -> show poster/fallback and return
        if not os.path.exists(filename):
            # show poster if exists
            poster = os.path.join(BASE_DIR, 'assets', 'ad', 'poster.png')
            if os.path.exists(poster):
                try:
                    pimg = Image.open(poster)
                    pimg = pimg.copy()
                    # scale similarly to show_input_image
                    w,h = pimg.size
                    max_w, max_h = 900, 520
                    scale = min(1.0, max_w / w, max_h / h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    pimg = pimg.resize((new_w, new_h), Image.LANCZOS)
                    tkimg = ImageTk.PhotoImage(pimg)
                    self._ad_frame_image = tkimg
                    self.input_label.configure(image=tkimg)
                    self.input_label.image = tkimg
                except Exception:
                    self.status_text.set("Ad file missing and poster failed.")
            else:
                self.status_text.set("Ad file missing.")
            return

        # try multiple backends to open video (order may help on Windows)
        backends_to_try = []
        try:
            backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        except Exception:
            backends_to_try = [cv2.CAP_ANY]

        cap = None
        for b in backends_to_try:
            try:
                cap = cv2.VideoCapture(filename, int(b))
                if cap is not None and cap.isOpened():
                    break
                else:
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
                    cap = None
            except Exception:
                cap = None

        # final attempt without backend param
        if cap is None:
            try:
                cap = cv2.VideoCapture(filename)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            except Exception:
                cap = None

        if cap is None or not cap.isOpened():
            # cannot open video -> show poster fallback
            poster = os.path.join(BASE_DIR, 'assets', 'ad', 'poster.png')
            if os.path.exists(poster):
                try:
                    pimg = Image.open(poster).copy()
                    w,h = pimg.size
                    max_w, max_h = 900, 520
                    scale = min(1.0, max_w / w, max_h / h)
                    pimg = pimg.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                    tkimg = ImageTk.PhotoImage(pimg)
                    self._ad_frame_image = tkimg
                    self.input_label.configure(image=tkimg)
                    self.input_label.image = tkimg
                    self.status_text.set("Cannot open ad video; showing poster.")
                except Exception:
                    self.status_text.set("Cannot open ad video.")
            else:
                self.status_text.set("Cannot open ad video.")
            return

        # At this point cap is opened; start thread loop
        self._ad_stop_event = threading.Event()

        def _ad_worker():
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                delay = 1.0 / max(1.0, fps)
                self.ad_playing = True
                while not self._ad_stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        # loop video by resetting position
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except Exception:
                            pass
                        time.sleep(0.01)
                        continue
                    # convert frame -> PIL -> PhotoImage as in show_input_image
                    try:
                        h, w = frame.shape[:2]
                        max_w, max_h = 900, 520
                        scale = min(1.0, max_w / w, max_h / h)
                        disp = cv2.resize(frame, (int(w*scale), int(h*scale)))
                        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        tkimg = ImageTk.PhotoImage(pil)
                    except Exception:
                        # on error skip frame
                        if self._ad_stop_event.wait(delay):
                            break
                        else:
                            continue

                    # update UI on main thread
                    def _set_frame(img=tkimg):
                        try:
                            self._ad_frame_image = img
                            self.input_label.configure(image=img)
                            self.input_label.image = img
                        except Exception:
                            pass
                    try:
                        self.root.after(0, _set_frame)
                    except Exception:
                        pass

                    # wait respecting fps but allow stop event to break early
                    if self._ad_stop_event.wait(delay):
                        break
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
                self.ad_playing = False

        self._ad_thread = threading.Thread(target=_ad_worker, daemon=True)
        self._ad_thread.start()

        # show close button overlay
        self._show_ad_close_button()


    def _stop_ad(self):
        """Stop ad thread and remove overlay button."""
        try:
            if hasattr(self, '_ad_stop_event') and self._ad_stop_event is not None:
                self._ad_stop_event.set()
        except Exception:
            pass
        try:
            if hasattr(self, '_ad_thread') and self._ad_thread is not None and self._ad_thread.is_alive():
                try:
                    self._ad_thread.join(timeout=0.5)
                except Exception:
                    pass
        except Exception:
            pass
        self.ad_playing = False
        # remove close button if present
        try:
            if getattr(self, '_ad_close_btn', None) is not None:
                try:
                    self._ad_close_btn.destroy()
                except Exception:
                    pass
                self._ad_close_btn = None
        except Exception:
            pass


    def _show_ad_close_button(self):
        """Place a small Close Ad button in top-right of input_frame (only one)."""
        try:
            if getattr(self, '_ad_close_btn', None) is not None:
                return
            btn = tk.Button(self.input_frame, text="Close Ad", command=self._on_close_ad, bg="#111", fg="white", bd=1)
            # place inside input_frame near top-right
            btn.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")
            self._ad_close_btn = btn
        except Exception:
            pass


    def _on_close_ad(self):
        """User clicked close: dismiss ad until login."""
        try:
            self.ad_dismissed = True
            self._stop_ad()
            # clear preview area (optional: show a neutral placeholder)
            try:
                self.input_label.configure(image='')
                self.input_label.image = None
            except Exception:
                pass
            self.status_text.set("Ad closed. Login to continue using the app.")
        except Exception:
            pass


    # ---------------------------
    # BUILD UI (header + main)
    # ---------------------------
    def build_ui(self):
        # ===========================
        # HEADER (Top Application Bar)
        # ===========================
        self.header_frame = tk.Frame(self.root, height=56, bg="#3e0f5f")
        self.header_frame.pack(side="top", fill="x")

        # App Name (left)
        self.app_title = tk.Label(
            self.header_frame,
            text="Vidim Processor",
            bg="#3e0f5f",
            fg="white",
            font=("Arial", 16, "bold")
        )
        self.app_title.pack(side="left", padx=16)

        # Dashboard button (only enabled when logged in)
        self.dashboard_btn = tk.Button(
            self.header_frame,
            text="Dashboard",
            bg="#2f1b4f",
            fg="white",
            font=("Arial", 11),
            state="disabled",  # initially disabled
            command=self.open_dashboard
        )
        self.dashboard_btn.pack(side="right", padx=10, pady=8)

        # Login / Logout button
        self.login_btn = tk.Button(
            self.header_frame,
            text="Login",
            bg="#6a1b9a",
            fg="white",
            font=("Arial", 12),
            command=self.open_login_window
        )
        self.login_btn.pack(side="right", padx=10, pady=8)

        # -----------------------------------------
        # Left control frame (unchanged logic)
        # -----------------------------------------
        ctrl = tk.Frame(self.root, width=320, bg='#2b0f40')
        ctrl.pack(side='left', fill='y')

        header = tk.Label(ctrl, text="R-RANSAC + SPRT Toolkit", bg='#4b1b6b', fg='white', font=('Arial', 14, 'bold'), pady=10)
        header.pack(fill='x', padx=8, pady=8)

        # Buttons: image
        self.btn_load_img = tk.Button(ctrl, text="Load Image", command=self.load_image)
        self.btn_load_img.pack(pady=6, padx=8, fill='x')
        self.btn_run_img = tk.Button(ctrl, text="Run Full Pipeline (Image)", command=self.run_pipeline_thread)
        self.btn_run_img.pack(pady=6, padx=8, fill='x')

        # Buttons: video
        self.btn_load_vid = tk.Button(ctrl, text="Load Video", command=self.load_video)
        self.btn_load_vid.pack(pady=6, padx=8, fill='x')
        self.btn_run_vid = tk.Button(ctrl, text="Run Video Pipeline", command=self.run_video_pipeline_thread)
        self.btn_run_vid.pack(pady=6, padx=8, fill='x')

        # Detector selection
        tk.Label(ctrl, text="Feature Detector", bg='#2b0f40', fg='white').pack(pady=(12,0))
        self.detector_var = tk.StringVar(value=DEFAULT_DETECTOR)
        ttk.Combobox(ctrl, textvariable=self.detector_var, values=['SIFT', 'ORB']).pack(padx=8, pady=4, fill='x')

        # Matcher selection
        tk.Label(ctrl, text="Matcher", bg='#2b0f40', fg='white').pack(pady=(8,0))
        self.matcher_var = tk.StringVar(value=DEFAULT_MATCHER)
        ttk.Combobox(ctrl, textvariable=self.matcher_var, values=['BF', 'FLANN']).pack(padx=8, pady=4, fill='x')

        # Model selection
        tk.Label(ctrl, text="Model", bg='#2b0f40', fg='white').pack(pady=(8,0))
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Combobox(ctrl, textvariable=self.model_var, values=['homography', 'fundamental', 'slam' , 'epipolar']).pack(padx=8,pady=4,fill='x')
        # Pretest selection
        tk.Label(ctrl, text="Pretest", bg='#2b0f40', fg='white').pack(pady=(8,0))
        self.pretest_var = tk.StringVar(value=DEFAULT_PRETEST)
        ttk.Combobox(ctrl, textvariable=self.pretest_var,values=['Td', 'SPRT', 'SPRT*', 'Bail-out', 'Wald', 'Wald-Opt', 'None']).pack(padx=8, pady=4,fill='x')

        # Threshold entry
        tk.Label(ctrl, text="Inlier threshold (px or Sampson)", bg='#2b0f40', fg='white').pack(pady=(8,0))
        self.thresh_entry = tk.Entry(ctrl); self.thresh_entry.insert(0,'3.0'); self.thresh_entry.pack(padx=8, pady=4, fill='x')

        # --- small developer box (Dark Gradient Purple→Aqua→Black) ---
        # قرار گرفتن دقیقاً بعد از فیلد threshold
        dev_box_frame = tk.Frame(ctrl, bg='#2b0f40')
        dev_box_frame.pack(padx=8, pady=(4,8), fill='x')

        # اندازه باکس کوچک — تقریبی و جمع و جور
        box_w = 220
        box_h = 30

        # Canvas برای نمایش گرادیانت و متن روی آن
        dev_canvas = tk.Canvas(dev_box_frame, width=box_w, height=box_h, bg='#2b0f40', highlightthickness=0)
        dev_canvas.pack(side='left')

        # متن داخل باکس (چپ‌چین)
        dev_text = "Developed by babak yousefian  |  github.com/babakyousefian"

        # آماده‌سازی تصویر گرادیان بلند (برای اسکرول عمودیِ بسیار ملایم)
        # این بخش از PIL استفاده می‌کند؛ بالای فایل هم از PIL ایمپورت شده است.
        long_h = max(box_h * 3, 180)
        long_w = box_w

        from PIL import Image, ImageDraw, ImageTk
        import math

        # رنگ‌ها (purple -> aqua -> black)
        cols = [(128, 0, 255), (0, 255, 255), (0,0,0)]

        def interp(c1, c2, t):
            return (int(c1[0] + (c2[0]-c1[0])*t),
                    int(c1[1] + (c2[1]-c1[1])*t),
                    int(c1[2] + (c2[2]-c1[2])*t))

        grad_img = Image.new("RGB", (long_w, long_h))
        draw = ImageDraw.Draw(grad_img)
        half = long_h // 2
        for y in range(long_h):
            if y < half:
                t = y / max(1, half-1)
                c = interp(cols[0], cols[1], t)
            else:
                t = (y - half) / max(1, (long_h-half-1))
                c = interp(cols[1], cols[2], t)
            draw.line([(0, y), (long_w, y)], fill=c)

        # برش اولیه برای نمایش
        start_y = 0
        crop = grad_img.crop((0, start_y, long_w, start_y + box_h))
        grad_photo = ImageTk.PhotoImage(crop)
        img_id = dev_canvas.create_image(0, 0, anchor='nw', image=grad_photo)

        # متن سفید روی گرادیانت
        # متن را کمی با padding به سمت چپ منتقل می‌کنیم
        text_x = 6
        text_y = box_h // 2
        text_id = dev_canvas.create_text(text_x, text_y, text=dev_text, anchor='w',
                                         font=("Arial", 9, "normal"), fill="white")

        # نگهداری رفرنس‌ها به instance تا GC نکند
        self._dev_grad_img = grad_img
        self._dev_grad_photo = grad_photo
        self._dev_canvas = dev_canvas
        self._dev_box_h = box_h

        # انیمیشن: حرکت آهسته عمودی (ده ثانیه یک دور)
        total_frames = 300
        frame_delay = int(10000 / total_frames)  # ~33ms
        self._dev_frame_idx = 0

        def animate_dev_box():
            fi = self._dev_frame_idx
            max_start = max(1, self._dev_grad_img.height - self._dev_box_h)
            t = fi / total_frames  # 0..1
            # سینوسی برای نرمی حرکت (از بالا به پایین و برعکس)
            s = 0.5 * (1 + math.sin(2 * math.pi * t - math.pi/2))
            start_y = int(s * max_start)
            crop = self._dev_grad_img.crop((0, start_y, long_w, start_y + self._dev_box_h))
            photo = ImageTk.PhotoImage(crop)
            # update image on canvas
            dev_canvas.itemconfig(img_id, image=photo)
            # preserve reference
            self._dev_grad_photo = photo
            self._dev_frame_idx = (self._dev_frame_idx + 1) % total_frames
            dev_canvas.after(frame_delay, animate_dev_box)

        animate_dev_box()

        # کلیک روی باکس لینک گیت‌هاب را باز می‌کند
        def _open_github(event=None):
            try:
                import webbrowser
                webbrowser.open("https://github.com/babakyousefian/")
            except Exception:
                pass

        dev_canvas.bind("<Button-1>", _open_github)
        # همچنین نشانگر ماوس را تغییر می‌دهیم تا قابل کلیک بودن مشخص شود
        dev_canvas.config(cursor="hand2")


        # Status box
        self.status_text = tk.StringVar(value="Ready")
        tk.Label(ctrl, textvariable=self.status_text, bg='#2b0f40', fg='white', wraplength=280, justify='left').pack(padx=8, pady=12)

        # Canvas frame on right - will contain two stacked labels: image/video preview and output preview
        self.canvas_frame = tk.Frame(self.root, bg='#0f0f10')
        self.canvas_frame.pack(side='right', expand=True, fill='both')

        # top: input preview (image or video first frame)
        self.input_frame = tk.Frame(self.canvas_frame, bg='#111111')
        self.input_frame.pack(side='top', fill='both', expand=True, padx=10, pady=8)
        self.input_label = tk.Label(self.input_frame)
        self.input_label.pack(expand=True)

        # bottom: output preview (annotated)
        self.output_frame = tk.Frame(self.canvas_frame, bg='#0f0f10', height=220)
        self.output_frame.pack(side='bottom', fill='x', padx=10, pady=(0,8))
        self.output_label = tk.Label(self.output_frame)
        self.output_label.pack(expand=True)

        # small controls for video preview play/stop
        self.preview_controls = tk.Frame(self.canvas_frame, bg='#0f0f10')
        self.preview_controls.pack(side='bottom', fill='x', padx=10, pady=(0,8))
        self.btn_preview_play = tk.Button(self.preview_controls, text="Play Preview", command=self.play_video_preview, state='disabled')
        self.btn_preview_play.pack(side='left', padx=6)
        self.btn_preview_stop = tk.Button(self.preview_controls, text="Stop Preview", command=self.stop_video_preview, state='disabled')
        self.btn_preview_stop.pack(side='left', padx=6)

    # ---------------------------
    # AUTH: login / logout / signup
    # ---------------------------
    def open_login_window(self):
        users = load_users()
        dlg = tk.Toplevel(self.root)
        dlg.title("Login / Sign up")
        dlg.geometry("360x300")
        dlg.resizable(False, False)
        dlg.configure(bg="#2b0f40")

        tk.Label(dlg, text="Username:", bg="#2b0f40", fg="white").pack(pady=(12,4))
        ent_user = tk.Entry(dlg); ent_user.pack(padx=12)
        tk.Label(dlg, text="Password:", bg="#2b0f40", fg="white").pack(pady=(8,4))
        ent_pass = tk.Entry(dlg, show="*"); ent_pass.pack(padx=12)
        tk.Label(dlg, text="Full name (for display):", bg="#2b0f40", fg="white").pack(pady=(8,4))
        ent_full = tk.Entry(dlg); ent_full.pack(padx=12)

        msg_var = tk.StringVar(value="")

        def do_login():
            u = ent_user.get().strip()
            p = ent_pass.get().strip()
            if not u or not p:
                msg_var.set("Enter username & password")
                return
            us = load_users()
            if u in us and us[u]['hash'] == hash_password(p):
                self.logged_user = u
                self.logged_fullname = us[u].get('fullname', u)
                # update header
                self.login_btn.config(text=f"User: {self.logged_fullname}", state="disabled")
                self.dashboard_btn.config(state="normal")
                self.status_text.set(f"Logged in as {self.logged_fullname}")
                dlg.destroy()
                # On login: allow unlimited processing
                self.update_access_controls()
                # play login sound if available
                try:
                    self._stop_ad()
                except Exception:
                    pass
                self.ad_dismissed = False
                try:
                    self._play_login()
                except Exception:
                    pass
            else:
                msg_var.set("Login failed")

        def do_signup():
            u = ent_user.get().strip()
            p = ent_pass.get().strip()
            full = ent_full.get().strip()
            if not u or not p or not full:
                msg_var.set("All fields required for sign up")
                return
            us = load_users()
            if u in us:
                msg_var.set("User exists")
                return
            us[u] = {'hash': hash_password(p), 'fullname': full, 'created': datetime.now().isoformat()}
            save_users(us)
            # create user output dir
            ensure_user_output_dir(u)
            msg_var.set("User created. Now login.")
        # Buttons
        tk.Button(dlg, text="Login", bg="#4b1b6b", fg="white", command=do_login).pack(pady=(12,6))
        tk.Button(dlg, text="Sign up", bg="#3a6a9a", fg="white", command=do_signup).pack(pady=(0,6))
        tk.Label(dlg, textvariable=msg_var, fg="yellow", bg="#2b0f40").pack(pady=6)
        dlg.transient(self.root)
        dlg.grab_set()

    def logout(self):
        self.logged_user = None
        self.logged_fullname = None
        self.login_btn.config(text="Login", state="normal")
        self.dashboard_btn.config(state="disabled")
        self.status_text.set("Logged out")
        # reset anonymous counters and access
        self.anon_images_processed = 0
        self.anon_videos_processed = 0
        self.update_access_controls()

    # ---------------------------
    # Access control helpers
    # ---------------------------
    def update_access_controls(self):
        # If logged in -> all enabled
        if self.logged_user:
            self.btn_run_img.config(state="normal")
            self.btn_run_vid.config(state="normal")
            self.btn_load_img.config(state="normal")
            self.btn_load_vid.config(state="normal")
            self.dashboard_btn.config(state="normal")
        else:
            # anonymous rules: allow at most ANON_IMAGE_LIMIT and ANON_VIDEO_LIMIT
            # anonymous rules: allow at most ANON_IMAGE_LIMIT and ANON_VIDEO_LIMIT
            if self.anon_images_processed < self.ANON_IMAGE_LIMIT:
                self.btn_run_img.config(state="normal")
                self.btn_load_img.config(state="normal")
            else:
                self.btn_run_img.config(state="disabled")
                self.btn_load_img.config(state="normal")

                # NEW: notify user only once
                if not self.warned_image_limit:
                    # اگر وارد نشده و ad قبلاً بسته نشده، آگهی را اجرا کن وقتی که هر دو محدودیت اتفاق افتاده‌اند
                    try:
                        # start ad if both run buttons are disabled OR at least one is disabled depending on UX
                        if (self.btn_run_img['state'] == 'disabled' or self.btn_run_vid[
                            'state'] == 'disabled') and not self.ad_dismissed:
                            # small delay to let messagebox close
                            self.root.after(200, lambda: self._start_ad())
                    except Exception:
                        pass

                    messagebox.showinfo(
                        "Login required",
                        "You have reached the free image-processing limit.\nPlease log in to continue."
                    )
                    self.warned_image_limit = True
                # at end of the else path for anon user:
                if (self.btn_run_img['state'] == 'disabled' or self.btn_run_vid[
                    'state'] == 'disabled') and not self.ad_dismissed:
                    self.root.after(200, lambda: self._start_ad())

            if self.anon_videos_processed < self.ANON_VIDEO_LIMIT:
                self.btn_run_vid.config(state="normal")
                self.btn_load_vid.config(state="normal")
            else:
                self.btn_run_vid.config(state="disabled")
                self.btn_load_vid.config(state="normal")

                # NEW: notify user only once
                if not self.warned_video_limit:
                    # اگر وارد نشده و ad قبلاً بسته نشده، آگهی را اجرا کن وقتی که هر دو محدودیت اتفاق افتاده‌اند
                    try:
                        # start ad if both run buttons are disabled OR at least one is disabled depending on UX
                        if (self.btn_run_img['state'] == 'disabled' or self.btn_run_vid[
                            'state'] == 'disabled') and not self.ad_dismissed:
                            # small delay to let messagebox close
                            self.root.after(200, lambda: self._start_ad())
                    except Exception:
                        pass

                    messagebox.showinfo(
                        "Login required",
                        "You have reached the free video-processing limit.\nPlease log in to continue."
                    )
                    self.warned_video_limit = True

                # at end of the else path for anon user:
                if (self.btn_run_img['state'] == 'disabled' or self.btn_run_vid[
                    'state'] == 'disabled') and not self.ad_dismissed:
                    self.root.after(200, lambda: self._start_ad())

    def _ensure_can_process_image_anon(self):
        if self.logged_user:
            return True
        if self.anon_images_processed >= self.ANON_IMAGE_LIMIT:
            messagebox.showinfo("Limited", f"Anonymous users can process up to {self.ANON_IMAGE_LIMIT} image(s). Please login to continue.")
            return False
        return True

    def _ensure_can_process_video_anon(self):
        if self.logged_user:
            return True
        if self.anon_videos_processed >= self.ANON_VIDEO_LIMIT:
            messagebox.showinfo("Limited", f"Anonymous users can process up to {self.ANON_VIDEO_LIMIT} video(s). Please login to continue.")
            return False
        return True

    # ---------------------------
    # IMAGE: load / show / process
    # ---------------------------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.png *.jpeg *.bmp"), ("All files","*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.status_text.set("Failed to read image.")
            return
        self.orig_img = img.copy()
        self.img = img.copy()
        self.show_input_image(self.img)
        self.status_text.set(f"Loaded image: {os.path.basename(path)}")

    def show_input_image(self, img):
        h, w = img.shape[:2]
        max_w, max_h = 900, 520
        scale = min(1.0, max_w / w, max_h / h)
        disp = cv2.resize(img, (int(w*scale), int(h*scale)))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._tk_input = ImageTk.PhotoImage(pil)
        self.input_label.configure(image=self._tk_input)
        self.input_label.image = self._tk_input

    def show_output_image(self, img):
        h, w = img.shape[:2]
        max_w, max_h = 900, 200
        scale = min(1.0, max_w / w, max_h / h)
        disp = cv2.resize(img, (int(w*scale), int(h*scale)))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._tk_output = ImageTk.PhotoImage(pil)
        self.output_label.configure(image=self._tk_output)
        self.output_label.image = self._tk_output

    def run_pipeline_thread(self):
        # Check anonymous limit for image
        if not self._ensure_can_process_image_anon():
            return
        t = threading.Thread(target=self.run_pipeline, daemon=True)
        t.start()

    def run_pipeline(self):
        if self.img is None and self.orig_img is None:
            self.status_text.set("No image loaded")
            return
        # ensure using orig_img
        img_to_use = self.orig_img if self.orig_img is not None else self.img
        self.status_text.set("Detecting features...")
        det_method = self.detector_var.get()
        self.detector = create_detector(det_method)
        kps, desc = detect_and_compute(img_to_use, self.detector)
        self.kps1, self.desc1 = kps, desc
        kp_img = draw_keypoints(img_to_use, kps, color=(0,255,0), radius=2, max_kp=500)
        self.show_input_image(kp_img)
        # INSERT (run_pipeline) — run epipolar if selected
        if self.model_var.get().lower() == 'epipolar':
            try:
                proc = EpipolarProcessor(detector_name=self.detector_var.get(), matcher_name=self.matcher_var.get())
                # For single-image case we do self-match demo (use same image as both views)
                res = proc.process_pair(self.orig_img, self.orig_img)
                # show annotated left image in input preview, and right annotated in output preview
                self.show_input_image(res['img1_annotated'])
                self.show_output_image(res['img2_annotated'])
                self.status_text.set(
                    f"Epipolar: matches={res['diagnostics'].get('n_matches', 0)} inliers={res['diagnostics'].get('n_inliers', 0)}")
            except Exception as e:
                self.status_text.set(f"Epipolar error: {e}")
            return

        self.status_text.set(f"Detected {len(kps)} keypoints. Matching...")
        matcher_method = self.matcher_var.get()
        matcher = create_matcher(matcher_method, det_method)
        matches = match_descriptors(self.desc1, self.desc1, matcher, ratio_thresh=0.75)
        self.status_text.set(f"Found {len(matches)} raw matches (self-match demo). Running RANSAC...")
        pts1 = []
        pts2 = []
        for m in matches:
            p1 = self.kps1[m.queryIdx].pt
            p2 = self.kps1[m.trainIdx].pt
            pts1.append(p1); pts2.append(p2)
        if len(pts1) == 0:
            self.status_text.set("No correspondences found.")
            return
        model_choice = self.model_var.get()
        threshold = float(self.thresh_entry.get())
        pretest = self.pretest_var.get()
        try:
            model, inliers_mask, stats = rransac_main(np.array(pts1), np.array(pts2), model=model_choice,
                                                      threshold=threshold, confidence=0.99, max_iter=1000,
                                                      pretest=(None if pretest=='None' else pretest),
                                                      d=1, sprt_params={'epsilon':0.02,'eta':0.5,'alpha':0.01,'beta':0.01})
            if model is None:
                self.status_text.set("No model found by R-RANSAC.")
                return
            inliers_count = int(inliers_mask.sum()) if hasattr(inliers_mask, 'sum') else int(np.sum(inliers_mask))
            self.status_text.set(f"Model found. Inliers: {inliers_count}/{len(inliers_mask)}. Iter: {stats.get('iter',0)}")
            out = draw_inliers_only(self.orig_img, pts1, inliers_mask)
            out2 = draw_matches_lines(out, np.array(pts1), np.array(pts2), inlier_mask=inliers_mask)
            self.show_output_image(out2)
            # if logged in: save output image to user's folder and record index
            if self.logged_user:
                try:
                    user_dir = ensure_user_output_dir(self.logged_user)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    fname = f'image_{ts}.png'
                    outpath = os.path.join(user_dir, fname)
                    # out2 is BGR numpy array
                    cv2.imwrite(outpath, out2)
                    entry = {'type': 'image', 'file': fname, 'ts': ts}
                    append_user_index(self.logged_user, entry)
                    self.status_text.set(self.status_text.get() + f" Saved to dashboard as {fname}")
                except Exception as e:
                    self.status_text.set(self.status_text.get() + f" (save error: {e})")
            else:
                # anonymous: increment counter and update controls (persist only in-session)
                self.anon_images_processed += 1
                self.update_access_controls()
            # optionally run object detection (ultralytics) and show result (no saving here)
            try:
                if ULTRALYTICS_AVAILABLE:
                    from ultralytics import YOLO
                    y = YOLO('yolov8n.pt')
                    res = y(self.orig_img[:, :, ::-1], imgsz=640)[0]
                    detections = []
                    names = res.names
                    for r in res.boxes:
                        xywh = r.xywh[0].cpu().numpy()
                        cx, cy, w, h = xywh
                        x = float(cx - w/2); y_ = float(cy - h/2)
                        cls = int(r.cls[0].cpu().numpy()) if hasattr(r, 'cls') else int(r.cls)
                        conf = float(r.conf[0].cpu().numpy()) if hasattr(r,'conf') else float(r.conf)
                        detections.append(([x,y_,w,h], cls, conf))
                    out3 = draw_bounding_boxes(self.orig_img, detections, class_names=names)
                    self.show_output_image(out3)
            except Exception:
                pass

            # play 'done' sound if available
            try:
                self._play_done()
            except Exception:
                pass

        except Exception as ex:
            self.status_text.set(f"Pipeline error: {ex}")

    # ---------------------------
    # VIDEO: load / preview / process
    # ---------------------------
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.status_text.set("Failed to open video.")
            return
        self.video_path = path
        self.video_cap = cap
        # read first frame for preview
        ret, frame = cap.read()
        if not ret:
            self.status_text.set("Cannot read video frame.")
            cap.release()
            self.video_cap = None
            return
        self.last_frame = frame.copy()
        self.show_input_image(frame)
        self.status_text.set(f"Loaded video: {os.path.basename(path)}")
        # enable preview buttons
        self.btn_preview_play.config(state='normal')
        self.btn_preview_stop.config(state='normal')
        # reset capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _video_preview_worker(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay = 1.0 / max(1.0, fps)
        self.video_playing = True
        while self.video_playing:
            ret, frame = cap.read()
            if not ret:
                break
            self.last_frame = frame.copy()
            self.show_input_image(frame)
            time.sleep(min(delay, 1/15))
        cap.release()
        self.video_playing = False

    def play_video_preview(self):
        if not self.video_path:
            self.status_text.set("No video loaded.")
            return
        if self.video_playing:
            return
        self.video_preview_thread = threading.Thread(target=self._video_preview_worker, daemon=True)
        self.video_preview_thread.start()
        self.status_text.set("Playing preview...")

    def stop_video_preview(self):
        if self.video_playing:
            self.video_playing = False
            self.status_text.set("Preview stopped.")
        else:
            self.status_text.set("Preview not running.")

    def run_video_pipeline_thread(self):
        # anonymous check
        if not self._ensure_can_process_video_anon():
            return
        if not self.video_path:
            self.status_text.set("No video loaded.")
            return
        # prevent multiple concurrent runs
        if self.video_processing_thread is not None and self.video_processing_thread.is_alive():
            self.status_text.set("Video processing already running.")
            return
        self.video_processing_thread = threading.Thread(target=self.run_video_pipeline, daemon=True)
        self.video_processing_thread.start()

    def run_video_pipeline(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status_text.set("Cannot open video for processing.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out_name = f"tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(OUTPUTS_DIR, out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, max(1,fps), (w,h))
        self.status_text.set("Starting video processing...")
        # init detector and matcher
        det_method = self.detector_var.get()
        self.detector = create_detector(det_method)
        matcher_method = self.matcher_var.get()
        matcher = create_matcher(matcher_method, det_method)
        prev_kps = None
        prev_desc = None
        frame_idx = 0
        process_frame_step = 1
        try:
            use_ultra = False
            ultra_model = None
            if ULTRALYTICS_AVAILABLE:
                try:
                    from ultralytics import YOLO
                    ultra_model = YOLO('yolov8n.pt')
                    use_ultra = True
                except Exception:
                    use_ultra = False

            # Initialize SLAM if selected
            slam_mode = (self.model_var.get().lower() == 'slam')
            slam_sys = None
            if slam_mode:
                try:
                    slam_sys = VisualSLAM(detector_name=self.detector_var.get(), matcher_name=self.matcher_var.get(),
                                          keyframe_interval=10, reproj_threshold=4.0)
                except Exception:
                    slam_sys = None

            self.last_video_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                do_process = (frame_idx % process_frame_step == 0)
                annotated = frame.copy()
                # inside video processing loop, after annotated prepared
                if self.model_var.get().lower() == 'epipolar':
                    try:
                        if hasattr(self, '_epip_proc'):
                            ep = self._epip_proc
                        else:
                            self._epip_proc = EpipolarProcessor(detector_name=self.detector_var.get(),
                                                                matcher_name=self.matcher_var.get())
                            ep = self._epip_proc
                        # use prev_frame if exists, else skip
                        if hasattr(self, 'last_video_frame') and self.last_video_frame is not None:
                            res = ep.process_pair(self.last_video_frame, frame)
                            # replace annotated preview with annotated right image (current frame) showing epilines
                            annotated = res['img2_annotated']
                            # optionally also set input to left annotated
                            self.show_input_image(res['img1_annotated'])
                            self.status_text.set(
                                f"Epipolar: matches={res['diagnostics'].get('n_matches', 0)} inliers={res['diagnostics'].get('n_inliers', 0)}")
                        # update last frame storage
                        self.last_video_frame = frame.copy()
                    except Exception:
                        pass

                if do_process:
                    kps, desc = detect_and_compute(frame, self.detector)
                    annotated = draw_keypoints(annotated, kps, color=(0,255,0), radius=1, max_kp=200)
                    if prev_desc is not None and desc is not None and len(prev_kps)>7 and len(kps)>7:
                        matches = match_descriptors(prev_desc, desc, matcher, ratio_thresh=0.75)
                        pts_prev = []; pts_curr = []
                        for m in matches:
                            pts_prev.append(prev_kps[m.queryIdx].pt)
                            pts_curr.append(kps[m.trainIdx].pt)
                        if len(pts_prev) >= 4:
                            try:
                                model_choice = self.model_var.get()
                                threshold = float(self.thresh_entry.get())
                                model, inliers_mask, stats = rransac_main(np.array(pts_prev), np.array(pts_curr),
                                                                          model=model_choice, threshold=threshold,
                                                                          confidence=0.95, max_iter=400,
                                                                          pretest=(None if self.pretest_var.get()=='None' else self.pretest_var.get()),
                                                                          d=1, sprt_params={'epsilon':0.02,'eta':0.5,'alpha':0.01,'beta':0.01})
                                if model is not None:
                                    annotated = draw_matches_lines(annotated, np.array(pts_curr), np.array(pts_prev), inlier_mask=inliers_mask)
                            except Exception:
                                pass
                    prev_kps, prev_desc = kps, desc
                    if use_ultra and ultra_model is not None:
                        try:
                            res = ultra_model(frame[:, :, ::-1], imgsz=640)[0]
                            dets = []
                            for r in res.boxes:
                                xywh = r.xywh[0].cpu().numpy()
                                cx, cy, w_, h_ = xywh
                                x = float(cx - w_/2); y_ = float(cy - h_/2)
                                cls = int(r.cls[0].cpu().numpy()) if hasattr(r, 'cls') else int(r.cls)
                                conf = float(r.conf[0].cpu().numpy()) if hasattr(r,'conf') else float(r.conf)
                                dets.append(([x,y_,w_,h_], cls, conf))
                            annotated = draw_bounding_boxes(annotated, dets, class_names=(res.names if hasattr(res,'names') else None))
                        except Exception:
                            pass
                    else:
                        cfg = os.path.join(BASE_DIR, 'assets', 'yolov4-tiny.cfg')
                        wts = os.path.join(BASE_DIR, 'assets', 'yolov4-tiny.weights')
                        names = os.path.join(BASE_DIR, 'assets', 'coco.names')
                        if os.path.exists(cfg) and os.path.exists(wts):
                            try:
                                net, classes = load_yolov4_tiny(cfg, wts, names)
                                dets = yolo_v4_tiny_detect(net, classes, frame, conf_thresh=0.35)
                                if dets:
                                    annotated = draw_bounding_boxes(annotated, dets, class_names=classes)
                            except Exception:
                                pass

                if slam_sys is not None:
                    try:
                        slam_info = slam_sys.process_frame(frame)
                        annotated = slam_sys.draw_overlay(annotated)
                        # optionally add textual info about status
                        st = slam_info.get('status', '')
                        mpcount = slam_info.get('map_points', 0)
                        inliers = slam_info.get('inliers', 0)
                        cv2.putText(annotated, f"SLAM:{st} MPs:{mpcount} In:{inliers}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
                    except Exception:
                        pass

                writer.write(annotated)
                if frame_idx % max(1, int(max(1, fps)//5)) == 0:
                    if slam_sys is not None:
                        try:
                            slam_info = slam_sys.process_frame(frame)
                            annotated = slam_sys.draw_overlay(annotated)
                            # optionally add textual info about status
                            st = slam_info.get('status', '')
                            mpcount = slam_info.get('map_points', 0)
                            inliers = slam_info.get('inliers', 0)
                            cv2.putText(annotated, f"SLAM:{st} MPs:{mpcount} In:{inliers}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
                        except Exception:
                            pass

                    self.show_output_image(annotated)
                    self.root.update_idletasks()
                if frame_count>0:
                    self.status_text.set(f"Processing video: frame {frame_idx}/{frame_count}")
                else:
                    self.status_text.set(f"Processing video: frame {frame_idx}")
            writer.release()
            cap.release()

            # if logged in -> move output into user's folder
            if self.logged_user:
                try:
                    user_dir = ensure_user_output_dir(self.logged_user)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    fname = f'video_{ts}.mp4'
                    dest = os.path.join(user_dir, fname)
                    # move file
                    try:
                        os.replace(out_path, dest)
                    except Exception:
                        import shutil
                        shutil.copy(out_path, dest)
                        os.remove(out_path)
                    entry = {'type':'video', 'file': fname, 'ts': ts}
                    append_user_index(self.logged_user, entry)
                    self.status_text.set(f"Video processed and saved to dashboard as {fname}")
                except Exception as e:
                    self.status_text.set(f"Video processed but save error: {e}")
            else:
                # anonymous: increment and update access controls
                self.anon_videos_processed += 1
                self.update_access_controls()
                self.status_text.set(f"Video processed -> {out_path} (not saved to account)")
            # show first frame of output as preview
            outcap = cv2.VideoCapture(out_path)
            ok, f0 = outcap.read()
            if ok:
                self.show_output_image(f0)
            outcap.release()

            # play 'done' sound if available
            try:
                self._play_done()
            except Exception:
                pass

        except Exception as e:
            try:
                writer.release()
            except Exception:
                pass
            try:
                cap.release()
            except Exception:
                pass
            self.status_text.set(f"Video processing error: {e}")

    # ---------------------------
    # Dashboard: view user outputs (save/remove/download)
    # ---------------------------
    def open_dashboard(self):
        if not self.logged_user:
            messagebox.showinfo("Login required", "Please login to access dashboard.")
            return
        user = self.logged_user
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Dashboard — {user}")
        dlg.geometry("900x560")

        frame = tk.Frame(dlg)
        frame.pack(fill='both', expand=True)

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        items = read_user_index(user)[::-1]  # newest first
        if not items:
            tk.Label(inner, text="No saved outputs yet.", pady=20).pack()
            return

        for it in items:
            row = tk.Frame(inner, bd=1, relief='ridge', padx=8, pady=6)
            row.pack(fill='x', padx=8, pady=6)
            lbl = tk.Label(row, text=f"{it.get('type')} — {it.get('file')} — {it.get('ts')}")
            lbl.pack(side='left')
            fpath = os.path.join(ensure_user_output_dir(user), it.get('file'))
            def make_view(p=fpath):
                def _view():
                    if not os.path.exists(p):
                        messagebox.showerror("Missing", "File not found")
                        return
                    if it.get('type')=='image':
                        img = cv2.imread(p)
                        if img is None:
                            messagebox.showerror("Error","Cannot open image")
                            return
                        # show in small window
                        top = tk.Toplevel(dlg)
                        top.title(it.get('file'))
                        h,w = img.shape[:2]
                        scale = min(1.0, 800/w, 600/h)
                        disp = cv2.resize(img, (int(w*scale), int(h*scale)))
                        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        tkimg = ImageTk.PhotoImage(pil)
                        lblimg = tk.Label(top, image=tkimg)
                        lblimg.image = tkimg
                        lblimg.pack()
                    else:
                        # video: open default system dialog to save or play externally
                        messagebox.showinfo("Video", f"Video located at: {p}\nUse Download to save a copy.")
                return _view
            def make_download(p=fpath):
                def _dl():
                    if not os.path.exists(p):
                        messagebox.showerror("Missing", "File not found")
                        return
                    dst = filedialog.asksaveasfilename(initialfile=os.path.basename(p))
                    if not dst:
                        return
                    try:
                        import shutil
                        shutil.copy(p, dst)
                        messagebox.showinfo("Saved", f"Saved to {dst}")
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                return _dl
            def make_remove(meta_entry=it):
                def _rm():
                    if not messagebox.askyesno("Confirm", "Remove this file from dashboard?"):
                        return
                    try:
                        # delete file
                        pth = os.path.join(ensure_user_output_dir(user), meta_entry.get('file'))
                        if os.path.exists(pth):
                            os.remove(pth)
                        # remove from index
                        arr = read_user_index(user)
                        arr2 = [e for e in arr if e.get('file')!=meta_entry.get('file')]
                        with open(user_index_file(user), 'w', encoding='utf-8') as f:
                            json.dump(arr2, f, indent=2, ensure_ascii=False)
                        row.destroy()
                        messagebox.showinfo("Removed", "File removed.")
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                return _rm
            btn_view = tk.Button(row, text="View", command=make_view())
            btn_view.pack(side='right', padx=6)
            btn_dl = tk.Button(row, text="Download", command=make_download())
            btn_dl.pack(side='right', padx=6)
            btn_rm = tk.Button(row, text="Remove", command=make_remove())
            btn_rm.pack(side='right', padx=6)

        # play dashboard sound when opened (best effort)
        try:
            self._play_dashboard()
        except Exception:
            pass

    # ---------------------------
    # Application close / cleanup
    # ---------------------------
    def on_close(self):
        try:
            # stop background music cleanly
            try:
                self._stop_background_music()
            except Exception:
                pass

            if hasattr(self, 'video_cap') and self.video_cap is not None:
                try:
                    self.video_cap.release()
                except Exception:
                    pass
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass


# -------------------------
# Launch
# -------------------------
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Vidim")
    root.iconbitmap("icon/favicon_io/favicon.ico")
    logo_img = PhotoImage(file="icon/favicon_io/favicon-16x16.png")
    label = tk.Label(root, image=logo_img)
    label.pack(pady=0)
    root.geometry("0x0")
    app = App(root)
    music_toggle = MusicToggle(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
