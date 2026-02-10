# ___```R-RANSAC within SPRT``` for real-time video and image processing Algorithm___ 
---
## ___```R-RANSAC within SPRT``` for real-time video and image processing Algorithm___ 
---
# Vidim Processor


---

## ✨ ویژگی‌ها

- رابط گرافیکی مبتنی بر Tkinter
- پخش موسیقی پس‌زمینه با امکان Mute / Unmute
- بسته‌بندی کامل به exe با PyInstaller
- نصب حرفه‌ای با Inno Setup
- مدیریت منابع (assets, icons, outputs)
- لاگ‌گیری بی‌صدا (Silent Logging)
- گزارش کرش (Crash Report)
- آماده برای Auto-Update
- سازگار با Windows 10 / 11

---

## 📁 ساختار پروژه


---


```angular2html

VidimProcessor/
│
├── gui_main.py # فایل اصلی رابط گرافیکی
├── logger.py # سیستم لاگ‌گیری
├── crash_report.py # مدیریت کرش
├── updater.py # بررسی آپدیت
├── requirements.txt # وابستگی‌ها
├── installer.iss # اسکریپت نصب Inno Setup
│
├── assets/
│ ├── sounds/
│ │ └── background.ogg # موسیقی پس‌زمینه
│ └── images/
│
├── icon/
│ └── favicon_io/
│ ├── favicon.ico
│ └── favicon-16x16.png
│
├── outputs/ # خروجی‌ها
├── dist/ # خروجی PyInstaller
└── build/


```


---

## 🧠 مفاهیم کلیدی پروژه

### resource_path
برای سازگاری با PyInstaller، تمام مسیر فایل‌ها باید از تابع زیر استفاده کنند:

```python
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
```



تمام فایل‌های زیر باید با این تابع لود شوند:

    آیکن‌ها

    تصاویر

    فایل‌های صوتی

    assets


🔊 مدیریت موسیقی (AudioManager)

پخش موسیقی با pygame انجام می‌شود:

مقداردهی اولیه فقط یک‌بار

توقف امن هنگام خروج

جلوگیری از کرش هنگام نبود فایل

مسیر موسیقی:

```python
assets/sounds/background.ogg
```


🛑 Clean Shutdown

هنگام بستن برنامه:

    موسیقی متوقف می‌شود

    pygame.quit() اجرا می‌شود

    برنامه بدون crash بسته می‌شود

📝 Silent Logging

لاگ‌ها بدون نمایش به کاربر ذخیره می‌شوند:

مسیر:

```python
C:\Users\<User>\AppData\Local\VidimProcessor\logs\vidim.log

```


شامل:

    زمان اجرا

    خطاها

    کرش‌ها

    رویدادهای مهم

💥 Crash Report

در صورت بروز خطای بحرانی:

    traceback کامل در لاگ ذخیره می‌شود

    پیام انسانی به کاربر نمایش داده می‌شود

    برنامه به‌صورت امن بسته می‌شود

🔄 Auto Update (اختیاری)

برنامه در شروع:

    نسخه فعلی را بررسی می‌کند

    version.txt را از GitHub می‌خواند

    در صورت وجود نسخه جدید، لینک دانلود را نمایش می‌دهد

🧪 آماده‌سازی محیط توسعه
1. ساخت Virtual Environment


```angular181svg
python -m venv venv

```

2. فعال‌سازی
```python
.\venv\Scripts\activate

```

3. نصب وابستگی‌ها
```angular17svg
pip install -r requirements.txt

```


🧱 ساخت فایل exe با PyInstaller

دستور صحیح:

```angular17svg
pyinstaller ^
 --onefile ^
 --windowed ^
 --icon=icon\favicon_io\favicon.ico ^
 --add-data "assets;assets" ^
 --add-data "icon;icon" ^
 --add-data "outputs;outputs" ^
 gui_main.py

```
پس از اتمام:

```azure
dist/gui_main.exe

```

📦 ساخت Installer با Inno Setup
پیش‌نیاز

نصب Inno Setup

اجرای installer.iss

فایل installer.iss را باز کرده و Compile کنید.

Installer:

فایل exe

پوشه assets

پوشه icon

پوشه outputs

را به‌درستی در مسیر نصب کپی می‌کند.

💿 نصب روی ویندوز

اجرای فایل Setup

Next → Next

مشاهده Progress Bar

نصب کامل در:

```bigquery
C:\Program Files (x86)\VidimProcessor

```


▶️ اجرای برنامه

    از Start Menu

    یا Desktop Shortcut

برنامه بدون نیاز به Python اجرا می‌شود.
🧩 رفع خطاهای رایج
❌ فایل صوتی پیدا نمی‌شود

    بررسی وجود فایل در assets/sounds

    استفاده از resource_path

❌ iconbitmap خطا می‌دهد

    استفاده از مسیر absolute با resource_path

    اطمینان از اضافه شدن icon در PyInstaller

👨‍💻 توسعه‌دهنده

Babak Yousefian
GitHub: https://github.com/babakyousefian
📜 لایسنس

این پروژه برای استفاده آموزشی و توسعه شخصی طراحی شده است.

```vue

---

این README:
- برای **ارائه دانشگاهی** کاملاً قابل دفاع است  
- برای **تحویل پروژه نرم‌افزاری** حرفه‌ای است  
- و برای **خودت در ۶ ماه آینده نجات‌بخش**

اگر خواستی، قدم بعدی می‌تونیم:
- README انگلیسی رسمی
- CONTRIBUTING.md
- یا CHANGELOG.md واقعی بسازیم

```

#### @Author by : ___babak yousefian___

---



