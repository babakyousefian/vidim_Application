from ultralytics import YOLO

# اگر فایل yolov8n.pt محلی نداشته باشی، با این فراخوانی وزن از ریموت دانلود می‌شود
model = YOLO("yolov8n.pt")  # یا YOLO("yolov8n")

# روی یک تصویر
results = model.predict(source="path/to/image.jpg", save=True)  # خروجی در runs/detect/... ذخیره می‌شود

# یا روی ویدیو
results = model.predict(source="path/to/video.mp4", save=True)

# دسترسی به نتایج برنامه‌نویسی
for r in results:
    print(r.boxes)   # باکس‌های تشخیص
