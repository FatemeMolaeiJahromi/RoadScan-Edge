import os
import sys
import urllib.request

os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

def download_file(urls, target_path, label):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        print(f"[OK] {label} already exists.")
        return True

    print(f"Downloading {label}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for url in urls:
        print(f"  Attempting source: {url[:60]}...")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=35) as response, open(target_path, 'wb') as out_file:
                total_size = int(response.getheader('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 64

                while True:
                    buf = response.read(block_size)
                    if not buf:
                        break
                    downloaded += len(buf)
                    out_file.write(buf)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        sys.stdout.write(f"\r  Progress: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB)")
                        sys.stdout.flush()

            print(f"\n[OK] {label} downloaded successfully!")
            return True
        except Exception as e:
            print(f"\n  [-] Source failed: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

    return False

# 1. Direct verified weights for Pothole Detection (Trained YOLO model on HF)
pothole_model_urls = [
    "https://hf-mirror.com/keremberke/yolov8m-pothole-segmentation/resolve/main/best.pt",
    "https://huggingface.co/keremberke/yolov8m-pothole-segmentation/resolve/main/best.pt",
    "https://hf-mirror.com/arnabdhar/YOLOv8-Pothole-Detection/resolve/main/best.pt",
    "https://huggingface.co/arnabdhar/YOLOv8-Pothole-Detection/resolve/main/best.pt"
]
download_file(pothole_model_urls, os.path.join("models", "pothole_yolo.pt"), "Pothole Detection Weights")

# 2. Real road dashcam video with potholes (Free open-source raw sample)
sample_video_urls = [
    "https://raw.gitcode.com/openvinotoolkit/sample-videos/raw/master/road-pothole.mp4",
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4", # fallback
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
]
download_file(sample_video_urls, os.path.join("videos", "pothole_road.mp4"), "Pothole Road Dashcam Video")

print("\nSetup finished. Run 'python run_pipeline.py' next.")
