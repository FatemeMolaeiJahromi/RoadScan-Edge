import urllib.request
import os

os.makedirs("videos", exist_ok=True)
dest = os.path.join("videos", "road_input.mp4")

# یک ویدیوی کوتاه dashcam با کیفیت تست از مخزن معتبر عمومی
url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/driver-action-recognition.mp4"
# لینک پشتیبان جاده در صورت نیاز:
road_sample_url = "https://github.com/amdegroot/ssd.pytorch/raw/master/demo/demo.mp4"

print("Downloading real road driving sample...")
try:
    # استفاده از لینک تست رانندگی جاده
    urllib.request.urlretrieve("https://github.com/openvinotoolkit/open_model_zoo/raw/master/data/dataset_classes/road_traffic.mp4", dest)
    print(f"✅ Video downloaded successfully to: {dest}")
except Exception as e:
    print(f"Fallback direct stream test...")
    urllib.request.urlretrieve(road_sample_url, dest)
    print(f"✅ Downloaded to: {dest}")