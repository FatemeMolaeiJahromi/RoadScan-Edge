import os
import urllib.request
import zipfile
import shutil

print("📥 در حال دریافت کامل ساختار Depth Anything V2...")

# آدرس آرشیو مخزن رسمی از میرور پرسرعت
zip_url = "https://github.com/DepthAnything/Depth-Anything-V2/archive/refs/heads/main.zip"
proxy_zip_url = f"https://ghproxy.net/{zip_url}"
zip_path = "repo.zip"

try:
    print("  در حال دانلود سورس کد (حجم بسیار کم: ~1MB)...")
    try:
        urllib.request.urlretrieve(proxy_zip_url, zip_path)
    except:
        urllib.request.urlretrieve(zip_url, zip_path)
    
    print("  در حال استخراج لایه‌ها...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_repo")

    # مسیر اصلی پوشه پکیج در فایل زیپ دانلود شده
    src_dir = os.path.join("temp_repo", "Depth-Anything-V2-main", "metric_depth", "depth_anything_v2")
    dest_dir = "depth_anything_v2"

    if os.path.exists(src_dir):
        # پاک‌کردن پوشه ناقص فعلی و جایگزینی با نسخه کامل رسمی
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
        print("✅ تمام لایه‌ها (dinov2_layers و ساختار اصلی) با موفقیت کپی شدند!")
    else:
        print("[!] مسیر در زیپ پیدا نشد.")

    # پاکسازی فایل‌های موقت
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if os.path.exists("temp_repo"):
        shutil.rmtree("temp_repo")

except Exception as e:
    print(f"خطا در دریافت خودکار: {e}")