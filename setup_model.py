import os
import urllib.request

# ساخت پوشه‌های مورد نیاز
os.makedirs("models", exist_ok=True)
os.makedirs("depth_anything_v2", exist_ok=True)

# ۱. دریافت ۴ فایل معماری اصلی از GitHub
base_url = "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/depth_anything_v2"
files = ["__init__.py", "dpt.py", "dinov2.py", "util.py"]

print("📥 در حال دریافت فایل‌های معماری Depth Anything V2...")
for f in files:
    target_path = os.path.join("depth_anything_v2", f)
    if not os.path.exists(target_path):
        url = f"{base_url}/{f}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp, open(target_path, 'wb') as out_file:
                out_file.write(resp.read())
            print(f"  [+] فایل {f} دریافت شد.")
        except Exception as e:
            print(f"  [!] تلاش مستقیم برای {f} ناموفق بود، در حال تست از آدرس جایگزین...")
            alt_url = f"https://ghproxy.net/{url}"
            try:
                urllib.request.urlretrieve(alt_url, target_path)
                print(f"  [+] فایل {f} از میرور دریافت شد.")
            except Exception as e2:
                print(f"  [-] خطا در دریافت {f}: {e2}")
    else:
        print(f"  [ok] {f} قبلاً وجود دارد.")

# ۲. دانلود وزن رسمی مدل جاده‌ای (Outdoor Metric - Small - حدود ۹۵ مگابایت)
weights_path = os.path.join("models", "depth_anything_v2_metric_vkitti_vits.pth")

if not os.path.exists(weights_path):
    print("\n📥 در حال دانلود وزن مدل متریک فضای باز (VKITTI Outdoor Small)...")
    # استفاده از میرور Hugging Face برای دانلود مستقیم و پایدار
    hf_mirror_url = "https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth"
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                print(f"\r  دانلود: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end="")

        urllib.request.urlretrieve(hf_mirror_url, weights_path, reporthook=show_progress)
        print("\n✅ وزن مدل با موفقیت دانلود شد!")
    except Exception as e:
        print(f"\n[!] خطا در دانلود میرور: {e}")
        print("💡 راهکار دستی: می‌توانی فایل را از لینک زیر دانلود کنی و در پوشه models قرار دهی:")
        print("https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth")
else:
    print("\n✅ فایل وزن مدل از قبل در پوشه models موجود است.")

print("\n🚀 آماده‌سازی کامل شد!")