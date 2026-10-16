import os
import csv
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

def main():
    print("=" * 60)
    print("RoadScan-Edge: AI Pothole Detection & Metric 3D Surveying")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚙️ Inference Device: {device.upper()}")

    # 1. Load Metric Depth Model (Depth Anything V2)
    depth_model_path = os.path.join("models", "depth_anything_v2_metric_vkitti_vits.pth")
    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    
    depth_model = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': 80.0})
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    print("🧠 Metric Depth Model Loaded.")

    # 2. Load Pothole Detection Model (YOLO)
    pothole_model_path = os.path.join("models", "pothole_yolo.pt")
    if os.path.exists(pothole_model_path):
        yolo_model = YOLO(pothole_model_path)
    else:
        yolo_model = YOLO("yolov8n.pt")
    print("🔍 Defect Detector Loaded.")

    # 3. Video Source & Output Setup
    input_video = os.path.join("videos", "pothole_road.mp4")
    if not os.path.exists(input_video):
        input_video = os.path.join("videos", "road_input.mp4")

    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = width * 2
    out_height = height
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "roadscan_pothole_detected.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # CSV Audit Report Setup
    csv_path = os.path.join("outputs", "road_audit_report.csv")
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Confidence', 'Distance_Z(m)', 'Width(m)', 'Height(m)', 'Asphalt_Liters', 'Severity'])

    fx = fy = width * 0.8
    cx = width / 2.0
    cy = height / 2.0

    frame_idx = 0
    print(f"🎬 Processing road survey video ({total_frames} frames)...")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # A. Metric Depth Estimation
            metric_depth = depth_model.infer_image(frame, input_size=518)
            depth_vis = np.clip(metric_depth.copy(), 0.5, 40.0)
            depth_norm = ((depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255.0).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

            # B. Pothole Detection & 3D Back-Projection
            results = yolo_model(frame, verbose=False, conf=0.25)[0]
            detected_potholes = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                patch = metric_depth[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                if patch.size == 0:
                    continue
                z_m = float(np.median(patch))

                u_c = (x1 + x2) / 2.0
                v_c = (y1 + y2) / 2.0
                x_m = (u_c - cx) * z_m / fx
                y_m = (v_c - cy) * z_m / fy

                box_w_m = abs(x2 - x1) * z_m / fx
                box_h_m = abs(y2 - y1) * z_m / fy
                est_depth_cm = min(15.0, max(3.0, 18.0 / (z_m + 1.0)))
                est_volume_liters = (box_w_m * box_h_m * 10000.0) * est_depth_cm * 0.001 * 0.65

                # Severity Classification
                if z_m < 6.0 and est_volume_liters > 15.0:
                    severity = "CRITICAL"
                    box_color = (0, 0, 255) # Red
                elif z_m < 12.0:
                    severity = "WARNING"
                    box_color = (0, 165, 255) # Orange
                else:
                    severity = "MINOR"
                    box_color = (0, 255, 255) # Yellow

                detected_potholes.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'pos': (x_m, y_m, z_m),
                    'dims': (box_w_m, box_h_m),
                    'volume': est_volume_liters,
                    'severity': severity
                })

                # Write to CSV Report
                csv_writer.writerow([frame_idx, f"{conf:.2f}", f"{z_m:.2f}", f"{box_w_m:.2f}", f"{box_h_m:.2f}", f"{est_volume_liters:.1f}", severity])

                # Draw Visual Tags
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (255, 255, 255), 2)

                tag = f"{severity} | Z:{z_m:.1f}m | {est_volume_liters:.1f}L"
                cv2.putText(frame, tag, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)
                cv2.putText(depth_colored, f"Z={z_m:.2f}m", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # HUD Telemetry Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 15), (440, 165), (20, 20, 25), -1)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

            cv2.putText(frame, "RoadScan-Edge : AUDIT & 3D TELEMETRY", (25, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.putText(frame, f"Detected Defects: {len(detected_potholes)}", (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 2)
            
            if detected_potholes:
                closest = min(detected_potholes, key=lambda p: p['pos'][2])
                cv2.putText(frame, f"Priority Target Z: {closest['pos'][2]:.2f}m [{closest['severity']}]", (25, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(frame, f"Asphalt Patch Needed: {closest['volume']:.1f} Liters", (25, 126),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "Scanning surface condition...", (25, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.putText(frame, "Pipeline: YOLOv8 + Depth Anything V2 Metric", (25, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

            combined_view = np.hstack((frame, depth_colored))
            out.write(combined_view)

            if frame_idx % 20 == 0:
                print(f"  Processed frame {frame_idx}/{total_frames}...")

    cap.release()
    out.release()
    csv_file.close()
    print("=" * 60)
    print(f"✅ Pipeline Completed Successfully!")
    print(f"🎥 Video Output: {output_path}")
    print(f"📊 Audit Report: {csv_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()