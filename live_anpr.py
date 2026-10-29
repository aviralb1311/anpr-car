#!/usr/bin/env python3
"""
Live Automatic Number Plate Recognition (ANPR)
Based on the YOLOv8 + EasyOCR approach in:
https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8

Run from the repository root:
    python live_anpr.py
    python live_anpr.py --source 0
    python live_anpr.py --source "rtsp://USER:PASS@CAMERA_IP:554/STREAM_PATH"

Expected files:
    yolov8n.pt
    license_plate_detector.pt
    sort/sort.py
Dependencies are listed in the repository's requirements.txt.
"""

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

try:
    from sort.sort import Sort
except ImportError as exc:
    raise SystemExit(
        "Could not import SORT. From the repository root, run:\n"
        "  git clone https://github.com/abewley/sort.git\n"
        "Then run this script from the repository root."
    ) from exc


VEHICLE_CLASSES = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck


def parse_source(value):
    """Convert numeric camera IDs such as '0' to integers; leave URLs/paths as strings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def box_center_inside(inner_box, outer_box):
    """True when the center of the plate box falls inside a vehicle box."""
    x1, y1, x2, y2 = inner_box
    vx1, vy1, vx2, vy2 = outer_box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return vx1 <= cx <= vx2 and vy1 <= cy <= vy2


def best_vehicle_for_plate(plate_box, tracked_vehicles):
    """
    Associate a plate with the tracked vehicle containing its center.
    Returns (vehicle_track_id, vehicle_box), or (None, None).
    """
    matches = []
    px1, py1, px2, py2 = plate_box
    plate_area = max(1, (px2 - px1) * (py2 - py1))

    for vehicle in tracked_vehicles:
        vx1, vy1, vx2, vy2, track_id = vehicle
        vehicle_box = (int(vx1), int(vy1), int(vx2), int(vy2))
        if box_center_inside(plate_box, vehicle_box):
            vehicle_area = max(1, (vx2 - vx1) * (vy2 - vy1))
            # Prefer the smallest containing vehicle box if boxes overlap.
            matches.append((vehicle_area, int(track_id), vehicle_box))

    if not matches:
        return None, None

    _, track_id, vehicle_box = min(matches, key=lambda item: item[0])
    return track_id, vehicle_box


def recognize_plate(reader, crop):
    """Run OCR and return the highest-confidence text and score."""
    if crop is None or crop.size == 0:
        return "", 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Upscale small plate crops to improve OCR readability.
    h, w = gray.shape[:2]
    if w < 240 and h > 0:
        scale = min(3.0, 240.0 / w)
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC
        )

    # EasyOCR's allowlist reduces irrelevant punctuation in vehicle plates.
    detections = reader.readtext(
        gray,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    if not detections:
        return "", 0.0

    candidates = []
    for _bbox, raw_text, confidence in detections:
        text = "".join(ch for ch in raw_text.upper() if ch.isalnum())
        if text:
            candidates.append((text, float(confidence)))

    if not candidates:
        return "", 0.0

    return max(candidates, key=lambda item: item[1])


def write_event(csv_writer, csv_file, event):
    csv_writer.writerow(event)
    csv_file.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Live YOLOv8 vehicle + license plate detection with EasyOCR and SORT."
    )
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (default: 0), video path, or RTSP/HTTP stream URL."
    )
    parser.add_argument("--vehicle-model", default="yolov8n.pt")
    parser.add_argument("--plate-model", default="license_plate_detector.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--vehicle-conf", type=float, default=0.35)
    parser.add_argument("--plate-conf", type=float, default=0.35)
    parser.add_argument(
        "--ocr-every", type=int, default=5,
        help="Run OCR at most once every N frames per tracked vehicle (default: 5)."
    )
    parser.add_argument(
        "--min-ocr-conf", type=float, default=0.40,
        help="Minimum OCR confidence before a reading is added to the vote."
    )
    parser.add_argument(
        "--csv", default="live_anpr_events.csv",
        help="CSV file for accepted plate observations."
    )
    parser.add_argument(
        "--save-video", default="",
        help="Optional output video path, e.g. live_output.mp4. Empty disables recording."
    )
    parser.add_argument(
        "--cpu-ocr", action="store_true",
        help="Force EasyOCR to use CPU (default). This flag is retained for clarity."
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not open an OpenCV display window (useful for headless runs)."
    )
    args = parser.parse_args()

    if args.ocr_every < 1:
        parser.error("--ocr-every must be at least 1")

    for model_path in (args.vehicle_model, args.plate_model):
        if not Path(model_path).is_file():
            raise SystemExit(
                f"Model file not found: {model_path}\n"
                "Run this script from the repository root and ensure both model files exist."
            )

    print("[INFO] Loading YOLO models...")
    vehicle_model = YOLO(args.vehicle_model)
    plate_model = YOLO(args.plate_model)

    print("[INFO] Loading EasyOCR (English, CPU)...")
    reader = easyocr.Reader(["en"], gpu=False)

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open source: {args.source}\n"
            "For a webcam, try --source 0. For CCTV, verify the RTSP URL and credentials."
        )

    # Lower buffering can reduce latency on supported camera backends.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)
    frame_number = 0
    last_ocr_frame = defaultdict(lambda: -10**9)
    plate_votes = defaultdict(Counter)
    plate_confidence_sum = defaultdict(float)
    plate_confidence_count = defaultdict(int)
    last_logged_plate = {}
    last_logged_time = defaultdict(float)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_path.stat().st_size == 0:
        csv_writer.writerow([
            "timestamp", "frame", "vehicle_track_id", "plate_text",
            "ocr_confidence", "plate_detector_confidence",
            "x1", "y1", "x2", "y2"
        ])
        csv_file.flush()

    video_writer = None
    fps_counter_start = time.time()
    fps_counter_frames = 0
    display_fps = 0.0
    last_frame_shape = None

    print("[INFO] Live ANPR started. Press Q in the video window to quit.")
    print(f"[INFO] Event CSV: {csv_path.resolve()}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Frame read failed; stopping stream.")
                break

            frame_number += 1
            height, width = frame.shape[:2]
            last_frame_shape = (width, height)

            # Vehicle inference
            vehicle_result = vehicle_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.vehicle_conf,
                verbose=False
            )[0]

            vehicle_detections = []
            if vehicle_result.boxes is not None:
                for box in vehicle_result.boxes:
                    class_id = int(box.cls[0].item())
                    if class_id not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    score = float(box.conf[0].item())
                    vehicle_detections.append([x1, y1, x2, y2, score])

            if vehicle_detections:
                tracker_input = np.asarray(vehicle_detections, dtype=np.float32)
            else:
                tracker_input = np.empty((0, 5), dtype=np.float32)

            tracked_vehicles = tracker.update(tracker_input)

            # Draw tracked vehicles.
            for vx1, vy1, vx2, vy2, track_id in tracked_vehicles:
                vb = clamp_box((vx1, vy1, vx2, vy2), width, height)
                if vb is None:
                    continue
                x1, y1, x2, y2 = vb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame, f"Vehicle ID {int(track_id)}",
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2
                )

            # Plate inference
            plate_result = plate_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.plate_conf,
                verbose=False
            )[0]

            if plate_result.boxes is not None:
                for box in plate_result.boxes:
                    raw_box = box.xyxy[0].tolist()
                    plate_box = clamp_box(raw_box, width, height)
                    if plate_box is None:
                        continue

                    plate_detector_conf = float(box.conf[0].item())
                    track_id, vehicle_box = best_vehicle_for_plate(
                        plate_box, tracked_vehicles
                    )

                    x1, y1, x2, y2 = plate_box
                    crop = frame[y1:y2, x1:x2]
                    shown_text = "Plate"

                    if track_id is not None:
                        # Throttle OCR per vehicle to reduce CPU load.
                        if frame_number - last_ocr_frame[track_id] >= args.ocr_every:
                            text, ocr_conf = recognize_plate(reader, crop)
                            last_ocr_frame[track_id] = frame_number

                            if text and ocr_conf >= args.min_ocr_conf:
                                plate_votes[track_id][text] += 1
                                plate_confidence_sum[track_id] += ocr_conf
                                plate_confidence_count[track_id] += 1

                                # Log a new/change in the majority reading, with a
                                # short cooldown to avoid duplicate events every frame.
                                voted_text, vote_count = plate_votes[track_id].most_common(1)[0]
                                now = time.time()
                                should_log = (
                                    voted_text != last_logged_plate.get(track_id)
                                    or now - last_logged_time[track_id] >= 5.0
                                )
                                if should_log and vote_count >= 1:
                                    avg_conf = (
                                        plate_confidence_sum[track_id]
                                        / max(1, plate_confidence_count[track_id])
                                    )
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    write_event(
                                        csv_writer, csv_file,
                                        [
                                            timestamp, frame_number, track_id,
                                            voted_text, f"{avg_conf:.3f}",
                                            f"{plate_detector_conf:.3f}",
                                            x1, y1, x2, y2
                                        ]
                                    )
                                    print(
                                        f"[PLATE] vehicle_id={track_id} "
                                        f"text={voted_text} "
                                        f"ocr_conf={avg_conf:.2f}"
                                    )
                                    last_logged_plate[track_id] = voted_text
                                    last_logged_time[track_id] = now

                        if plate_votes[track_id]:
                            shown_text = plate_votes[track_id].most_common(1)[0][0]
                        else:
                            shown_text = f"Vehicle {track_id}: reading..."
                    else:
                        shown_text = "Unassociated plate"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, shown_text[:40], (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                    )

            # Optional annotated video writer.
            if args.save_video:
                if video_writer is None:
                    source_fps = cap.get(cv2.CAP_PROP_FPS)
                    if not source_fps or source_fps <= 1:
                        source_fps = 20.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        args.save_video, fourcc, source_fps, (width, height)
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {args.save_video}")
                video_writer.write(frame)

            # Display measured processing FPS (not camera's advertised FPS).
            fps_counter_frames += 1
            elapsed = time.time() - fps_counter_start
            if elapsed >= 1.0:
                display_fps = fps_counter_frames / elapsed
                fps_counter_frames = 0
                fps_counter_start = time.time()

            cv2.putText(
                frame, f"Processing FPS: {display_fps:.1f}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2
            )

            if not args.no_display:
                cv2.imshow("Live ANPR - Press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        csv_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"[INFO] Saved plate observations to: {csv_path.resolve()}")
    if args.save_video:
        print(f"[INFO] Saved annotated video to: {Path(args.save_video).resolve()}")


if __name__ == "__main__":
    main()
