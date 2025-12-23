import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ================= DRAW UTILITIES =================

def draw_counter_panel(frame, x, y, title, count, color):
    cv2.rectangle(frame, (x, y), (x + 360, y + 110), color, -1)
    cv2.putText(frame, title, (x + 15, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Count: {count}", (x + 15, y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)


def draw_end_screen(frame, title, persons, vehicles):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, title, (w // 4, h // 2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    cv2.putText(frame, f"Persons (Left): {persons}", (w // 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"Vehicles (Right): {vehicles}", (w // 4, h // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return frame


def write_summary(out, frame, title, persons, vehicles, fps, seconds=10):
    summary = draw_end_screen(frame, title, persons, vehicles)
    for _ in range(int(fps * seconds)):
        out.write(summary)

# ================= MAIN PIPELINE =================

def process_video(video_path):

    model = YOLO("yolov8n.pt")
    deepsort = DeepSort(max_age=25)

    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 🔥 CRITICAL FIX
    if fps is None or fps <= 5:
        fps = 25

    mid_x = w // 2

    # 🔥 ENSURE OUTPUT DIRECTORY
    os.makedirs("static/output", exist_ok=True)

    out_ds = cv2.VideoWriter(
        "static/output/deepsort_result.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h)
    )

    out_bt = cv2.VideoWriter(
        "static/output/bytetrack_result.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h)
    )

    if not out_ds.isOpened() or not out_bt.isOpened():
        raise RuntimeError("VideoWriter failed")


    ds_person_ids, ds_vehicle_ids = set(), set()
    bt_person_ids, bt_vehicle_ids = set(), set()

    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()

        # ================= YOLO DETECTION =================
        results = model(frame, imgsz=960, conf=0.2, iou=0.2)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls not in [0, 1, 2, 3]:
                continue

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # ================= DEEPSORT =================
        ds_frame = frame.copy()
        tracks = deepsort.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = (x1 + x2) // 2
            cls = track.det_class

            if cx < mid_x and cls == 0:
                ds_person_ids.add(track.track_id)
                cv2.rectangle(ds_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if cx >= mid_x and cls in [2, 3]:
                ds_vehicle_ids.add(track.track_id)
                cv2.rectangle(ds_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # ================= BYTETRACK =================
        bt_frame = frame.copy()
        bt_results = model.track(
            bt_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4
        )[0]

        if bt_results.boxes.id is not None:
            for i, box in enumerate(bt_results.boxes):
                track_id = int(bt_results.boxes.id[i])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2

                if cx < mid_x and cls == 0:
                    bt_person_ids.add(track_id)
                    cv2.rectangle(bt_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

                if cx >= mid_x and cls in [2, 3]:
                    bt_vehicle_ids.add(track_id)
                    cv2.rectangle(bt_frame, (x1, y1), (x2, y2), (200, 0, 0), 2)

        for f in (ds_frame, bt_frame):
            cv2.line(f, (mid_x, 0), (mid_x, h), (0, 255, 255), 2)

        out_ds.write(ds_frame)
        out_bt.write(bt_frame)

    # ================= END SUMMARY =================
    if last_frame is not None:
     write_summary(
        out_ds,
        last_frame.copy(),
        "DeepSORT Analysis Summary",
        len(ds_person_ids),
        len(ds_vehicle_ids),
        fps
    )

    write_summary(
        out_bt,
        last_frame.copy(),
        "ByteTrack Analysis Summary",
        len(bt_person_ids),
        len(bt_vehicle_ids),
        fps
    )

    out_ds.release()
    out_bt.release()
    cap.release()

    return {
        "DeepSORT": {
            "persons_left": len(ds_person_ids),
            "vehicles_right": len(ds_vehicle_ids)
        },
        "ByteTrack": {
            "persons_left": len(bt_person_ids),
            "vehicles_right": len(bt_vehicle_ids)
        }
    }
