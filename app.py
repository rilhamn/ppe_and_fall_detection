import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import time
import math

st.set_page_config(layout="wide")
st.title("PPE + Fall Monitoring Demo")

# --------------------------------------------------
# Load models once
# --------------------------------------------------
@st.cache_resource
def load_models():
    ppe = YOLO("runs/detect/train/weights/best.pt")
    fall = YOLO("runs/detect/fall/weights/best.pt")
    return ppe, fall

ppe_model, fall_model = load_models()


# --------------------------------------------------
# Video processor
# --------------------------------------------------
class YOLOVideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.ppe_model = ppe_model
        self.fall_model = fall_model

        self.ppe_hits = []
        self.fall_hits = []

        # temporal stability
        self.fall_streak = 0

    def transform(self, frame):

        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            h, w, _ = img.shape
            now = time.time()

            # --------------------------------------------------
            # PPE MODEL (person + violation)
            # --------------------------------------------------
            ppe_res = self.ppe_model(img, verbose=False)[0]
            annotated = ppe_res.plot()

            person_boxes = []
            violation_boxes = []

            if ppe_res.boxes is not None:
                for box, cls in zip(ppe_res.boxes.xyxy,
                                    ppe_res.boxes.cls):

                    label = self.ppe_model.names[int(cls)]
                    x1, y1, x2, y2 = map(int, box)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)

                    if label == "person":
                        person_boxes.append((x1, y1, x2, y2))

                    if label in ["no_helmet", "no_boots"]:
                        violation_boxes.append((x1, y1, x2, y2))

            def center(b):
                return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

            def dist(b1, b2):
                c1 = center(b1)
                c2 = center(b2)
                return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

            person_near_violation = False
            DIST_THRESHOLD = 180

            for p in person_boxes:
                for v in violation_boxes:
                    if dist(p, v) < DIST_THRESHOLD:
                        person_near_violation = True
                        break
                if person_near_violation:
                    break

            if person_near_violation:
                self.ppe_hits.append(now)

            self.ppe_hits = [t for t in self.ppe_hits if now - t <= 5]

            ppe_count = len(self.ppe_hits)
            ppe_alarm = ppe_count >= 25

            # --------------------------------------------------
            # FALL MODEL (run only inside person boxes)
            # --------------------------------------------------
            found_fall = False

            for (px1, py1, px2, py2) in person_boxes:

                crop = img[py1:py2, px1:px2]

                if crop.size == 0:
                    continue

                res = self.fall_model(crop, conf=0.5, verbose=False)[0]

                best_label = None
                best_conf = 0.0

                if res.boxes is not None:
                    for cls, conf in zip(res.boxes.cls, res.boxes.conf):

                        raw_label = self.fall_model.names[int(cls)]

                        if float(conf) > best_conf:
                            best_conf = float(conf)
                            best_label = raw_label

                # draw fall result on PERSON box
                if best_label is not None:

                    cv2.putText(
                        annotated,
                        f"{best_label} ({best_conf:.2f})",
                        (px1, max(py1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    if best_label.strip().lower() == "fall":
                        found_fall = True

            # --------------------------------------------------
            # temporal stabilisation for fall
            # --------------------------------------------------
            if found_fall:
                self.fall_streak += 1
            else:
                self.fall_streak = 0

            if self.fall_streak >= 5:
                self.fall_hits.append(now)

            self.fall_hits = [t for t in self.fall_hits if now - t <= 5]

            fall_count = len(self.fall_hits)
            fall_alarm = fall_count >= 25

            # --------------------------------------------------
            # DRAW STATUS
            # --------------------------------------------------
            cv2.putText(
                annotated,
                f"PPE hits (5s): {ppe_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Fall hits (5s): {fall_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            if ppe_alarm:
                cv2.putText(
                    annotated,
                    "ALARM : PPE VIOLATION",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

            if fall_alarm:
                cv2.putText(
                    annotated,
                    "ALARM : FALL DETECTED",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

            return annotated

        except Exception as e:
            print("Transform error:", e)
            return frame.to_ndarray(format="bgr24")


# --------------------------------------------------
# WebRTC
# --------------------------------------------------
webrtc_streamer(
    key="ppe-fall-demo",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True,
)
