from src.video.stream_reader import StreamReader
from src.video.frame_sampler import FrameSampler
from src.vision.detector import PersonDetector

reader = StreamReader(source="your_video.mp4", loop=False, reconnect=False)
sampler = FrameSampler(frame_skip=2)  # 每2帧处理1帧
detector = PersonDetector(model_name="yolov8n.pt", conf_threshold=0.35, device="cuda")

assert reader.open(), "视频打开失败"
detector.load_model()

while True:
    ok, frame_obj = reader.read()
    if not ok:
        break

    frame_id = frame_obj["frame_id"]
    if not sampler.should_process(frame_id):
        continue

    image = frame_obj["image"]
    person_dets = detector.predict(image)
    print(frame_id, frame_obj["timestamp"], person_dets)

reader.release()
