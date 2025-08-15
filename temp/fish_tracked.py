import json, cv2, torch, time
from ultralytics import YOLO
from picamera2 import Picamera2

def convert_tensor(obj):
        if isinstance(obj, (float, int)):
            return obj
        elif isinstance(obj, list):
            return [convert_tensor(i) for i in obj]
        elif isinstance(obj, dict):
            return {key: convert_tensor(value) for key, value in obj.items()}
        elif isinstance(obj, torch.Tensor):
            return obj.item()  # Tensor에서 값을 추출
        return obj

model = YOLO('./best.pt')  # 학습된 물고기 탐지 모델

picam2 = Picamera2()

# 카메라 초기화 (해상도/FPS는 환경에 맞게)
camera_config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},  # RGB로 받음
    controls={"FrameDurationLimits": (33333, 33333)} # ≈30fps
)
picam2.configure(camera_config)
picam2.start()

# 첫 프레임 확인 및 크기 설정
frame = picam2.capture_array()               # RGB
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
height, width = frame_bgr.shape[:2]

last_preview_save = 0.0   # 프리뷰 이미지 주기 저장용

# 🔁 영상 전체 추적 실행
try:
    while True:
        # 프레임 획득 (RGB → BGR)
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 회전 (필요 없으면 지우세요)
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 🔍 객체 추적
        results = model.track(
            source=frame,
            conf=0.6,
            persist=True,
            verbose=False,
            save_txt=False,
            tracker="./bytetrack.yaml"
            )

        # 결과 시각화
        annotated_frame = results[0].plot()

        now = time.time()
        if now - last_preview_save > 1.0:
            cv2.imwrite("./tracked_preview.jpg", annotated_frame)
            last_preview_save = now

        # 추적된 객체 정보 수집
        frame_data = []
        for result in results[0].boxes:
            # 각 객체의 ID, bbox, confidence 값을 추출
            obj_id = result.id if result.id is not None else -1  # 객체 ID가 None이면 -1로 처리
            bbox = result.xywh[0].cpu().numpy().tolist()  # 바운딩 박스 (x, y, w, h)
            conf = result.conf[0].cpu().item()  # 신뢰도 (Tensor에서 값을 추출)

            # 객체 정보 딕셔너리 생성
            frame_data.append({
                'id': obj_id,
                'bbox': bbox,
                'confidence': conf
            })

        json_obj = json.dumps([convert_tensor(frame) for frame in frame_data], indent=4)
        print(json_obj)


except KeyboardInterrupt:
    pass

picam2.stop()
cv2.destroyAllWindows()
