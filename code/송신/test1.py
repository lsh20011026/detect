import cv2
from ultralytics import YOLO

def initialize_system(model_path="yolo11n.pt"):
    """YOLO 모델과 웹캠 초기화"""
    model = YOLO(model_path)  # YOLOv11 Nano
    cap = cv2.VideoCapture(0)  # /dev/video0
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")
    return model, cap

def process_frame(model, frame, frame_count):
    """YOLO 탐지 실행 및 콘솔 출력"""
    print(f"\n=== Frame {frame_count} ===")
    results = model.predict(frame, device=0, verbose=False)  # GPU 추론
    return results

def visualize_results(results):
    """탐지 결과 시각화 및 객체 정보 출력"""
    for r in results:
        annotated_frame = r.plot()  # 자동 박싱/라벨

        # 객체 상세 출력
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"Class ID: {cls_id}, Confidence: {conf:.2f}, Box: {xyxy}")

        cv2.imshow("YOLOv11 Detection", annotated_frame)

def main_loop(model, cap):
    """메인 루프: 프레임 읽기 + 처리 + 종료 제어"""
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        frame_count += 1
        results = process_frame(model, frame, frame_count)
        visualize_results(results)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def cleanup(cap):
    """리소스 정리"""
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model, cap = initialize_system()
    try:
        main_loop(model, cap)
        print(f"\n✅ 총 {cap.get(cv2.CAP_PROP_FRAME_COUNT) or 'N/A'} 프레임 분석 완료!")
    finally:
        cleanup(cap)
