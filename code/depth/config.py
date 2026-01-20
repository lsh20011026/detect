"""HybridTracker 설정 관리"""
from dataclasses import dataclass

@dataclass
class TrackerConfig:
    ROI_W: int = 60
    ROI_H: int = 60
    REDETECT_INTERVAL: int = 10
    TEMPLATE_CONF_THRESH: float = 0.65
    MAX_LOST_FRAMES: int = 45
    KALMAN_ONLY_FRAMES: int = 15
    TX_INTERVAL: int = 5
    YOLO_CONF: float = 0.35
    YOLO_IMGSZ: int = 256
    YOLO_MAX_DET: int = 5
    CAM_WIDTH: int = 640
    CAM_HEIGHT: int = 480
    MODEL_PATH: str = "/home/nes/yolo11n.engine"
    
    # 🔥 거리 측정 캘리브레이션 (실제값으로 수정!)
    PIXEL_PER_METER_X: float = 120.0  # 1m당 X 픽셀
    PIXEL_PER_METER_Y: float = 90.0   # 1m당 Y 픽셀
    BASE_DISTANCE: float = 5.0        # 기준 거리(m)
    FOV_HORIZONTAL: float = 60.0      # 카메라 수평 FOV(도)
    
    # 🔥 거리 추정용 (5m에서 1도 회전시 Δx 픽셀 - 실험측정!)
    BASE_DELTA_X: float = 120.0       # 캘리브레이션 필요
    MIN_DELTA_FRAMES: int = 5         # 최소 변화 프레임



