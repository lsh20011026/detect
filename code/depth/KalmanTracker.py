import cv2
import numpy as np

class KalmanTracker:
    """ROI 중심 기반 칼만 필터 추적기"""
    
    def __init__(self):
        self.kalman = None
        self.initialized = False
        self.use_for_tracking = False
        
    def init_kalman(self, cx, cy):
        """ROI 중심 기준 칼만 필터 초기화"""
        self.kalman = cv2.KalmanFilter(4, 2)
        
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
        
        self.kalman.statePost = np.array([[cx], [cy], [0.], [0.]], np.float32)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1
        self.initialized = True
        self.use_for_tracking = False
    
    def reset(self):
        """칼만 필터 리셋"""
        self.kalman = None
        self.initialized = False
        self.use_for_tracking = False
    
    def predict_roi(self, frame_w, frame_h, roi_w=60, roi_h=60):
        """칼만 예측으로 ROI 업데이트"""
        if not self.initialized:
            return False, None
            
        prediction = self.kalman.predict()
        kx, ky = prediction[0, 0], prediction[1, 0]
        
        x1 = max(0, int(kx - roi_w / 2))
        y1 = max(0, int(ky - roi_h / 2))
        x2 = min(frame_w - 1, int(kx + roi_w / 2))
        y2 = min(frame_h - 1, int(ky + roi_h / 2))
        
        roi = (x1, y1, x2, y2)
        return (x2 > x1 and y2 > y1), roi
    
    def correct(self, meas_x, meas_y):
        """측정값으로 칼만 보정"""
        if self.initialized:
            measurement = np.array([[np.float32(meas_x)], [np.float32(meas_y)]])
            self.kalman.correct(measurement)
    
    def get_position(self):
        """현재 칼만 위치 반환"""
        if self.initialized and self.kalman is not None:
            return int(self.kalman.statePost[0, 0]), int(self.kalman.statePost[1, 0])
        return None, None




