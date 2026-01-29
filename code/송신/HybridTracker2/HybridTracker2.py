import cv2
import numpy as np
import time
from ultralytics import YOLO
import serial
import sys
import struct
sys.path.append("/home/nes/.local/lib/python3.10/site-packages")
from KalmanTracker import KalmanTracker
from serial_manager import SerialManager
from config import TrackerConfig 
from dataclasses import dataclass 

class HybridTracker:
    """YOLO + Template Matching 하이브리드 추적 시스템 (🚀 YOLO 재탐지 ON + BBOX 숨김 + 카메라 전환)"""

    def __init__(self):
        # 🔥 설정 객체 (기존 10줄 하드코딩 → 1줄)
        self.config = TrackerConfig()
        
        # 🔥 SerialManager로 교체
        self.serial_mgr = SerialManager()
        self.last_tx_frame = 0
        
        # 🔥 KalmanTracker 인스턴스 추가
        self.kalman_tracker = KalmanTracker()
        
        # 🔥 카메라 전환 관련 변수 추가
        self.current_cam_index = 0
        self.available_cameras = []
        self.cap = None
        
        # 추적 상태
        self.current_roi = None
        self.template = None
        self.tracking_mode = "NONE"  # "NONE" / "TEMPLATE" / "KALMAN_ONLY"
        self.yolo_enabled = False
        self.roi_tracking_active = False  # 🔥 ROI 추적 시작 여부
        self.show_yolo_boxes = True  # 🔥 YOLO BBOX 표시 여부 (클릭 후 OFF)

        # 상태 변수
        self.frame_h = 0
        self.frame_w = 0
        self.lost_frame_count = 0
        self.frame_count = 0
        self.last_conf = 0.0
        self.kalman_only_count = 0

        # 하드웨어
        self.model = None

        # 마우스 콜백용
        self.mouse_param = {"frame": None, "boxes": None}

    # ================= 🔥 카메라 전환 기능 ==================
    def detect_available_cameras(self):
        """사용 가능한 USB 카메라 자동 감지"""
        self.available_cameras = []
        for i in range(4):  # 0~3번 카메라 테스트
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 100 and h > 100:  # 유효한 카메라
                    self.available_cameras.append(i)
                cap.release()
        print(f"📹 사용 가능 카메라: {self.available_cameras}")

    def switch_camera(self):
        """다음 카메라로 전환"""
        if len(self.available_cameras) <= 1:
            print("❌ 전환할 카메라가 없습니다")
            return
            
        # 현재 카메라 해제
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # 다음 카메라 인덱스 계산
        self.current_cam_index = (self.current_cam_index + 1) % len(self.available_cameras)
        new_cam_id = self.available_cameras[self.current_cam_index]
        
        # 새 카메라 초기화
        self.init_single_camera(new_cam_id)
        self.reset_tracking()  # 추적 리셋 (새 카메라에서는 새로 시작)
        
        print(f"🔄 카메라 전환: {new_cam_id} ({self.current_cam_index+1}/{len(self.available_cameras)})")
        
    def init_single_camera(self, cam_index):
        """단일 카메라 초기화"""
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise ValueError(f"❌ Cannot open camera index {cam_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAM_HEIGHT)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Camera {cam_index}: {self.frame_w}x{self.frame_h}")

    # ================= Kalman 래퍼 ==================
    def _init_kalman(self, cx, cy):
        """KalmanTracker 초기화 래퍼"""
        self.kalman_tracker.init_kalman(cx, cy)

    def _reset_kalman(self):
        """KalmanTracker 리셋 래퍼"""
        self.kalman_tracker.reset()

    def _predict_kalman_roi(self):
        """칼만 예측 ROI 래퍼"""
        success, roi = self.kalman_tracker.predict_roi(
            self.frame_w, self.frame_h, self.config.ROI_W, self.config.ROI_H
        )
        if success:
            self.current_roi = roi
            self.kalman_only_count += 1
            self.lost_frame_count = 0
            self.tracking_mode = "KALMAN_ONLY"
            self.kalman_tracker.use_for_tracking = True
        return success

    # ============== 하드웨어 초기화 ==============
    def init_hardware(self, cam_index=0):
        """YOLO + 카메라 초기화 (시리얼은 생성자에서!)"""
        # 🔥 사용 가능 카메라 감지
        self.detect_available_cameras()
        if not self.available_cameras:
            raise ValueError("❌ 사용 가능한 카메라가 없습니다")
        
        # 🔥 YOLO 모델 안전 로드
        try:
            self.model = YOLO(self.config.MODEL_PATH, task='detect')
            print("🚀 TensorRT YOLO loaded")
        except Exception as e:
            print(f"❌ YOLO model load failed: {e}")
            self.model = None

        # 첫 번째 카메라 초기화
        self.current_cam_index = 0
        self.init_single_camera(self.available_cameras[self.current_cam_index])

    # 🔥 SerialManager 사용
    def send_serial_data(self, frame_id, roi, conf, mode, fps, status):
        """SerialManager 위임"""
        if not self.serial_mgr.is_connected():
            return
        self.serial_mgr.send_tracking_data(frame_id, roi, conf, mode, fps, status)

    # ============== UI / 마우스 ==============
    def setup_window(self):
        """윈도우 및 마우스 콜백 설정"""
        win_name = "HybridTracker (Drone) - YOLO Redetect ON + CAM_SWITCH"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setMouseCallback(win_name, self.mouse_callback, self.mouse_param)
        return win_name

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param["frame"]
            boxes = param["boxes"]

            clicked_on_object = self._handle_yolo_click(x, y, boxes, frame)
            if not clicked_on_object:
                self._handle_manual_roi(x, y, frame)

        elif event == cv2.EVENT_MOUSEWHEEL:
            self._handle_zoom(flags)

    def _handle_yolo_click(self, x, y, boxes, frame):
        """YOLO 박스 클릭 처리 - 🔥 BBOX 숨기고 재탐지는 계속"""
        if boxes is None or len(boxes) == 0 or not self.yolo_enabled:
            return False

        for box in boxes:
            try:
                b_xyxy = box.xyxy[0].tolist()
                if (b_xyxy[0] <= x <= b_xyxy[2] and 
                    b_xyxy[1] <= y <= b_xyxy[3]):
                    self._set_roi_from_box(b_xyxy, frame, shrink=0.1)
                    print(f"[YOLO→TEMPLATE] ROI: {self.current_roi}")
                    self.lost_frame_count = 0
                    self.roi_tracking_active = True
                    self.show_yolo_boxes = False  # 🔥 BBOX 완전 숨김
                    self.mouse_param["boxes"] = None  # 클릭 후 초기화
                    return True
            except:
                continue
        return False

    def _handle_manual_roi(self, x, y, frame):
        """수동 ROI 설정"""
        x1 = max(0, int(x - self.config.ROI_W / 2))
        y1 = max(0, int(y - self.config.ROI_H / 2))
        x2 = min(self.frame_w - 1, int(x + self.config.ROI_W / 2))
        y2 = min(self.frame_h - 1, int(y + self.config.ROI_H / 2))

        if x2 > x1 and y2 > y1:
            self.current_roi = (x1, y1, x2, y2)
            self.template = frame[y1:y2, x1:x2].copy()
            self.tracking_mode = "TEMPLATE"
            self.lost_frame_count = 0
            self.kalman_only_count = 0
            self.roi_tracking_active = True
            self.show_yolo_boxes = False  # 🔥 수동 ROI도 BBOX 숨김
            print(f"[MANUAL] ROI: {self.current_roi}")

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self._init_kalman(cx, cy)

    def _set_roi_from_box(self, xyxy, frame, shrink=0.1):
        """박스에서 ROI 생성"""
        x1, y1, x2, y2 = map(int, xyxy)
        w, h = x2 - x1, y2 - y1
        x1 = int(x1 + w * shrink)
        x2 = int(x2 - w * shrink)
        y1 = int(y1 + h * shrink)
        y2 = int(y2 - h * shrink)

        self.current_roi = (max(0, x1), max(0, y1),
                            min(self.frame_w - 1, x2), min(self.frame_h - 1, y2))
        self.template = frame[y1:y2, x1:x2].copy()
        self.tracking_mode = "TEMPLATE"
        self.kalman_only_count = 0
        self.roi_tracking_active = True

        cx = (self.current_roi[0] + self.current_roi[2]) / 2
        cy = (self.current_roi[1] + self.current_roi[3]) / 2
        self._init_kalman(cx, cy)

    def _handle_zoom(self, flags):
        """마우스 휠 줌"""
        win_name = "HybridTracker (Drone) - YOLO Redetect ON + CAM_SWITCH"
        rect = cv2.getWindowImageRect(win_name)
        w, h = rect[2], rect[3]

        if flags > 0:
            new_w, new_h = min(1920, w + 100), min(1080, h + 100)
        else:
            new_w, new_h = max(640, w - 100), max(480, h - 100)

        cv2.resizeWindow(win_name, new_w, new_h)

    # ============== 템플릿 매칭 ==============
    def template_matching(self, frame):
        """템플릿 매칭 추적 + 칼만 보완"""
        if self.template is None or self.current_roi is None:
            return False, 0.0

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tpl_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            th, tw = tpl_gray.shape[:2]

            rx1, ry1, rx2, ry2 = self.current_roi
            roi_cx, roi_cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2

            margin = 80
            sx1 = max(0, rx1 - margin)
            sy1 = max(0, ry1 - margin)
            sx2 = min(self.frame_w, rx2 + margin)
            sy2 = min(self.frame_h, ry2 + margin)

            if (sx2 - sx1) > tw and (sy2 - sy1) > th:
                search_roi = gray[sy1:sy2, sx1:sx2]
                res = cv2.matchTemplate(search_roi, tpl_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                meas_x = max_loc[0] + sx1 + tw / 2.0
                meas_y = max_loc[1] + sy1 + th / 2.0
                drift_dist = np.sqrt((meas_x - roi_cx) ** 2 + (meas_y - roi_cy) ** 2)

                self._log_template(
                    frame_count=self.frame_count,
                    max_val=max_val,
                    roi=self.current_roi,
                    new_pos=(meas_x, meas_y),
                    drift=drift_dist
                )

                self.last_conf = max_val

                if max_val > self.config.TEMPLATE_CONF_THRESH:
                    x1 = int(meas_x - tw / 2)
                    y1 = int(meas_y - th / 2)
                    x2 = x1 + tw
                    y2 = y1 + th

                    self.current_roi = (max(0, x1), max(0, y1),
                                        min(self.frame_w - 1, x2), min(self.frame_h - 1, y2))
                    self.template = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                    self.lost_frame_count = 0
                    self.kalman_only_count = 0
                    self.tracking_mode = "TEMPLATE"

                    if self.kalman_tracker.initialized:
                        self.kalman_tracker.correct(meas_x, meas_y)

                    self.kalman_tracker.use_for_tracking = False
                    return True, max_val
                else:
                    self.lost_frame_count += 1
                    self._fallback_to_kalman()
                    return False, max_val
            else:
                self.lost_frame_count += 1
                self._fallback_to_kalman()
                return False, 0.0

        except Exception as e:
            print(f"💥 Template error: {e}")
            self._fallback_to_kalman()
        return False, 0.0

    def _fallback_to_kalman(self):
        """템플릿 실패시 칼만 추적으로 폴백"""
        if self.kalman_tracker.initialized and self._predict_kalman_roi():
            print(f"🔥 KALMAN_ONLY[{self.kalman_only_count}] activated")
        else:
            self.lost_frame_count += 1

    def _log_template(self, frame_count, max_val, roi, new_pos, drift):
        """템플릿 로그 출력"""
        print(f"F{frame_count:4d} | TMP:{max_val:.3f} | "
              f"ROI{roi}→NEW{new_pos} | DRIFT:{drift:.1f}px")

    # ============== YOLO ==============
    def yolo_detection(self, frame):
        """YOLO 객체 탐지 - 🔥 재탐지는 계속, BBOX는 show_yolo_boxes에 따라"""
        self.mouse_param["boxes"] = None  # 항상 초기화

        if not self.yolo_enabled or self.model is None:
            return

        # 🔥 재탐지는 항상 실행
        try:
            results = self.model.predict(
                source=frame, device=0, verbose=False,
                conf=self.config.YOLO_CONF, imgsz=self.config.YOLO_IMGSZ, max_det=self.config.YOLO_MAX_DET
            )

            boxes = None
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    self.mouse_param["boxes"] = boxes
                    
                    # 🔥 BBOX 그리기 제어 (클릭 후 숨김)
                    if self.show_yolo_boxes:
                        self._draw_yolo_boxes(r, frame)
                    
                    break
                self.mouse_param["boxes"] = boxes

            self._yolo_redetect(boxes, frame)

        except Exception as e:
            print(f"YOLO error: {e}")

    def _draw_yolo_boxes(self, result, frame):
        """YOLO 박스 그리기 - show_yolo_boxes=True일 때만"""
        for box in result.boxes:
            try:
                xyxy = box.xyxy[0].tolist()
                cv2.rectangle(frame,
                              (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])),
                              (128, 128, 128), 1)
            except:
                continue

    def _yolo_redetect(self, boxes, frame):
        """ROI 내 YOLO 재탐지 - 🔥 항상 동작"""
        if (self.frame_count % self.config.REDETECT_INTERVAL != 0 or
                self.current_roi is None):
            return

        rx1, ry1, rx2, ry2 = self.current_roi
        roi_cx = (rx1 + rx2) / 2
        roi_cy = (ry1 + ry2) / 2

        best_box, best_score, best_conf = self._find_best_roi_box(
            boxes, roi_cx, roi_cy, rx1, rx2, ry1, ry2
        )

        if best_box is not None:
            self._set_roi_from_box(best_box, frame)
            print(f"[REDETECT✓] conf={best_conf:.3f}")
            self.lost_frame_count = 0
            self.kalman_only_count = 0

    def _find_best_roi_box(self, boxes, roi_cx, roi_cy, rx1, rx2, ry1, ry2):
        """ROI 내 최적 박스 찾기"""
        best_box = None
        best_score = -1
        best_conf = 0

        if boxes is None or len(boxes) == 0:
            return best_box, best_score, best_conf

        for box in boxes:
            try:
                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                conf = float(box.conf[0])

                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    dist2 = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
                    score = conf * 1000 - dist2

                    if score > best_score:
                        best_score = score
                        best_box = xyxy
                        best_conf = conf
            except:
                continue

        return best_box, best_score, best_conf

    # ============== 그리기 / 상태 ==============
    def draw_roi(self, frame):
        """ROI 시각화 + Kalman 위치 점찍기"""
        if self.current_roi is not None:
            x1, y1, x2, y2 = map(int, self.current_roi)
            
            if self.tracking_mode == "TEMPLATE":
                color = (0, 255, 255)
            elif self.tracking_mode == "KALMAN_ONLY":
                color = (255, 0, 255)
            else:
                color = (0, 128, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, self.tracking_mode, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 🔥 KalmanTracker 위치 표시
            kx, ky = self.kalman_tracker.get_position()
            if kx is not None:
                kf_color = (0, 0, 255) if self.kalman_tracker.use_for_tracking else (0, 255, 0)
                cv2.circle(frame, (kx, ky), 5 if self.kalman_tracker.use_for_tracking else 3, kf_color, -1)
                cv2.putText(frame, "KF" + ("*" if self.kalman_tracker.use_for_tracking else ""), 
                           (kx + 8, ky), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kf_color, 1)

    def draw_status(self, frame, fps):
        """상태 표시 - 🔥 카메라 정보 추가"""
        cam_info = f"CAM{self.available_cameras[self.current_cam_index]}"
        bbox_status = "BBOX:OFF" if not self.show_yolo_boxes else "BBOX:ON "
        status = (f"M:{self.tracking_mode[:4]} Y:{'ON' if self.yolo_enabled else 'OFF'} "
                 f"{bbox_status}L:{self.lost_frame_count} K:{self.kalman_only_count} "
                 f"T:{'ON' if self.roi_tracking_active else 'OFF'}")
        cv2.putText(frame, status, (10, self.frame_h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{cam_info} ({len(self.available_cameras)}cams)", (10, self.frame_h - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "t:YOLO b:BBOX n:NEXT_CAM r:reset q:quit Wheel:ZOOM TX:ON", 
                   (10, self.frame_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS:{fps:.1f} CONF:{self.last_conf:.2f}", (10, self.frame_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ============== 메인 처리 ==============
    def process_frame(self, frame):
        """단일 프레임 처리"""
        self.frame_count += 1
        self.mouse_param["frame"] = frame

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            progress = self.frame_count / total_frames * 100
            cv2.putText(frame, f"F:{self.frame_count}/{total_frames} ({progress:.1f}%)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, f"F:{self.frame_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        fps_est = 0.0
        tracking_success = False

        if (self.current_roi is not None and self.tracking_mode in ["TEMPLATE", "KALMAN_ONLY"] 
            and self.template is not None):
            
            if self.tracking_mode == "TEMPLATE":
                success, conf = self.template_matching(frame)
                self.last_conf = conf
                tracking_success = success
                fps_est = 30.0
            else:
                tracking_success = True
                self.last_conf = 0.75
                fps_est = 60.0

        if not tracking_success:
            if self.lost_frame_count > self.config.MAX_LOST_FRAMES:
                print("💥 MAX_LOST → FULL RESET")
                self.reset_tracking()
            elif self.kalman_only_count > self.config.KALMAN_ONLY_FRAMES:
                print("💥 KALMAN_TIMEOUT → YOLO REDETECT")
                self.template = None

        self.yolo_detection(frame)  # 🔥 항상 YOLO 재탐지 실행
        self.draw_roi(frame)

        if self.frame_count % self.config.TX_INTERVAL == 0:
            status = 'LOST' if self.lost_frame_count > 10 else 'OK'
            self.send_serial_data(
                frame_id=self.frame_count,
                roi=self.current_roi,
                conf=self.last_conf,
                mode=self.tracking_mode,
                fps=fps_est,
                status=status
            )

        return frame

    def handle_keys(self, key, win_name):
        """키 입력 처리 - 🔥 'n'키로 카메라 전환 추가"""
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.reset_tracking()
            self.show_yolo_boxes = True  # 🔥 리셋시 BBOX 복원
            print("🔄 Reset - BBOX 복원")
        elif key == ord('t'):
            self.yolo_enabled = not self.yolo_enabled
            print(f"YOLO {'ON' if self.yolo_enabled else 'OFF'}")
        elif key == ord('b'):  # 🔥 BBOX 토글
            self.show_yolo_boxes = not self.show_yolo_boxes
            print(f"BBOX {'ON' if self.show_yolo_boxes else 'OFF'}")
        elif key == ord('n'):  # 🔥 카메라 전환
            self.switch_camera()
        return True

    def reset_tracking(self):
        """추적 리셋 - 🔥 BBOX 상태 복원"""
        self.current_roi = None
        self.template = None
        self.tracking_mode = "NONE"
        self.lost_frame_count = 0
        self.kalman_only_count = 0
        self.roi_tracking_active = False
        self.show_yolo_boxes = True  # 🔥 리셋시 BBOX 복원
        self._reset_kalman()

    # 🔥 변경된 cleanup
    def cleanup(self):
        """SerialManager 정리 추가"""
        if self.cap:
            self.cap.release()
        self.serial_mgr.close()
        cv2.destroyAllWindows()
        print("👋 Tracker ended")

    def run(self, cam_index=0):
        """메인 루프 (USB 카메라용 - 자동 감지)"""
        self.init_hardware(cam_index)
        win_name = self.setup_window()

        print(f"🎬 Camera stream | t=YOLO b=BBOX n=NEXT_CAM r=RESET q=QUIT")
        print(f"🔥 현재 카메라: {self.available_cameras[self.current_cam_index]}")
        print(f"🔥 YOLO REDETECT:ON | BBOX:클릭후OFF | 'n'로 카메라전환 | 📡 Serial TX:ON")

        prev_time = time.time()
        while True:
            if self.cap is None or not self.cap.isOpened():
                print("💥 카메라 연결 오류 - 재시작")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("💥 Camera read failed")
                time.sleep(0.1)
                continue

            frame = self.process_frame(frame)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if self.frame_count > 1 else 0
            prev_time = curr_time
            self.draw_status(frame, fps)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if not self.handle_keys(key, win_name):
                break

        self.cleanup()

if __name__ == "__main__":
    tracker = HybridTracker()
    tracker.run(cam_index=0)


