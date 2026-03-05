import os
import cv2
import numpy as np
import torch
import gc
from ultralytics import YOLO
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# ============= Adaptive Fusion Tracker =============

class AdaptiveFusionTracker:
    """Adaptive fusion tracker for YOLO and SAM2 with state-machine switching"""
    def __init__(self, trend_window=10, long_window=30):
        self.sam_conf_history = []
        self.trend_window = trend_window
        self.long_window = long_window
        self.last_box = None
        self.initialized = False
        
    def initialize(self, bbox):
        """Initialize with the first detection box [x1, y1, x2, y2]"""
        self.last_box = bbox
        self.initialized = True
        self.sam_conf_history = []
    
    def update_sam_conf_history(self, sam_conf):
        """Update SAM2 confidence history"""
        self.sam_conf_history.append(sam_conf)
        if len(self.sam_conf_history) > self.long_window:
            self.sam_conf_history.pop(0)
    
    def get_baseline_stats(self):
        if len(self.sam_conf_history) < 2:
            return 0.2, 0.05, 0.15, 0.25
        return (
            np.mean(self.sam_conf_history),
            np.std(self.sam_conf_history),
            np.min(self.sam_conf_history),
            np.max(self.sam_conf_history)
        )
    
    def detect_trend(self):
        if len(self.sam_conf_history) < self.trend_window:
            return 'unknown', 0.0, 0.0
        
        recent = self.sam_conf_history[-self.trend_window:]
        x = np.arange(len(recent))
        
        slope, intercept = np.polyfit(x, recent, 1)
        
        y_pred = slope * x + intercept
        ss_res = np.sum((recent - y_pred) ** 2)
        ss_tot = np.sum((recent - np.mean(recent)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        baseline_mean, baseline_std, _, _ = self.get_baseline_stats()
        threshold = baseline_std * 0.1 if baseline_std > 0 else 0.01
        
        if slope < -threshold:
            trend = 'declining'
        elif slope > threshold:
            trend = 'rising'
        else:
            trend = 'stable'
        
        return trend, slope, max(0, r_squared)
    
    def compute_sam_reliability(self, current_conf):
        info = {
            'trend': 'unknown',
            'slope': 0.0,
            'percentile': 0.5,
            'decline_severity': 0.0
        }
        
        if len(self.sam_conf_history) < 3:
            return 0.5, info
        
        trend, slope, trend_confidence = self.detect_trend()
        info['trend'] = trend
        info['slope'] = slope
        
        baseline_mean, baseline_std, hist_min, hist_max = self.get_baseline_stats()
        if hist_max > hist_min:
            percentile = (current_conf - hist_min) / (hist_max - hist_min)
            percentile = np.clip(percentile, 0, 1)
        else:
            percentile = 0.5
        info['percentile'] = percentile
        
        base_reliability = percentile
        
        if trend == 'declining':
            decline_severity = min(1.0, abs(slope) / (baseline_std + 1e-6) * trend_confidence)
            info['decline_severity'] = decline_severity
            reliability = base_reliability * (1 - decline_severity * 0.5)
        elif trend == 'rising':
            reliability = min(1.0, base_reliability * 1.1)
        else:
            reliability = base_reliability
        
        if current_conf < baseline_mean - 2 * baseline_std:
            reliability *= 0.5
        
        return np.clip(reliability, 0, 1), info
    
    def get_normalized_confidence(self, sam_conf):
        reliability, info = self.compute_sam_reliability(sam_conf)
        normalized_conf = reliability
        is_reliable = reliability > 0.3
        return normalized_conf, is_reliable, info
    
    def update(self, bbox):
        self.last_box = bbox
    
    def get_last_box(self):
        return self.last_box if self.last_box is not None else [0, 0, 0, 0]

def get_center_and_negative_points(bbox, margin_ratio=0.15, inner_ratio=0.25):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    inner_margin_x = width * inner_ratio
    inner_margin_y = height * inner_ratio
    
    top_point = [center_x, y1 + inner_margin_y]
    bottom_point = [center_x, y2 - inner_margin_y]
    left_point = [x1 + inner_margin_x, center_y]
    right_point = [x2 - inner_margin_x, center_y]
    
    positive_points = [
        [center_x, center_y],
        top_point,
        bottom_point,
        left_point,
        right_point
    ]
    
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    
    negative_points = [
        [x1 - margin_x, y1 - margin_y],
        [x2 + margin_x, y1 - margin_y],
        [x1 - margin_x, y2 + margin_y],
        [x2 + margin_x, y2 + margin_y],
        [center_x, y1 - margin_y],
        [center_x, y2 + margin_y],
        [x1 - margin_x, center_y],
        [x2 + margin_x, center_y]
    ]
    
    return positive_points, negative_points

def bbox_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def adaptive_fuse_detections(yolo_box, yolo_conf, sam_box, sam_conf, fusion_tracker):
    YOLO_CONF_THRESH = 0.1
    YOLO_HIGH_CONF = 0.5
    SAM_IOU_THRESH = 0.5
    
    fusion_info = {
        'yolo_used': False,
        'sam_used': False,
        'scenario': '',
        'confidence': 0.0,
        'need_sam_reset': False,
        'prediction_only': False,
        'yolo_rejected': False
    }
    
    if sam_conf > 0:
        fusion_tracker.update_sam_conf_history(sam_conf)
    
    if yolo_conf < YOLO_CONF_THRESH:
        if sam_box is not None:
            fused_box = sam_box
            fusion_info['sam_used'] = True
            fusion_info['scenario'] = 'yolo_lost'
            norm_conf, is_reliable, reliability_info = fusion_tracker.get_normalized_confidence(sam_conf)
            fusion_info['confidence'] = norm_conf
            fusion_info['sam_reliable'] = is_reliable
            fusion_info['reliability_info'] = reliability_info
            fusion_info['need_sam_reset'] = not is_reliable and reliability_info['trend'] == 'declining'
        else:
            fused_box = fusion_tracker.get_last_box()
            fusion_info['scenario'] = 'no_detection'
            fusion_info['confidence'] = 0.4
            fusion_info['prediction_only'] = True
    
    elif yolo_box is not None and sam_box is not None:
        iou = bbox_iou(yolo_box, sam_box)
        
        if iou > SAM_IOU_THRESH:
            norm_sam_conf, is_reliable, reliability_info = fusion_tracker.get_normalized_confidence(sam_conf)
            
            total_weight = yolo_conf + norm_sam_conf
            if total_weight > 0:
                weight_yolo = yolo_conf / total_weight
                weight_sam = norm_sam_conf / total_weight
            else:
                weight_yolo, weight_sam = 0.5, 0.5
            
            fused_box = [
                yolo_box[i] * weight_yolo + sam_box[i] * weight_sam
                for i in range(4)
            ]
            
            fusion_info['yolo_used'] = True
            fusion_info['sam_used'] = True
            fusion_info['scenario'] = 'agreement'
            fusion_info['confidence'] = (yolo_conf + norm_sam_conf) / 2
            fusion_info['sam_reliable'] = is_reliable
            fusion_info['need_sam_reset'] = False
        
        else:
            if yolo_conf > YOLO_HIGH_CONF:
                fused_box = yolo_box
                fusion_info['yolo_used'] = True
                fusion_info['scenario'] = 'conflict_yolo_wins'
                fusion_info['confidence'] = yolo_conf
                fusion_info['need_sam_reset'] = True
            else:
                fused_box = sam_box
                fusion_info['sam_used'] = True
                fusion_info['scenario'] = 'conflict_sam_wins'
                norm_sam_conf, is_reliable, reliability_info = fusion_tracker.get_normalized_confidence(sam_conf)
                fusion_info['confidence'] = norm_sam_conf
                fusion_info['sam_reliable'] = is_reliable
                fusion_info['need_sam_reset'] = not is_reliable and reliability_info['decline_severity'] > 0.5
    
    elif yolo_box is not None:
        fused_box = yolo_box
        fusion_info['yolo_used'] = True
        fusion_info['scenario'] = 'yolo_only'
        fusion_info['confidence'] = yolo_conf
        fusion_info['need_sam_reset'] = False
    
    elif sam_box is not None:
        fused_box = sam_box
        fusion_info['sam_used'] = True
        fusion_info['scenario'] = 'sam_only'
        norm_sam_conf, is_reliable, reliability_info = fusion_tracker.get_normalized_confidence(sam_conf)
        fusion_info['confidence'] = norm_sam_conf
        fusion_info['sam_reliable'] = is_reliable
        fusion_info['need_sam_reset'] = not is_reliable
    
    else:
        fused_box = fusion_tracker.get_last_box()
        fusion_info['scenario'] = 'no_detection'
        fusion_info['confidence'] = 0.4
        fusion_info['need_sam_reset'] = False
        fusion_info['prediction_only'] = True
    
    fusion_tracker.update(fused_box)
    
    return fused_box, fusion_info

def get_gpu_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0

def aggressive_memory_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_pretrained_model(yolo_path):
    model = YOLO(yolo_path, verbose=False)
    return model

def load_sam_predictor(model_name="sam2_t.pt"):
    overrides = dict(
        conf=0.01,
        task="segment",
        mode="predict",
        imgsz=512, 
        model=model_name,
        save=False,
        verbose=False
    )
    predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=5)
    return predictor

def check_high_confidence(results, threshold=0.75):
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        bbs = boxes.xyxy.cpu().numpy().tolist()
        
        max_conf_idx = np.argmax(confidences)
        max_conf = confidences[max_conf_idx]
        
        if max_conf > threshold:
            return True, [bbs[max_conf_idx]]
    
    return False, []

def process_video(video_path, yolo_model, sam_predictor, save_dir, 
                 conf_threshold=0.75, update_interval=1, reset_interval=10000, 
                 memory_threshold_mb=6000, update_cond_memory=True, max_memory_size=7):
    
    video_name = os.path.basename(video_path)
    print(f"Processing {video_name}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output file
    save_txt_name = f'{os.path.splitext(video_name)[0]}-fusion-{height}-{width}.txt'
    save_path = os.path.join(save_dir, save_txt_name)
    
    f_out = open(save_path, 'w')
    
    frame_idx = 0
    sam_triggered = False
    fusion_tracker = AdaptiveFusionTracker()
    
    # Variables for SAM2 loop
    last_update_frame = -1
    update_counter = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # YOLO Detection
        yolo_results = yolo_model(frame, verbose=False)
        yolo_boxes = []
        yolo_confidences = []
        
        if len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes
            yolo_boxes = boxes.xyxy.cpu().numpy().tolist()
            yolo_confidences = boxes.conf.cpu().numpy().tolist()
            
        if not sam_triggered:
            # Check for high confidence to start SAM2
            has_high_conf, high_conf_box = check_high_confidence(yolo_results, conf_threshold)
            
            if has_high_conf:
                print(f"  High confidence detection at frame {frame_idx}. Initializing SAM2.")
                sam_triggered = True
                initial_box = high_conf_box[0]
                fusion_tracker.initialize(initial_box)
                
                # Initialize SAM2
                aggressive_memory_cleanup()
                
                # Need to init with previous frame if possible, but here we use current frame
                # Actually fusion4 code goes back one frame. Let's try to just use current frame for simplicity or mimic fusion4
                # Fusion4: cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                # Here we are at frame_idx. Let's just init with current frame.
                
                positive_points, negative_points = get_center_and_negative_points(initial_box)
                points = positive_points + negative_points
                labels = [1] * len(positive_points) + [0] * len(negative_points)
                
                try:
                    _ = sam_predictor(source=frame, points=[points], labels=[labels], obj_ids=[1], update_memory=True)
                except Exception as e:
                    print(f"  Error initializing SAM predictor: {e}")
                    break
                
                last_update_frame = frame_idx
                
                # Write this frame's result (it's the initial box)
                x1, y1, x2, y2 = initial_box
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w_norm = (x2 - x1) / width
                h_norm = (y2 - y1) / height
                # Use max confidence from YOLO for this frame
                conf = max(yolo_confidences) if yolo_confidences else 1.0
                f_out.write(f"{frame_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.6f}\n")
                
            else:
                # Not triggered yet, write YOLO result if available (best one)
                if yolo_boxes:
                    # Find best box
                    max_conf_idx = np.argmax(yolo_confidences)
                    box = yolo_boxes[max_conf_idx]
                    conf = yolo_confidences[max_conf_idx]
                    
                    x1, y1, x2, y2 = box
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w_norm = (x2 - x1) / width
                    h_norm = (y2 - y1) / height
                    f_out.write(f"{frame_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.6f}\n")
        
        else:
            # SAM2 is active
            sam_results = sam_predictor(source=frame)
            sam_boxes = []
            box_conf = []
            if sam_results and len(sam_results) > 0 and hasattr(sam_results[0], 'boxes') and len(sam_results[0].boxes) > 0:
                boxes = sam_results[0].boxes
                sam_boxes = boxes.xyxy.cpu().numpy().tolist()
                box_conf = boxes.conf.cpu().numpy().tolist()
            
            # Prepare for fusion
            best_yolo_box = None
            best_yolo_conf = 0
            if yolo_boxes:
                max_conf_idx = np.argmax(yolo_confidences)
                best_yolo_box = yolo_boxes[max_conf_idx]
                best_yolo_conf = yolo_confidences[max_conf_idx]
            
            best_sam_box = None
            best_sam_conf = 0.8
            if sam_boxes:
                best_sam_box = sam_boxes[0]
                if box_conf:
                    best_sam_conf = box_conf[0]
            
            # Fuse
            fused_box, fusion_info = adaptive_fuse_detections(
                best_yolo_box, best_yolo_conf, best_sam_box, best_sam_conf, fusion_tracker
            )
            
            # Update SAM2 memory if needed
            should_update_sam_with_fusion = False
            
            if fusion_info.get('need_sam_reset', False):
                should_update_sam_with_fusion = True
            elif fusion_info['scenario'] == 'agreement' and fusion_info['confidence'] > 0.7:
                if (frame_idx - last_update_frame) > update_interval:
                    should_update_sam_with_fusion = True
            elif fusion_info['scenario'] == 'yolo_lost' and fusion_info['confidence'] > 0.6:
                if (frame_idx - last_update_frame) > update_interval * 2:
                    should_update_sam_with_fusion = True
            
            if should_update_sam_with_fusion and fused_box is not None:
                update_counter += 1
                last_update_frame = frame_idx
                
                _, reserved = get_gpu_memory_info()
                need_reset = (update_counter % reset_interval == 0) or (reserved > memory_threshold_mb)
                
                if need_reset:
                    if hasattr(sam_predictor, 'reset_predictor'):
                        sam_predictor.reset_predictor()
                    elif hasattr(sam_predictor, 'memory_bank'):
                        sam_predictor.memory_bank = []
                        if hasattr(sam_predictor, 'obj_idx_set'):
                            sam_predictor.obj_idx_set = set()
                    aggressive_memory_cleanup()
                
                positive_points, negative_points = get_center_and_negative_points(fused_box)
                points = positive_points + negative_points
                labels = [1] * len(positive_points) + [0] * len(negative_points)
                
                _ = sam_predictor(
                    source=frame,
                    points=[points],
                    labels=[labels],
                    obj_ids=[1],
                    update_memory=update_cond_memory
                )
                
                if hasattr(sam_predictor, 'memory_bank') and len(sam_predictor.memory_bank) > max_memory_size:
                    sam_predictor.memory_bank.pop(0)
                
                aggressive_memory_cleanup()
            
            # Write result
            if fused_box is not None:
                x1, y1, x2, y2 = fused_box
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w_norm = (x2 - x1) / width
                h_norm = (y2 - y1) / height
                conf = fusion_info['confidence']
                f_out.write(f"{frame_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.6f}\n")
        
        frame_idx += 1
        
    cap.release()
    f_out.close()
    aggressive_memory_cleanup()

def main():
    # Configuration — update these paths before running
    yolo_path = "ckpt/yolo_weight.pt"
    sam_model_name = "sam2.1_l.pt"          # SAM2 model name (downloaded automatically by ultralytics)
    video_folder = "videos/"                # folder containing input .mp4 files
    save_dir = "output/"                    # folder for output tracking results
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Loading models...")
    yolo_model = load_pretrained_model(yolo_path)
    sam_predictor = load_sam_predictor(sam_model_name)
    
    video_list = os.listdir(video_folder)
    video_list.sort()
    
    for video_name in video_list:
        if not video_name.endswith('.mp4'):
            continue
            
        video_path = os.path.join(video_folder, video_name)
        process_video(video_path, yolo_model, sam_predictor, save_dir)
        
    print("All videos processed.")

if __name__ == "__main__":
    main()
