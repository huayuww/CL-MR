import os
import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path: str):
    model = YOLO(model_path)
    return model

def run_videos(folder_dir: str, model, save_dir: str = None):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_list = os.listdir(folder_dir)
    for video_name in video_list:
        if not video_name.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing {video_name}, FPS: {fps}")
        
        save_txt_name = f'{os.path.splitext(video_name)[0]}-bytetrack-{frame_height}-{frame_width}.txt'
        save_path = os.path.join(save_dir, save_txt_name)
        
        with open(save_path, 'w') as f:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.track(frame, tracker='bytetrack.yaml', persist=True, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Format: frame_idx class_id x_center y_center width height conf
                        # Using normalized coordinates xywhn
                        x, y, w, h = box.xywhn[0].tolist()
                        conf = float(box.conf[0])
                        f.write(f"{frame_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
                
                frame_idx += 1
        cap.release()
            
    

def main():
    yolo_path="/work/andyee1997/huayu/vocal/tracking/checkpoints/best_retrain.pt"
    
    video_folder = '/work/andyee1997/huayu/data/videos/harborview/'
    
    save_dir = '/work/andyee1997/huayu/vocal/tracking/output/bytetrack_results'
    model = load_yolo_model(yolo_path)
    run_videos(video_folder, model, save_dir)

if __name__ == "__main__":
    main()