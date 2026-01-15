from ultralytics import YOLO
from pathlib import Path

def main():
    model_path = Path('runs/detect/yolov12m_final_tuned_model4/weights/best.pt')

    model = YOLO(model_path)
    print(f"Loaded trained model from {model_path}")

    print("Running predictions on the test image folder")
    model.predict(
        source='dataset/test/images/',
        save=True,  
        conf=0.5  
    )
    print("Test set predictions saved in 'runs/detect/predict/'")

    video_file = 'vid.mp4'
    print(f"Running prediction on video: {video_file}")
    

    results = model.predict(
        source=video_file,
        save=True,
        conf=0.5,
        stream=True
    )
    for r in results:
        pass  
    print(f"Video prediction saved in 'runs/detect/predict/'")

if __name__ == '__main__':
    main()