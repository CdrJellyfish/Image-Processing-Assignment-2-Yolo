from ultralytics import YOLO
from pathlib import Path
import yaml 

def main():

    params_file_path = Path('runs/detect/yolov12m_minimal_tuning4/best_hyperparameters.yaml') 

    if not params_file_path.exists():
        print(f"Error: Parameter file not found at {params_file_path}")
        return

    print(f"Loading best parameters from {params_file_path}")
    with open(params_file_path, 'r') as f:
        best_params = yaml.safe_load(f)

    model = YOLO('yolov12s.pt')

    print("Starting training with loaded parameters")
    results = model.train(
        data='data.yaml',

        lr0=best_params['lr0'],
        weight_decay=best_params['weight_decay'],
        hsv_s=best_params['hsv_s'],
        hsv_v=best_params['hsv_v'],

        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name='yolov12m_final_tuned_model' 
    )

    print("Final training complete.")
    print(f"Model is saved in: {results.save_dir}")

if __name__ == '__main__':
    main()