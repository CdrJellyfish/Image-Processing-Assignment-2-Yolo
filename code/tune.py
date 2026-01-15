from ultralytics import YOLO

def main():

    model = YOLO('yolov12s.pt')

    search_space = {
        'lr0': (1e-4, 1e-1),          
        'weight_decay': (0.0, 0.001), 
        'hsv_s': (0.0, 0.9),          
        'hsv_v': (0.0, 0.9)          
    }

    print("Starting hyperparameter tuning...")
    
 
    results = model.tune(
        data='data.yaml',
        space=search_space,
        epochs=15,      
        iterations=30,  
        optimizer='AdamW',
        imgsz=640,
        batch=8,
        device=0,
        
        name='yolov12s_tuning' 
    )
    
    print("Minimal tuning complete.")
    print(f"Best hyperparameters are saved in: {results.save_dir}/best_hyperparameters.yaml")

if __name__ == '__main__':
    main()