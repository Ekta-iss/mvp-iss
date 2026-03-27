from ultralytics import YOLO
import os

def main():
    # 1. Load the model
    model = YOLO('yolov8n.pt')

    # 2. Define relative paths 
    # Since you are running the command FROM the 'src' folder:
    # '../' goes up to 'smartVision', then into 'datasets'
    data_rel_path = '../datasets/port_crane/data.yaml'
    project_rel_path = '../models'

    # 3. Start Training
    results = model.train(
        data=data_rel_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project=project_rel_path,
        name='smart_port_base_model',
        device='cpu' 
    )

if __name__ == '__main__':
    main()