from ultralytics import YOLO
import os
import glob

def test_agv_and_corners():
    # 1. Load the best model
    model_path = 'runs/models/smart_port_base_model3/weights/best.pt'
    model = YOLO(model_path)

    # 2. Path to your new AGV images
    agv_image_folder = '../raw_data/AGV'
    
    # Get all jpg/png images
    images = glob.glob(os.path.join(agv_image_folder, "*.[jp][pn]g"))

    if not images:
        print(f"❌ No images found in {agv_image_folder}. Check your file extensions!")
        return

    print(f"🚀 Testing on {len(images)} images from the AGV folder...")

    # 3. Run Inference with "High Sensitivity"
    # imgsz=1280: Scales the image up to see tiny corners
    # conf=0.15: Shows objects even if the AI is only 15% sure (good for debugging)
    # iou=0.5: Helps distinguish between Container 1 and Container 2 if they overlap
    results = model.predict(
        source=images,
        conf=0.15,          
        imgsz=1280,         
        save=True,           
        save_txt=True,      
        project='../verification',
        name='agv_precision_test'
    )

    print(f"✅ Test complete! Results saved in: smartVision/verification/agv_precision_test")

if __name__ == '__main__':
    test_agv_and_corners()