from ultralytics import YOLO

def train_model(data_yaml_path, epochs, save_path, name_of_future_model):
    
    model = YOLO('yolov8l-pose.pt')
    
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        save_period=50,
        batch=16,
        degrees=15,
        flipud=0.5,
        augment=True,
        patience=50,
        #device=[0, 1],
        project=save_path,
        name=name_of_future_model
    )

