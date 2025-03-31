from ultralytics import YOLO

def train_model(data_yaml_path, epochs, save_path, name_of_future_model):
    
    model = YOLO('yolov8n-pose.pt')
    
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        save_period=3,
        batch=8,
        degrees=15,
        flipud=0.5,
        augment=True,
        patience=50,
        #device=[0, 1],
        project=save_path,
        name=name_of_future_model
    )

#az elkepzeles az hogy a streamlit majd nezi a datasets foldert es akkor amikor elkeszit egy datasetet a prepare.py akkor meg tudjuk hivni ezt ra

yaml_path = r'C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\models\training_data\datasets\ds_integrated_1\config\data.yaml'
output_model = r'C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\models\trained'

if __name__ == "__main__":
    train_model(yaml_path,10,output_model,'demoYOLOOOOO')