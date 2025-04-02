from logs import log, warn
from ultralytics import YOLO
import shutil

def predict_with_yolo(input_picture_folder, yolomodel_path, conf_threshold = 0.7, batch_size = 16, stream = True):
    
    log("Loading YOLO model...")
    model = YOLO(yolomodel_path).to("cpu")
    
    log("Performing YOLO predictions...")
    predictions = model.predict(
        source=input_picture_folder,
        conf=conf_threshold,
        batch=batch_size,
        stream=stream,
        verbose=True
    )
    yolo_pred = []
    for r in predictions:
        yolo_pred.append(r)
    log("YOLO predictions complete...")
    return yolo_pred

def save_initial_prediction(result, predictions_path):
    saved_path = result.save()
    
    shutil.move(saved_path, predictions_path)
    
    log(f"Saved a prediction to: {predictions_path}/{saved_path}")
    
def save_initial_predictions(predictions, saveflag, predictions_path):
    
    if not saveflag:
        return
    
    for result in predictions:
        save_initial_prediction(result, predictions_path)