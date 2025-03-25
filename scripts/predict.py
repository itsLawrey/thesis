from scripts.pipeline import log, warn
from ultralytics import YOLO
import shutil

def predict_with_yolo(input_picture_folder, yolomodel_path, conf_threshold = 0.7, batch_size = 1, stream = False):
    
    log("Loading YOLO model...")
    model = YOLO(yolomodel_path).to("cpu")
    
    log("Performing YOLO predictions...")
    predictions = model.predict(
        source=input_picture_folder,
        conf=conf_threshold,
        batch=batch_size,
        stream=stream,
        #project=PREDICTIONS_PATH,
        verbose=True
    )
    
    log("YOLO predictions complete...")
    return predictions

def save_initial_prediction(result, predictions_path):
    saved_path = result.save()
    
    shutil.move(saved_path, predictions_path)
    
    log(f"Saved a prediction to: {predictions_path}/{saved_path}")
    
def save_initial_predictions(predictions, saveflag, predictions_path):
    
    if not saveflag:
        return
    
    for result in predictions:
        save_initial_prediction(result, predictions_path)
        
#legacy
def extract_prediction_keypoints(predictions):
    
        # print(result.keypoints.xy)
        # print("\n")
        # print(result.keypoints.xy[0][4][1])
        # print("\n")
        # print(result.keypoints.xy[0][4][1].item())
        # print("\n")
        
        # #object 1, keypoint 5, coordinate y
        # #result.keypoints.xy[0][4][1].item() --> python float
        # print(result.keypoints.xy[0][4][1].numpy())
        # print(result.keypoints.xy[0][4][1].item())
    
    #65. kep 3. kulcspontjanak coordinatai: keypoints['x'][65][3]
    keypoints = {
        "x": [],
        "y": []
    }
    
    for result in predictions:
        
        result = result.summary()
            
        keypoints["x"].append(result[0]["keypoints"]["x"])
        keypoints["y"].append(result[0]["keypoints"]["y"])
        
    return keypoints