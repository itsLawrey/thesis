from logs import log, warn
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os


def union_masks(masks):
    """Take a list of binary masks and return their union (logical OR)."""
    if len(masks) == 0:
        return None
    union = masks[0].copy().astype(bool)
    for m in masks[1:]:
        union = np.logical_or(union, m.astype(bool))
    return union.astype(np.uint8)

def load_sam_model(model_type, checkpoint, device):
    """
    Loads the SAM model from the specified checkpoint and moves it to the chosen device.
    Returns a SamPredictor instance ready for segmentation.
    """
    log("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    log(f"SAM model loaded and moved to {device}.")
    return SamPredictor(sam)

def get_masks_with_sam(predictions, model_type, checkpoint, device):
    
    sam_predictor = load_sam_model(model_type, checkpoint, device)
    
    log("Computing SAM masks for each image...")
    
    sam_masks = []
    
    for result in predictions:
        
        #original images!!!
        img_path = result.path
        log(f"Processing {os.path.basename(img_path)} for mask.")
        img = cv2.imread(img_path)

        if img is None:
            warn(f"Could not load image {img_path} for SAM segmentation. Skipping.")
            sam_masks.append(None)
            continue

        if not hasattr(result, 'keypoints') or result.keypoints is None or result.keypoints.xy is None:
            warn(f"No keypoints for image {img_path}. Skipping SAM segmentation.")
            sam_masks.append(None)
            continue

        keypoints = result.keypoints.xy.cpu().numpy()
        
        if keypoints.shape[1] < 5:
            warn(f"Not enough keypoints for image {img_path}. Skipping SAM segmentation.")
            sam_masks.append(None)
            continue

        # spine es tailbase SAM-nek
        kp3 = keypoints[0][3]
        kp4 = keypoints[0][4]
        prompt1 = np.array([kp3])
        prompt2 = np.array([kp4])
        prompt3 = np.array([kp3, kp4])
        masks = []
        
        for prompt in [prompt1, prompt2, prompt3]:
            sam_predictor.set_image(img)
            pts_labels = np.ones(len(prompt))#amelyik indexen van egy kulcspont, ha ugyanitt 1 van azt jeletni hogy ez foreground
            m, _, _ = sam_predictor.predict(point_coords=prompt, point_labels=pts_labels)
            masks.append(m[0])
        union_mask = union_masks(masks)
        sam_masks.append(union_mask)

    return sam_masks

def save_sam_masks(masks, saveflag, masks_path):
    if saveflag:
        for idx, union_mask in enumerate(masks):
            mask_vis = (union_mask * 255).astype(np.uint8)
            mask_save_path = os.path.join(masks_path, f"mask_{idx:05d}.png")
            cv2.imwrite(mask_save_path, mask_vis)
            log(f"Saved SAM mask: {mask_save_path}")
