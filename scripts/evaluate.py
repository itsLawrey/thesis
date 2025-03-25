
#TODO REWRITE THIS
def draw_predictions(label_folder, input_picture_folder, output_folder):    
    
    #also save images but keep them in a folder and return it
    
    for result in label_folder:
        # Get image path and base filename
        image_path = result.path
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_img_path = os.path.join(output_folder, f"{img_name}_predicted.png")
        
        # Load the image and get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_path}")
            continue
        img_h, img_w = img.shape[:2]
        
        label_lines = []  # List to hold label lines for this image
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_all = result.keypoints.xy
            try:
                keypoints_conf = result.keypoints.conf
            except AttributeError:
                keypoints_conf = None
                
            # Process each detection
            for i in range(len(result.boxes)):
                # Get bounding box in xyxy format
                bbox = result.boxes.xyxy[i].cpu().numpy().flatten()  # [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = bbox
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Normalize bounding box values
                norm_center_x = center_x / img_w
                norm_center_y = center_y / img_h
                norm_width = width / img_w
                norm_height = height / img_h
                
                # Get object id if available, else default to 0
                try:
                    object_id = int(result.boxes.cls[i].cpu().numpy())
                except Exception:
                    object_id = 0
                
                # Get keypoints for this detection
                keypoints = keypoints_all[i].cpu().numpy()
                if keypoints_conf is not None:
                    confs = keypoints_conf[i].cpu().numpy()
                else:
                    confs = [1.0] * len(keypoints)
                
                # Build the label line with normalized values (6 decimal places)
                line = f"{object_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                for keypoint_idx, kp in enumerate(keypoints):
                    norm_kp_x = kp[0] / img_w
                    norm_kp_y = kp[1] / img_h
                    norm_conf = confs[keypoint_idx]
                    line += f" {norm_kp_x:.6f} {norm_kp_y:.6f} {norm_conf:.6f}"
                label_lines.append(line)
                
                # Draw skeleton connections on the image
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        start_point = keypoints[start_idx]
                        end_point = keypoints[end_idx]
                        if (start_point[0] == 0 and start_point[1] == 0) or (end_point[0] == 0 and end_point[1] == 0):
                            continue
                        start_pt = (int(start_point[0]), int(start_point[1]))
                        end_pt = (int(end_point[0]), int(end_point[1]))
                        if start_idx in left_leg_keypoints or end_idx in left_leg_keypoints:
                            line_color = darker_color(color_mapping["left_leg"])
                        elif start_idx in right_leg_keypoints or end_idx in right_leg_keypoints:
                            line_color = darker_color(color_mapping["right_leg"])
                        else:
                            line_color = color_mapping["skeleton"]
                        cv2.line(img, start_pt, end_pt, line_color, 2)
                # Draw keypoints and annotate them
                for keypoint_idx, keypoint in enumerate(keypoints):
                    x, y = int(keypoint[0]), int(keypoint[1])
                    if x == 0 and y == 0:
                        continue
                    if keypoint_idx in left_leg_keypoints:
                        color = color_mapping["left_leg"]
                    elif keypoint_idx in right_leg_keypoints:
                        color = color_mapping["right_leg"]
                    else:
                        color = color_mapping["default"]
                    cv2.circle(img, (x, y), 3, color, -1)
                    cv2.putText(img, str(keypoint_idx), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the annotated image
        cv2.imwrite(output_img_path, img)
        print(f"Image saved to {output_img_path}")
        
        # Write the normalized label file for this image
        label_file_path = os.path.join(labels_folder, f"{img_name}.txt")
        with open(label_file_path, "w") as f:
            for line in label_lines:
                f.write(line + "\n")
        print(f"Label file saved to {label_file_path}")
        
        
    return predicted_images
#update this to work with a list instead
def create_video_from_images(predicted_images, output_video_path, fps=30):
    # # Get a list of image files in the folder, explicitly excluding subdirectories
    # image_files = predicted_images
    
    # # Sort files in numerical order using the custom sort function
    # image_files = sort_nicely(image_files)
    
    # if not image_files:
    #     print("No images found in the folder.")
    #     return
    
    # # Load the first image to get video properties
    # first_image_path = os.path.join(folder_path, image_files[0])
    # first_image = cv2.imread(first_image_path)
    # if first_image is None:
    #     print(f"Error loading the first image {first_image_path}")
    #     return
    
    # height, width, _ = first_image.shape

    # # Initialize video writer (with frame rate and proper codec)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # # Process each image and write it to the video
    # for idx, image_file in enumerate(image_files):
    #     image_path = os.path.join(folder_path, image_file)
    #     image = cv2.imread(image_path)
        
    #     if image is None:
    #         print(f"Error loading image {image_path}")
    #         continue
        
    #     # Resize image to match the first image's resolution (to avoid dimension mismatches)
    #     resized_image = cv2.resize(image, (width, height))

    #     # Write the resized image as a frame to the video
    #     out.write(resized_image)

    #     # Logging to verify the correct order of processing
    #     print(f"Writing frame {idx + 1} of {len(image_files)}: {image_file}")

    # # Release the video writer
    # out.release()
    # print(f"Video saved at {output_video_path}")
    pass

def plot_results():
    pass



