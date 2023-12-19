import cv2
import numpy as np 
from segment_anything import sam_model_registry, SamPredictor
import torch
import matplotlib.pyplot as plt
import keyboard
import tkinter as tk 
from tkinter import simpledialog,Tk
import os 

def normalize_points(contour_points, image_width, image_height):
    # Normalize the points
    print(type(contour_points))
    normalized_points = contour_points.astype(float) / np.array([image_width, image_height])
    return normalized_points

def convert_contour_to_yolov8(contour_points, class_index):
    # Flatten the contour points to a 1D array
    flattened_points = contour_points.reshape(-1, 2)

    # Create YOLOv8 label string
    label_string = f"{class_index}"
    for point in flattened_points:
        label_string += f" {point[0]} {point[1]}"

    return label_string

def on_key_press(event):
    # Check if the pressed key is the "Space" key (keycode 32)
    
    root_dialog = Tk()
    # Create a dialogue box to take user input
    user_input = simpledialog.askstring("Input", "Enter something:")
    if user_input != 'None':
        obj_type.append(user_input)
        normalized_points = normalize_points(contour[0], original_img_width, original_img_height)
        label_string = convert_contour_to_yolov8(normalized_points, user_input)
        label_string_list.append(label_string)
    # Print the user input
    print("User input:", user_input)
    root_dialog.destroy()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def resize_img(image_path, window_size=(854, 480)):
    # Read the image from file
    image = cv2.imread(image_path)
    window_width, window_height = window_size
    canvas = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255
    # Resize the image to fit the specified window size
    img_height, img_width = image.shape[:2]
    if img_height >= img_width:
        new_img_height = 480
        scale_factor = new_img_height/img_height
        new_img_width = int(img_width*scale_factor)
        img_size = (new_img_width, new_img_height)

        if new_img_width > 854:
            new_img_width_2 = 854
            scale_factor = 854/new_img_width
            new_img_height_2 = int(new_img_height*scale_factor)
            img_size = (new_img_width_2, new_img_height_2)

    if img_height < img_width:
        new_img_width = 854
        scale_factor = new_img_width/img_width
        new_img_height = int(img_height*scale_factor)
        img_size = (new_img_width, new_img_height)

        if new_img_height > 480:
            new_img_height_2 = 480
            scale_factor = new_img_height/img_height
            new_img_width_2 = int(img_width*scale_factor)
            img_size = (new_img_width_2, new_img_height_2)

    image = cv2.resize(image, img_size)
    y_offset = (canvas.shape[0] - image.shape[0]) // 2
    x_offset = (canvas.shape[1] - image.shape[1]) // 2

    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

    return canvas, image

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel coordinates: ({x}, {y})")
        points_list.append([int((x - x_offset)*(img.shape[1]/resized_image.shape[1])), int((y - y_offset)*(img.shape[0]/resized_image.shape[0]))])
        label_list.append(1)
        print(points_list)
        print(label_list)

    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Clicked at pixel coordinates: ({x}, {y})")
        points_list.append([int((x - x_offset)*(img.shape[1]/resized_image.shape[1])), int((y - y_offset)*(img.shape[0]/resized_image.shape[0]))])
        label_list.append(0)
        print(points_list)
        print(label_list)

if __name__ == '__main__':
# -------------Inputs------------------
    img_folder = '/path/to/img/folder'
    save_folder = '/path/to/save/folder'
# -------------------------------------
    root = Tk()
    root.bind("<KeyPress>", on_key_press)
    
    sam_checkpoint = "models\\sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    img_list = os.listdir(img_folder)
    count = 0
    while count < len(img_list):

        image_path = os.path.join(img_folder, img_list[count])
        
        img = cv2.imread(image_path)
        original_img_height, original_img_width = img.shape[:2]

        predictor.set_image(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        canvas, resized_image = resize_img(image_path)
        y_offset = (canvas.shape[0] - resized_image.shape[0]) // 2
        x_offset = (canvas.shape[1] - resized_image.shape[1]) // 2

        
        cv2.namedWindow('Fixed Size Window', cv2.WINDOW_NORMAL)
        window_size = (854, 480)
        points_list = []
        label_list = []
        obj_type = []
        label_string_list = []
        cv2.resizeWindow('Fixed Size Window', window_size[0], window_size[1])
        cv2.setMouseCallback('Fixed Size Window', on_mouse_click)

        # Display the image in the window
        cv2.imshow('Fixed Size Window', canvas)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if chr(key) == 'a':
                print("I have pressed a key")
                input_point = np.array(points_list)
                input_label = np.array(label_list)
                masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
                )
                print(masks)
                contour, _ = cv2.findContours(masks[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(contour)
                plt.figure(figsize=(10,10))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                show_mask(masks, plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.axis('off')
                plt.show()
                points_list = []
                label_list = [] 
                print(label_string_list)
                # root.mainloop()

            if chr(key) == 'n':
                output_path = os.path.splitext(img_list[count])[0] + '.txt'
                with open(os.path.join(save_folder, output_path), 'w') as file:
                # Write each element of the list on a new line
                    for item in label_string_list:
                        file.write(str(item) + '\n')
                count += 1
                break
                
        print("Outside Loop")



        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
