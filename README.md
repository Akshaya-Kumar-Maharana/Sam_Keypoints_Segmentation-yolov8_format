# Sam_Keypoints_Segmentation-yolov8_format

Steps to be followed:
Step 1: Provide the input image folder and the path to the folder where you wish to save the labels and run the code.<br>
Step 2: A image window appears on the screen, select points on the object you wish to segment using left mouse click, and use right mouse clicks to exculde regions from the object you are interested to generate the mask of./n
Step 3: After clicking the desired number of points, click the 'a' button on the keyboard. This would open a window showing the mask of gthe object generated along with a tkinter window.
Steo 4: If the mask generated is satisfactory, click on the tkinter window and click the 'space bar'. This would open a dailogue box to take the class id as input. Enter the class id in the dailogue box and click enter. Important - Don't close the tkinter window. Once you enter the class id, close the window showing the mask and click the other objects you wish to segment. Important - Click on the tkinter window before clicking the space bar.
Step 5: If the generated mask is not satisfactory then simply close the window showcasing the mask and select points again on the image window and follow from step 3.
Step 6: Once all the objects have been segmented, click on the image window and press 'n' button on the keyboard. This would save the labels file in the folder path you provided. Important - Click on the image window before clicking 'n'.
Step 7: Once the labels file is saved i.e. you have clicked 'n', close the image window. This would load the next image. Reapeat the step for all the other images in the folder. Once all the images are covered, the code will stop automatically.
