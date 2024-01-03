# YOLOv5-Object-Detection-And-Cropping
Lets do object detection, and cropping in few steps with help of  this repository demonstrates object detection using YOLOv5. Below are the steps to run object detection, display images, and crop detected objects.

![obj_det](https://github.com/Afnanfreelancer/YOLOv5-Object-Detection-And-Cropping/assets/92038975/b3cd211c-31a9-413c-b180-19e561e19c1f)


Steps:
# Step 1: Clone YOLOv5 Repository

!git clone https://github.com/ultralytics/yolov5.git

# Step 2: Change Directory to yolov5
%cd yolov5/

# Step 3: Install Dependencies

!pip install -U -r requirements.txt

# Step Most important: After running Third cell 
Re-run cell one and two (you might have to try 2-3 to run cell one again after Restart Session
after running this cell make sure to re-run or repeat step1 and step 2 cells again then proceed to step4

# Step 4: Run Object Detection on Images
Run object detection on an image by specifying the image path. 
replace path after --Source .... with your uploaded image
you can upload direct image to repo in defined repositry or mount google drive and select image from there too.
path defined after project for example(project /content/yolov5/data/images/detected) is where image will be saved after objects are detected

!python detect.py --source /content/yolov5/yolov5/data/images/zidane.jpg --weights yolov5s.pt --conf 0.4 --save-txt --save-conf --save-crop --exist-ok --project /content/yolov5/data/images/detected



# Step 5: Display Images in Colab
Run both of cells in this step it will display orignal image and then Object detected image
Display the original and detected images using matplotlib.
import cv2
import matplotlib.pyplot as plt

## Display the original image
org_image = cv2.imread('/content/yolov5/data/images/zidane.jpg')
plt.imshow(cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')
plt.show()

## Display the detected image
detected_image = cv2.imread('/content/yolov5/yolov5/data/images/detected/exp/zidane.jpg')
plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Objects')
plt.show()

# Step 6: Crop Detected Objects
Crop detected objects from the original image using bounding box coordinates saved in the label file.

from PIL import Image
import matplotlib.pyplot as plt

# Load the original image and label file
image = Image.open('/content/yolov5/yolov5/data/images/zidane.jpg')
with open('/content/yolov5/data/images/detected/exp/labels/zidane.txt', 'r') as file:
    lines = file.readlines()

# Crop detected objects
cropped_images = []
for line in lines:
    # Extract label information for each object detected
    # ...

    # Crop the region of interest
    # ...
    cropped_images.append(image.crop((x1, y1, x2, y2)))

# Display cropped images
num_images = len(cropped_images)
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
for i in range(num_images):
    axes[i].imshow(cropped_images[i])
    axes[i].axis('off')

plt.suptitle('Cropped Images', x=0.12, y=0.95, ha='left', fontsize=16)
plt.show()


Step 6: Test and Predict Function
Test the object detection and prediction functions.

# Test function for object detection
this function doesten take argument just run it to get an idea how thing works here, it will return original, obejcts detected and cropped images 
of default image
def test():

# Prediction 
this function return labels of detected object and score of confidence

Fusing layers... 
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
Adding AutoShape... 
***Label: person, 
 ***Score: 0.88 

***Label: tie, 
 ***Score: 0.68 

***Label: person, 
 ***Score: 0.67 

***Label: tie, 
 ***Score: 0.26 
