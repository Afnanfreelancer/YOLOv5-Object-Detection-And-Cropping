# YOLOv5-Object-Detection-And-Cropping
Lets do obhect detection, and cropping in few steps with help of  this repository demonstrates object detection using YOLOv5. Below are the steps to run object detection, display images, and crop detected objects.

![obj_det](https://github.com/Afnanfreelancer/YOLOv5-Object-Detection-And-Cropping/assets/92038975/b3cd211c-31a9-413c-b180-19e561e19c1f)


Steps:
# Step 1: Clone YOLOv5 Repository

!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5/

# Step 2: Install Dependencies

!pip install -U -r requirements.txt

after running this cell make sure to re run or rpeat step1 and step 2 again then proceed to step4

# Step 3: Run Object Detection on Images
Run object detection on an image by specifying the image path. Ensure to upload the image in the inference/images/ folder.

!python detect.py --source /content/yolov5/yolov5/data/images/zidane.jpg --weights yolov5s.pt --conf 0.4 --save-txt --save-conf --save-crop --exist-ok --project /content/yolov5/data/images/detected

# Step 4: Display Images in Colab
Display the original and detected images using matplotlib.
import cv2
import matplotlib.pyplot as plt

# Display the original image
org_image = cv2.imread('/content/yolov5/data/images/zidane.jpg')
plt.imshow(cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')
plt.show()

# Display the detected image
detected_image = cv2.imread('/content/yolov5/yolov5/data/images/detected/exp/zidane.jpg')
plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Objects')
plt.show()

Step 5: Crop Detected Objects
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
def test():

  # it takes Example image
  #do object detection and saved on dersir yolo5 runs
  !python detect.py --source /content/yolov5/data/images/zidane.jpg --weights yolov5s.pt --conf 0.4 --save-txt --save-conf --save-crop --exist-ok --project /content/yolov5/yolov5/runs/detect



  #original image Display
  orgiginal_image_path = '/content/yolov5/data/images/zidane.jpg'
  orgiginal_image_path = cv2.imread(orgiginal_image_path)
  # Display the image using Matplotlib
  plt.imshow(cv2.cvtColor(orgiginal_image_path, cv2.COLOR_BGR2RGB))
  plt.axis('off')  # Hide the axis
  plt.title('Original Image')
  plt.show()


  #Objects Detected Image Display
  # Path to the detected image
  detected_image_path = '/content/yolov5/yolov5/data/images/detected/exp/zidane.jpg'
  # Load the detected image
  detected_image = cv2.imread(detected_image_path)
  # Display the image using Matplotlib
  plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
  plt.axis('off')  # Hide the axis
  plt.title('Object Detected Image')
  plt.show()
![image](https://github.com/Afnanfreelancer/YOLOv5-Object-Detection-And-Cropping/assets/92038975/3616a7e4-d764-4798-9206-d6bc83af34c5)



#Cropping

  #labels
  labels_path = '/content/yolov5/yolov5/runs/detect/exp/labels/zidane.txt'

  mage = Image.open(image_path)
  # Open and read the content of the label file
  with open(label_path, 'r') as file:
    lines = file.readlines()

  # Initialize a list to store cropped images
  cropped_images = []

  # Iterate through each line in the label file
  for line in lines:
    # Extract label information for each object detected
    class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())

    # Convert YOLO format to pixel coordinates
    img_width, img_height = image.size
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate bounding box coordinates
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Crop the region of interest
    cropped_images.append(image.crop((x1, y1, x2, y2)))

# Display the cropped images horizontally
  num_images = len(cropped_images)
  fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
  for i in range(num_images):
    axes[i].imshow(cropped_images[i])
    axes[i].axis('off')

  plt.suptitle('Cropped Images', x=0.12, y=0.95, ha='left', fontsize=16)  # Set title on the left side

  plt.show()

#Calling Function

test()



# Predict function for inference
def predict(image_path_or_url):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    if image_path_or_url.startswith('http'):
        # Load image from URL
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        # Load image from local path
        image = Image.open(image_path_or_url)

    # Perform inference
    results = model(image)

    # Iterate over detected objects and display labels and scores
    for obj in results.xyxy[0]:
        label = results.names[int(obj[5])]
        score = obj[4].item()
        print(f'***Label: {label}, \n ***Score: {score:.2f}','\n')

# Example usage
image_path_or_url = '/content/yolov5/yolov5/data/images/zidane.jpg'
predict(image_path_or_url)

![image](https://github.com/Afnanfreelancer/YOLOv5-Object-Detection-And-Cropping/assets/92038975/3c86e1d1-03a0-4039-b299-6703e2c9f088)


