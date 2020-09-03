## Classification and Segmentation of Longitudinal Road Marking using Convolutional Neural Networks for Dynamic Retroreflection Estimation

### Requirements
glob2==0.7    
numpy==1.16.5      
pandas==1.0.4      
torch==1.2.0      
torchsummary==1.5.1      
torchvision==0.4.0a0+6b959ee      
tqdm==4.32.1

### Abstract
The dynamic retroreflection estimation method of the longitudinal road marking from the luminance camera using convolutional neural networks (CNNs). From the image captured by the luminance camera, a classification and regression CNN model is proposed to find out whether the longitudinal road marking was accurately acquired. If the longitudinal road marking is present in the captured image, a segmentation model that appropriately presents the longitudinal road marking and the reference plate is also proposed.

### Overall Structure
First, it is determined whether retroreflection prediction is possible from the luminance image. If possible, the luminance image is cropped as the small image to find the retroreflection. After that, the reference plate and the longitudinal road marking are precisely segmented.
<img src="https://user-images.githubusercontent.com/23445222/92067885-615a8180-ede0-11ea-8b43-28076b1c26da.png" width="600">

### Classification Model
The CNN models used in this paper are those that have shown very good performance in ImageNet. The original models classify into 1,000 classes, but in this paper only binary classification exists. Furthermore, There is also an output that predicts the value of h for cropping to a small image size. Therefore, the last layers were modified such that a hidden layer with 256 and 128 units was added, and only 3 units were estimated at the output layer. 
<img src="https://user-images.githubusercontent.com/23445222/92069157-ae8c2280-ede3-11ea-9578-c03fbc7bc458.png" width="600">

### Acknowledgement
This research was supported by a grant from Technology Business Innovation Program (TBIP) funded by Ministry of Land, Infrastructure and Transport of Korean government (No. 18TBIP-C144255-01) [Development of Road Damage Information Technology based on Artificial Intelligence].