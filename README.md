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

### Dynamic Retroreflection Estimation
The dynamic retroreflection was predicted during actual drivingon the road. Note that when the classification model is determined to be negative, the ratio is 0. If the captured luminance image has a blue frame, it is a result predicted as positive, and if it is a red frame, it is a result predicted as negative. It is possible to confirm that only the retroreflection of the longituidnal road marking was extracted.
![fig5_1](https://user-images.githubusercontent.com/23445222/92069511-7e914f00-ede4-11ea-9b80-fd0438bd6d3b.png)

### Acknowledgement
This research was supported by a grant from Technology Business Innovation Program (TBIP) funded by Ministry of Land, Infrastructure and Transport of Korean government (No. 18TBIP-C144255-01) [Development of Road Damage Information Technology based on Artificial Intelligence].