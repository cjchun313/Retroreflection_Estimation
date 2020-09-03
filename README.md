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
![fig2_1](https://user-images.githubusercontent.com/23445222/92067885-615a8180-ede0-11ea-8b43-28076b1c26da.png)