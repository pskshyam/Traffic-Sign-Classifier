# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/visualization.jpg "Visualization"
[image2]: ./output_images/equalized.jpg "Equalized"
[image4]: ./web_images/bumpy.jpg "Traffic Sign 1"
[image5]: ./web_images/100kmspeed.jpg "Traffic Sign 2"
[image6]: ./web_images/caution-roadworks.jpg "Traffic Sign 3"
[image7]: ./web_images/no-vehicles.jpg "Traffic Sign 4"
[image8]: ./web_images/TurnRightAhead.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/pskshyam/Self-Driving-Car-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the data distribution of the training, testing and validation sets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to correct the underexposure of the images by equalizing their pixel intensities therefore all the pixels are 
rendered with correct intensity. I have used cv2.equalizeHist() function for this.

Here is an example of a traffic sign image before and after histogram equalization.

![alt text][image2]

As a second step, I have augmented the training data because the frequency distribution of each label is imbalanced. I have applied methods like image rotation and translation with minimum count of augmentation as 5000.

As a last step, I have normalized the image data because the training runs faster with complex architectures like LeNet.

The difference between the original data set and the augmented data set is the following ... 

Class label 0, original sample count 180, updated sample count 5040
Class label 1, original sample count 1980, updated sample count 5940
Class label 2, original sample count 2010, updated sample count 6030
Class label 3, original sample count 1260, updated sample count 5040
Class label 4, original sample count 1770, updated sample count 5310
Class label 5, original sample count 1650, updated sample count 6600
Class label 6, original sample count 360, updated sample count 5040
Class label 7, original sample count 1290, updated sample count 5160
Class label 8, original sample count 1260, updated sample count 5040
Class label 9, original sample count 1320, updated sample count 5280
Class label 10, original sample count 1800, updated sample count 5400
Class label 11, original sample count 1170, updated sample count 5850
Class label 12, original sample count 1890, updated sample count 5670
Class label 13, original sample count 1920, updated sample count 5760
Class label 14, original sample count 690, updated sample count 5520
Class label 15, original sample count 540, updated sample count 5400
Class label 16, original sample count 360, updated sample count 5040
Class label 17, original sample count 990, updated sample count 5940
Class label 18, original sample count 1080, updated sample count 5400
Class label 19, original sample count 180, updated sample count 5040
Class label 20, original sample count 300, updated sample count 5100
Class label 21, original sample count 270, updated sample count 5130
Class label 22, original sample count 330, updated sample count 5280
Class label 23, original sample count 450, updated sample count 5400
Class label 24, original sample count 240, updated sample count 5040
Class label 25, original sample count 1350, updated sample count 5400
Class label 26, original sample count 540, updated sample count 5400
Class label 27, original sample count 210, updated sample count 5040
Class label 28, original sample count 480, updated sample count 5280
Class label 29, original sample count 240, updated sample count 5040
Class label 30, original sample count 390, updated sample count 5070
Class label 31, original sample count 690, updated sample count 5520
Class label 32, original sample count 210, updated sample count 5040
Class label 33, original sample count 599, updated sample count 5391
Class label 34, original sample count 360, updated sample count 5040
Class label 35, original sample count 1080, updated sample count 5400
Class label 36, original sample count 330, updated sample count 5280
Class label 37, original sample count 180, updated sample count 5040
Class label 38, original sample count 1860, updated sample count 5580
Class label 39, original sample count 270, updated sample count 5130
Class label 40, original sample count 300, updated sample count 5100
Class label 41, original sample count 210, updated sample count 5040
Class label 42, original sample count 210, updated sample count 5040

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| outputs 400       									|
| Dense Layer 1 | outputs 120 |
| Dense Layer 2 | outputs 84 |
| Dense Layer 3 | outputs 43 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer to minimize the cost functionw which is softmax_cross_entropy_with_logits. I have used below hyperparameters to train the network.

* epochs = 130
* batch_size = 128
* learning_rate = 0.001
* dropout_probability = 0.3

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.999
* test set accuracy of 0.955

* I have chosen a LeNet architecture with number of classes as 43.
* The original LeNet architecture trained on MNIST images would be suitable for traffic signs dataset as well because the architecture performs well enough in identifying digits in most of the traffic signs and other shapes of traffic signs. The original architecture without any changes itself was provided accuracy of 89%.
* The final model's accuracy on the training, validation and test set provide an evidence that the model is working well since the training and validation accuracies are high enough without much variance and is performed well on the test set as well with good accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because there are other areas in the image including road, grass along with 100 km/h traffic sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| 100 kmph     			| Bumpy road 										|
| Road work					| Road work											|
| No vehicles	      		| No vehicles					 				|
| Turn right ahead			| Turn right ahead     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95% which also has few misclassifications in predicting 100 kmph signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 32nd cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a bump sign (probability of 1.0), and the image does contain a bump sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.1         			| Bumpy road   									| 
| 0.0     				| Traffic signals 										|
| 0.0					| Road work											|
| 0.0	      			| Bicycles crossing					 				|
| 0.0				    | Dangerous curve to the left      							|


For the second image which is 100 kmph sign, the model is only 11% confident that this is a 100 kmph sign (probability of 0.11) and is 77% confident that it is 120 kmph sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.77         			| Bumpy road   									| 
| 0.11    				| Traffic signals 										|
| 0.1					| Dangerous curve to the right											|
| 0.0	      			| No entry					 				|
| 0.0				    | Road work      							|

For the third image, the model is relatively sure that this is a road work sign (probability of 1.0), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.1         			| Road work   									| 
| 0.0     				| Road narrows on the right 										|
| 0.0					| Beware of ice/snow											|
| 0.0	      			| Right-of-way at the next intersection					 				|
| 0.0				    | Wild animals crossing  							|

For the fourth image, the model is relatively sure that this is a no-vehicle sign (probability of 1.0), and the image does contain a no-vehicle sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.1         			| No vehicles   									| 
| 0.0     				| Speed limit (70 km/h) 										|
| 0.0					| Speed limit (120 km/h)											|
| 0.0	      			| Speed limit (30 km/h)					 				|
| 0.0				    | Speed limit (100 km/h)      							|

For the fifrth image, the model is relatively sure that this is a turn right ahead sign (probability of 1.0), and the image does contain a turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.1         			| Turn right ahead   									| 
| 0.0     				| Keep left 										|
| 0.0					| Go straight or right											|
| 0.0	      			| Ahead only					 				|
| 0.0				    | Roundabout mandatory      							|
