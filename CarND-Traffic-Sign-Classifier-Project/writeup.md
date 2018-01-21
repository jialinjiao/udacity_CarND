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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/before_preprocessing.png "Before Preprocessing"
[image3]: ./examples/after_preprocessing.png "After Preprocessing"

[test_image1]: ./test_images/11_right_of_way.jpg "right of way"
[test_image2]: ./test_images/12_priority_road.jpg "priority road"
[test_image3]: ./test_images/13_yield_sign.jpg "yield sign"
[test_image4]: ./test_images/14_stop_sign.jpg "Stop Sign"
[test_image5]: ./test_images/15_no_vehicles.jpg "no vehicles"
[test_image6]: ./test_images/18_general_caution.jpg "general caution"
[test_image7]: ./test_images/25_road_work.jpg "road work"
[test_image8]: ./test_images/34_turn_left_ahead.jpg "turn left ahead"
[test_image9]: ./test_images/3_speed_limit_60kmh.jpg "speed limit (60 kmh)"
[test_image10]: ./test_images/9_no_passing.jpg "no passing"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jialinjiao/udacity_CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the train data distribute over classes:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I believe the color information isn't necesary for classifying traffic sign.
As a last step, I normalized the image data because it is critical for ConvNet to have image data normalized within [0,1] or [-1, 1] and using same scale for both training and prediction.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image2] ![alt text][image3]

I also sample the first image of each class and display them in the notebook.

Although I understand the benefits of data augmentation, I have not done that for this project just to see how it works out.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is LeNet with dropouts, which consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution Layer   | 5x5 convolution, 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 16x16x6 				|
| Convolution layer	| 5x5 convolution, 1x1 stride, valid padding, outputs 10x10x16      		|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x16 				|
| Flatten layer		    | input 5x5x16, Output 400x1        									|
| Fully connected		| input 400x1, output 120x1        									|
| ReLU				    |          									|
| Dropout				| keep_pro =0.5												|
| Fully connected		| input 120x1, output 84x1												|
| ReLU				    |          									|
| Dropout				| keep_pro =0.5												|
| Fully connected		| input 84x1, output 43x1												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, and other hyperparameters as following:
*number of epochs = 60
*batch size = 128
*learning rate = 0.001 # learning rate


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.934
* test set accuracy of 0.917


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used the original LeNet as the first architecture, the reason I chose it was because:
 (1) it is classic
 (2) I heard that even today, LeNet is still not out of fashion yet (although not as fancy as AlexNet, VGG, GoogleLeNet, ResNet) and many more recent predominant nets are still based on LeNet
 (3) I want to see how the performance it will be if just using this simple architecture, and if it is not good, I could use it as a baseline

* What were some problems with the initial architecture?
Without dropouts, even validation accurracy is > 0.93 and test accuracy is > 0.91, when testing on the new images I download from internet, the accuracuy for the 10 images I downloaded are pretty poor, something around 10% ~ 30%; another issues with the original LeNet archicture was that it requires a lot of epoches (sometimes > 100) to achieve > 0.93 validation accuracy.

another interesting observation I had was that re-traininng the nets will give pretty different performances from one trained net to another trained net,  I believe this is an evidence of the effectiveness of overfitting prevention techniques such as dropout or batch normalization.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I simply adjusted the architecture by adding dropouts after the two fully connected layers before the final FC layer. And the training times (number of epoches needed) was reduced greatly, I could reached >0.93 validation accuracy in 60 epoches, and also the accuracy in the new testing images I downloaded is reaching 80% ( 8 out of 10 were successfully detected).


* Which parameters were tuned? How were they adjusted and why?
I mostly tuned 3 hyperparameters:
* number of epoches
* learning rate
when I make it too smaller , such as 0.0001, it requires a lot of epoches to reach 0.93 validation accuracy;
* keep_pro for dropout
I feel that 0.5 ~ 0.6 is provide better generalization than higher keep_pro


### Test a Model on New Images

#### 1. Choose 10 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

Right of way <br />
![alt text][test_image1] <br />
Priority road <br />
![alt text][test_image2] <br />
Yield sign <br />
![alt text][test_image3] <br />
Stop sign <br />
![alt text][test_image4] <br />
No vehicle <br />
![alt text][test_image5] <br />
General caution <br />
![alt text][test_image6] <br />
Road work <br />
![alt text][test_image7] <br />
Turn left ahead <br />
![alt text][test_image8] <br />
Speed limit (60 kmh) <br />
![alt text][test_image9] <br />
No passing <br />
![alt text][test_image10] <br />

I feel the priority road, no vehicle, general causion and road work might be difficult to clasify because of complicated background and noise, and also the speed limit sign might be also hard due to the image size is very small.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			        |     Prediction	        					| Correct?|
|:---------------------:|:---------------------------------------------:|:------:|
| No Vehicle      		| No Vehicle   									| Yes |
| Road work     			| Speed Limit (30km/h) 										| No|
| Stop sign					| Stop sign											|Yes|
| Right of way	      		| Right of way					 				| Yes|
| Speed limit (60km/h)		| No passing      							| No |
| Turn left ahead      		| Turn left ahead   									| Yes |
| No passing     			| No passing 										| Yes |
| Priority road					| Priority road											| Yes |
| General caution	      		| General caution					 				| Yes |
| Yield sign			| Yield sign      							| Yes |



The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


Actually all the ten tests have overwhelming high probablities for the highest probablity, usually 1.0 or more than 0.9999; even for the wrong predictions (road work and speed limit 60km/h), the wrong prediction got high probablity and the probabilistics for the correct class are not even in the top 5.



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


