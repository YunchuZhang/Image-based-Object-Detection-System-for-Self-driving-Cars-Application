# Image-based-Object-Detection-System-for-Self-driving-Cars-Application
The Goal of this project: Detect and track 4 different objects includes vehicle, pedestrian, cyclist and traffic lights (labeled as 1, 2, 3, 20).   
And I have done some things below:  

 Base on Deep learning (Mxnet) to implement object detection and tracking system on self-driving car system      

 Based on given dataset and Yolo algorithm to construct special neural network model and update a new loss function      

 Utilize GPU for training and tune parameters to converge and optimize the result        

 Optimize feedforward inference network and realize object detection and tracking in real time on camera     


(In the future) Move the system to robots for avoiding obstacle   


See demo below or see result in jupyter notebook's result       

Yolo algorithm. There are 2 verson for it. [v1](https://arxiv.org/pdf/1506.02640.pdf), [v2](https://arxiv.org/pdf/1612.08242.pdf).


# Dependencies
Python,Mxnet 1.0, cuda8.0, tensorboardX,cudnn,opencv,GPU:nvidia 1070T


# Yolo-v1


Data and Model
https://drive.google.com/drive/u/0/folders/0BwXw1vJFiBDaZ1IwNjlEd0RZMFU     

For asking training Dataset and testing Dataset, you could send me email.		

To run the code:  
!!!First you need to change the code's path and make it suitable in your Pc  

    Download data , model and label  
    mkdir and cd to the path of "new_data"  
    put generate_data.ipynb to folder"new_data" and run it to generate 50k new data  
    mkdir and cd to the path of "DATA_rec/" and the json you need is in new_data  
    run "python data_util/py" for data preparaion with train and val recfile  
    cd to the src root path and run "pyton run_train.py"  
    
  
  For the test:
    Please take a look of wild_test.ipynb and demo_test.ipynb in src first.
    
    And then run test.py which could output a json file for results and draw the bbox in image
    
    For real time predict, to run "pythonw real_time_object_detection" on Mac/ "python real_time_object_detection"
    
# Result
After about 24 hour's training(350 epoch),accuracy is about 0.99,precision is 0.83, recall is 0.98,h_diff is 1.3, w_diff is 1.23  
The result is shown below:  

![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/1.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/2.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/3.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/4.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/5.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/6.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/7.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/8.png)  
![image](https://github.com/YunchuZhang/Image-based-Object-Detection-System-for-Self-driving-Cars-Application/blob/master/readme/9.png)  

Demo for test data:


Real-time test:  
1.video data test: https://drive.google.com/open?id=1a9H8viB03dgJFk3aSzO0xqn8tFwSbmVm   
2.real-time test on road:
# Refer:

http://blog.topspeedsnail.com/archives/2068

https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/
# Yolo-v2
Is writting and updating
# Result
Demo for test data:		

Real-time test:		
