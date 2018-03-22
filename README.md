# Image-based-Object-Detection-System-for-Self-driving-Cars-Application
In this research project, I have done some things below:        

Base on Deep learning (Mxnet) to implement object detection and tracking system on self-driving car system      

Based on given dataset and Yolo algorithm to construct special neural network model and update a new loss function      

Utilize GPU for training and tune parameters to converge and optimize the result        

Optimize feedforward inference network and realize object detection and tracking in real time on camera     


(In the future) Move the system to robots for avoiding obstacle   


See demo below or see result in jupyter notebook's result       

Yolo algorithm. There are 2 verson for it. [v1](https://arxiv.org/pdf/1506.02640.pdf), [v2](https://arxiv.org/pdf/1612.08242.pdf)./


# Dependencies
Python,Mxnet 1.0, cuda8.0, tensorboardX,cudnn,opencv,GPU:nvidia 1070T


# Yolo-v1


Data and Model
https://drive.google.com/drive/u/0/folders/0BwXw1vJFiBDaZ1IwNjlEd0RZMFU     

For asking training Dataset and testing Dataset, you could send me email.		

To run the code:		

    mkdir and cd to the path of "DATA_rec/"
    run "python data_util/py" for data preparaion with train and val recfile
    cd to the src root path and run "pyton run_train.py"
  
  For the test:
    Please take a look of wild_test.ipynb and demo_test.ipynb in src first.
    
    And then run test.py which could output a json file for results and draw the bbox in image
    
    For real time predict, to run "pythonw real_time_object_detection" on Mac/ "python real_time_object_detection"
    
# Result
Demo for test data:		

Real-time test:		
# Refer:

http://blog.topspeedsnail.com/archives/2068

https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/
# Yolo-v2
Is writting and updating
# Result
Demo for test data:		

Real-time test:		
