
# Week 1
In this week, our goal is to learn how to evaluate and analysis video frames:  
1. Understand and become familiar with the programming framework used in the project (Python).  
2. Learn about the databases to be used  
3. Implement the evaluation metrics and graphs used during the module.  
4. Read / write sequences of images and associated segmentation ground truth.  

To check all the tasks, there is a week1_launch file to direatly get the results in each task. 
Here are some explaination of the python files:
1. evaluation.py: including these function which relate to task1,2,4(highway images)-  
evaluateAllFrames, evaluateOneFrame, temproalTP, temproalFscore
2. OpticalFlow.py: including these funtion which relate to task3,5(optical flow)-  
MSEN_PEPN, histSquareError, visulizeError

## task1 precision and recall  
task1.1 Given the ground truth and results of two methods(A andd B), to get the precision, recall and f1-score and analysis the performance.  
task1.2 Background Substraction (qualitative)  

## task2 Temporal analysis of the results
task2.1 TP and TF vs frames  
task2.2 F1-score vs frames  

## task3 quantitative evaluation of optical flow
task3.1 MSEN  
task3.2 PEPN  
task3.3 Analysis & Visualizations  

## task4 De-synchronization of results
task4 Force de-synchronized results for background substraction and study the results     

## task5 Optical flow plot
task5 plot and visualize the optical flow  
