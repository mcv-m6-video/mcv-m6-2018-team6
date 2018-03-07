# Video Surveillance for Road Traffic Monitoring
Master in Computer Vision - M6 Visual Analysis

 
# Week 2
In this week, our goal is to learn how to model the background video frames:  
1. Understand and become familiar with the background/foreground. 
2. Learn about how to choose a background 
3. Implement state of the art methods for background modelling.  
4. Do it in grayscale images and color images.


## Task 1: Gaussian Distribution  
- Task1.1 Gaussian modelling. The dataset will be divided in two, the first half will be used for modelling the background (for
extracting mean and std) and the second half will be used to segment the foregorund.

Instructions: task1.py | given one of the three datasets it will provide a .gif with the performance and mean and std values plotted.

- Task1.2 F1_score vs alpha. When modelling the background, it is needed to choose an alpha as setting a threshold for the
permisivity of the algorithm to classify one pixel as a background or foreground. A plot of F1 vs alpha will help to see and 
choose the best alpha over a range of it.

  Instructions:task1.py | given one of the three datasets it will provide a plot with the F1 vs alpha curve.

- Task1.3 Evaluation AUC-PR: The Precision Recall curve is plotted and the Area Under the Curve is calculated. This value
will asses the performance of the background classifier.

  Instructions: task1.py | given one of the three datasets it will provide a plot Precision-Recall curve and AUC values.

## Task 2: Adaptive Modelling
- Task2.1 Adaptive Modelling (Baseline). In this case there is a grid search for the best alpha and rho parameters that
gives the best F1 value score. The mean and std of the background is first estimated with the half of the frames.
  
  Instructions: mainV3.py | given a dataset it will provide the grid search graph and the best values to get the best performance.
  
- Task2.2 Comparison adaptive vs non 

## Task 3: Comparison with state-of-the-art
- Task3 Comparison of the Single Gaussian developed versus the BackgroundSubstractorMOG from OpenCV (adaptive model)
  
  Instructions: sota.py | given a dataset it will provide a .gif with the performance and the F1 score of the MOG technique.  

## Task 4: Color sequences
- Task4 Update of the task 1 with an implementation to support color sequences:

Case 1- rgb:Red Green Blue channels.

  Instructions: mainV3.py | given a datast and RGB chosen, it provides the F1 values vs Alpha to see which is the largest F1-scor (prediction of the foreground).
  
Case 2- hsv: Hue Saturation Value channels.

Instructions: mainV3.py | given a datast and HSV chosen, it provides the F1 values vs Alpha to see which is the largest F1-score.(prediction of the foreground).
 
