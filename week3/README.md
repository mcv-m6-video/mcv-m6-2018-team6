# Video Surveillance for Road Traffic Monitoring
Master in Computer Vision - M6 Visual Analysis  

 
# Week 3
In this week, our goal is to learn how to :  
1. Implement a robust foreground segmentation algorithm.  
2. Basic post filtering strategies as Hole filling, are filling and morphological operators.  
3. A strategy to remove shadows.  



## Task 1: Hole filling   
- Task1: post process the previous week method with hole filling, with 4 and 8 connectivity and provide gain on AUC.  

   Instructions:  task1.py  

## Task 2: Area filtering
- Task2.1: AUC vs #pixels: Plot the AUC vs the number of pixels of are filtering.  

  Instructions:  later will be uploaded   
  
- Task2.2: Argument max of AUC depending on the P, to find the highest mean of AUC for each sequence.  

  Instructions:   

## Task 3: Morphological processing
- Task3: Morphological filtering: Apply various morphological operations (opening and closing) with different structuring elements.   

  Instructions:  task3.py  | given a foreground it applies the morph operation desired with the desired structuring element.  


## Task 4: Shadow removal
- Task4: try to remove shadows based on the previos methods for foreground detection.  

Instructions: main.py  


## Task 5: Comparison PR curve (AUC)
- Task5: A comparison between the  previous (week2) and the model after morphological filtering has been applied. 

Instructions: we run the task3.py and the task 1 from last week and plot them together.

