# FYP
Deep Learning for Medical Ultra-Sound Image Classification/Project Number: A3282-201


# Explanation of code:
Importing datasets and defining them into dataloader
Defining Models from torchvision.models 

Defining test model functions: 
#conducts test run using testloader database
#tests the model based on highest validation F1 score(test_f1)/smallest validation loss(test_valid)
#test results are then compiled and saved into dataset directory

Defining graph model functions: 
#plots and saves the figures into respective directories
#3 types of graphs functions are created=

  graph: basic individual graph plotting of performing parameters against epoch/iteration
  lrgraph: combined graph plotting to show performing parameter across different lr
  modelgraph: combined graph plotting to show performing parameter across different models
  
Defining calculating functions:
#calculates the mean and standard deviation
#Saves the respective numpy array of performing parameters values and calculation results into dataset directory

Defining training model function:
#Runs 50 epoch and 15600 iterations of training and validation loop
#For every epoch, run training loop according to length of trainloader(iterations).
  #For every 10 iterations, run validation loop and save and record the highest F1/smallest valid loss value
#At the end of training, call the graph model functions, mean functions and test functions

Running models and learning rates:
#RESNET18 lr0,lr1,lr2,lr3 - RESNET50, lr1 - ALEXNET,lr1

Plotting the graphs for report:
#Load the saved numpy values from directory and call graph model functions 
