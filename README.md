# FYP
Deep Learning for Medical Ultra-Sound Image Classification/Project Number: A3282-201

# How to run code:
This code is run from Visual Studio Code and the output is printed as a ipynb folder. 

To run this code on your selected environment, ensure the following is met:
  1. Respective libraries have been installed
        pytorch lightning, torch, numpy etc. libraries can be found on the top of each cell
        search for "import" through the code for quick search on all the libraries needed
        
  2. Model paths and image root folder path directory is linked to the correct path in the system
        image_root=r"[path]\Dataset_BUSI_with_GT" -> link to image folder "Dataset_BUSI_with_GT"
        model_path=r"[path]\FYP\[model]" -> link to every respective models
        
  3. File "Dataset_BUSI_with_GT" has no masks images and should be downloaded
  
# Summary Explanation of code:
For the first few lines, we would be doing data importing, loading and splitting to a ratio of 80% of the data is used for training, 10% is used for validation and 10% is used for testing.

Then, the classification and importing of convolution, layering and blocks obtained for the models from model zoo documentation in pytorch.org.

Two experiments are splitted and carried out:
  1. Training the same model type RESNET18 with different learning rates to choose the best learning rate
  2. Training the different model types RESNET50 and ALEXNET at the chosen learning rate to choose the best model to represent our dataset

The Training loop consisted of running over 50 epoches and 15600 iterations. 
With every 10 iteration, a validation test is carried out.
At the end of the training loop, 
  the following parameter values are calculated and produced:
    training & validation losses,accuracy,F1 score
    mean and standard deviation losses of the training&validation loss,accuracy and F1 scores
    
  these are the outputs are saved:
    1. model state with the highest validation F1 score as "highest_f1.pt" as a checkpoint for test run
    2. model state with the smallest validation loss as "smallest_valid_loss.pt" as a checkpoint for test run
    3. training results per epoch is saved and compiled into a numpy array and saved into localized folders as .npy file
    4. validation results per iteration is saved and compiled into a numpy array and saved into localized folders
    5. Every parameter values is compiled and saved into localized folders as "data_results.txt"
    6. Every graph plotted is saved into their respective localized folders as .png file
    
Two seperate test runs are then conducted with the model states with the best performance scores - highest validation F1 score and smallest validation loss. Similarly, the test results are then printed and saved in the same file that is collating the model output parameter values in "data_results.txt"
 
With all the values of the training, validation saved into a localized folder, we would be able to easily plot the combined graphs that we need to conclude our experiments and choose the best performing model for our dataset.
