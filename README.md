# ec503_final_project
## How to run our code
### 1. Download dataset
Download the PASCAL VOC 2012 dataset at (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)<br />
### 2. Get the original data
Run the initial_imp.m at (https://github.com/wasimk1995/ec503_final_project/blob/master/implementation/custom/initial_imp.m)<br />
For this file, you should change the image_dir and label_dir to your own absolute path of the folders JPEGimages and Annotations inside the VOC2012.<br />
This program will generate a data_300_300.mat which contains the feature vector and the ground-truth label for every sample.
## 3. Run the model
Put the data_300_300.mat in the same folder with the models.<br />
Run the models and they will generate the training and test results like the CCR, precision, recall and F-score.
