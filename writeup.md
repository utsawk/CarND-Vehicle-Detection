## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image0000.png "vehicle image"
[image2]: ./output_images/extra14.png "non-vehicle image"
[image3]: ./output_images/hog.png "hog features"
[image4]: ./output_images/heatmap.png


### Files
My project includes the following files:
* P5.ipynb (Jupyter notebook) containing the project code related of detection of vehicles only. This file can be used to train the SVM and run it on a video to detect vehicles. Jupyter notebook is used to show all code and visualizations in the same document. To run the project, its preferable to use the python files provided because it combines both lane detection and vehicle detection
* train_svm.py containing code to train the svm
* lane_lines_and_vehicle_detection.py contains code to detect lane lines and vehicles
* utils_lane_finding.py contains utility functions to detect lane lines
* utils_vehicle_detection.py contains utility functions for detecting cars/vehicles on the road

### Training the model
The data to train the model can be downloaded from [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). The given SVM can be trained by executing
```sh
python train_svm.py
```

### Detecting lane lines and vehicles
Using the trained SVM model, lane lines and vehicles can be detected by executing
```sh
python lane_lines_and_vehicle_detection.py
```
The output is saved in a file named "lane_and_vehicles.mp4".

### Feature extraction

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes.

![alt text][image1]
![alt text][image2]


The code for feature extraction and training the classifier is in cell 5 of the IPython notebook. Hog feature extraction is done in the function get_hog_features() in cell 2 of the notebook. I decided to use the histogram of color and spatial binning feature because that gave me higher accuracy on the test data as compared to when the classifier was trained only using the HOG feature (~99% as compared to ~96%). An example of the hog feature is shown in figure below for "YUV" color space, HOG parameters of `orientations=12`, `pixels_per_cell = (16, 16)` and `cells_per_block = (2,2)` are used.

![alt text][image3]

#### Final choice of HOG parameters.

I experimented with different color spaces and chose "YUV" though some of the other color spaces (like YCrCb, HSV, etc.) also gave very similar results. I tried various combinations of parameters and converged on using `pixels_per_cell = (16, 16)` because increasing from the lecture suggested value of `(8,8)` did not affect the performance while at the same time running much faster both when training and when detecting vehicles in the video. I also increased the number of orientations to 12 because it gave marginally better test accuracy. 

### SVM

I trained a linear SVM after normalizing the training set using the sklearn StandardScaler() utility class that subtracts the mean and scales the dataset to make it unit variance. Using the above mentioned parameters, I got an accuracy of 98.9% on the test set. The code for this is in cell 5 of the notebook. 

### Sliding Window Search

Instead of extracting hog features independently for each window selection, I used hog subsampling like described in the class notes. The code of this is in the function find_cars() in cell 6 of the notebook. I used three window sizes for my search as shown in cell 15 of the notebook:
* scale 1.5, ystart = 400, y_stop = 500
* scale 2.5, ystart = 400, y_stop = 575
* scale 3.5, ystart = 400, y_stop = 620

This gives a series of boxes corresponding to the overlapping windows used to look for vehicles.  

### Heatmap & False Positives
The overlapping boxes can be then combined via a heatmap thresholding to construct one box showing detected vehicle on the road. The thresholding helps remove false positives as well. An example of detected boxes around vehicles is given in figure below.

![alt text][image4]
---

### Video Implementation

Here's a [link to my video result](./project_video_output.mp4) with only detected vehicles and here is a [link to my video result](./lane_and_vehicles.mp4) combined with lane detection from last project. Note that I also do camera calibration and distortion correction when combining lane and vehicle detection. I have written python files for combining the approaches because doing everything on Jupyter notebook may not be the best approach.


#### Dealing with false positives

To deal with false positives on the video is easier than in individual images. I create a class called bounding_box() in cell 15 to track boxes over multiple frames. I keep track of the last 10 frames and bounding boxes for those 10 images. I threshold the heatmap at 4, which is obtained empirically. Since false positives occur only in some frames, this thresholding kills the false positives while smoothing the output boxes across frames as well.


---

### Discussion

I started looking at the deep learning approaches after spending considerable time on the computer vision-SVM approach to solve this problem and it seems that you can possibly run it way faster than the approach outlined here. In the interest of time, I decided to submit the project as is and continue exploring the deep learning approach. I feel that the thresholding is going to definitely fail in a lot of corner cases. I also tried exploring SVM tricks to limit the number of false postivies and tried playing around with C and gamma to shift the hyperplane separating the two classes to limit false positives, but was not successful doing so.  

