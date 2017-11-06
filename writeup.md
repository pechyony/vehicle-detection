**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/cars_not_cars.png
[image2]: ./examples/hog_features.png
[image3]: ./examples/search_windows.png
[image4]: ./examples/test_images.png
[image5]: ./examples/bboxes_heat_labels.png
[video1]: ./output_videos/project_video.mp4

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used `hog` function of `skimage.feature` package to compute HOG features. This function is called from the function `get_hog_features` located in cell `#2` of the Jupyter notebook.  `get_hog_features` is called from `extract_features` function, which is in cell `#5` of the Jupyter notebook.

I started by reading in all the `vehicle` and `non-vehicle` images. These images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. Here is an example of ten random images of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

All images have 64x64 pixels. There are 8792 `vehicle` images and 8968 `non-vehicle` images. `vehicle` images contain a single car. Most of the time the car is shown from behind. `non-vehicle` images show various parts of empty road, traffic signs and the areas surrounding the road. Both `vehicle` and `non-vehicle` images vary in brightness and contrast.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` and `LUV` color spaces and HOG parameters of ``orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` and `transform_sqrt=False`:

![alt text][image2]

HOG images show the egde that corresponds to the strongest gradient in each cell of 16x16 pixels. The brightness of the edge is proportional to the magnitude of the gradient. 

Surprisingly, the orignal images in R, G, B and L dimensions and their HOG transforms look very similar. However U and V dimensions contain additional information that is not stored elsewhere. In general, the HOG transform in R, G, B, and L dimensions has many cells where horizontal edge is the strongest one. There are also a few cells in these dimension with the edges having a slope around +/- 45 degrees. The prevalence of these gradients in the HOG transform looks like a very distinctive feature of images with a car. 

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various color spaces and values of `orientation`, `pixels_per_cell` and `cell_per_block` parameters. I used cross-validation error and detection of object in the test images to choose the best color space and the best values of parameters. The results of these experiments are described in the next section and section 2 of Sliding Window Search chapter. 

Since `U` and `V` dimensions of `LUV` color space might have negative values, I set `transform_sqrt` parameter of `skimage.hog()` to `False`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC` class of `scikit-learn`. The code for training a model is in the cell `#10` of the Jupyter notebook. The training pipeline has multiple steps:

* Compute HOG features, histogram features and spatial binning features. Each type of features is computed in a separate color space. The final feature set is a concatenation of all features from these 3 feature sets.

* Normalize all features to have zero mean and standard deviation 1.

* Define a grid of possible values of hyperparameter C of Linear SVM. I used the grid `[0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]`.

* Use `GridSearchCV` class of `scikit-learn` to find the value of hyperparameter C that maximizes cross-validation accuracy. Since the dataset is pretty big, I used 2-fold cross validation.

* Use the optimal value of hyperparameter C to train the final model over the entire dataset. Notice that this and previous steps are performed inside a single call to `fit` method of `GridSearchCV` class.

I compared the final cross-validation accuracy for different combinations of the color spaces of each of the three feature sets. Within HOG features I tried to use either a single dimension of the color space or the entire set of three dimensions. 
The search over different values color space for different feature sets was not exhastive and I checked only about 20 different combinations of the color spaces and dimensions. 
I found that cross-validation accuracy was maximized with HOG features of L dimension of LUV color space + histogram feautures in RGB space + spatial binning features in RGB space. These features resulted in cross-validation accuracy of 88.75\%. 

In all these experiments I used `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` parameter values of HOG features, `hist_bins=16` and `hist_range=(0,1)` parameter values of histogram features and `spatial_size=(16,16)` parameter value of spatial binning features. There values of parameters induced 2580 features. I tried different values of parameters of feature sets, but didn't observe any improvement of the cross-validation accuracy.
 
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code of sliding window search is in function `find_cars()` in the cell `#7` of Jupyter notebook. I did a sliding window search with 96x96 windows. Such windows were big enough to enclose a full car when it is viewed from the rear and about half a car when it is viewed from a side. I tried also smaller and larger window sizes, but got worse object detection in terms of the numbers of true and false positives. 

Remember that the training images had a size 64x64. Hence to use the car classification model developed in the previous section, we need to scale a 96x96 search window to the size 64x64. Instead of scaling each search window separately, I scaled the original image by the factor of 1/(96/64) and used 64x64 search windows inside the scaled image. When a 64x64 search window is classified as a car, I convert it into a 96x96 window in the original image by multiplying the coordinates by 96/64. 

I slided the search window by two cells (parameter `cell_to_step` in the cell `#7`), with each cell having 8x8 size. Hence two successive windows have an overlap of 64-2*8=48 pixels in the search direction. I tried to slide windows by one cell. While the object detection results were about the same as with two cells, the object detection pipeline was slowed down by a factor of 2. I also tried to slide windows by three cells, but this resulted in a mediocre object detection results. 

The image below shows all overlapping 96x96 search windows in the original image:

![alt text][image3]

Since in the scaled image the windows are slided with the step of 2 cells x 8 pixels = 16 pixels, in the original image the sliding step is 16/64*96=24 pixels. This is exactly the distance between two successive parallel blue lines in the above image. To speed up a sliding window search I searched over the road area of the image, which has y-coornate between 400 and 650. 

I used HOG subsampling to speed up the computation of HOG features for multiple search windows. HOG subsampling ensures that the HOG features of each block (which is a 2x2 array of cells, i.e. 16x16 pixels) in the original image are computed only once. Initially HOG features are precomputed for each block of the entire image. Then the values of the HOG features for a search window are taken from the precomputed values of HOG features of the blocks that are inside it.

I classified each search window using Linear SVM model described in the previous chapter. For each positively classified window I stored its raw score, gived by `decision_function()` function of lienar SVM. This raw score indicates how strong is the car detection. This score is leveraged when processing video streams. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As described in Section 3 of "Histogram of Oriented Gradients" chapter, I chose the values of parameters of feature generation and classifier using cross-validation error. Here are examples of the object detection in test images:

![alt text][image4]

In these test images there are no false positives. The black car is detected perfectly in all images, the white car was fully detected in 3 images and partially detected in the other 2 images. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video and their scores. From the positive detections I created a heatmap, where each pixel is a maximum value of the scores of all boxes covering it. Then I thresholded that map with the `threshold=0.35` to identify vehicle positions.  In the next step I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I assumed each blob corresponded to a vehicle. Finally, I constructed bounding boxes to cover the area of each blob detected.  

Here's an example showing the positively classified boxes in a series of video frames, the induced heatmap, the result of `scipy.ndimage.measurements.label()` and the boxes rendered in the video frame:

![alt text][image5]

In the heatmap pictures the brightness of each pixel is proportional to the maximal raw score of the bounding box containing it. The white color corresponds to detections with high scores and the red color corresponds to detections with low scores. Heatmap images show that both cars are detected most of the time with very high raw scores. We also see in the heatmap and labelled regions images that the fourth and fifth images of the sequence have false positive detections. However the final rendered image (rightmost columns in the above picture) does not contain false positive. This happens due to object tracking mechanism described below and implemented in `process_image_continuous` function in cell #16:

0. I maintained a list of currently detected objects (`detected_objects` variable in cell `#16`). Each detected object is an instance of class `Object`, defined in cell `#14`, and contains 
    * bounding box coordinates
    * center of the bounding box
    * age (in terms of video frames where it was detected)
    * indicator if it updated in the last video frame

1. After detecting all bounding boxes in the current frame, I try to match them with the objects detected in the previous frames. Previously detected object and a newly detected bounding box are matched when their centers are at most 70 pixels apart from each other in each direction. If there is a match then the bounding box of the current object is updated the new bounding box using the smoothing parameter alpha:

        xmin(current object) = (1-alpha) * xmin(current object) + alpha * xmin(new bounding box)
        xmax(current object) = (1-alpha) * xmax(current object) + alpha * xmax(new bounding box)
        ymin(current object) = (1-alpha) * ymin(current object) + alpha * ymin(new bounding box)
        ymax(current object) = (1-alpha) * ymax(current object) + alpha * ymax(new bounding box)

    I found emprically that the best performance (in terms of the number of true and false positives) is obtained with `alpha=0.5`. Also, in case of the match the age of the current object is increased by 1. 

    If there is no match for the newly detected bounding box then a new detected object is created with the coordinates of the bounding box and `age=1`.

2. If the object was not detected in the last frame, i.e. if there was no match between the object and any bounding box from the last frame, then the age of the object is divided by 1.1 . If the age becomes less than one then the object is deleted from the list of the currently detected objects. 

3. If the age of the object is more than 20 then the object is drawn in the current video frame. 

In the example above the false positives in rows 4 and 5 were not drawn in the current video frame because their age was less than 21.

---

### Discussion

I described an object detecion pipeline that uses a simple machine learning model and a manually engineered feature set. While I was able to tune the pipeline to work well in the given project video, there no guarantee that it work well with other videos. In particular, the existing pipeline might render a large number of false positives and a low number of true positives. Here is a partial list of improvements that will make the pipeline more robust:

* Automatic joint tuning of all parameters of the pipeline to optimize the number of true positives and false positives. This can be achieved by using [Pipeline objects](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) of scikit-learn.

* Usage of more powerful machine learning algorithms (e.g. convolutional neural networks) that don't require manual feature engineering.

Also, the pipeline was tuned using a video that has a straight road segment with just two cars. To improve the robustness of car detection, the pipeline should also be tuned using videos of curvy roads with multiple cars. 

