Project 3        
Morphology transformation, Image Segmentation and    
Point detection, Hough Transformation
==========================
Manish Reddy Challamala,
December 3, 2018 ,manishre@buffalo.edu

For detailed explaination, please visit the below link:

[link for report pdf](https://github.com/manish216/CSE-573-Project3/blob/master/proj3_cse573.pdf)

## 1 Task1 - Morphology image processing


### Abstract
The goal of this task is stated point wise below:

1. Remove the noise from the given image ’noise.jpg’ by using the two
    Morphology image processing algorithm.
2. compare the output of above two resultant images and specify weather
    the two resultant images are same or not.
3. Extract the boundary of the two images and save the result.

## 1.1 Experimental set-up:

1. For this task, we are using the following libraries:
    1. Cv2, numpy, matplotlib
2. The morphology operations are stated below:
    1.1. Erosion - Erodes the image.
    1.2. Dilation - Dilates the image.
    1.3. Opening - It is nothing but Erosion followed by dilation operation.
    1.4. Closing - It is nothing but dilation followed by erosion operation.
3. The dilation operation, takes care of pepper noise.
4. The erosion operation, takes care of salt noise.
5. Algorithm for Erosion:

    5.1. consider a image matrix
    5.2. consider a structuring element.
    
          | 1  |  1  | 1  |
          | --- | --- | --- |
          | 1  | 1  | 1  |
          | 1  | 1  | 1  |

        
    5.3. Initialize a matrix with zero of size of image let it be Newimage
    
    5.4. pad the original image.
    
    5.5. Convolve the structuring element on the padded image and check weather the ones in structuring element overlaps only with the ones in padded image.
    
    5.6. If the condition is True, update the newimage matrix with one and slide to the next position else zero.
    
    5.7 Repeat the step 4.4 and 4.5 for all the pixel values in padded image.
    
6. Algorithm for dilation:
    
    6.1. Steps are same still step 4.4.
    
    6.2. Convolve the structuring element on padded image and perform a logical AND operation.
    
    6.3. If all the values of resultant Logical AND operation is zero then update the newimage matrix position with zero. else with one.
    
    6.4. Repeat the step 5.2 and 5.3 for all the pixel values in padded image.


## 2 Task 2 - Image Segmentation and Point detection.

### Abstract
  The goal of this task is stated point wise below:

  1. Detect the porosity by using the point detection algorithm and point
    the coordinates of the detected point on the image.
  2. Segment the object from background by choosing a threshold value
    and draw a rectangular bounding box around the object.

### 2.1 Experimental set-up:

1. For this task, we are using the following libraries:
    1. Cv2, numpy, matplotlib
2. Point Detection algorithm is a common way to calculate the gray-level
    discontinuities of a image by running a mask through the image. The
    formula to measure the weighted difference can be refered from report link given above. 

3. The kernel for the point detection is
 
 | -1  | -1  | -1  |
 | --- | --- | --- |
 | -1  |  8  | -1  |
 | -1  | -1  | -1  |

4. Algorithm for Point Detection:

    4.1. Consider the image matrix.
    
    4.2. Consider the kernel shown above in step 3.
    
    4.3. Initialize a matrix with zeros of size of the image. Let it be the new image.
    
    4.4. Convolve the kernel on the image and calculate the sum of the neighbouring elements
    by using the above formula and update the sum value in the respective pixel in the new image.
    
    4.5. Repeat the step 4.4 for all the pixel values in original image.

5. Algorithm for segmenting the object and background:
    
    5.1. Plot the histogram by finding out the intensities of all the pixel value in the image.
    
    5.2. choose a threshold value manually and threshold the image by using that threshold value.
    
    5.3. Find out the pixel coordinates of the segmented object in the image and draw the bounding box around the object.


## 3 Task3 - Hough transform

### Abstract
  The goal of this task is stated point wise below:

  1. Detect all the red lines in an image by using Hough transform algorithm and specify how many red lines are detected.
  2. Use the same above algorithm and detect all the diagonal blue lines in a image.

### 3.1 Experimental setup:

1. Hough transform is a technique used to isolate features of a particular
    shape in an image such as lines, circles an ellipse and etc.,
2. Parametric line notion of hough transform is given by :

```
r = xcosθ+ysinθ
```

    2.1. where r is the length of a normal from origin to this line and θ is
    orientation of r with respect to point (x,y) on line.


    2.2. Here we are converting the pixel co-ordinates of an image to a point in hough space.
    If we plot all the possible (r,θ) values to point in hough space, 
    The (x,y) point in the Cartesian image space maps to a curve in hough space, 
    So this point-to-curve transformation is called hough transformation for straight line.

3. Algorithm for Hough Transformation:
    
    3.1. Each (Xi,Yi) coordinates in image space is converted into a discrete
    (r,θ) curve and the accumulator which line along the curve are incremented.
    
    3.2. Transverse through the accumulator array and find out the peaks
    which results that there exists a straight line in the image.

## 6. Result:

  The results are available here:
  [Results]https://github.com/manish216/CSE-573-Project3/blob/master/proj3_cse573.pdf)
