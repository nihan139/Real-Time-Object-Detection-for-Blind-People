# Real-Time-Object-Detection-for-Blind-People
Neural Network Model Based (YOLO V3)


## **Dataset Description**
-   Pascal VOC 2007 dataset is used to train the model.
-   In Pascal VOC dataset there are total 9963 images and 24640 annotated objects in train and test dataset. In train dataset there are 5011 images and 15662 annotated files. Also there are 4952 images in test dataset.
-   All the images of dataset are RGB images. There are 20 classes to detect in the dataset.

<br>

Here are some image with annotations from dataset
<br>
<br>

<p align="center">
   <img src="./image1.jpg" width="600" height="300"/>
</p>


<br> <br>

## **YOLO V3 Architecture**

<br>

<p align="center">
   <img src="./architecture.jpg" width="600" height="300"/>
</p>

<br> <br>

## **Working Method**
-   The model splits the input image into a grid.
-   The model is applied to an image at different scales and locations.
-   Prediction occurs in 3 scales such as 13×13, 26×26 𝑎𝑛𝑑 52×52.

<br>

<p align="center">
   <img src="./working_method.jpg" width="600" height="300"/>
</p>

<br> <br>


## **Loss Function & Metrics**
-   Square Loss
-   Logistic Regression Loss

Intersection Over Union (IoU) metric is used which reflects how good the predicted bounding box is for a particular object.

<br>

<p align="center">
   <img src="./loss_vs_epoch.jpg" width="400" height="300"/>
</p>

<br> <br>


## **Predicted Results**
<br>

<p align="center">
   <img src="./prediction.jpg" width="600" height="300"/>
</p>

<br> 

So the trained model perfectly predicts the objects.

<br> 

## **Contributors**
1. Md. Tasnim Azad
2. Sabbir Hossain Ujjal
3. Md. Tajbid Zaman Rhythm
4. Kazi Moheuddin Alamgir
5. Md. Tanveer Aziz Durbar
