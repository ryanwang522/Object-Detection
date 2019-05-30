# Object-Detection
The task is to detect and clssify the objects present in the aerial images by determining their bounding boxes.
![](https://github.com/ryanwang522/Object-Detection/blob/master/resource/intro.jpg)

## Network Architecture
I took VGG16 with batch normalization as backbone model, and further trained the last few FC layers for classification. 
![](https://github.com/ryanwang522/Object-Detection/blob/master/resource/arch.png)

## Method
Devide the image to 7x7(or 8x8) grids, each cell predicts two bounding-boxes and object confidence.
![](https://github.com/ryanwang522/Object-Detection/blob/master/resource/grid_cell.png)

## Dataset
I use aerial images in [DOTA dataset](https://captain-whu.github.io/DOTA/) for object detection.
And there are total 16 classes in this implementation.
```
OBJECT_CLASSES = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "small-vehicle": 9,
    "large-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
    "container-crane": 15
}
```

## Results
The model outputs in different training stages (i.e. early, middle, final stages) are as below:
* Tennis-court
![](https://github.com/ryanwang522/Object-Detection/blob/master/resource/result-1.png)
* Plane
![](https://github.com/ryanwang522/Object-Detection/blob/master/resource/result-2.png)
* Storage-tank
![](https://github.com/ryanwang522/Object-Detection/blob/master/resource/result-3.png)




