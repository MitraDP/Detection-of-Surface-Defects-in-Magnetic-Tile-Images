# Detection of Surface Defects in Magnetic Tile Images
## Dataset

The [magnetic tile surface defect dataset](https://github.com/abin24) is originally used by Huang et al [1]. Based on the defect types, the dataset is divided into six smaller datasets: blowhole, crack, break, fray, uneven (grinding uneven) and free (no defects). Each image is accompanied by its pixel-level ground-truth image. 



<p align="center">
<image src= "assets/magnetic_tile_surface_defects.png" width="600"> 
</p> 
<p align="center">               
Examples of magnetic tile surface defects, labelled with the pixel-level ground truths [1].
</p>

## Objective
One of the most laborious parts of quality control of magnetic tile manufacturing is surface defect detection. Since blowhole and crack have a crucial impact on the quality of the tiles. The goal of this study is binary semantic segmentation of the Blowhole_Crack_Free dataset. 

## Evaluation Metrics

The evaluation metrics are the maximum F_beta measure (beta^2=0.3) and the mean absolute error (MAE) associated with it.

## Solution Approach

In semantic segmentation, every pixel in the image is assigned to a class. To train a network that detects defects at the pixel level (binary semantic segmentation), I define two main classes: defective and flawless. I label the defective pixels as True (1) and the rest as False (0). 

Table 1 shows the number of images in each class. Since the dataset is highly imbalanced, I perform undersampling on the “Free” dataset and sample randomly 80 images. Using the stratified random sampling technique, I divide the datasets into train, validation and test sets using a 70/10/20 split. The images and their corresponding masks, which are in various sizes, are all resized to 224×224 pixels. Each pair of images and mask of the train dataset are randomly flipped horizontally or vertically or rotated at an angle of (0, -90, 90, 180) degrees.  

Table 1

|Class|Number of images|
|----|----|
|blowhole|115|
|crack|57|
|break|85|
|fray|32|
|uneven|103|

Apart from undersampling “Free” images, more action is required to handle the data imbalance. To do so, I use either Tversky loss [3] or the Weighted BCE loss function.
I perform the pixel classification using the UNET architecture which is developed by Olaf Ronneberger et al. for BioMedical Image Segmentation [2]. This architecture, which is a Fully Convolutional Network, contains an encoder and a decoder path. The encoder path captures the context of the image and the decoder path enables localization. The contracting path is a stack of convolutional and max-pooling layers and the symmetric expanding path uses transposed convolutions. Figure 2 summarizes the model that is used in this project which has four resolution steps. I use 32 in-features and a dropout probability of 20%.


<p align="center">
<image src= "assets/2D_UNet_architecture.png" width="600"> 
 </p>


I did experimentation with various optimizers (SGD, Adam), batch sizes, and loss functions (Weighted BCE, Tversky). For the training schedule, I use Leslie Smith’s One Cycle Learning Rate Policy [4] with 200 epochs. In each training, the best model is the one with the lowest validation loss.
With a batch size of 32, Adam optimizer, Tversky loss, and maximum learning rate of 0.02, I get  0.937 and 8.07e-04 for the maximum F_beta measure and its mean absolute error (MAE), respectively.
Samples of images, masks, and the binary prediction counterparts of the Crack and Blowhole defects are shown down below. Pixels with probabilities of 0.5 (threshold) or higher are considered defective.
 
<p align="center">
<image src="assets/Crack_prediction_sample1.png" width="600"> 
 </p>
  
<p align="center">
<image src="assets/Crack_prediction_sample2.png" width="600"> 
 </p>
  
<p align="center">
<image src="assets/Crack_prediction_sample3.png" width="600"> 
 </p>
  
<p align="center">
<image src="assets/Blowhole_prediction_sample1.png" width="600"> 
 </p>
  
<p align="center">
<image src="assets/Blowhole_prediction_sample2.png" width="600"> 
 </p>
  
<p align="center">
<image src="assets/Blowhole_prediction_sample3.png" width="600"> 
 </p>


## Run 

- Download the dataset from https://github.com/abin24/Magnetic-tile-defect-datasets to your Google Drive.
- You may change the batch size, optimizer type, loss function, threshold probability limit, classes that you might want to keep, and the number of epochs in the “Set the training parameters” section. If you wish to include more classes update the list in the “partitioning” function of “dataset.py” as well.  
- If you wish to find the maximum learning rate, set lr_find to true and run the code up to the “Train and validate the model” section. Change the scheduler “max_lr” value to the suggested rate and set the “lr” in the optimizer definition to 1/10 of the “max_lr”. Set “lr_find” to False and rerun the code.
- The notebook will generate a CSV file for the history of train and validation loss.
- The notebook performs classification and visualization on the “test” data.

## References:

[1] Huang, Y., Qiu, C. & Yuan, K. Surface defect saliency of magnetic tile. Vis Comput 36, 85–96 (2020). https://doi.org/10.1007/s00371-018-1588-5.

[2] Ronneberger, O.; Fischer, P.; Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Proceedings of the International Conference on Medical image computing and computer-assisted intervention, Munich, Germany, 5–9 October 2015; pp. 234–241.

[3] Salehi, Seyed Sadegh Mohseni et al. “Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks.” ArXiv abs/1706.05721 (2017): n. pag.

[4] L. N. Smith, A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay, US Naval Research Laboratory Technical Report 5510-026, arXiv:1803.09820v2. 2018.

[5] Achanta et al.,"Frequency-tuned salient region detection," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami,FL,USA, 2009, pp.1597-1604, doi: 10.1109/CVPR.2009.5206596.

[6] https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114/2

[7] https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

[8] https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb

[9] https://discuss.pytorch.org/t/pytorch-how-to-initialize-weights/81511/4

[10] https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24

[11] https://github.com/joe-siyuan-qiao/WeightStandardization

[12] https://github.com/davidtvs/pytorch-lr-finder

[13] https://github.com/frankkramer-lab/MIScnn

[14] https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6

