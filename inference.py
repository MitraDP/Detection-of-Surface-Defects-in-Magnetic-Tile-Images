#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transforms import ToTensor
from metrics import performance_metrics 

#-----------------------------------------------------------------------#
#                      plot_prediction_results                          #
#  Performs prediction on the test dataset, plots and saves the images  #
#-----------------------------------------------------------------------#
# model:      trained model                                          #  
# loaders:    Test dataloader                                           #
# threshold:  Threshold value to create binary image                    #
#-----------------------------------------------------------------------#
def plot_prediction_results(model, train_on_gpu, loaders, threshold=0.5):
    for batch_idx, (images, targets) in enumerate(loaders):
        # Move image and mask Pytorch Tensor to GPU if CUDA is available.
        if train_on_gpu:
            images, targets = images.cuda(), targets.cuda()
        # Set the model to inference mode
        model.eval()
        # Forward pass (inference) to get the output
        output = model(images)
        # Define the size of images (3 images in a row)
        fig = plt.figure(figsize=(15,4))
        # Prep images for display
        images = images.cpu().numpy()     
        targets = targets.cpu().numpy()
        output = output.cpu().detach().numpy()
        # Binarize the output
        output_b = (output>threshold)*1
        # Plot the image trio: image, mask and prediction
        ax = fig.add_subplot(1, 3, 1, xticks=[], yticks=[])        
        ax.imshow(np.squeeze(images), cmap='gray')
        ax.set_title('image'+str(batch_idx)) 
        ax = fig.add_subplot(1, 3, 2, xticks=[], yticks=[])
        ax.imshow(np.squeeze(targets), cmap='gray')
        ax.set_title('target') 
        ax = fig.add_subplot(1, 3, 3, xticks=[], yticks=[])
        plt.imshow(np.squeeze(output_b), cmap='gray')
        ax.set_title('binary prediction')
    # Save the image trio: image, mask and prediction
    plt.savefig('b_'+str(batch_idx)+'.png')
    plt.clf()

#-----------------------------------------------------------------------#
#                 get_inference_performance_metrics                     #
#  Performs prediction on the test dataset, plots and saves the images  #
#-----------------------------------------------------------------------#
# model:      trained model                                             #  
# loaders:    Test dataloader                                           #
# threshold:  Threshold value to create binary image                    #
#-----------------------------------------------------------------------#
  
def get_inference_performance_metrics(model, train_on_gpu, loaders, threshold= 0.5):
    # A list to keep track of test performance metrics
    test_metrics =[]
    # Set the model to inference mode
    model.eval()
    # Initialize variables to monitor performance metrics
    specificity_val = 0
    sensitivity_val = 0
    precision_val = 0
    F1_score_val = 0
    F2_score_val = 0
    DSC_val = 0
    F_beta_score_val = 0
    MAE_val = 0
    accuracy_val =0
    # initialize the number of test instances
    test_cnt = 0

    for batch_idx, (data, target) in enumerate(loaders):
        # Move image and mask Pytorch Tensor to GPU if CUDA is available.
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass (inference) to get the output
        output = model(data)
        output = output.cpu().detach().numpy()
        # Binarize the output
        output_b = (output>threshold)*1
        output_b = np.squeeze(output_b)
        batch_l = output_b.size
        # update the total number of test pairs
        test_cnt += batch_l
        t1 = ToTensor()
        # Transform output back to Pytorch Tensor and move it to GPU
        output_b = t1(output_b)
        output_b = output_b.cuda()
        # Get average metrics per batch
        m = performance_metrics(smooth = 1e-6)
        specificity, sensitivity, precision, F1_score, F2_score, DSC, F_beta_score, MAE, accuracy = m( output_b, target)    
        
        specificity_val += specificity * batch_l
        sensitivity_val += sensitivity * batch_l
        precision_val += precision * batch_l
        F1_score_val += F1_score * batch_l
        F2_score_val += F2_score * batch_l 
        DSC_val += DSC * batch_l 
        F_beta_score_val += F_beta_score * batch_l
        MAE_val += MAE * batch_l
        accuracy_val += accuracy * batch_l
        # Calculate the overall average metrics
        specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val , F_beta_score_val, MAE_val , accuracy_val= specificity_val/test_cnt, sensitivity_val/test_cnt, precision_val/test_cnt, F1_score_val/test_cnt, F2_score_val/test_cnt, DSC_val/test_cnt, F_beta_score_val/test_cnt, MAE_val/test_cnt, accuracy_val/test_cnt
        
        test_metrics.append((specificity_val, sensitivity_val, precision_val, F1_score_val, F2_score_val, DSC_val, F_beta_score_val, MAE_val, accuracy_val))
    #save the test metrics as a Pandas DataFrame
    df=pd.DataFrame.from_records(test_metrics, columns=[ 'specificity', 'sensitivity', 'precision', 'F1_score', 'F2_score', 'DSC','F_beta', 'MAE', 'acc' ])
    # Save the test metrics
    df.to_csv('test_metrics.csv', index=False)       
    return df

