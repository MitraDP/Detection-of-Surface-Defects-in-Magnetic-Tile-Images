#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch

#-----------------------------------------------------------------------#
#                        Performance metrics                            #
# Reference:                                                            #
# Salehi et al.,“Tversky Loss Function for Image Segmentation           #
#    Using 3D Fully Convolutional Deep Networks.”                       #
#    ArXiv abs/1706.05721 (2017)                                        #
# Achanta et al.,"Frequency-tuned salient region detection,"            #
#    2009 IEEE Conference on Computer Vision and Pattern Recognition,   #
#    Miami,FL,USA, 2009, pp.1597-1604, doi: 10.1109/CVPR.2009.5206596.  #
#-----------------------------------------------------------------------#
# Returns: specificity, sensitivity, precision, F1_score, F2_score,     #
# Dice similarity coefficient, F_beta_score, mean absolute error,       #
# and accuracy per batch as a numpy array                               #
#-----------------------------------------------------------------------#
# smooth: a very small number added to the denomiators to               #
#         prevent the division by zero                                  #
# beta_2: beta^2 in the F_beta_score formula
# tp:     number of true positives                                      #
# fp:     number of false positives                                     #
# tn:     number of true negatives                                      #
# fn:     number of false negatives                                     #
# DSC:    Dice_Similarity_Coefficient                                   #
# MAE:    mean absolute error
#-----------------------------------------------------------------------#


class performance_metrics():
    def __init__(self, smooth = 1e-10, beta_2=0.3):
        super().__init__()
        self.smooth = smooth
        self.beta_2 = 0.3
            
    def __call__(self, y_pred, y_true):
        # Flatten both prediction and GT tensors
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true)
        # calculate the parameters
        tp = (y_pred_flat * y_true_flat).sum()
        tn = ((1 - y_pred_flat) * (1- y_true_flat)).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        # continue the calculation in numpy
        tp = tp.cpu().detach().numpy()
        fp = fp.cpu().detach().numpy()
        tn = tn.cpu().detach().numpy()
        fn = fn.cpu().detach().numpy()
        #calculate the metrics
        if tn == 0.0:
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)
        if tp == 0:
            sensitivity = 0.0
            precision = 0.0
            F2_score = 0.0
            DSC = 0.0
        else:
            sensitivity = tp/(tp + fn)
            precision =  tp/(tp + fp)
            F2_score = (5*tp)/(5*tp + 4*fn +  fp)
            DSC = (2*tp)/(2*tp + fn + fp)
        if precision==0 or sensitivity==0:
            F1_score = 0.0 
            F_beta_score = 0.0
        else:
            F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
            F_beta_score =((1+self.beta_2)* precision * sensitivity) / \
                           ((self.beta_2*precision) + sensitivity)
        MAE = (torch.abs(y_pred_flat - y_true_flat)).sum()/(len(y_pred_flat))
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        return specificity, sensitivity, precision, F1_score, F2_score, DSC, \
               F_beta_score, MAE.cpu().numpy(), accuracy

