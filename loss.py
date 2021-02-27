#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
import torch.nn as nn


#-----------------------------------------------------------------------#
#                          WeightedBCELoss                              #
#     Calculate and return the weighted binary croos entropy loss       #
#-----------------------------------------------------------------------#
# adopted from:                                                         #
# https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114/2  #
#-----------------------------------------------------------------------#
# weight:   weight tensor per class                                     #
# smooth:   a small number to avoid log(0.0)                            #
#-----------------------------------------------------------------------#
class WeightedBCELoss(nn.Module):
    def __init__(self, weights, smooth = 1e-6):
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        loss = self.weights[1] * (targets * torch.log(inputs + self.smooth)) + \
               self.weights[0] * ((1 - targets) * torch.log(1 - inputs + self.smooth))
        return torch.neg(torch.mean(loss))

#-----------------------------------------------------------------------#
#                             TverskyLoss                               #
#                 Calculate and return the Tversky Loss                 #
#-----------------------------------------------------------------------#
# Reference:                                                            #
# Salehi al. “Tversky Loss Function for Image Segmentation              #
# Using 3D Fully Convolutional Deep Networks.”                          #
# ArXiv abs/1706.05721 (2017)                                           #
#-----------------------------------------------------------------------#
# alpha and beta control the magnitude of penalties on false            #
# positives and false negatives, respectively.                          #
# alpha = beta = 0.5 is equilvalent to Dice coefficient                 #
# alpha = beta = 1 is equilvalent to Tanimoto index/Jaccard coefficient #
# alpha + beta = 1 leads to the set of F_beta scores                    #
# smooth: a very small number added to the denomiators to               #
#         prevent the division by zero                                  #
# tp: number of true positives                                          #
# fp: number of false positives                                         #
# fn: number of false negatives                                         #
#-----------------------------------------------------------------------#
class TverskyLoss(nn.Module):
    #returns the Tversky loss per batch
    def __init__(self, smooth = 1e-10, alpha = 0.5, beta = 0.5):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        # Flatten both prediction and GT tensors
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true)
        # calculate the number of true positives, false positives and false negatives
        tp = (y_pred_flat * y_true_flat).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        # calculate the Tversky index
        tversky = tp/(tp + self.alpha * fn + self.beta * fp + self.smooth)
        # return the loss
        return 1 - tversky

