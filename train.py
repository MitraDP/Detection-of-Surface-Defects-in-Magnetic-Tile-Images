#-----------------------------------------------------------------------#
#                          Library imports                              #
#-----------------------------------------------------------------------#
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------#
#                                train_2D                               #
#              Train 2D UNet for some number of epochs                  #
#-----------------------------------------------------------------------#
def train_2D(n_epochs, loaders, model, optimizer, criterion, train_on_gpu, path):
    #keep track of train and validation losses
    loss_epoch=[]
    # initialize tracker for minimum validation loss
    valid_loss_min = np.inf
    show_every = 10
    # Epoch training loop
    for epoch in tqdm( range(1, n_epochs+1), total = n_epochs+1):
        print(f'=== Epoch #{epoch} ===')
        # Initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        valid_cnt = 0
        
        ###################
        # train the model #
        ###################
        model.train()
        print('=== Training ===')
        # Batch training loop
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # Move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            if batch_idx % show_every == 0:
                print(f'{batch_idx + 1} / {len(loaders["train"])}...')
            # Clear the gradients of all optimized variable
            optimizer.zero_grad() 
            # Forward pass (inference) to get the output
            output = model(data) 
            # Calculate the batch loss
            loss = criterion(output, target) 
            # Backpropagation
            loss.backward() 
            # Update weights
            optimizer.step() 
            # Update training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss)) 
                         
        ######################    
        # validate the model #
        ######################
        print('=== Validation ===')
        model.eval()
        with torch.no_grad():
            # Batch training loop
            for batch_idx, (data, target) in enumerate(loaders['val']):
                if batch_idx % show_every == 0:
                    print(f'{batch_idx + 1} / {len(loaders["val"])}...')
                # Move to GPU 
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # Forward pass (inference)
                output = model(data)
                # Calculate the batch loss
                loss = criterion (output, target)
                # Update validation loss
                valid_loss +=  ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        # print training/validation losses
        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
                
        if valid_loss < valid_loss_min:
            print('Validation loss decreased.  Saving model ...')            
            torch.save(model.state_dict(), path)
            valid_loss_min = valid_loss

        loss_epoch.append((epoch, train_loss.cpu().detach().numpy(), valid_loss.cpu().detach().numpy()))

    # Save the loss_epoch history
    df=pd.DataFrame.from_records(loss_epoch, columns=['epoch', 'Training Loss', 'Validation Loss'])
    df.to_csv('loss_epoch.csv', index=False)  

    # Return the trained model
    return model

