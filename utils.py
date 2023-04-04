import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, device, win=None):
        self.win = win
        self.device = device

    def forward(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        # cc = torch.mean(cross * cross) / torch.mean((I_var * J_var + 1e-5))
        return -torch.mean(cc)

class DSC:
    """Calculates the Dice Similarity Coefficient
    
    Calculates the Dice Similarity Coefficient of two binary masks
    defined as 2*intersection/(area1+area2)
    
    Args:
        y_true: binary mask of ground truth
        y_pred: binary mask of predicted transformation
    
    Returns:
        para1:
    
    """
    def __init__(self,device):
        self.device = device
        
    def forward(self, y_true, y_pred):
        y_true, y_pred = y_true.to(self.device), y_pred.to(self.device)
        intersection = y_true*y_pred
        intersection = torch.sum(intersection)
        sum_area = torch.sum(y_true)+torch.sum(y_pred)
        dsc = (2*intersection)/sum_area
        return dsc.item()


    
    
def set_seed(seed_value, pytorch=True):
    """
    Set seed for deterministic behavior

    Parameters
    ----------
    seed_value : int
        Seed value.
    pytorch : bool
        Whether the torch seed should also be set. The default is True.

    Returns
    -------
    None.
    """
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    if pytorch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True



#%%
def getdiff(model,dataset,test_set,i,slicenr,device):
    """Plot the difference of two images
    
    Imports the model weights and plots the differences in warped images.
    
    Args:
        model: pytorch model
        dataset (class): dataset
        test_set: torch.utils.data.dataset.Subset
        i (int): index of data
        slicenr (int): slice of data
        device (str): computation hardware
        
    
    Returns:
        None
    
    """
    img_moving, img_fixed = test_set[i][0].to(device), test_set[i][1].to(device)
    img_moving, img_fixed = img_moving.unsqueeze(0), img_fixed.unsqueeze(0)
    mask_moving = test_set[i][2]
    mask_fixed = test_set[i][3]
    
    model.load_state_dict(torch.load('save/ncc/epochs150_lr1e-9/weights.pth'))
    img_warped, T = model(img_moving, img_fixed)
    mask_warped = dataset.transform_rigid(T,mask_moving.unsqueeze(0).to(device))
    mask_warped = torch.where(mask_warped < 0.5, torch.zeros_like(mask_warped), torch.ones_like(mask_warped))    
    mask_warped = mask_warped+1
    yep = mask_fixed.to(device)+mask_warped.to(device)


    model.load_state_dict(torch.load('save/mse_supervised/epochs150_lr1e-9/weights.pth'))
    img_warped2, T2 = model(img_moving, img_fixed)
    mask_warped2 = dataset.transform_rigid(T2,mask_moving.unsqueeze(0).to(device))
    mask_warped2 = torch.where(mask_warped2 < 0.5, torch.zeros_like(mask_warped2), torch.ones_like(mask_warped2))    
    mask_warped2 = mask_warped2+1
    yep2 = mask_fixed.to(device)+mask_warped2.to(device)
    
    model.load_state_dict(torch.load('save/mse_unsupervised/epochs150_lr1e-9/weights.pth'))
    img_warped3, T3 = model(img_moving, img_fixed)
    mask_warped3 = dataset.transform_rigid(T3,mask_moving.unsqueeze(0).to(device))
    mask_warped3 = torch.where(mask_warped3 < 0.5, torch.zeros_like(mask_warped3), torch.ones_like(mask_warped3))    
    mask_warped3 = mask_warped3+1
    yep3 = mask_fixed.to(device)+mask_warped3.to(device)
    
    
    


    plt.figure()
    plt.figure().set_figwidth(10)
    fig1, axs1 = plt.subplots(1, 2,figsize=(10, 10))
    plt.subplots_adjust(wspace=0)
    axs1[0].imshow(img_moving.squeeze().cpu().numpy()[:, :, 60], cmap='gray')
    axs1[0].set_title('Moving image')
    axs1[1].imshow(img_fixed.squeeze().cpu().numpy()[:, :, 60], cmap='gray')
    axs1[1].set_title('Fixed image')
    for ax in axs1:
        ax.set_xticks([])
        ax.set_yticks([])    
    plt.show()
    
    
    plt.figure()
    plt.figure().set_figwidth(25)
    fig, axs = plt.subplots(1, 6,figsize=(25, 25))
    plt.subplots_adjust(wspace=0)
    axs[0].imshow(img_warped2.squeeze().detach().cpu().numpy()[:, 60, :], cmap='gray')
    axs[0].set_title('Warped MSE-s')
    axs[1].imshow(yep2.squeeze().cpu().numpy()[:, 60, :])
    axs[1].set_title('Masks NCC')
    axs[2].imshow(torch.square(img_warped2-img_fixed).squeeze().detach().cpu().numpy()[:, :, 60], cmap='gray')
    axs[2].set_title('D_images')  
    axs[3].imshow(torch.square(img_warped2-img_warped).squeeze().detach().cpu().numpy()[:, :, 60], cmap='gray')
    axs[3].set_title('D_NCC')  
    axs[4].imshow(torch.square(img_warped2-img_warped3).squeeze().detach().cpu().numpy()[:, :, 60], cmap='gray')
    axs[4].set_title('D_Unsup')  
    axs[5].imshow(torch.square(img_warped2-img_warped2).squeeze().detach().cpu().numpy()[:, :, 60], cmap='gray')
    axs[5].set_title('D_sup')  
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])    
    plt.show()
    
   
