"""
Training/validation functions
"""
import sys
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils import NCC, DSC
import monai
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.transform import Rotation as R
# import time
import numpy as np


def train_epoch(model, data_loader, dataset, optimizer, device, lossfn):
    """
    Train for one epoch
    """
    total_ncc_batch = 0
    total_mse_batch = 0
    total_dsc_batch = 0
    total_mse_img_batch = 0
    total_hd95_batch = 0
    
    # Initialize loss functions
    similarity_loss = NCC(device)
    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    
    model.train()
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(data_loader, file=sys.stdout)):
        # Take the img_moving and fixed images to the GPU
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device), img_fixed.to(device), mask_moving.to(device), mask_fixed.to(device)
        optimizer.zero_grad(set_to_none=True)################

        img_warped, T,_,_ = model(img_moving, img_fixed)
        mask_warped = dataset.transform_rigid(T,mask_moving)
        
        loss = similarity_loss.forward(img_fixed, img_warped)
        T_error = mse_loss(T, T_ground_truth.to(device))
        MSE_img = mse_loss(img_warped, img_fixed)
        dice = dsc.forward(mask_warped, mask_fixed)
        # hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed, percentile=95)
        # print(loss)
        total_ncc_batch += loss.item()
        total_mse_batch += T_error.item()
        total_dsc_batch += dice
        total_mse_img_batch += MSE_img.item()
        # total_hd95_batch += hd95
        if lossfn=='ncc':
            loss.backward()
        if lossfn=='mse_u':
            MSE_img.backwards()
        if lossfn=='mse_s':
            T_error.backwards()
            
        optimizer.step()
        del loss, T_error, img_moving, img_fixed, img_warped, T_ground_truth, dice, mask_moving, mask_fixed, mask_warped, MSE_img


    train_ncc_loss = total_ncc_batch / len(data_loader)   
    train_T_error = total_mse_batch / len(data_loader)
    train_dsc = total_dsc_batch / len(data_loader)
    mse_img = total_mse_img_batch / len(data_loader)
    # hd95_train = total_hd95_batch / len(data_loader)
    
    if lossfn=='ncc':
        lossprint = train_ncc_loss
    if lossfn=='mse_u':
        lossprint = mse_img
    if lossfn=='mse_s':
        lossprint = train_T_error

           
    """ Print loss """
    print("Train Loss = %.5f" % lossprint)
    return train_ncc_loss, train_T_error, train_dsc, mse_img

def validate_epoch(model, val_loader, dataset, device, lossfn):

    
    val_ncc_batch = 0
    val_T_error_batch = 0
    total_dsc_batch = 0
    total_mse_img_batch = 0
    total_hd95_batch = 0

    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    similarity_loss = NCC(device)
    
    #Disable training
    model.train(mode=False)
    torch.no_grad()
    
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(val_loader, file=sys.stdout)):
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device), img_fixed.to(device), mask_moving.to(device), mask_fixed.to(device)

        img_warped, T,_,_ = model(img_moving, img_fixed)
        mask_warped = dataset.transform_rigid(T,mask_moving)

        val_loss = similarity_loss.forward(img_fixed, img_warped)  
        dice = dsc.forward(mask_warped, mask_fixed)
        MSE_img = mse_loss(img_warped, img_fixed).item()
        T_error = mse_loss(T, T_ground_truth.to(device))
        # hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed, percentile=95)

        val_ncc_batch += val_loss.item()
        val_T_error_batch += T_error.item()
        total_dsc_batch += dice
        total_mse_img_batch += MSE_img
        # total_hd95_batch += hd95

        del val_loss, img_moving, img_fixed, img_warped, T_error, T_ground_truth, dice, mask_moving, mask_fixed, mask_warped, MSE_img

    val_ncc_loss = val_ncc_batch/len(val_loader)
    val_T_error = val_T_error_batch /len(val_loader)
    val_dsc = total_dsc_batch / len(val_loader)
    mse_img = total_mse_img_batch / len(val_loader)
    # hd95_val = total_hd95_batch / len(val_loader)
    if lossfn=='ncc':
        lossprint = val_ncc_loss
    if lossfn=='mse_u':
        lossprint = mse_img
    if lossfn=='mse_s':
        lossprint = val_T_error
    print("Validation Loss = %.5f" % lossprint)
    return val_ncc_loss, val_T_error, val_dsc, mse_img


def test_model(model, test_loader, dataset, device):
    test_ncc_batch = []
    test_T_error_batch = []
    total_dsc_batch = []
    total_mse_img_batch = []
    total_hd95_batch = []
    total_ssim_batch =[]
    # initial_ncc = []
    # initial_dsc = []
    # initial_mse_img = []
    # initial_mse_T = []
    # initial_hd95 = []
    
    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    similarity_loss = NCC(device)
    
    #Disable training
    model.train(mode=False)
    torch.no_grad()
    
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(test_loader, file=sys.stdout)):
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device), img_fixed.to(device), mask_moving.to(device), mask_fixed.to(device)

        
        img_warped, T = model(img_moving, img_fixed)
        mask_warped = dataset.transform_rigid(T,mask_moving)
        mask_warped = torch.where(mask_warped < 0.5, torch.zeros_like(mask_warped), torch.ones_like(mask_warped))
        
        # initial_ncc += [similarity_loss.forward(img_fixed, img_moving)]
        # initial_dsc += [dsc.forward(mask_moving, mask_fixed)]
        # initial_mse_img += [mse_loss(img_moving, img_fixed).item()]

        # initial_mse_T += [mse_loss(T.squeeze(), T_augment.to(device)).item()]
        # initial_hd95 += [monai.metrics.compute_hausdorff_distance(mask_moving, mask_fixed, percentile=95)]
        
        # initialavg = [initial_ncc, initial_dsc, initial_mse_img, initial_mse_T, initial_hd95]

        test_loss = similarity_loss.forward(img_fixed, img_warped)  
        dice = dsc.forward(mask_warped, mask_fixed)
        MSE_img = mse_loss(img_warped, img_fixed).item()
        T_error = mse_loss(T, T_ground_truth.to(device)).item()      
        hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed, percentile=95)
        ssim_img = ssim(img_warped.squeeze().cpu().detach().numpy(),img_fixed.squeeze().cpu().detach().numpy(),win_size=9)
        
        test_ncc_batch.append(test_loss.item())
        test_T_error_batch.append(T_error)
        total_dsc_batch.append(dice)
        total_mse_img_batch.append(MSE_img)
        total_hd95_batch.append(hd95)
        total_ssim_batch.append(ssim_img)

        del test_loss, img_moving, img_fixed, img_warped, T_error, T_ground_truth, dice, mask_moving, mask_fixed, mask_warped, MSE_img, hd95, ssim_img

    # test_ncc_loss = test_ncc_batch/len(test_loader)
    # test_T_error = test_T_error_batch /len(test_loader)
    # test_dsc = total_dsc_batch / len(test_loader)
    # mse_img = total_mse_img_batch / len(test_loader)
    # hd95_test = total_hd95_batch / len(test_loader)
    return test_ncc_batch, total_dsc_batch, total_mse_img_batch , test_T_error_batch, total_hd95_batch, total_ssim_batch

def test_initial(model, test_loader, dataset, device):
    initial_ncc = []
    initial_dsc = []
    initial_mse_img = []
    #initial_mse_T = []
    initial_hd95 = []
    initial_ssim = []
    
    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    similarity_loss = NCC(device)
    
    #Disable training
    model.train(mode=False)
    torch.no_grad()
    
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(test_loader, file=sys.stdout)):
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device), img_fixed.to(device), mask_moving.to(device), mask_fixed.to(device)
 
        img_warped, T = model(img_moving, img_fixed)
        
        initial_ncc += [similarity_loss.forward(img_fixed, img_moving)]
        initial_dsc += [dsc.forward(mask_moving, mask_fixed)]
        initial_mse_img += [mse_loss(img_moving, img_fixed).item()]
        #initial_mse_T += [mse_loss(T.squeeze(), T_augment.to(device).squeeze()).item()]
        initial_hd95 += [monai.metrics.compute_hausdorff_distance(mask_moving, mask_fixed, percentile=95)]
        initial_ssim += [ssim(img_moving.squeeze().cpu().numpy(),img_fixed.squeeze().cpu().numpy(),win_size=9)]
        

        del img_moving, img_fixed, mask_moving, mask_fixed
    initialavg = [initial_ncc, initial_dsc, initial_mse_img, initial_hd95, initial_ssim]
    # test_ncc_loss = test_ncc_batch/len(test_loader)
    # test_T_error = test_T_error_batch /len(test_loader)
    # test_dsc = total_dsc_batch / len(test_loader)
    # mse_img = total_mse_img_batch / len(test_loader)
    # hd95_test = total_hd95_batch / len(test_loader)
    return initialavg

def plot_test(model, test_set, dataset, plotlist, slicenr, device, modelname):
    similarity_loss = NCC(device)
    # test_loss_batch = 0 
    mse_loss = torch.nn.MSELoss()
    dsc=DSC(device)
    model.train(mode=False)
    torch.no_grad()

    
    for idx,i in enumerate(tqdm(plotlist,file=sys.stdout)):
        fig, axs = plt.subplots(1, 3)
        plt.subplots_adjust(bottom=-0.19)

        img_moving, img_fixed = test_set[i][0].to(device), test_set[i][1].to(device)
        img_moving, img_fixed = img_moving.unsqueeze(0), img_fixed.unsqueeze(0)
        img_warped, T = model(img_moving, img_fixed)
        
        mse = mse_loss(img_fixed, img_warped) 
        initial_mse = mse_loss(img_fixed,img_moving)
        
        initial_ssim = ssim(img_moving.squeeze().cpu().numpy(),img_fixed.squeeze().cpu().numpy())
        ssim_img = ssim(img_warped.squeeze().cpu().detach().numpy(),img_fixed.squeeze().cpu().detach().numpy())
        ncc = similarity_loss.forward(img_fixed, img_warped)
        initial_ncc = similarity_loss.forward(img_fixed, img_moving)
        mask_moving = test_set[i][2]
        mask_fixed = test_set[i][3]
        mask_warped = dataset.transform_rigid(T,mask_moving.unsqueeze(0).to(device))
        mask_warped = torch.where(mask_warped < 0.5, torch.zeros_like(mask_warped), torch.ones_like(mask_warped))
        
        dice = dsc.forward(mask_warped, mask_fixed)
        dice_initial = dsc.forward(mask_moving, mask_fixed)
        # print(mask_warped.shape, mask_moving.shape, mask_fixed.shape)
        hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed.unsqueeze(0), percentile=95)
        
        hd95_initial = monai.metrics.compute_hausdorff_distance(mask_moving.unsqueeze(0), mask_fixed.unsqueeze(0), percentile=95)
        img_moving , img_fixed = img_moving.detach(), img_fixed.detach()
        axs[0].imshow(img_moving.squeeze().cpu().numpy()[:, slicenr, :], cmap='gray')
        axs[0].set_title('Moving image')
        axs[1].imshow(img_fixed.squeeze().cpu().numpy()[:,slicenr, :], cmap='gray')
        axs[1].set_title('Fixed image')
        axs[2].imshow(img_warped.squeeze().detach().cpu().numpy()[:,slicenr , :], cmap='gray')
        axs[2].set_title('Warped image')
        img_warped = img_warped.detach()
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.suptitle(f'MSE: {round(initial_mse.item(),4)} | {round(mse.item(),4)} \nNCC: {round(initial_ncc.item(),4)} | {round(ncc.item(),4)} \nDSC: {round(dice_initial,4)} | {round(dice,4)} \nHD95: {round(hd95_initial.item(),4)} | {round(hd95.item(),4)}\nSSIM: {round(initial_ssim.item(),4)} | {round(ssim_img.item(),4)}\n')
        plt.savefig(f'save/mse_unsupervised/{modelname}/oasis_{i}.png')
        plt.close()
        del img_moving, img_fixed, img_warped, T, mse, ncc, dice, mask_moving, mask_fixed, mask_warped, ssim_img, initial_ssim
    fig.show()


def rotateonly(model, test_loader, angleslist, device):
    """Test model for rotation in angleslist
    
    Computes the predicted angle of rotation along 3rd axis.
    
    Args:
        model: pytorch model
        test_loader: data loader
        angleslist: list of angles to predict
        device (str): computation hardware
        
    
    Returns:
        pred_angles (list): list of angles predicted for dataset of a specific angle in angleslist
        
    
    """
    model.train(mode=False)
    pred_angles=[]
    # pred_trans=[]
    # ttestl=[]
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(test_loader):
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device), img_fixed.to(device), mask_moving.to(device), mask_fixed.to(device)        
        # tstart=time.time()
        img_warped, T, angles, trans = model(img_moving, img_fixed)
    
        pred_angles+=angles.detach().cpu()
        # pred_trans+=trans.detach().cpu()
        r = R.from_rotvec(angles.cpu().detach().numpy().squeeze())       
        pred_angle = r.as_rotvec(degrees=True)[2]
        pred_angles.append(pred_angle)
        # trans=trans.detach().cpu().numpy()
        # print(trans[0][2])
        # pred_trans.append(trans[0][2])
        # ttestl.append(ttest)
        del img_moving, img_fixed, mask_moving, mask_fixed, img_warped, T_ground_truth, T_augment, T, angles, trans,
    print(f'Pred angle: {np.mean(pred_angles)}')    
    # print(f'Average test time: {np.mean(ttestl)} s')    
    return pred_angles 

