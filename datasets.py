import os
import SimpleITK as sitk
import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from model.RegistrationNetworks import euler_angles_to_matrix
from glob import glob
from utils import set_seed

class OASIS(torch.utils.data.Dataset):
    def __init__(self, 
                 data_path = 'data/imagesTr/OASIS_*',
                 mask_path = 'data/masksTr/OASIS_*',
                 max_trans=0.25,
                 max_angle=30,
                 device='cuda',
                 rotateonly=False):
        self.rotateonly=rotateonly
        self.device = device
        self.data_paths, self.mask_paths = self.get_paths(data_path, mask_path)
        self.max_trans, self.max_angle = max_trans, max_angle    
        self.get_T()
        self.inshape, self.voxel_spacing = self.get_image_header(self.data_paths[0])
        self.adjust_shape(32, self.data_paths)
        
        

    def augmentation(self, idx):
        fixed_img_np = self.read_image_np(self.data_paths[idx])
        fixed_img = torch.from_numpy(fixed_img_np).unsqueeze(0)
        moving_img = self.transform_rigid(self.T_real[idx],fixed_img.unsqueeze(0)) 
        fixed_mask_np = self.read_image_np(self.mask_paths[idx])
        fixed_mask = torch.from_numpy(fixed_mask_np).unsqueeze(0)
        moving_mask = self.transform_rigid(self.T_real[idx],fixed_mask.unsqueeze(0))
        moving_mask = torch.where(moving_mask < 0.5, torch.zeros_like(moving_mask), torch.ones_like(moving_mask))
        return moving_img, fixed_img, moving_mask, fixed_mask

 

    def get_paths(self, data_path, mask_path):
        data_paths = glob(data_path)
        if len(data_paths)==0:
            raise Exception("Data not found. Check image/mask path")
        data_paths.sort()
        mask_paths = glob(mask_path)
        mask_paths.sort()
        #return data_paths
        return data_paths, mask_paths
    
    def get_T(self):
        self.T_real = []
        self.T_inv = []
        for i in range(len(self.data_paths)):
            set_seed(i)
            rand_trans = np.random.uniform(low=-self.max_trans, high=self.max_trans, size=(3,)).astype('float32')
            # print(rand_trans.shape)
            rand_angles = np.random.uniform(low=-self.max_angle, high=self.max_angle, size=(3,)).astype('float32')
            # print(rand_trans)
            if self.rotateonly!=False:
                rand_angles = np.array([0,0,self.rotateonly])
                rand_trans = np.array([0,0,0]).astype('float32')
            translation = torch.from_numpy(rand_trans)

            euler_angles = np.pi * torch.from_numpy(rand_angles) / 180.

            rot_mat = euler_angles_to_matrix(euler_angles=euler_angles, convention="XYZ")
            # print(translation,'\n',rot_mat)
            T = torch.cat((rot_mat, translation.view(3, 1)), axis=1)
            # print(T)
            T = T.view(-1, 3, 4)
            T4x4 = torch.cat((T.squeeze(), torch.Tensor([0,0,0,1]).unsqueeze(0)),0)
            Tinv=torch.inverse(T4x4)   
            Tinv=Tinv[:-1]
            self.T_real.append(T)
            self.T_inv.append(Tinv)
    def transform_rigid(self, T, input_tensor):
        grid = F.affine_grid(T, input_tensor.size(),align_corners=False) #N*3*4 & N*C*D*H*W = 1,1,192,224,160
        input_aug_tensor = F.grid_sample(input_tensor, grid,align_corners=False).squeeze(0)   
        return input_aug_tensor               
        


                
    def get_image_header(self,path):
        image = sitk.ReadImage(path)
        dim = np.array(image.GetSize())
        voxel_sp = np.array(image.GetSpacing())
        return dim[::-1], voxel_sp[::-1]
           
    def adjust_shape(self, multiple_of, data_paths):
        old_shape, _ = self.get_image_header(data_paths[0])
        new_shape = tuple([int(np.ceil(shp / multiple_of) * multiple_of) for shp in old_shape])
        self.inshape = new_shape
        self.offsets = [shp - old_shp for (shp, old_shp) in zip(new_shape, old_shape)]
    
    def read_image_sitk(self, path):
        if os.path.exists(path):
            image = sitk.ReadImage(path)
        else:
            print('Image does not exist')
        return image
    
    def read_image_np(self, path):
        if os.path.exists(path):
            image = sitk.ReadImage(path)
            image_np = sitk.GetArrayFromImage(image).astype('float32')
        else:
            print('Image does not exist')
        return image_np
      
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):  
        # self.augmentation(self.data_paths,self.max_trans,self.max_trans)        
        moving_img, fixed_img, moving_mask, fixed_mask = self.augmentation(idx)
        # return self.moving_t[idx], self.fixed_t[idx], self.Tinv[idx], idx
        return moving_img, fixed_img, moving_mask, fixed_mask, self.T_inv[idx], self.T_real[idx]
