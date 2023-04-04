""" THIS IS A SCRIPT TO TRAIN A SWIN TRANSFORMER FOR RIGID REGISTRATION """
from datasets import OASIS
from model.RegistrationNetworks import RegTransformer
from model.configurations import get_VitBase_config
from train_val_test import train_epoch, validate_epoch, test_model, plot_test, test_initial, rotateonly
from torchinfo import summary
from utils import getdiff
from tqdm import tqdm
import sys
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch
import time
import numpy as np

if __name__ == '__main__':
        
    def train_model(
            learning_rate = 1e-9, # Tune this hyperparameter
            batch_size = 1,
            epochs = 150,
            device = 'cuda',
            data_path = 'data/imagesTr/OASIS_*',
            mask_path = 'data/masksTr/OASIS_*',
            max_trans=0.25,
            max_angle=30,
            lossfn='ncc',
            ):
            """Initiates model and dataset
            
            Loads the model architecture and data preprocessing. Generate transformation matrices for each data set
            and splits data set in three sets, for training, validation and testing. 
            
            Args:
                learning_rate (float): step size of gradient
                batch_size (int): nr of images in batch
                epochs (int): nr of times model iterates through whole data set
                device (str): computation hardware
                lossfn (str): Type of loss function. 'ncc','mse_s','mse_u'
            
            Returns:
                None
            
            """
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True              
            dataset = OASIS(data_path, mask_path, max_trans, max_angle)
            train_set, val_set, test_set = data.random_split(dataset,[0.7,0.1,0.2], generator=torch.Generator().manual_seed(42))
            print('Train: ', len(train_set),'\nVal set: ', len(val_set), '\nTest: ', len(test_set))    
        
            config = get_VitBase_config(img_size=tuple(dataset.inshape))
            model = RegTransformer(config)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=0)
            
            """ TRAINING """
            print(f'\n----- Training with {lossfn} -----')
            train_NCC_list = list()
            train_MSEt_list = list()
            train_dsc_list = list()
            train_mse_list = list()
            train_hd95_list = list()
            
            val_NCC_list = list()
            val_MSEt_list = list()
            val_dsc_list = list()
            val_mse_list = list()
            val_hd95_list = list()
            
            epoch = 1
            start = time.time()
            while epoch <= epochs:
                print(f'\n[epoch {epoch} / {epochs}]')
                train_ncc, train_MSEt , train_dice, mse_train = train_epoch(model, train_loader, dataset, optimizer, device, lossfn)
                train_NCC_list.append(train_ncc)
                train_MSEt_list.append(train_MSEt)
                train_dsc_list.append(train_dice)
                train_mse_list.append(mse_train)
                #train_hd95_list.append(hd95_train)
                
                val_ncc, val_MSEt, val_dice, mse_val = validate_epoch(model, val_loader, dataset, device, lossfn)
                val_NCC_list.append(val_ncc)
                val_MSEt_list.append(val_MSEt)
                val_dsc_list.append(val_dice)
                val_mse_list.append(mse_val)
                #val_hd95_list.append(hd95_val)
                
                epoch += 1
            end = time.time()
            traintime = round(end - start)
            print('Total time training: ', traintime, ' seconds')
            output = [train_NCC_list,train_MSEt_list,train_dsc_list,train_mse_list,train_hd95_list,val_NCC_list,val_MSEt_list,val_dsc_list,val_mse_list,val_hd95_list]
            return(output, model, config, train_set, train_loader, val_set, val_loader, test_set, test_loader,dataset)
            
#%% Rotate experiment

    def test_rotate(device, 
                    path_weights='save/mse_unsupervised/epochs150_lr1e-9/weights.pth',
                    learning_rate=1e-9,
                    batch_size = 1):
        
        """Test model for rotation in angleslist
        
        Computes the predicted angle of rotation along 3rd axis.
        
        Args:
            model: pytorch model
            test_loader: data loader
            angleslist: list of angles to predict
            device (str): computation hardware
            path_weights (str): filepath of weights
        
        Returns:
            predlist (list): list of angles predicted for dataset of a specific angle in angleslist
            
        
        """
        predlist =[]
        angleslist=range(0,5,180)
        
        for idx,angle in enumerate(tqdm(angleslist, file=sys.stdout)):
            dataset = OASIS(rotateonly=angle)
            train_set, val_set, test_set = data.random_split(dataset,[0.7,0.1,0.2], generator=torch.Generator().manual_seed(42))
            config = get_VitBase_config(img_size=tuple(dataset.inshape))
            model = RegTransformer(config)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.load_state_dict(torch.load(path_weights))
        
            # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
            # val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=0)
            pred_angles, pred_trans = rotateonly(model, test_loader, angleslist, device)
            predlist.append(pred_trans)
        return predlist
    
#%% Save all to file

    def save_all(model, 
                 output, 
                 learning_rate, 
                 epochs,
                 savepath):
        import json
        d = {
             0: output[0], #train_NCC_list,
             1: output[1], #train_MSEt_list,
             2: output[2], #train_dsc_list,
             3: output[3], #train_mse_list, 
             4: output[4], #train_hd95_list, 
             5: output[5], #val_NCC_list, 
             6: output[6], #val_MSEt_list, 
             7: output[7], #val_dsc_list, 
             8: output[8], #val_mse_list, 
             9: output[9], #val_hd95_list, 
            10: learning_rate,
            11: epochs,
            #"k" : train_hd95_list,
            #"l" : val_hd95_list}
            }
        json.dump(d,open(f'{savepath}/variables/json',"w"))
        torch.save(model.state_dict(), f'{savepath}/weights.pth')

#%%
def plot_rotate():
    import json
    import scienceplots
    import matplotlib as mpl
    from matplotlib.pyplot import figure
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    import numpy as np
    plt.style.use(['science','ieee'])
    plt.style.use(['science','no-latex'])
    d = json.load(open("save/translation_ncc.json","r"))
    ncc_all = d['0']
    d = json.load(open("save/translation_mseu.json","r"))
    mseu_all = d['0']
    d = json.load(open("save/translation_mses.json","r"))
    mses_all = d['0']
    
    
    fig, ax = plt.subplots()
    
    figure(figsize=(4, 3), dpi=300)
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    
    
    
    # plt.ylim(-15, 40)
    # plt.xlim(0, 180)
    
    p1,=plt.plot(np.linspace(0,1,21),np.mean(mseu_all,axis=1)*-1, color='black', label='MSE unsupervised')
    p2,=plt.plot(np.linspace(0,1,21),np.mean(mses_all,axis=1)*-1, color='blue', label='MSE supervised')
    p3,=plt.plot(np.linspace(0,1,21),np.mean(ncc_all,axis=1)*-1, color='red', label='LNCC')
    p4,=plt.plot(np.linspace(0,1,21),np.linspace(0,1,21),color='grey',linestyle='--')
    # p2,=plt.plot(np.mean(mseu_all,axis=1), color='blue')
    # p2,=plt.plot(mseu8, color='blue')
    #plt.plot(mseu7, color='red')
    
    # p4,=plt.plot(mse9,linestyle='--', color='black')
    # p5,=plt.plot(mse8,linestyle='--', color='blue')
    # p6,=plt.plot(mse7,linestyle='--', color='red')
    
    plt.ylabel('Predicted translation (-)',fontweight='bold')
    plt.xlabel('Target translation (-)',fontweight='bold')
    plt.legend(['MSE unsupervised','MSE supervised','LNCC'])
    # plt.legend([(p1, p4), (p2, p5)], ['1e-9 unsup/sup', '1e-8 unsup/sup'], numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3)
    plt.show()


#%%
def plot_training(output):
    train_NCC_list,train_MSEt_list,train_dsc_list,train_mse_list,train_hd95_list,val_NCC_list,val_MSEt_list,val_dsc_list,val_mse_list,val_hd95_list=output[0],output[1],output[2],output[3],output[4],output[5],output[6],output[7],output[8],output[9]

    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(10,18), gridspec_kw={'hspace':0.5})
    x_tll = range(0,len(train_NCC_list))
    ax[0].plot(x_tll,train_NCC_list, label='Loss training')
    ax[0].plot(x_tll, val_NCC_list, label='Loss validation')
    ax[0].set_xticks(range(0,len(x_tll)+1,10))
    ax[0].set_title('Negative NCC Loss ',fontweight='bold')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('loss (NCC)')
    ax[0].legend()

    x_tel = range(0,len(train_MSEt_list))
    ax[1].plot(x_tel, train_MSEt_list, label='MSE training')
    ax[1].plot(x_tel, val_MSEt_list, label='MSE validation')
    ax[1].set_xticks(range(0,len(x_tel)+1,10))
    ax[1].set_title('MSE Transformation Matrix (Ground truth - predicted)²', fontweight='bold')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MSE')
    ax[1].legend()
    
    x_tdl = range(0,len(train_dsc_list))
    ax[2].plot(x_tdl, train_dsc_list, label='DSC training')
    ax[2].plot(x_tdl, val_dsc_list, label='DSC validation')
    ax[2].set_xticks(range(0,len(x_tdl)+1,10))
    ax[2].set_title('Dice Similarity Coefficient (mask_warped, mask_fixed)', fontweight='bold')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('DSC')
    ax[2].legend()

    x_tml = range(0,len(train_mse_list))
    ax[3].plot(x_tml, train_mse_list, label='MSE training')
    ax[3].plot(x_tml, val_mse_list, label='MSE validation')
    ax[3].set_xticks(range(0,len(x_tml)+1,10))
    ax[3].set_title('MSE (img_warped - img_fixed)²', fontweight='bold')
    ax[3].set_xlabel('Epochs')
    ax[3].set_ylabel('MSE')
    ax[3].legend()
    
    # x_thdl = range(0,len(train_hd95_list))
    # ax[4].plot(x_thdl, train_hd95_list, label='HD95 training')
    # ax[4].plot(x_thdl, val_hd95_list, label='HD95 validation')
    # ax[4].set_xticks(range(0,len(x_thdl)+1,5))
    # ax[4].set_title('Hausdorff distance 95% percentile (mask_warped, mask_fixed)', fontweight='bold')
    # ax[4].set_xlabel('Epochs')
    # ax[4].set_ylabel('Hausdorff distance')
    # ax[4].legend()



    # fig.suptitle(f'Epochs: 150 | LR: {learning_rate} | )
    # mpl.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['ps.fonttype'] = 42
    # mpl.rcParams['font.family'] = 'Arial'
    fig.show()
#%% EXECUTE
"""------------- EXECUTE PROGRAM -------------"""
device='cuda'
output, model, config, train_set, train_loader, val_set, val_loader, test_set, test_loader, dataset =train_model(lossfn='ncc')    
plot_training(output)
#%% SAVE
save_all('save/mse_unsupervised/epochs150_lr1e-9/')

#%% TEST MODEL
test_ncc_batch, total_dsc_batch, total_mse_img_batch , test_T_error_batch, total_hd95_batch, total_ssim_batch = test_model(model, test_loader, dataset, device)
def calcavg(inputlist):
    a = torch.tensor(inputlist)
    return torch.mean(a), torch.std(a)



#%% SUMMARY MODEL
summary(model=RegTransformer(config),
    input_size=((1,1,192, 224, 160),(1,1,192, 224, 160)), # (batch_size, color_channels, depth, height, width)
    #col_names=["input_size"], # uncomment for smaller output
    col_names=["input_size", "output_size", "num_params"],
    # col_width=20,
    # row_settings=["var_names"])
    )
#%%



# #%% LOAD
# import json
# d = json.load(open("save/mse_unsupervised/epochs150_lr1e-7/variables.json","r"))
# train_NCC_list = d['0']
# train_MSEt_list = d['1']
# train_dsc_list = d['2']
# train_mse_list = d['3'] 
# train_hd95_list = d['4'] 
# val_NCC_list = d['5'] 
# val_MSEt_list = d['6'] 
# val_dsc_list = d['7']
# val_mse_list = d['8']
# val_hd95_list = d['9']
# learning_rate = d['10']
# epochs = d['11']
# traintime = d['12']
# splitlen = d['13']
# #train_hd95_list = d["k"]
# #val_hd95_list = d["l"]
# #%%
# import scienceplots
# plt.style.use(['science','ieee'])
# plt.style.use(['science','no-latex'])
# import matplotlib as mpl
# from matplotlib.pyplot import figure
# import matplotlib.pyplot as plt
# from matplotlib.legend_handler import HandlerTuple
# d = json.load(open("save/mse_unsupervised/epochs150_lr1e-9/variables.json","r"))
# mseu9 = d['3']
# d = json.load(open("save/mse_unsupervised/epochs150_lr1e-8/variables.json","r"))
# mseu8 = d['3']
# # d = json.load(open("save/mse_unsupervised/epochs150_lr1e-7/variables.json","r"))
# # mseu7 = d['3']

# d = json.load(open("save/mse_supervised/epochs150_lr1e-9/variables.json","r"))
# mse9 = d['1']
# d = json.load(open("save/mse_supervised/epochs150_lr1e-8/variables.json","r"))
# mse8 = d['1']
# d = json.load(open("save/mse_supervised/epochs150_lr1e-7/variables.json","r"))
# mse7 = d['1']

# fig, ax = plt.subplots()

# figure(figsize=(4, 3), dpi=300)
# plt.rcParams["axes.edgecolor"] = "black"
# plt.rcParams["axes.linewidth"] = 2.50
# p1,=plt.plot(mseu9, color='black')
# p2,=plt.plot(mseu8, color='blue')
# #plt.plot(mseu7, color='red')

# p4,=plt.plot(mse9,linestyle='--', color='black')
# p5,=plt.plot(mse8,linestyle='--', color='blue')
# p6,=plt.plot(mse7,linestyle='--', color='red')

# plt.ylabel('MSE',fontweight='bold')
# plt.xlabel('Epochs',fontweight='bold')
# plt.legend([(p1, p4), (p2, p5)], ['1e-9 unsup/sup', '1e-8 unsup/sup'], numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3)
# plt.show()

# #%%
# import scienceplots
# plt.style.use(['science','ieee'])
# plt.style.use(['science','no-latex'])
# import matplotlib as mpl
# from matplotlib.pyplot import figure
# import matplotlib.pyplot as plt
# from matplotlib.legend_handler import HandlerTuple


# d = json.load(open("save/ncc/epochs150_lr1e-9/variables.json","r"))
# ncc9 = d['0']
# d = json.load(open("save/ncc/epochs150_lr1e-8/variables.json","r"))
# ncc8 = d['0']
# # d = json.load(open("save/ncc/epochs150_lr1e-7/variables.json","r"))
# # ncc7 = d['0']

# fig, ax = plt.subplots()

# figure(figsize=(4, 3), dpi=300)
# plt.rcParams["axes.edgecolor"] = "black"
# plt.rcParams["axes.linewidth"] = 2.50
# p1,=plt.plot(ncc9, color='black')
# p2,=plt.plot(ncc8, color='blue')
# #plt.plot(ncc7, color='red')

# plt.ylabel('-LNCC',fontweight='bold')
# plt.xlabel('Epochs',fontweight='bold')
# plt.legend(['1e-9','1e-8'], numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3)
# plt.show()
#%%
#%%
# import scienceplots
# plt.style.use(['science','ieee'])
# plt.style.use(['science','no-latex'])
# import matplotlib as mpl
# from matplotlib.pyplot import figure
# import matplotlib.pyplot as plt
# from matplotlib.legend_handler import HandlerTuple


# d = json.load(open("save/mse_supervised/epochs150_lr1e-9/variables.json","r"))
# mset_train = d['1']
# d = json.load(open("save/mse_supervised/epochs150_lr1e-9/variables.json","r"))
# mset_val = d['6']
# # d = json.load(open("save/mse_supervised/epochs150_lr1e-7/variables.json","r"))
# # ncc7 = d['0']

# fig, ax = plt.subplots()

# figure(figsize=(4, 3), dpi=300)
# plt.rcParams["axes.edgecolor"] = "black"
# plt.rcParams["axes.linewidth"] = 2.50
# p1,=plt.plot(mset_train, color='black')
# p2,=plt.plot(mset_val, color='orange')
# #plt.plot(ncc7, color='red')

# plt.ylabel('MSE',fontweight='bold')
# plt.xlabel('Epochs',fontweight='bold')
# plt.legend(['Training','Validation'], numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3)
# plt.show()
#%%
# torch.save(model.state_dict(), f'save/mse_unsupervised/{modelname}/weights.pth')


#%%


#%%

