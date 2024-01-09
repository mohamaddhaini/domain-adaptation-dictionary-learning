import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from torchvision import transforms, datasets
import os

from data import MNISTM
from models import Net,Model_Regression
import random

import sys
import model_dsprites
import transform_dsprites as tran
import argparse
import time
import numpy as np
torch.set_num_threads(1)
import matplotlib.pyplot as plt 
import random
from read_data import ImageList_r as ImageList



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_data(root_path, domain, batch_size, phase,shuffle):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    data = datasets.ImageFolder(root=os.path.join(root_path, domain), transform=transform_dict[phase])
    # data = datasets.ImageFolder(root=os.path.join(root_path, domain))
    # data.targets = torch.tensor(data.targets)
    # data.targets = F.one_hot(data.targets)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    return data_loader
data_folder = '/content/drive/MyDrive/Colab Notebooks/Wasserstein GANS/pytorch-domain-adaptation/data/office31'


def Regression_test(loader, model):
    MSE = [0, 0, 0]
    MAE = [0, 0, 0]
    number = 0
    list_umap=[]
    list_values=[]
    with torch.no_grad():
        for (imgs, labels) in loader['test']:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            labels = torch.cat((labels1, labels2), dim=1)
            labels = labels.float() / 39
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
            # a=model.feature_extractor(imgs).detach().cpu().numpy()
            # list_umap.append(a)
            # list_values.append(labels[:,0].detach().cpu().numpy())
    for j in range(3):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    # print("\tMSE : {0},{1}\n".format(MSE[0], MSE[1]))
    # print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    # print("\tMSEall : {0}\n".format(MSE[2]))
    # print("\tMAEall : {0}\n".format(MAE[2]))
    return MAE[2]
    
    
    
def Regression_test_dsprites(loader, model):
    MSE = [0, 0, 0, 0]
    MAE = [0, 0, 0, 0]
    number = 0
    with torch.no_grad():
        for (imgs, labels) in loader['test']:
            imgs = imgs.to(device)
            labels_source = labels.to(device)
            labels1 = labels_source[:, 0]
            labels3 = labels_source[:, 2]
            labels4 = labels_source[:, 3]
            labels1 = labels1.unsqueeze(1)
            labels3 = labels3.unsqueeze(1)
            labels4 = labels4.unsqueeze(1)
            labels_source = torch.cat((labels1, labels3, labels4), dim=1)
            labels = labels_source.float()
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred[:, 2], labels[:, 2])
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred[:, 2], labels[:, 2])
            MSE[3] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[3] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(4):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    # print("\tMSE : {0},{1},{2}\n".format(MSE[0],MSE[1],MSE[2]))
    # print("\tMAE : {0},{1},{2}\n".format(MAE[0], MAE[1], MAE[2]))
    # print("\tMSEall : {0}\n".format(MSE[3]))
    # print("\tMAEall : {0}\n".format(MAE[3]))
    return MAE[3]


def set_seed():
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3)
    random.seed(3)
    os.environ['PYTHONHASHSEED'] = str(3)


def main(args):
    set_seed()
    # dataset = MNISTM(train=False)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
    #                         drop_last=False, num_workers=1, pin_memory=True)
    dataloader = load_data(data_folder,'webcam', args.batch_size, phase='tar',shuffle=True)
    model = Model_Regression().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))
    model.eval()

    if (args.classification==1):
      total_accuracy = 0
      with torch.no_grad():
          for x, y_true in tqdm(dataloader, leave=False):
              x, y_true = x.to(device), y_true.to(device)
              y_pred = model(x)
              total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
      
      mean_accuracy = total_accuracy / len(dataloader)
      print(f'Accuracy on target data: {mean_accuracy:.4f}')
    else:
      os.chdir(r'/content/Data')
      data_transforms = {'train': tran.rr_train(resize_size=224),'val': tran.rr_train(resize_size=224),
          'test': tran.rr_eval(resize_size=224),}
      # set dataset
      batch_size = {"train": 36, "val": 36, "test": 4}
      c="/content/Domain-Adaptation-Regression/DAR-RSD/dSprites/color.txt"
      n="/content/Domain-Adaptation-Regression/DAR-RSD/dSprites/noisy.txt"
      s="/content/Domain-Adaptation-Regression/DAR-RSD/dSprites/scream.txt"

      c_t="/content/Domain-Adaptation-Regression/DAR-RSD/dSprites/color_test.txt"
      n_t="/content/Domain-Adaptation-Regression/DAR-RSD/dSprites/noisy_test.txt"
      s_t="/content/Domain-Adaptation-Regression/DAR-RSD/dSprites/scream_test.txt"

      dsets = {"train": ImageList(open(n).readlines(), transform=data_transforms["train"]),
              "val": ImageList(open(s).readlines(),transform=data_transforms["val"]),
              "test": ImageList(open(s_t).readlines(),transform=data_transforms["test"])}
      dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                                    shuffle=True, num_workers=0)
                      for x in ['train', 'val']}
      dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                        shuffle=False, num_workers=0)
      reg_result=Regression_test(dset_loaders, model)
      print(f'Results on target data: {reg_result:.4f}')
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model on MNIST-M')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--classification')
    arg_parser.add_argument('--batch-size', type=int, default=256)
    args = arg_parser.parse_args()
    main(args)
