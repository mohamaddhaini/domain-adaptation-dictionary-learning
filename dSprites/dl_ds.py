"""
Implements WDGRL:
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import pickle
import datetime

import argparse
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange
from torchvision import transforms, datasets
import os
import dl_loss
import sys
import time
import numpy as np
import random
import config
from data import MNISTM
from models import Net,Model_Regression_dsprites
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb
from test_model import Regression_test_dsprites
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
def set_seed(a):
    torch.manual_seed(a)
    torch.cuda.manual_seed_all(a)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(a)
    random.seed(a)
    os.environ['PYTHONHASHSEED'] = str(a)

def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

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

def RSD(Feature_s, Feature_t):
    u_s, s_s, v_s = torch.svd(Feature_s.t())
    u_t, s_t, v_t = torch.svd(Feature_t.t())
    p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return torch.norm(sinpa,1)+0.03*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

def main(args):
    print(args.path)
    tim=time.time()
    set_seed(args.seed)
    # clf_model = Net().to(device)
    clf_model = Model_Regression_dsprites().to(device)
    # clf_model.load_state_dict(torch.load(args.MODEL_FILE))
    
    feature_extractor = clf_model.feature_extractor
    discriminator = clf_model.classifier

    critic = nn.Sequential(
        nn.Linear(512, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    

    half_batch = args.batch_size // 2
    if (args.classification==1):
      # >>>>>> Case of Classificatio <<<<<<<
      data_folder = '/content/drive/MyDrive/Colab Notebooks/Wasserstein GANS/pytorch-domain-adaptation/data/office31'
      n_class = 31
      domain_src, domain_tar = 'amazon', 'webcam'
      source_loader = load_data(data_folder, domain_src, half_batch, phase='src',shuffle=True)
      target_loader = load_data(data_folder, domain_tar, half_batch, phase='tar',shuffle=True)
      clf_criterion = nn.CrossEntropyLoss()
    else:
      # >>>>> Case of Regression <<<<<<<<<
      os.chdir(os.path.join(args.path,'Data'))
      data_transforms = {'train': tran.rr_train(resize_size=224),'val': tran.rr_train(resize_size=224),
          'test': tran.rr_eval(resize_size=224),}
      # batch_size = {"train": 36, "val": 36, "test": 4}
      color=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/dSprites/color.txt")
      noisy=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/dSprites/noisy.txt")
      scream=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/dSprites/scream.txt")
      color_t=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/dSprites/color_test.txt")
      noisy_t=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/dSprites/noisy_test.txt")
      scream_t=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/dSprites/scream_test.txt")

      if args.src =='color':
          source_path = color
      elif args.src =='noisy':
          source_path = noisy
      elif args.src =='scream':
          source_path = scream

      if args.tgt =='color':
          target_path = color
          target_path_t = color_t
      elif args.tgt =='noisy':
          target_path = noisy
          target_path_t = noisy_t
      elif args.tgt =='scream':
          target_path = scream
          target_path_t = scream_t

        
      dsets = {"train": ImageList(open(source_path).readlines(), transform=data_transforms["train"]),
              "val": ImageList(open(target_path).readlines(),transform=data_transforms["val"]),
              "test": ImageList(open(target_path_t).readlines(),transform=data_transforms["test"])}

      """#>>>>>>>>> Use all dataloader <<<<<<<<<<<"""
      if not args.subset:
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=half_batch,
                                                      shuffle=True, num_workers=0)
                        for x in ['train', 'val']}
        dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=half_batch,
                                                          shuffle=False, num_workers=0)

      else:
        """ >>>>>>>> Create Subsets for tuning only <<<<<<<<<<<<"""
        size=args.subset_size
        dset_train, _ = torch.utils.data.random_split(dsets["train"], (size, len(dsets["train"])-size))
        dset_val, _ = torch.utils.data.random_split(dsets["val"], (size, len(dsets["val"])-size))
        dset_test, _ = torch.utils.data.random_split(dsets["test"], (size, len(dsets["test"])-size))
        dsets_subset = {"train": dset_train, "val": dset_val,"test": dset_test}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets_subset[x], batch_size=half_batch,
                                                      shuffle=True, num_workers=0)
                        for x in ['train', 'val']}
        dset_loaders["test"] = torch.utils.data.DataLoader(dsets_subset["test"], batch_size=half_batch,
                                                          shuffle=False, num_workers=0)
        """#>>>>>>>>>>>>>....................<<<<<<<<<<<<<<<<<<<<<<<<<<<"""

      source_loader = dset_loaders["train"]
      target_loader = dset_loaders["val"]
      len_source = len(dset_loaders["train"]) - 1
      len_target = len(dset_loaders["val"]) - 1
      clf_criterion = nn.MSELoss()
      test_loss=np.inf

    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, feature_extractor.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, discriminator.parameters()), "lr": 1}]
    clf_optim = torch.optim.SGD(optimizer_dict ,  lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    # clf_optim = torch.optim.RMSprop(clf_model.parameters() ,  lr=args.lr)

    # >>>>>> Training Loop <<<<<<<<
    print(len_source)
    print('Starting',flush=True)
    for epoch in range(1, args.epochs+1):
        param_lr = []
        for param_group in clf_optim.param_groups:
            param_lr.append(param_group["lr"])
        total_loss = 0
        total_accuracy = 0
        generator_loss=[]
        discriminator_loss=[]
        reg_loss=[]
        if not args.subset:
          n_iter = len_source
        else:
          n_iter = args.iterations
        # for iter_num in trange(len_source, leave=False):
        for iter_num in range(n_iter):
            clf_model.train(True)
            # if args.rsd==1:
            clf_optim = inv_lr_scheduler(param_lr, clf_optim, iter_num, init_lr=0.1, gamma=0.0001, power=0.75,weight_decay=0.0005)
            if iter_num % len_source == 0:
              iter_source = iter(dset_loaders["train"])
            if iter_num % len_target == 0:
              iter_target = iter(dset_loaders["val"])
            (source_x, source_y), (target_x, _) = next(iter_source),next(iter_target)
            if iter_num==0:
                print(f'First batch mean is {source_x.mean()}')
            source_x, target_x = source_x.to(device), target_x.to(device)
            labels1 = source_y[:, 0]
            labels3 = source_y[:, 2]
            labels4 = source_y[:, 3]
            labels1 = labels1.unsqueeze(1)
            labels3 = labels3.unsqueeze(1)
            labels4 = labels4.unsqueeze(1)
            source_y = torch.cat((labels1,labels3,labels4),dim=1)
            source_y = source_y.float().to(device)
            clf_optim.zero_grad()
            h_s = feature_extractor(source_x).view(source_x.shape[0], -1)
            h_t = feature_extractor(target_x).view(target_x.shape[0], -1)
            source_preds = discriminator(h_s)
            clf_loss = clf_criterion(source_preds, source_y)
            if args.rsd==1:
              rsd_loss =  RSD (h_s,h_t)
            else:
              if args.wd_clf==0:
                rsd_loss=0
              else:
                rsd_loss,_= dl_loss.match_dl(h_s,h_t,args.rank,args.alpha,args.sp1,args.sp2)
              
            loss = clf_loss + args.wd_clf*(rsd_loss)
            loss.backward()
            # print(clf_model.feature_extractor.layer3[1].conv1.weight.grad,flush=True)
            clf_optim.step()
            # Evaluate
            if ((iter_num+1)%args.print_interval==0):
              clf_model.eval()
              reg_result=Regression_test_dsprites(dset_loaders,clf_model)
              if reg_result<test_loss:
                test_loss=reg_result
              print(f'Epoch {iter_num}/{n_iter} ({int((iter_num/n_iter)*100)}%) Test Result is  {reg_result:.4f}, best is {test_loss:.4f}',flush=True)

    ##### Extract Source and target dictionaries of the dataset at the end of the training 
    iter_source = iter(dset_loaders["train"])
    iter_target = iter(dset_loaders["val"])
    source_features=[]
    for i in range(len_source):
      source_x, source_y = next(iter_source)
      source_x = source_x.to(device)
      h_s = feature_extractor(source_x).view(source_x.shape[0], -1).detach()
      source_features.append(h_s)
    source_features = torch.cat(source_features)
    target_features=[]
    for i in range(len_target):
      target_x, _ =next(iter_target)
      target_x = target_x.to(device)
      h_t = feature_extractor(target_x).view(target_x.shape[0], -1)
      target_features.append(h_t)
    target_features = torch.cat(target_features)
    # M_source,_,_,_  = dl_loss.dictionary_learning(50,source_features.t(),rank=18,lambda_sp=0.9,lambda_reg=1,lamda = 0.1)
    # M_target,_,_,_  = dl_loss.dictionary_learning(50,target_features.t(),rank=18,lambda_sp=0.9,lambda_reg=1,lamda = 0.1)
    _,delta = dl_loss.match_dl(source_features,target_features,args.rank,args.alpha,args.sp1,args.sp2)

    print(f' Best Result rsd {args.rsd} src {args.src} tgt {args.tgt} lr {args.lr:.4f} seed {args.seed} rsd {args.rsd} batch size {args.batch_size:04d} is {test_loss:.4f}')
    print(f'Elapsed time is {((time.time()-tim)/3600):.4f} hours')
    key1=str(datetime.date.today())
    key2 = str(datetime.datetime.now().time())
    try:
      f= open(os.path.join(args.path2,'training_dict_hist_DL.pkl') ,'rb')
      df=pickle.load(f)
      f.close()
    except:
      df = {}
    try:
      df[key1][key2]={'src':args.src,'tgt':args.tgt,'batch-size':args.batch_size,'best-score':test_loss,'rsd':args.rsd,'seed':args.seed,\
          'delta':delta}
    except:
      df[key1]={}
      df[key1][key2]={'src':args.src,'tgt':args.tgt,'batch-size':args.batch_size,'best-score':test_loss,'rsd':args.rsd,'seed':args.seed,\
              'delta':delta}
    save=False
    while (save==False):
      try:
        f=open(os.path.join(args.path2,'training_dict_hist_DL.pkl'),'wb')
        pickle.dump(df,f)
        f.close()
        save=True
      except:
        pass

if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description='Domain adaptation using WDGRL')
  # arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
  arg_parser.add_argument('--classification',type=int,default=0)
  arg_parser.add_argument('--rsd' , type=int, default= 0)
  arg_parser.add_argument('--subset' , type=int, default= 0)
  arg_parser.add_argument('--src', type=str, default='t', metavar='S',help='source dataset')
  arg_parser.add_argument('--tgt', type=str, default='rl', metavar='S',help='target dataset')
  arg_parser.add_argument('--seed',type=int,default=3)
  arg_parser.add_argument('--manifold',type=int,default=2)
  arg_parser.add_argument('--adam',type=int,default=1)
  arg_parser.add_argument('--trial',type=int,default=0)
  arg_parser.add_argument('--print-interval', type=int, default=100)
  arg_parser.add_argument('--lr', type=float, default=0.0001)
  arg_parser.add_argument('--batch-size', type=int, default=72)
  arg_parser.add_argument('--iterations', type=int, default=100)
  arg_parser.add_argument('--epochs', type=int, default=1)
  arg_parser.add_argument('--gamma', type=float, default=0.1)
  arg_parser.add_argument('--wd-clf', type=float, default=0.001)
  arg_parser.add_argument('--subset-size', type=int, default=100)
  arg_parser.add_argument('--alpha', type=int, default=1000)
  arg_parser.add_argument('--rank', type=int, default=18)
  arg_parser.add_argument('--sp1', type=float, default=0.9)
  arg_parser.add_argument('--sp2', type=float, default=0.9)
  arg_parser.add_argument('--path',type=str)
  arg_parser.add_argument('--path2',type=str)
  args = arg_parser.parse_args()
  main(args)

