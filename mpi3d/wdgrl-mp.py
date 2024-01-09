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
import transport
import transport_old
import sys
import time
import numpy as np
import random
import config
from data import MNISTM
from models import Net,Model_Regression_mp
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb
from test_model import Regression_test_mp
import model_mp
import transform_mp as tran
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

def riemannian_distance(Feature_s, Feature_t):
  u_s, s_s, v_s = torch.svd(Feature_s.t())
  u_t, s_t, v_t = torch.svd(Feature_t.t())
  p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
  return torch.norm(torch.log10(cospa))

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
    tim=time.time()
    set_seed(args.seed)
    # clf_model = Net().to(device)
    clf_model = Model_Regression_mp().to(device)
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
      rc=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/MPI3D/realistic.txt")
      rl=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/MPI3D/real.txt")
      t=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/MPI3D/toy.txt")
      rc_t=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/MPI3D/realistic_test.txt")
      rl_t=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/MPI3D/real_test.txt")
      t_t=os.path.join(args.path,"Domain-Adaptation-Regression/DAR-RSD/MPI3D/toy_test.txt")

      if args.src =='rl':
          source_path = rl
      elif args.src =='rc':
          source_path = rc
      elif args.src =='t':
          source_path = t

      if args.tgt =='rl':
          target_path = rl
          target_path_t = rl_t
      elif args.tgt =='rc':
          target_path = rc
          target_path_t = rc_t
      elif args.tgt =='t':
          target_path = t
          target_path_t = t_t

        
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
    if args.rsd==0:
      if (args.adam==0):
        critic_optim = torch.optim.RMSprop(critic.parameters(), lr=args.critic_lr)
        clf_optim = torch.optim.RMSprop(clf_model.parameters() ,  lr=args.lr)
      else:
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr,betas=(0.5,0.99))
        # clf_optim = torch.optim.Adam(clf_model.parameters() ,  lr=args.lr,betas=(0.5,0.99))
        clf_optim = torch.optim.SGD(optimizer_dict ,  lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    else:
      clf_optim = torch.optim.SGD(optimizer_dict ,  lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    # clf_optim = torch.optim.RMSprop(clf_model.parameters() ,  lr=args.lr)

    # >>>>>> Training Loop <<<<<<<<
    print(len_source)
    print('Starting',flush=True)
    for epoch in range(1, args.epochs+1):

        # batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
        # iter_source = iter(dset_loaders["train"])
        # iter_target = iter(dset_loaders["val"])
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
            labels1 = source_y[:,0]
            labels2 = source_y[:,1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            source_y  = torch.cat((labels1,labels2),dim=1)
            source_y  = source_y .float().to(device)/39

            if args.rsd==0:
              # >>>>>>>>>Train critic<<<<<<<<<
              set_requires_grad(feature_extractor, requires_grad=False)
              set_requires_grad(critic, requires_grad=True)
              with torch.no_grad():
                h_s =  feature_extractor(source_x).data.view(source_x.shape[0], -1)
                h_t = feature_extractor(target_x).data.view(target_x.shape[0], -1)
              disc_loss=0.0
              for _ in range(args.k_critic):
                  critic_optim.zero_grad()
                  gp = gradient_penalty(critic, h_s, h_t)
                  critic_s = critic(h_s)
                  critic_t = critic(h_t)
                  wasserstein_distance = critic_s.mean() - critic_t.mean()
                  riemannian1 = torch.mean(torch.norm(torch.norm(torch.norm(source_x-target_x,dim=1),dim=1),dim=1) - torch.norm(h_s-h_t,dim=1))
                  riemannian2 = torch.mean(torch.norm(h_s-h_t,dim=1) - torch.norm(source_y +discriminator(h_t)))
                  critic_cost = args.lambda_critic*(-(wasserstein_distance)+ args.gamma*gp)
                  critic_cost.backward()
                  # print(critic[0].weight.grad)
                  critic_optim.step()
                  total_loss += critic_cost.item()
                  disc_loss+=(wasserstein_distance).item()

              # >>>>> Weight Clipping <<<<<<
              # for p in critic.parameters():
              #   p.data.clamp_(-0.01,0.01)
              # >>>>>>>>>Train classifier<<<<<<<<<
              set_requires_grad(feature_extractor, requires_grad=True)
              set_requires_grad(critic, requires_grad=False)
              gen_loss=0.0
              for _ in range(args.k_clf):
                  clf_optim.zero_grad()
                  source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
                  target_features = feature_extractor(target_x).view(target_x.shape[0], -1)
                  source_preds = discriminator(source_features)
                  clf_loss = clf_criterion(source_preds, source_y)
                  wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()
                  # riemannian=torch.mean((torch.matmul(source_features,target_features.T)/(torch.norm(target_features)*torch.norm(source_features)))-(critic(source_features)-critic(target_features))**2)
                  if args.manifold==1:
                    riemannian1 = torch.mean(torch.norm(torch.norm(torch.norm(source_x-target_x,dim=1),dim=1),dim=1) - torch.norm(source_features-target_features,dim=1))
                    riemannian2 = torch.mean(torch.norm(source_features-target_features,dim=1) - torch.norm(source_y-source_preds))
                    loss = clf_loss + args.wd_clf*(wasserstein_distance+ args.coeff*(riemannian2+riemannian1))
                  elif args.manifold==2:
                    loss_transport,gamma=transport.optimal_trans(source_x,target_x,source_features,target_features,source_y,source_preds,args.abl_trial,regr=args.regr,nb_iter=args.nb_iter)
                    loss = clf_loss + args.wd_clf*(wasserstein_distance) + args.coeff*(loss_transport)
                    # print(gamma,flush=True)
                  else:
                    loss = clf_loss + args.wd_clf*(wasserstein_distance)
                  
                  loss.backward()
                  # print(clf_model.feature_extractor.layer3[1].conv1.weight.grad)
                  gen_loss+=loss.item()
                  clf_optim.step()

              generator_loss.append(gen_loss/args.k_clf)
              discriminator_loss.append(disc_loss/args.k_critic)
              reg_loss.append(clf_loss.item())
            else:
              clf_optim.zero_grad()
              h_s = feature_extractor(source_x).view(source_x.shape[0], -1)
              h_t = feature_extractor(target_x).view(target_x.shape[0], -1)
              source_preds = discriminator(h_s)
              clf_loss = clf_criterion(source_preds, source_y)
              rsd_loss =  RSD (h_s,h_t)
              loss = clf_loss + args.wd_clf*(rsd_loss)
              loss.backward()
              # print(clf_model.feature_extractor.layer3[1].conv1.weight.grad)
              clf_optim.step()


            # Evaluate
            if ((iter_num+1)%args.print_interval==0):
              clf_model.eval()
              reg_result=Regression_test_mp(dset_loaders,clf_model)
              if reg_result<test_loss:
                test_loss=reg_result
              print(f'Epoch {iter_num}/{n_iter} ({int((iter_num/n_iter)*100)}%) Test Result is  {reg_result:.4f}, best is {test_loss:.4f}',flush=True)
            #   if args.manifold==2:
            #     fig=plt.figure()
            #     plt.imshow(gamma.cpu().detach().numpy())
            #     plt.savefig(r'/content/drive/MyDrive/Mohamad/GANS/pytorch-domain-adaptation/saved data/ot-gamma.png', bbox_inches='tight')
            #     plt.close(fig)
            # os.makedirs(os.path.join(r'/content/drive/MyDrive/Mohamad/GANS/pytorch-domain-adaptation/saved data','trial_'+str(args.trial)), exist_ok=True)
            # np.save(os.path.join(r'/content/drive/MyDrive/Mohamad/GANS/pytorch-domain-adaptation/saved data','trial_'+str(args.trial)+'/gen_loss.npy'),generator_loss)
            # np.save(os.path.join(r'/content/drive/MyDrive/Mohamad/GANS/pytorch-domain-adaptation/saved data','trial_'+str(args.trial)+'/disc_loss.npy'),discriminator_loss)
            # np.save(os.path.join(r'/content/drive/MyDrive/Mohamad/GANS/pytorch-domain-adaptation/saved data','trial_'+str(args.trial)+'/reg_loss.npy'),reg_loss)


        
        # os.chdir(r'/content/drive/MyDrive/Mohamad/GANS/pytorch-domain-adaptation')
        mean_loss = total_loss / (args.iterations * args.k_critic)
    print(f' Best Result rsd {args.rsd} src {args.src} tgt {args.tgt} lr {args.lr:.4f} manifold {args.manifold:02d} regr sinkhorn {args.regr:.4f} coeff {args.coeff:.4f} batch size {args.batch_size:04d} is {test_loss:.4f}')
    print(f'Elapsed time is {((time.time()-tim)/3600):.4f} hours')
    key1=str(datetime.date.today())
    key2 = str(datetime.datetime.now().time())
    try:
      f= open(os.path.join(args.path2,'training_dict_hist.pkl') ,'rb')
      df=pickle.load(f)
      f.close()
    except:
      df = {}
    try:
      df[key1][key2]={'src':args.src,'tgt':args.tgt,'manifold':args.manifold,'regr-sinkhorn':args.regr,'coeff':args.coeff,'batch-size':args.batch_size,'best-score':test_loss}
    except:
      df[key1]={}
      df[key1][key2]={'src':args.src,'tgt':args.tgt,'manifold':args.manifold,'regr-sinkhorn':args.regr,'coeff':args.coeff,'batch-size':args.batch_size,'best-score':test_loss}
    save=False
    while (save==False):
      try:
        f=open(os.path.join(args.path2,'training_dict_hist.pkl'),'wb')
        pickle.dump(df,f)
        f.close()
        save=True
      except:
        pass
        # tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_loss:.4f}')

        # if args.manifold==1:
        #   torch.save(clf_model.state_dict(), 'trained_models/wdgrl_with_manifold.pt')
        # else:
        #   torch.save(clf_model.state_dict(), 'trained_models/wdgrl_without_manifold.pt')


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
  arg_parser.add_argument('--critic-lr', type=float, default=0.01)
  arg_parser.add_argument('--lr', type=float, default=0.0001)
  arg_parser.add_argument('--batch-size', type=int, default=72)
  arg_parser.add_argument('--iterations', type=int, default=100)
  arg_parser.add_argument('--epochs', type=int, default=1)
  arg_parser.add_argument('--k-critic', type=int, default=5)
  arg_parser.add_argument('--k-clf', type=int, default=1)
  arg_parser.add_argument('--gamma', type=float, default=0.1)
  arg_parser.add_argument('--wd-clf', type=float, default=0.001)
  arg_parser.add_argument('--coeff', type=float, default=1.0)
  arg_parser.add_argument('--lambda-critic', type=float, default=1.0)
  arg_parser.add_argument('--abl-trial', type=int, default=6)
  arg_parser.add_argument('--subset-size', type=int, default=100)
  arg_parser.add_argument('--nb-iter', type=int, default=1000)
  arg_parser.add_argument('--regr', type=float, default=0.001)
  arg_parser.add_argument('--wd-clf2', type=float, default=1.0)
  arg_parser.add_argument('--path',type=str)
  arg_parser.add_argument('--path2',type=str)
  args = arg_parser.parse_args()
  main(args)
