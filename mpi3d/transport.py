import ot
import torch
from itertools import combinations

# L2 distance
def L2_dist(x,y):
    '''
    compute the squared L2 distance between two matrics
    '''
    distx = torch.reshape(torch.sum(x*x,1), (-1,1))
    disty = torch.reshape(torch.sum(y*y,1), (1,-1))
    dist = distx + disty
    dist -= 2.0*torch.matmul(x, torch.transpose(y,0,1))  
    return dist

def cosine_similarity(x, y):
    """
    Compute the cosine similarity between two matrices.
    """
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    return torch.matmul(x, torch.transpose(y, 0, 1))


def optimal_trans(xs,xt,gs,gt,ys,ft_pred,trial,regr=10,nb_iter=1000,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
  with torch.no_grad():
    s= xs.shape
    s1= xt.shape
    gs = gs /torch.norm(gs, dim=1, keepdim=True)
    gt = gt /torch.norm(gt, dim=1, keepdim=True)
    xs=  xs.reshape(-1, s[1] * s[2] * s[3])
    xt = xt.reshape(-1, s1[1] * s1[2] * s1[3])
    xs = xs /torch.norm(xs, dim=1, keepdim=True)
    xt = xt /torch.norm(xt, dim=1, keepdim=True)

    C0 = torch.cdist(gs, gt, p=2.0)**2
    C1 = torch.cdist(xs,xt,p=2.0)**2
    C2 = torch.cdist(ys, ft_pred, p=2)**2
    combin = list(combinations([C0,C1,C2],1)) + list(combinations([C0,C1,C2],2)) + list(combinations([C0,C1,C2],3))
    C=0
    selected = combin[trial]
    for i in selected:
      C +=i
    C=C.cpu().numpy()
    gamma=ot.sinkhorn(ot.unif(gs.shape[0]),ot.unif(gt.shape[0]),C,reg=regr, numItermax=nb_iter)
    gamma = torch.tensor(gamma).to(device)
  gdist = L2_dist(gs,gt) + L2_dist(ys,ft_pred)
  loss = torch.mean(gamma * (gdist)) 
  return loss ,gamma
  
  # return torch.sum(gamma0 * (L2_dist(gs,gt))) + torch.sum(gamma2 * (L2_dist(ys,ft_pred))),gamma2
