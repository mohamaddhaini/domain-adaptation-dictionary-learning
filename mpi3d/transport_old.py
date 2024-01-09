import ot
import torch

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


def optimal_trans(xs,xt,gs,gt,ys,ft_pred):
  with torch.no_grad():
    s= xs.shape
    C0 = torch.cdist(gs, gt, p=2.0)**2
    C1 = torch.cdist(xs.reshape(-1, s[1] * s[2] * s[3]),xt.reshape(-1, s[1] * s[2] * s[3]),p=2.0)**2
    C2 = torch.cdist(ys, ft_pred, p=2)**2
    C= C0 + C1 + C2
    C=C.cpu().numpy()
    gamma=ot.sinkhorn(ot.unif(gs.shape[0]),ot.unif(gt.shape[0]),C,reg=10)
    gamma = torch.tensor(gamma).cuda()

  gdist = L2_dist(gs,gt) + L2_dist(ys,ft_pred)
  return torch.sum(gamma * (gdist)) ,gamma


