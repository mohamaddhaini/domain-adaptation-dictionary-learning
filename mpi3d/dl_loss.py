import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh

def _lipschitz_constant(W):
    #L = torch.linalg.norm(W, ord=2) ** 2
    WtW = torch.matmul(W.t(), W)
    #L = torch.linalg.eigvalsh(WtW)[-1]
    L = eigsh(WtW.detach().cpu().numpy(), k=1, which='LM',
              return_eigenvectors=False).item()
    return L

def ridge(b, A, alpha=1e-4):
    # right-hand side
    rhs = torch.matmul(A.T, b)
    # regularized gram matrix
    M = torch.matmul(A.T, A)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    if info != 0:
        raise RuntimeError("The Gram matrix is not positive definite. "
                           "Try increasing 'alpha'.")
    x = torch.cholesky_solve(rhs, L)
    return x

def ista(x, weight, alpha=1.0, fast=True, lr='auto', maxiter=100,
         tol=1e-5, verbose=False,initial=None):
    
    n_samples = x.size(0)
    n_components = weight.size(1)
    if initial is None:
        z0 = ridge(x.T, weight, alpha=alpha).T
    else:
        z0=initial
    # z0 = x.new_zeros(n_samples, n_components)
    if lr == 'auto':
        # set lr based on the maximum eigenvalue of W^T @ W; i.e. the
        # Lipschitz constant of \grad f(z), where f(z) = ||Wz - x||^2
        L = _lipschitz_constant(weight)
        lr = 1 / L
    tol = z0.numel() * tol

    def loss_fn(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        loss = 0.5 * resid.pow(2).sum() + alpha * z_k.abs().sum()
        return loss / x.size(0)

    def rss_grad(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        return torch.matmul(resid, weight)
    
    # optimize
    z = z0
    if fast:
        y, t = z0, 1
    loss=[]
    for _ in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z))

        # ista update
        z_prev = y if fast else z
        
        z_next = F.softshrink(z_prev - lr * rss_grad(z_prev), alpha * lr)

        # check convergence
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break

        # update variables
        if fast:
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        z = z_next
        loss.append(loss_fn(z).clone().detach().cpu().numpy())
    return z,loss

def dictionary_learning(max_iter,R,rank,lambda_sp,lambda_reg,lamda = 0.000001):
    with torch.no_grad():
        # u, s, v = torch.svd(R)
        # M=u[:,:rank].float()
        # A ,_= ista(R.T,M,alpha=lambda_sp,maxiter=10)
        M=torch.randn(R.shape[0],rank).cuda()
        M = M/torch.norm(M,dim=0,p=2)
        A=torch.randn(R.shape[1],rank).cuda()
        loss=[]
        v = torch.zeros(R.shape[0],rank).cuda()
    for i in range(max_iter):
        error = R - torch.matmul(M, A.T)
        #Batch Update
        v= lambda_reg*v - lamda * (torch.matmul(error, A))
        M=M-v
        M = M/torch.norm(M,dim=0,p=2)
        A,_ = ista(R.T,M,alpha=lambda_sp,maxiter=10)
        loss.append(torch.linalg.norm(error).cpu().detach().numpy())
    error = R - torch.matmul(M, A.T)
    return M,A,error,np.array(loss)

# def dictionary_learning(max_iter,R,rank,lambda_sp,lambda_reg,lamda = 0.000001):
#     with torch.no_grad():
#         M=torch.randn(R.shape[0],rank).cuda()
#         M = M/torch.norm(M,dim=0,p=2)
#         A=torch.randn(R.shape[1],rank).cuda()
#         loss=[]
#         v = torch.zeros(R.shape[0],rank).cuda()
#         for i in range(max_iter):
#             error = R - torch.matmul(M, A.T)
#             #Batch Update
#             v= lambda_reg*v - lamda * (torch.matmul(error, A))
#             M=M-v
#             M = M/torch.norm(M,dim=0,p=2)
#             A,_ = ista(R.T,M,alpha=lambda_sp,maxiter=10)
#             loss.append(torch.linalg.norm(error).cpu().detach().numpy())
#     error = R - torch.matmul(M, A.T)
#     #Batch Update
#     v= lambda_reg*v - lamda * (torch.matmul(error, A))
#     M=M-v
#     M = M/torch.norm(M,dim=0,p=2)
#     A,_ = ista(R.T,M,alpha=lambda_sp,maxiter=1,initial=A)
#     loss.append(torch.linalg.norm(error).cpu().detach().numpy())
#     error = R - torch.matmul(M, A.T)
#     return M,A,error,np.array(loss)

def match_dl(Feature_s, Feature_t,rank,alpha,sp1,sp2):
    with torch.no_grad():
        b_t,_,_,_=dictionary_learning(50,Feature_t.t(),rank=rank,lambda_sp=sp1,lambda_reg=1,lamda = 0.1)
    b_s,A_s,error_s,loss_s=dictionary_learning(50,Feature_s.t(),rank=rank,lambda_sp=sp1,lambda_reg=1,lamda = 0.1)
    coeffs1,loss= ista(Feature_t,b_s,alpha=sp2,maxiter=50)
    # plt.plot(loss_s)
    # print(A_s,coeffs1)
    res_s=Feature_t.T-b_s.mm(coeffs1.T)
    delta=res_s.mm(coeffs1.mm(torch.linalg.pinv(alpha*torch.ones(rank,rank).cuda()+coeffs1.T.mm(coeffs1))))
    return torch.norm(delta,p='fro'),delta


# def match_dl(Feature_s, Feature_t,alpha,sp1,sp2):
#     b_s,A_s,error_s,loss_s=dictionary_learning(50,Feature_s.t(),rank=18,lambda_sp=sp1,lambda_reg=1,lamda = 0.1)
#     with torch.no_grad():
#         coeffs1,loss= ista(Feature_t,b_s,alpha=sp2,maxiter=50)
#     coeffs1,loss= ista(Feature_t,b_s,alpha=sp2,maxiter=1,initial=coeffs1)
#     # plt.plot(loss_s)
#     # print(A_s,coeffs1)
#     res_s=Feature_t.T-b_s.mm(coeffs1.T)
#     delta=res_s.mm(coeffs1.mm(torch.linalg.pinv(alpha*torch.ones(18,18).cuda()+coeffs1.T.mm(coeffs1))))
#     return torch.norm(delta,p='fro'),b_s.detach().cpu().numpy(),b_s.detach().cpu().numpy()