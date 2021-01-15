import numpy as np



def hinge_loss(X,A,B):
    """
    Support vectorize and scalar computation
    """
    batch_size = len(B)

    product = np.dot(A,X)
    loss = np.maximum(np.zeros(batch_size), 1 - product*B)
    correct = np.where(loss == 0)[0]
    return loss, correct


def reg_hinge_loss(X,A,B,alpha=0,test=False):
    loss, correct = hinge_loss(X,A,B)
    if test:
        grad = []
    else:
        grad = -A*B.reshape(len(B),1)
        grad[correct] = 0
    accuracy = len(correct)/len(B)
    return np.mean(loss) + (alpha/2)*sum(X**2), np.mean(grad,0) + alpha*X, accuracy


def simplex_proj(x):
    x_s = np.sort(x, kind='quicksort')[::-1]
    cum_s = np.cumsum(x_s)
    res = x_s - (cum_s-1)/(np.arange(len(x))+1)
    d0 = np.max(np.where(res>0)[0]) + 1
    theta = (cum_s[d0-1] -1)/d0

    return np.maximum(0,x-theta)

def l1_ball_proj(x,z=1,d=0,weighted=False):
    if sum(abs(x)) <= z:
        return x
    else:
        if weighted:
            w = weighted_simplex_proj(abs(x)/z,d)
        else :
            w = simplex_proj(abs(x)/z)
        return z*np.sign(x)*w

def weighted_simplex_proj(x,D):
    """
    x : vector, D : diag matrix
    """
    dx = np.abs(np.dot(D,x))
    sorted_indices = np.argsort(-dx, kind='quicksort')
    sx = np.cumsum(x[sorted_indices])
    sd = np.cumsum(1/np.diag(D)[sorted_indices])
    res = dx[sorted_indices] - (sx-1)/sd
    d0 = np.max(np.where(res>0)[0])
    theta = (sx[d0]-1)/sd[d0]

    return np.dot(np.linalg.inv(D),np.maximum(0,dx-theta))

