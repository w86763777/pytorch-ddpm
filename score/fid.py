import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3


DIM = 2048
device = torch.device('cuda:0')


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        K = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        K = K.type(dtype)
        Z = Z.type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * K - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6,
                               use_torch=False):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    if use_torch:
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        # Run 50 itrs of newton-schulz to get the matrix sqrt of
        # sigma1 dot sigma2
        covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50)
        if torch.any(torch.isnan(covmean)):
            return float('nan')
        covmean = covmean.squeeze()
        out = (diff.dot(diff) +
               torch.trace(sigma1) +
               torch.trace(sigma2) -
               2 * torch.trace(covmean)).cpu().item()
    else:
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        out = (diff.dot(diff) +
               np.trace(sigma1) +
               np.trace(sigma2) -
               2 * tr_covmean)
    return out


def get_statistics(images, num_images=None, batch_size=50, use_torch=False,
                   verbose=False, parallel=False):
    """when `images` is a python generator, `num_images` should be given"""

    if num_images is None:
        try:
            num_images = len(images)
        except:
            raise ValueError(
                "when `images` is not a list like object (e.g. generator), "
                "`num_images` should be given")

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx1]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    if use_torch:
        fid_acts = torch.empty((num_images, 2048)).to(device)
    else:
        fid_acts = np.empty((num_images, 2048))

    iterator = iter(tqdm(
        images, total=num_images,
        dynamic_ncols=True, leave=False, disable=not verbose,
        desc="get_inception_and_fid_score"))

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            if use_torch:
                fid_acts[start: end] = pred[0].view(-1, 2048)
            else:
                fid_acts[start: end] = pred[0].view(-1, 2048).cpu().numpy()
        start = end

    if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        s1 = torch_cov(fid_acts, rowvar=False)
    else:
        m1 = np.mean(fid_acts, axis=0)
        s1 = np.cov(fid_acts, rowvar=False)
    return m1, s1


def get_fid_score(stats_cache, images, num_images=None, batch_size=50,
                  use_torch=False, verbose=False, parallel=False):
    m1, s1 = get_statistics(
        images, num_images, batch_size, use_torch, verbose, parallel)

    f = np.load(stats_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    if use_torch:
        m2 = torch.tensor(m2).to(m1.dtype)
        s2 = torch.tensor(s2).to(s1.dtype)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2, use_torch=use_torch)

    if use_torch:
        fid_value = fid_value.cpu().item()
    return fid_value
