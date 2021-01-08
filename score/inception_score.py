import numpy as np
import torch
from tqdm import trange

from .inception import InceptionV3


device = torch.device('cuda:0')


def get_inception_score(images, splits=10, batch_size=32, use_torch=False,
                        verbose=False, parallel=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    preds = []
    iterator = trange(
        0, len(images), batch_size, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_score")

    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)[0]
        if use_torch:
            preds.append(pred)
        else:
            preds.append(pred.cpu().numpy())
    if use_torch:
        preds = torch.cat(preds, 0)
    else:
        preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[
            (i * preds.shape[0] // splits):
            ((i + 1) * preds.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        is_mean, is_std = (
            torch.mean(scores).cpu().item(), torch.std(scores).cpu().item())
    else:
        is_mean, is_std = np.mean(scores), np.std(scores)
    del preds, scores, model
    return is_mean, is_std
