import torch.nn as nn

from utils import StaticDict

def train_AE(target, optimizer, enc, dec, reg=None, input=None,
             enc_mode='train', dec_mode='train', reg_mode=None):
    if enc_mode is 'train':
        enc.train()
    elif enc_mode is 'eval':
        enc.eval()
    else:
        raise ValueError('dec_mode not recognized.')
    if dec_mode is 'train':
        dec.train()
    elif dec_mode is 'eval':
        dec.eval()
    else:
        raise ValueError('dec_mode not recognized.')
    optimizer.zero_grad()

    recons = dec(enc(target))

    mse = nn.MSELoss()
    loss = mse(recons, target)
    loss.backward()
    optimizer.step()

    return loss


def train_RG(input, target, optimizer, reg, enc, dec=None,
             reg_mode='train', enc_mode='eval', dec_mode=None):
    if reg_mode is 'train':
        reg.train()
    elif reg_mode is 'eval':
        reg.eval()
    else:
        raise ValueError('reg_mode not recognized.')
    if enc_mode is 'train':
        enc.train()
    elif enc_mode is 'eval':
        enc.eval()
    else:
        raise ValueError('dec_mode not recognized.')
    optimizer.zero_grad()

    repres = reg(input)
    dnsamp = enc(target)

    mse = nn.MSELoss()
    loss = mse(repres, dnsamp)
    loss.backward()
    optimizer.step()

    return loss


def train_SR(input, target, optimizer, reg, dec, enc=None,
             reg_mode='train', dec_mode='train', enc_mode=None):
    if reg_mode is 'train':
        reg.train()
    elif reg_mode is 'eval':
        reg.eval()
    else:
        raise ValueError('reg_mode not recognized.')
    if dec_mode is 'train':
        dec.train()
    elif dec_mode is 'eval':
        dec.eval()
    else:
        raise ValueError('dec_mode not recognized.')
    optimizer.zero_grad()

    upsamp = dec(reg(input))

    mse = nn.MSELoss()
    loss = mse(upsamp, target)
    loss.backward()
    optimizer.step()

    return loss


###


def train(data, optimizers,
          enc=None, dec=None, reg=None,
          batch_size=16,
          modes=StaticDict({'train_AE': {'enc_mode': 'train', 'dec_mode': 'train'},
                            'train_RG': {'reg_mode': 'train', 'enc_mode': 'eval'},
                            'train_SR': {'reg_mode': 'train', 'dec_mode': 'train'}}),
          cuda=False):
    epoch_loss = 0.0
    N = 0
    trainers = {}
    for base in optimizers.keys():
        def base_train(input, target):
            return getattr(globals(), base)(input=input, target=target, optimizer=optimizers[base],
                                            enc=enc, dec=dec, reg=reg, **modes[base])
        trainers[base] = base_train

    for b, (inputs, targets) in enumerate(data):
        if inputs.size()[0] == batch_size and targets.size()[0] == batch_size:
            # count
            N += batch_size
            # cuda
            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            # run
            for base in optimizers.keys():
                loss = trainers[base](inputs, targets)
                # cuda
                if cuda:
                    loss = loss.cpu()
                epoch_loss += loss.item()
    return epoch_loss / N


def test(data, enc=None, dec=None, reg=None,
         batch_size=16, cuda=False):
    [model.eval() for model in (enc, dec, reg) if model is not None]
    epoch_loss = 0.0
    N = 0
    for b, (inputs, targets) in enumerate(data):
        if inputs.size()[0] == batch_size and targets.size()[0] == batch_size:
            # count
            N += batch_size
            # cuda
            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            # run
            upsamps = dec(reg(inputs))
            mse = nn.MSELoss()
            loss = mse(upsamps, targets)
            # cuda
            if cuda:
                loss = loss.cpu()
            epoch_loss += loss.item()
    return epoch_loss / N

