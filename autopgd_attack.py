import os
import argparse
import time

import numpy as np
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

class Butterfly_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', transform=None):
        super(Butterfly_Dataset, self).__init__()
        self.mode = mode
        self.imgs = []
        self.labels = []
        self.transform = transform

        self.class_to_idx = {key:value for value, key in enumerate(os.listdir('./data/ButterflyCls/Butterfly20'))}

        if self.mode == 'train':
            path = './data/ButterflyCls/Butterfly20'
            for cls in os.listdir(path):
                file_path = os.path.join(path, cls)
                for img in os.listdir(file_path):
                    file_name = os.path.join(file_path, img)
                    self.imgs.append(file_name)
                    self.labels.append(self.class_to_idx[cls])

        elif self.mode == 'test':
            path = './data/ButterflyCls/Butterfly20_test'
            img_list = os.listdir(path)
            img_list.sort(key=lambda x: int(x.split('.')[0]))
            for img in img_list:
                file_name = os.path.join(path, img)
                self.imgs.append(file_name)
            clsfile = './data/ButterflyCls/result.txt'
            with open(clsfile, 'r') as file_to_read:
                for cls in file_to_read.readlines():
                    self.labels.append(self.class_to_idx[cls[:-1]])

        else:
            raise ValueError('please check you data mode!!!')

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        if self.transform != None:
            return self.transform(img), self.labels[index]
        else:
            return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)

class MyResNet(nn.Module):
    def __init__(self, num_classes, mode='resnet18', in_channels=3):
        super(MyResNet ,self).__init__()
        if mode == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif mode == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
        elif mode == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError('please check your model!!!')
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, inputs):
        return self.classifier(self.resnet(inputs))

class MyAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', device='cuda'):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.device = device

        from autopgd_base import APGDAttack
        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device)

        from autopgd_base import APGDAttack_targeted
        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device)

        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)

    def get_logits(self, x):
        return self.model(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def run_standard_evaluation(self, x_orig, y_orig, bs=250):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x)
                correct_batch = y.eq(output.max(dim=1)[1])
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

            if self.verbose:
                print('initial accuracy: {:.2%}'.format(robust_accuracy))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True

                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True

                    else:
                        raise ValueError('Attack not supported')

                    output = self.get_logits(adv_curr)
                    false_batch = ~y.eq(output.max(dim=1)[1]).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)
                        print('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                if self.verbose:
                    print('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().view(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).view(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().view(x_orig.shape[0], -1).sum(dim=-1)
                print('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                print('robust accuracy: {:.2%}'.format(robust_accuracy))

        return x_adv

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))

        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))

        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False

        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            adv[c] = self.run_standard_evaluation(x_orig, y_orig, bs=bs)
            if verbose_indiv:
                acc_indiv  = self.clean_accuracy(adv[c], y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                print('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))

        return adv

    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))

        if version == 'standard':
            # self.attacks_to_run = ['apgd-ce', 'apgd-t']
            self.attacks_to_run = ['apgd-ce']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.apgd_targeted.n_restarts = 1

        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'apgd-t']
            self.apgd.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.apgd_targeted.n_target_classes = 9
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--dataset', type=str, default='Butterfly')
    parser.add_argument('--model', type=str, default='resnet18')

    args = parser.parse_args()

    print('We are attacking {} at {} by {}_{}'.format(args.model, args.dataset, args.version, args.norm))

    if args.dataset == 'Butterfly':
        model = MyResNet(20, mode=args.model)
        transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
        transform_chain = transforms.Compose(transform_list)
        item = Butterfly_Dataset(mode='test', transform=transform_chain)
        
    elif args.dataset == 'cifar100':
        model = MyResNet(100, mode=args.model)
        transform_list = [transforms.ToTensor()]
        transform_chain = transforms.Compose(transform_list)
        item = datasets.CIFAR100(root='./data/cifar100', train=False, transform=transform_chain, download=True)

    elif args.dataset == 'fashionmnist':
        model = MyResNet(10, mode=args.model, in_channels=1)
        transform_list = [transforms.ToTensor()]
        transform_chain = transforms.Compose(transform_list)
        item = datasets.FashionMNIST(root='./data/fashionmnist', train=False, transform=transform_chain, download=True)

    else:
        raise ValueError('please check your dataset!!!')

    ckpt = torch.load('./model/{}/{}/model.pth'.format(args.model, args.dataset))
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=0)

    save_path = os.path.join(args.save_dir, args.model)

    # create save dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adversary = MyAttack(model, norm=args.norm, eps=args.epsilon, version=args.version)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)

            torch.save({'adv_complete': adv_complete}, '{}/{}/{}_{}_{}.pth'.format(
                args.save_dir, args.model, args.dataset, args.version, args.norm))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test, y_test, bs=args.batch_size)

            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))

