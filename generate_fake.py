import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as T

from PIL import Image

import os
import argparse

class Butterfly_Dataset(Dataset):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Butterfly')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--norm', type=str ,default='Linf')
    parser.add_argument('--model', type=str, default='resnet18')

    args = parser.parse_args()

    print('We are generate fake_img for {} at {}'.format(args.model, args.dataset))

    if args.dataset == 'Butterfly':
        transform_list = [T.Resize((224, 224)), T.ToTensor()]
        transform_chain = T.Compose(transform_list)
        item = Butterfly_Dataset(mode='test', transform=transform_chain)
        
    elif args.dataset == 'cifar100':
        transform_list = [T.ToTensor()]
        transform_chain = T.Compose(transform_list)
        item = datasets.CIFAR100(root='./data/cifar100', train=False, transform=transform_chain, download=True)

    elif args.dataset == 'fashionmnist':
        transform_list = [T.ToTensor()]
        transform_chain = T.Compose(transform_list)
        item = datasets.FashionMNIST(root='./data/fashionmnist', train=False, transform=transform_chain, download=True)

    else:
        raise ValueError('please check your dataset!!!')

    ckpt = torch.load('./results/{}/{}_{}_{}.pth'.format(args.model, args.dataset, args.version, args.norm))
    x_advs = ckpt['adv_complete']
    inv_transform = T.ToPILImage()

    img_save_path = './fake_sample/{}/fake_img'.format(args.dataset)
    noise_save_path = './fake_sample/{}/noise'.format(args.dataset)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if not os.path.exists(noise_save_path):
        os.makedirs(noise_save_path)

    loader = DataLoader(item, batch_size=1, shuffle=False, num_workers=0)

    for batch_id, (X, y) in enumerate(loader):
        x_adv = x_advs[batch_id]

        X.squeeze_()

        noise = X - x_adv

        fake_img = inv_transform(x_adv)
        noise_img = inv_transform(noise)

        fake_img.save(img_save_path + '/{}.jpg'.format(batch_id+1))
        noise_img.save(noise_save_path + '/{}.jpg'.format(batch_id+1))
