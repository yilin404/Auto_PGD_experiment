from urllib import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torchsummary import summary

import os
import argparse

from PIL import Image
import numpy as np

def adjust_learning_rate(optimizer, epoch, dsr_epoch):
    if epoch in dsr_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.3

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Butterfly')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int ,default=10)
    parser.add_argument('--model', type=str, default='resnet18')

    args = parser.parse_args()

    print('We are traing {} at {}'.format(args.model, args.dataset))
    print('num_epochs:{} batch_size:{}'.format(args.num_epochs, args.batch_size))

    if args.dataset == 'Butterfly':
        model = MyResNet(20, mode=args.model)
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
         ])
        train_dataset = Butterfly_Dataset(mode='train', transform=transform_train)
        test_dataset = Butterfly_Dataset(mode='test', transform=transform_test)          
        
    elif args.dataset == 'cifar100':
        model = MyResNet(100, mode=args.model)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR100(root='./data/cifar100', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, transform=transform_test, download=True)

    elif args.dataset == 'fashionmnist':
        model = MyResNet(10, mode=args.model, in_channels=1)
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.FashionMNIST(root='./data/fashionmnist', train=True, transform=transform_train, download=True)
        test_dataset = datasets.FashionMNIST(root='./data/fashionmnist', train=False, transform=transform_test, download=True)


    else:
        raise ValueError('please check your dataset!!!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    learning_rate = 1e-3
    dsr_epoch = [3, 6, 9, 12, 15]
    save_path = './model/{}/{}/'.format(args.model, args.dataset)
  
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gl_max_acc = 0.0

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        adjust_learning_rate(optimizer, epoch, dsr_epoch)

        model.train()
        size = len(train_loader.dataset)
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        model.eval()
        num_batches = len(test_loader)
        size = len(test_loader.dataset)
        with torch.no_grad():
            test_loss, correct_top1, correct_top5 = 0, 0, 0
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                y_resize = y.view(-1, 1).to(device)
                pred = model(X)
                _, pred_top5 = pred.topk(5, 1, True, True)

                test_loss += criterion(pred, y).item()
                correct_top1 += (pred.argmax(1) == y).type(torch.float).sum().item()
                correct_top5 += (pred_top5 == y_resize).type(torch.float).sum().item()

            test_loss /= num_batches
            correct_top1 /= size
            correct_top5 /= size
            print(f"Test Error: \n Top1 Error: {(100*(1 - correct_top1)):>0.1f}%, Top-5 Error: {(100*(1-correct_top5)):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        if correct_top1 > gl_max_acc:
            gl_max_acc = correct_top1
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path+'/model.pth')

    print('Done!')
