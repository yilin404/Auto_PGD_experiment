import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

import os
import argparse
from PIL import Image

class MyInception(nn.Module):
    def __init__(self, num_classes):
        super(MyInception ,self).__init__()
        self.inception = models.inception_v3(aux_logits=False, pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, inputs):
        return self.classifier(self.inception(inputs))

class MyMobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MyMobileNet ,self).__init__()
        self.net = models.mobilenet_v2(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, inputs):
        return self.classifier(self.net(inputs))

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

class Butterfly_Adversarial_Dataset(torch.utils.data.Dataset):
    def __init__(self, adv_mode='Inception'):
        super(Butterfly_Adversarial_Dataset, self).__init__()
        self.imgs = []
        self.labels = []
        self.adv_mode = adv_mode

        self.class_to_idx = {key:value for value, key in enumerate(os.listdir('./data/ButterflyCls/Butterfly20'))}

        path = './results/resnet18/Butterfly_standard_Linf.pth'
        ckpt = torch.load(path)
        self.imgs = ckpt['adv_complete']
        if self.adv_mode == 'Inception':
            self.imgs = F.interpolate(self.imgs, 299)

        clsfile = './data/ButterflyCls/result.txt'
        with open(clsfile, 'r') as file_to_read:
            for cls in file_to_read.readlines():
                self.labels.append(self.class_to_idx[cls[:-1]])

    def __getitem__(self, index):
            return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Inception')

    args = parser.parse_args()

    if args.model == 'Inception':
        model = MyInception(20)
        test_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])
    
    elif args.model == 'MobileNet':
        model = MyMobileNet(20)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    elif args.model == 'resnet18':
        model = MyResNet(20)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    else:
        raise ValueError('please check your model!!!')

    ckpt = torch.load('./model/{}/Butterfly/model.pth'.format(args.model))
    model.load_state_dict(ckpt)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_dataset = Butterfly_Dataset(mode='test', transform=test_transform)
    adv_dataset = Butterfly_Adversarial_Dataset(adv_mode=args.model)

    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    adv_loader = data.DataLoader(adv_dataset, batch_size=32, shuffle=False, num_workers=0)

    size = len(test_loader.dataset)
    with torch.no_grad():
        correct_top1, correct_top5 = 0, 0
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            _, logits_top5 = logits.topk(5, 1, True, True)
            y_resize = y.view(-1, 1).to(device)

            correct_top1 += (logits.argmax(1) == y).type(torch.float).sum().item()
            correct_top5 += (logits_top5 == y_resize).type(torch.float).sum().item()

        correct_top1 /= size
        correct_top5 /= size
        print(f"Before Attack:\n Top1 Error: {(100*(1 - correct_top1)):>0.1f}%, Top-5 Error: {(100*(1-correct_top5)):>0.1f}%\n")

    size = len(adv_loader.dataset)
    with torch.no_grad():
        correct_top1, correct_top5 = 0, 0
        for X, y in adv_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            _, logits_top5 = logits.topk(5, 1, True, True)
            y_resize = y.view(-1, 1).to(device)

            correct_top1 += (logits.argmax(1) == y).type(torch.float).sum().item()
            correct_top5 += (logits_top5 == y_resize).type(torch.float).sum().item()

        correct_top1 /= size
        correct_top5 /= size
        print(f"After Attack:\n Top1 Error: {(100*(1 - correct_top1)):>0.1f}%, Top-5 Error: {(100*(1-correct_top5)):>0.1f}%\n")