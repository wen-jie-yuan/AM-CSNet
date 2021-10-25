import argparse

import torchvision
from torch.utils.data import DataLoader

from CSNet.slover import Trainer

# set Training parameter

parser = argparse.ArgumentParser(description='Compressed sensing with NN')

# Set super parameters
parser.add_argument('--batchSize', type=int, default=16, help='Small batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='Test batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='Iterations')
parser.add_argument('--imageSize', type=int, default=128, metavar='N')
parser.add_argument('--samplingRate', type=int, default=10, help='sampling_rate = 1/4/10/25')
parser.add_argument('--resBlock', type=int, default=8, help='res blocks')
parser.add_argument('--samplingPoint', type=int, default=102, help='1% - 10 4% - 41 10% - 102 25% - 256')
parser.add_argument('--trainPath', default='./dataset/train/')
parser.add_argument('--val_set5', default='./dataset/test_images/Set5/')
parser.add_argument('--val_set11', default='./dataset/test_images/Set11/')
parser.add_argument('--val_set14', default='./dataset/test_images/Set14/')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cuda', action='store_true', default=True)

args = parser.parse_args()
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(args.imageSize),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(args.trainPath, transform=transforms)
val_set5 = torchvision.datasets.ImageFolder(args.val_set5, transform=transforms_test)
val_set11 = torchvision.datasets.ImageFolder(args.val_set11, transform=transforms_test)
val_set14 = torchvision.datasets.ImageFolder(args.val_set14, transform=transforms_test)
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
val_loader_set5 = DataLoader(val_set5, batch_size=args.testBatchSize, shuffle=False)
val_loader_set11 = DataLoader(val_set11, batch_size=args.testBatchSize, shuffle=False)
val_loader_set14 = DataLoader(val_set14, batch_size=args.testBatchSize, shuffle=False)

# model select
model = Trainer(args, train_loader, val_loader_set5, val_loader_set11, val_loader_set14)

model.run()
