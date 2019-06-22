"""
We will do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using 'torchvision'
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import argparse

### 학습한 모델 test용 ###
# model_name = 'best_model.pt'

### 학습용 ###
model_name = None

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=model_name, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--batch_size_test', default=128, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
args = parser.parse_known_args()[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################
# 1. Loading and normalizing CIFAR-10
############################################

transforms_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_worker)

############################################
# 2. Define a Convolutional Neural Network
############################################

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10):
        super(ResNet, self).__init__()
        self.num_layers = num_layers    # 5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # feature map size = 32x32x16
        self.layers_2n = self.get_layers(block, 16, 16, stride=1)
        # feature map size = 16x16x32
        self.layers_4n = self.get_layers(block, 16, 32, stride=2)
        # feature map size = 8x8x64
        self.layers_6n = self.get_layers(block, 32, 64, stride=2)

        # output layers
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def get_layers(self, block, in_channels, out_channels, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, stride, down_sample)])

        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)


    def forward(self, x):
        x = self.conv1(x) # 32x32
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers_2n(x) # 
        x = self.layers_4n(x)
        x = self.layers_6n(x)

        x = self.avg_pool(x)
        x = self.conv_out(x)
        x = x.view(x.size(0),-1)
        return x

def resnet():
    block = ResidualBlock
    # total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
    model = ResNet(2, block)
    return model


net = resnet()
net = net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)

# 학습된 모델 저장되는 곳
if args.resume is not None:
    checkpoint = torch.load(args.resume)
#     checkpoint = torch.load('./save_model/' + args.resume)
    net.load_state_dict(checkpoint['net'])

############################################
# 3. Define a Loss function and optimizer
############################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

decay_epoch = [40000, 48000]
# decay_epoch = [32000, 48000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, 
								 milestones=decay_epoch, gamma=0.1)

############################################
# 4. Train the network
############################################

def train(epoch, global_steps):
	net.train()

	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		global_steps += 1
		step_lr_scheduler.step()
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		
	acc = 100 * correct / total
	print('\n\ntrain epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
		   epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), acc))

	return global_steps

############################################
# 5. Test the network on the test data
############################################

def test(epoch, best_acc, global_steps):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    best_model = None

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
        epoch, batch_idx, len(testloader), test_loss / (batch_idx + 1), acc))

    if acc > best_acc:
        print('\n==> Saving model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(state, './save_model/best_model.pt')
        if epoch % 5 == 0:
            torch.save(state, './save_model/ckpt_epoch{}.pt'.format(epoch))
        best_model = state
        best_acc = acc

    else:
        print("\naccuracy is worse than best")

    return best_acc, best_model

if __name__ == '__main__':
    best_acc = 0
    epoch = 0
    global_steps = 0
    stop_step = 60000
    best_model = None

    if args.resume is not None:
        test(epoch=0, best_acc=0,global_steps=0)
    else:
        while True:
            epoch += 1
            global_steps = train(epoch, global_steps)
            best_acc, best_model = test(epoch, best_acc, global_steps)
            print('best test accuracy is ', best_acc)
            print('global steps: ',global_steps)

            if global_steps >= stop_step:
                break