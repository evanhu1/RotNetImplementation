import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader  # For custom datasets

import yaml
from data import Data
from resnet import ResNet

import time
import shutil
import argparse

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    total_loss = 0

    for i, (input, target) in enumerate(train_loader):
        B, N, H, W, C = input.shape
        input = input.view(-1, C, H, W)
        target = target.view(-1, 4)
        #print(input.shape, target.shape)

        optimizer.zero_grad()
        output = model(input)
        print(output.shape, target.shape)
        loss = criterion(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()

    return sum(total_loss)

def validate(val_loader, model, criterion):
    model.eval()

    total_loss = 0
    total_accuracy = 0

    for i, (input, target) in enumerate(val_loader):
        B, N, H, W, C = input.shape
        input = input.view(-1, C, H, W)

        outputs = model(input)
        loss = criterion(output, target)
        total_loss += loss
        total_accuracy += (output == target).sum().data[0]

    return total_loss, total_accuracy/(i+1)

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
    if torch.cuda.is_available():
        print(torch.cuda.device_count(), "gpus available")
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        print("no gpus available")
        device = torch.device('cpu')

    n_epochs = config["num_epochs"]
    print("num_epochs: ", n_epochs)

    model = ResNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), momentum=config["momentum"],
                          lr=config["learning_rate"], weight_decay=config["weight_decay"])

    path = "/Users/Gina Wu/Desktop/RotNetImplementation/data/cifar-10-batches-py/"
    #path = "/Users/evanhu/code/RotNetImplementation/data/cifar-10-batches-py/"
    train_dataset = Data(path)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
    val_dataset = Data(path, True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    best_loss = float('inf')

    for epoch in range(n_epochs):
        print("Epoch:{0}".format(epoch))

        #TODO: make your loop which trains and validates. Use the train() func
        total_loss = train(train_loader, model, criterion, optimizer, epoch)
        print("Total Loss:{0}".format(total_loss))

        val_loss, curr_accuracy = validate(val_loader, model, criterion)
        print("Validation Loss:{0} || Accuracy:{1}".format(curr_loss, curr_accuracy))

        #TODO: Save your checkpoint (if current loss is better than current best)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model.state_dict(), True)
        else:
            save_checkpoint(model.state_dict(), False)


if __name__ == "__main__":
    main()
