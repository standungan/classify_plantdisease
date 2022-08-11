# load configuration
import config as cfg
import torch
import torch.nn as nn
from data_loader.dataset import dset_imageFolder
from torch.utils.data import DataLoader
from models.lenet import LeNet5
from utilities.metrics import accuracy

if __name__ == '__main__':

    # Load dataset from folders using Dataloaders
    train_dataset, valid_dataset, test_dataset = dset_imageFolder()

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model from models folder
        # implement arx :
            # lenet --- done
            # alexnet
            # vgg
            # resnet
    model = LeNet5(n_classes=cfg.n_classes).to(device)
    print(model)

    
    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimization algorithm
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9)

    # create tensorboard
    for epoch in range(cfg.epochs):

        model.train()
        
        current_loss = 0

        for data, target in train_loader:

            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            # forward pass
            predict, _ = model(data)
            
            #calculate loss
            loss = criterion(predict, target)
            current_loss += loss.item() * data.size(0)

            # backward pass
            loss.backward()

            # update model's parameter
            optimizer.step()
        
        epoch_loss = current_loss / len(train_loader.dataset)
        print("Done")

        break

    # create training loop

        # training process

        # validation process if valid dir isn't empty

        # save model with validation > save_threshold

