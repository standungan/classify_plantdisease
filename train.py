# load configuration
import config as cfg
import torch
import torch.nn as nn
from tqdm import tqdm
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model from models folder
    model = LeNet5(n_classes=cfg.n_classes).to(device)
    print(model)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimization algorithm
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9)

    # create tensorboard -> coming soon
    
    # Training Loop
    for epoch in range(cfg.epochs):
        print("Epoch : ", epoch)
        
        # train
        print("Training...")
        model.train()  
        current_loss = 0
        current_acc = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)): # use tqdm progress bar
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            # forward pass
            predict, _ = model(data)
            # calculate loss
            loss = criterion(predict, target)
            current_loss += loss.item()
            # backward pass
            loss.backward()
            # update model's parameter
            optimizer.step()
        
        train_loss = current_loss / len(train_loader)
        print("Training Loss : {:.6f}".format(train_loss))
        # validation
        print("Validation...")
        model.eval()
        current_loss = 0
        current_acc = 0
        for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
            
            data = data.to(device)
            target = target.to(device)

            # forward pass
            predict, _ = model(data)
            loss = criterion(predict, target)
            current_loss += loss.item() / data.size(0)

        valid_loss = current_loss / len(valid_loader)
        print("Validation Loss : {:.6f}".format(valid_loss))

    print("Done...")