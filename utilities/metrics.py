import torch

def accuracy(target, prediction):

    with torch.no_grad():

        pred = torch.argmax(prediction, dim=1)
        assert pred.shape[0] == len(target)
    
        correct = 0
        correct += torch.sum(pred == target).item()
    
        acc = correct / len(target)
    
    return acc