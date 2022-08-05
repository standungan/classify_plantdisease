# image dataset location
train_dir = 'D:\Dataset\PlantDisease\Train'
valid_dir = 'D:\Dataset\PlantDisease\Validation'
test_dir = 'D:\Dataset\PlantDisease\Test'

# dataset parameters
img_size = 128

# dataloader parameters
num_workers = 2
shuffle = True
batch_size = 4

# computing device
device = "cpu"

# training parameters
model_arx = ""
lossFunc = ""
optim = ""
epochs = 5


# validation parameters
save_threshold = 0.85

# save model 
filename = "" #modelName_time_acc_loss.pth