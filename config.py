train_dir = '..\..\Dataset\PlantDisease\Train'
valid_dir = '..\..\Dataset\PlantDisease\Validation'
test_dir = '..\..\Dataset\PlantDisease\Test'

img_transform = ""
# { train, test}
# resize, flip. crop, normalize

device = "cpu"

# train params
model_arx = ""
lossFunc = ""
optim = ""
epochs = 5
batch_size = 4

# validation
save_threshold = 0.85

# save model 
filename = "" #modelName_time_acc_loss.pth