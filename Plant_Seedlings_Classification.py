import torch.nn as nn
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torch.autograd import Variable
import math
DEVICE = 6

#======================= model =======================#

class CNN(nn.Module):                   # 官方步驟，引入nn.Module
    def __init__(self, num_classes):    # 官方步驟
        super(CNN, self).__init__()     # 官方步驟，繼承nn.Module的__init__功能
        
        # 建立自己的layers:
        self.cnn_layers_1 = nn.Sequential(    # cnn_layers為自訂名稱，cnn layers
            
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*112*112
            
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128*56*56
            )
        
        self.cnn_layers_2 = nn.Sequential(
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 512*7*7
            )
        

        self.dense_layers = nn.Sequential(  # fully connected layers
        
            nn.Linear(in_features=512* 7* 7, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=num_classes)
            
            )

    
    def forward(self, x):                  # 官方步驟，fit入資料x，堆疊layers
        
        x = self.cnn_layers_1(x)             # 資料x放入cnn_layers
        x = self.cnn_layers_2(x)
        x = x.view(x.size(0), -1)          # flatten
        output = self.dense_layers(x)           # flatten後放入fully connected layers，並output類別組
        
        return output



#======================= dataset =======================#

path = '/home/tingyu/User/Plant_Seedlings_Classification/train'
category = os.listdir(path) # all folders name under the path
train_img = []       # train image path
train_label = []     # train image label index
val_img = []         # validation image path
val_label = []       # validation image label indax
test_img = []         # test image path
test_label = []       # test image label indax
label_num = 0

# dataset prepare
for classes in category:
    img_path = path + '/' + classes
    classes_img = os.listdir(img_path)  # all files name under the path
    for i in range(len(classes_img)):
        if len(classes_img)-i > 50:     # 資料倒數50筆，前30筆存入validation的list，後20筆存入test的list
            train_img.append(img_path + '/' + classes_img[i])
            train_label.append(label_num)
        elif len(classes_img)-i >20:
            val_img.append(img_path + '/' + classes_img[i])
            val_label.append(label_num)
        else:
            test_img.append(img_path + '/' + classes_img[i])
            test_label.append(label_num)
    label_num = label_num + 1



#======================= optimizer and loss function =======================#

CLASSES = 12   # number of classes
BATCH_SIZE = 64

model = CNN(num_classes=CLASSES).cuda(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()




#======================= dataloader and training =======================#

data_transform = transforms.Compose([   # preprocesing
    transforms.RandomResizedCrop(224),  # resize image to 224*224
    transforms.RandomRotation(60),      # rotate image
    transforms.ToTensor(),              # to tensor, shape = (3,224,224) and /255
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]) # normalize

def default_loader(path):
    image =  Image.open(path).convert('RGB') # avoid RGBA
    img_tensor = data_transform(image)
    return img_tensor

class trainset(Dataset):    # training data's dataset

    def __init__(self, loader=default_loader):      # define path of image and label
        self.images = train_img
        self.target = train_label
        self.loader = loader

    def __getitem__(self, index):
        img = self.images[index]
        train_x = self.loader(img)      # do transforms
        train_y = self.target[index]
        return train_x, train_y

    def __len__(self):
        return len(self.images)
    

class validationset(Dataset):    # validation data's dataset

    def __init__(self, loader=default_loader): 
        self.val_images = val_img
        self.val_target = val_label
        self.loader = loader

    def __getitem__(self, index):
        img = self.val_images[index]
        val_x = self.loader(img)
        val_y = self.val_target[index]
        return val_x, val_y

    def __len__(self):
        return len(self.val_images)


train_data  = trainset()
val_data  = validationset()
data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


# for curve painting
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

# initial values
count = 0
best_acc = 0.0

# set earlystopping step number and epoch number
earlystopping = 5
num_epochs = 100

# start training
for epoch in range(num_epochs): 
    print(f'Epoch: {epoch + 1}/{num_epochs}')
    print('-' * 13)
    training_loss = 0.0
    training_corrects = 0.0
    for i, (inputs, labels) in enumerate(data_loader, 0):
#         get values from dataloader
        inputs = Variable(inputs.cuda(DEVICE))
        labels = Variable(labels.cuda(DEVICE))
        
#         learning
        optimizer.zero_grad()   # initial gradient
        
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = loss_function(outputs, labels)  # loss 
        loss.backward()         # Backpropagation
        optimizer.step()        # optimize (leanring rate 0.0001)
        
#         sum loss and acc
        training_loss += loss.data * inputs.size(0)
        training_corrects += sum(preds == labels.data)
        
#     validations' loss and acc    
    val_loss = 0.0
    val_acc = 0.0
    
    for j, (val_inputs, val_labels) in enumerate(val_data_loader, 0):
        val_inputs = Variable(val_inputs.cuda(DEVICE))
        val_labels = Variable(val_labels.cuda(DEVICE))
        val_output = model(val_inputs)
        _, val_preds = torch.max(val_output.data, 1)
        v_loss = loss_function(val_output, val_labels)

        val_loss += v_loss.data * val_inputs.size(0)
        val_acc += sum(val_preds == val_labels.data)
        
#     calculate loss and acc

    training_loss = training_loss/len(train_data)
    training_acc = training_corrects/len(train_data)
    val_loss = val_loss/len(val_data)
    val_acc = val_acc/len(val_data)

#     result append to list    
    
    train_loss_list.append(float(training_loss))
    train_acc_list.append(float(training_acc))
    val_loss_list.append(float(val_loss))
    val_acc_list.append(float(val_acc))
           
    print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\tval_loss: {val_loss:.4f}\tval_acc:{val_acc:.4f}\n')
    
#     earlystopping

    if(best_acc < val_acc): 
        count = 0
        best_acc = val_acc
        torch.save(model.state_dict(),'/home/tingyu/Jupyter/models/Plant_Seedlings_Classification_model.pkl')
    else: 
        count = count + 1

    if(count == earlystopping):
        print("Early Stopping")
        break


#======================= training curve =======================#

import matplotlib.pyplot as plt

plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training loss', 'Validation loss'])
plt.savefig('curve_loss.png')
plt.figure()

plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title('Acc')
plt.ylabel('Acc')
plt.xlabel('Epochs')
plt.legend(['Training acc', 'Validation acc'])
plt.savefig('curve_acc.png')
plt.show()


#======================= test data =======================#

import pandas as pd

dataset_root = '/home/tingyu/User/Plant_Seedlings_Classification/test'
classes = os.listdir('/home/tingyu/User/Plant_Seedlings_Classification/train')

model = CNN(num_classes=CLASSES).cuda(DEVICE)
model.load_state_dict(torch.load('/home/tingyu/Jupyter/models/Plant_Seedlings_Classification_model.pkl'))
model.eval()

sample_submission = pd.read_csv('/home/tingyu/User/Plant_Seedlings_Classification/sample_submission.csv')
submission = sample_submission.copy()
for i, filename in enumerate(sample_submission['file']):
    image = Image.open(dataset_root+'/'+filename).convert('RGB')
    image = data_transform(image).unsqueeze(0)
    inputs = Variable(image.cuda(DEVICE))
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    print(classes[preds[0]])
    submission['species'][i] = classes[preds[0]]

submission.to_csv('/home/tingyu/User/Plant_Seedlings_Classification/submission.csv', index=False)

