import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
import os
from models import *
from models.efficientNet import EfficientNet
import torch

# torch.autograd.set_detect_anomaly(True)
# torch.cuda.empty_cache()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Using {} device".format(device))

cut_size = 44
epoch = 10
bs = 4
lr = 0.0001
shape = (112, 112)
num_classes = 7

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(shape),
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(shape),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#
# training_data = datasets.FashionMNIST(
#     root="data2",
#     train=True,
#     download=True,
#     transform=transform_test
# )
#
# test_data = datasets.FashionMNIST(
#     root="data2",
#     train=False,
#     download=True,
#     transform=transform_test
# )
trainset = datasets.ImageFolder('data/total/train', transform=transform_train)
# trainset2 = torch.utils.data.Subset(training_data, torch.randperm(len(training_data))[:300])
# # trainset2 = torch.utils.data.Subset(trainset,torch.randperm(len(trainset)))
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=0)

testset = datasets.ImageFolder('data/total/test', transform=transform_test)
# # testset2 = torch.utils.data.Subset(testset,torch.randperm(len(testset))[:100] )
# testset2 = torch.utils.data.Subset(test_data, torch.randperm(len(test_data))[:100])
testloader = torch.utils.data.DataLoader(testset, batch_size=bs)

net = EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0, dropout_ratio=0.2,
                   se_ratio=0.25, stochastic_depth=True).to(device)

loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)
# optimizer = optim.Adam(net.parameters(),lr = lr )

scheduler = lr_sch.ExponentialLR(optimizer, gamma=0.9)


def train(dataloader, net, loss_fn, opt):
    net.train()
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # reset all calculated gradiant
        opt.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, net, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # freeze dropout layers (using for testing state)
    net.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = net(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"size {size} num of bathces : {num_batches}")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#train loop
for t in range(epoch):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(trainloader, net, loss_fn, optimizer)
    test(testloader, net, loss_fn)
    # decay learning rate
    scheduler.step()
print("Training Done!")

# saving model in the outModels folder
models_path = './outModels'
dirListing = os.listdir(models_path)
model_number = len(dirListing)
torch.save(net.state_dict(), "./outModels/model_{}.pth".format(model_number))
print("Saved PyTorch Model State to model_{}.pth".format(model_number))

# network accuracy on each class

classes = trainset.classes
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
                total += 1
            total_pred[classes[label]] += 1
    print("total accuracy : {}".format(total / len(testset)))
# print accuracy
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                         accuracy))
