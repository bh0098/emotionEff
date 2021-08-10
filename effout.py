import torch
from torchvision import datasets,transforms
from PIL import Image
from models.efficientNet import EfficientNet

device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cpu"
model= EfficientNet(num_classes=7, width_coef=1.0, depth_coef=1.0, scale=1.0, dropout_ratio=0.2,
                     se_ratio=0.25, stochastic_depth=True).to(device)


'''enter trained model path'''
model_path = './outModels/model_1.pth'
model.load_state_dict(torch.load(model_path))

#batch size
bs = 4

shape = (112, 112)
data_transforms = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(shape),
        transforms.ToTensor()
        ])
print("Using {} device".format(device))


'''enter image path'''
img = Image.open('./data/total/test/fear/PrivateTest_85557728.jpg')
img = data_transforms(img)

'''in the eval mode dropout layeres are freeze'''
model.eval()
testset = datasets.ImageFolder('data/total/test', transform=data_transforms)
# # testset2 = torch.utils.data.Subset(testset,torch.randperm(len(testset))[:100] )
# testset2 = torch.utils.data.Subset(test_data, torch.randperm(len(test_data))[:100])
testloader = torch.utils.data.DataLoader(testset, batch_size=bs)

'''you can enter class names automaticly by loading dataset'''
class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

with torch.no_grad():
        '''calculate network output for input image'''
        for x, y in testloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)

                # print(torch.unsqueeze(img, 0).shape)
                # outputs = model(torch.unsqueeze(img, 0))

                # print(outputs.shape)

                # the class with the highest energy is what we choose as prediction
                cnt = 0
                for out in  outputs :
                        cnt+=1
                        _, predicted= torch.max(out.data, 0)
                        class_label = predicted.item()
                        print("image {} class  : ".format(cnt),class_names[class_label])
                break



