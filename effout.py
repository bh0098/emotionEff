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




shape = (112, 112)
data_transforms = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(shape),
        transforms.ToTensor()
        ])
print("Using {} device".format(device))


'''enter image path'''
img = Image.open('./data/archive/test/fear/PrivateTest_85557728.jpg')
img = data_transforms(img)
print(img.shape)

'''in the eval mode dropout layeres are freeze'''
model.eval()

'''you can enter class names automaticly by loading dataset'''
class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

with torch.no_grad():
        '''calculate network output for input image'''
        outputs = model(torch.unsqueeze(img, 0))

        print(outputs)

        # the class with the highest energy is what we choose as prediction
        _, predicted= torch.max(outputs.data, 1)
        class_label = predicted[0].item()
        print(class_names[class_label])



