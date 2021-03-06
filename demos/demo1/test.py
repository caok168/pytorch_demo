import torch
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from demos.demo1.base_resnet import resnet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = resnet()

model_path = os.path.abspath(os.path.join(os.getcwd(), "models/resnet/159.pth"))

net.load_state_dict(torch.load(model_path))

im_list = glob.glob("../../datasets/cifar-10/test/*/*")

np.random.shuffle(im_list)
net.to(device)

label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

test_transform = transforms.Compose([
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

for im_path in im_list:
    net.eval()
    im_data = Image.open(im_path)

    inputs = test_transform(im_data)
    inputs = torch.unsqueeze(inputs, dim=0)

    inputs = inputs.to(device)
    outputs = net.forward(inputs)

    _, pred = torch.max(outputs.data, dim=1)
    print(label_name[pred.cpu().numpy()[0]])

    img = np.asarray(im_data)
    img = img[:, :, [1, 2, 0]]

    img = cv2.resize(img, (300, 300))
    cv2.imshow("im", img)
    cv2.waitKey()
