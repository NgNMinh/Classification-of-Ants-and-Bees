import cv2
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dict', '-i', type=str, required=True)
    arg = parser.parse_args()
    return arg


def inference(arg):
    categories = ['ant', 'bee']
    ori_img = cv2.imread(arg.img_dict, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(ori_img)
    img_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = img_transforms(img)
    img = img.unsqueeze(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = resnet18()
    model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    model.load_state_dict(torch.load('model/best.pt',  map_location=torch.device(device)))
    model.eval()

    model.to(device)
    img = img.to(device)
    softmax = torch.nn.Softmax()

    with torch.no_grad():
        out = model(img)
        preds = softmax(out[0])
    predicted_prob, predicted_idx = torch.max(preds, dim=0)

    # Display result
    cv2.imshow("{}: {:0.2f} %".format(categories[predicted_idx], predicted_prob.item() * 100), ori_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    arg = get_arg()
    inference(arg)
