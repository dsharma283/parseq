from torchvision import transforms
from torchvision import models
from torch import optim
import os, numpy as np
import argparse, tqdm
import torch.nn as nn
import torch, copy


'''
    Author: Devesh Sharma
    Email: sharma.98@iitj.ac.in

    - Data directory organization
        data-path/train/<language>/<images>
        data-path/test/<language>/<images>
        data-path/val/<language>/<images>
    - Derive number of class from <language> directory count.
'''


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def build_transform(imsize, norm):
    transform = transforms.Compose([
                                    #transforms.ToPILImage(),
                                    transforms.Resize((imsize, imsize)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm[0], norm[1]),
                                    ])
    return transform


def build_resnet(nclass):
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    for param in model.parameters():
        param.require_grad = False
    #print(model)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(512, 128),
                             nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(128, nclass))
    return model


def resume_model(mod_path):
    if os.path.exists(mod_path) is False:
        return None
    entry = torch.load(mod_path)
    return entry


def load_model(mod_path):
    entry = resume_model(mod_path)

    nclass = entry['classmap']['nclass']
    classes = entry['classmap']['classes']
    stats = entry['classmap']['stats']

    model = build_resnet(nclass)
    model = model.to(device)
    model.load_state_dict(entry['model_state_dict'])

    epoch = entry['epoch']
    print(f'Resumed {mod_path} from epoch {epoch} with addressable classes {classes}')
    xform = build_transform(224, stats)

    return model, entry['classmap'], xform


def predict_one(img, model, classes, nograd=False):
    if nograd:
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).detach().to('cpu').numpy().item()
    else:
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True).detach().to('cpu').numpy().item()
    return classes[pred]


def predict_crops(crops, model, classmap, xform):
    preds = {}
    model.eval()

    if crops is None:
        return preds
    with torch.no_grad():
        for idx, crop in enumerate(crops):
            crop = xform(crop).unsqueeze(0).to(device)
            pred = predict_one(crop, model, classmap['classes'])
            #print(idx, pred)
            if pred not in preds:
                preds[pred] = [idx]
            else:
                preds[pred].append(idx)
            #preds.append((idx, pred))
    return preds


def identify_script(mod_path, crops):
    model, classmap, xform = load_model(mod_path)
    return predict_crops(crops, model, classmap, xform)

'''
def start_main():
    mod_path = f'../../IndicScriptID/results/checkpoints/resnet18-4-checkpoint.pt' 
    identify_script(mod_path, None)


if __name__ == '__main__':
    start_main()
'''
