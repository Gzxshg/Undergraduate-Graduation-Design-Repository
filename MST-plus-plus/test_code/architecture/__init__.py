import torch
from .MST_Plus_Plus import MST_Plus_Plus


def model_generator(method, pretrained_model_path=None):

    model = MST_Plus_Plus().cuda()
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        print(checkpoint['state_dict'].keys())
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model
