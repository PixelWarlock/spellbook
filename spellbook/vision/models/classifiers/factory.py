import torch

class ClassifierFactory:

    @classmethod
    def get(cls, key:str, till:int=None, **kwargs)->torch.nn.Module:
        all_classifiers = {
            'efficientnet_b0':cls.get_efficientnet_b0,
            'efficientnet_b1':cls.get_efficientnet_b1,
            'dinov2_s':cls.get_dinov2_s
        }
        model = all_classifiers[key](**kwargs)
        if till is not None:
            return model[:till]
        else:
            return model

    @staticmethod
    def get_efficientnet_b0(pretrained:bool)->torch.nn.Module:
        # https://github.com/rwightman/gen-efficientnet-pytorch/issues/13#issuecomment-550009556
        model = torch.hub.load(
            'rwightman/gen-efficientnet-pytorch',
            'efficientnet_b0',
            pretrained=pretrained,
        )
        return torch.nn.Sequential(*list(model.as_sequential()))

    @staticmethod
    def get_efficientnet_b1(pretrained:bool)->torch.nn.Module:
        model = torch.hub.load(
            'rwightman/gen-efficientnet-pytorch',
            'efficientnet_b1',
            pretrained=pretrained,
        )
        return torch.nn.Sequential(*list(model.as_sequential()))

    @staticmethod
    def get_dinov2_s()->torch.nn.Module:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        return torch.nn.Sequential(*list(model.as_sequential()))
