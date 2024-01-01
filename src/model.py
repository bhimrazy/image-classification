import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DenseNetModel, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = DenseNetModel(num_classes=5)

    summary(model, (3, 224, 224))
