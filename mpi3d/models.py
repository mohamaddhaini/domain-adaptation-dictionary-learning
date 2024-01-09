from torch import nn
from torchvision import models
from torchvision.models import resnet50


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(3, 10, kernel_size=5),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(10, 20, kernel_size=5),
        #     nn.MaxPool2d(2),
        #     nn.Dropout2d(),
        # )
        self.feature_extractor=resnet50(weights="IMAGENET1K_V2") 
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),nn.Linear(512,50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 3),nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

class Resnet18Fc(nn.Module):
    def __init__(self):
        super(Resnet18Fc, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class Model_Regression_mp(nn.Module):
    def __init__(self):
        super(Model_Regression_mp,self).__init__()
        self.feature_extractor = Resnet18Fc()
        self.classifier  = nn.Linear(512, 2)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)
        self.classifier  = nn.Sequential(self.classifier,  nn.Sigmoid())

    def forward(self,x):
        feature = self.feature_extractor(x)
        feature = feature.view(x.shape[0], -1)
        outC= self.classifier(feature)
        return outC

class Model_Regression_dsprites(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.feature_extractor = Resnet18Fc()
        self.classifier  = nn.Linear(512, 3)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)
        self.classifier  = nn.Sequential(self.classifier,  nn.Sigmoid())

    def forward(self,x):
        feature = self.feature_extractor(x)
        feature = feature.view(x.shape[0], -1)
        outC= self.classifier(feature)
        return outC
