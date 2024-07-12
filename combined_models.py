import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_CNLK(nn.Module):
    """
    The CNLK model from "Uncovering doubly charged scalars with dominant three-body decays using machine learning"
    """
    def __init__(self, inchannels=3):
        super(CNN_CNLK, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(inchannels, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(4),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(5),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(5),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.AvgPool2d(5),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(5),
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * 32 * 2, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 600),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(66, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 600),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(1200, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 2),
        )

    def forward(self, x, y):
        x = self.layer1(x.unsqueeze(1)).squeeze()
        x = self.layer2(x)
        x1 = self.layer3(x)
        x1 = self.layer4(x1)
        x2 = self.layer5(x)
        x2 = self.layer6(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        y = self.fc2(y)
        x = torch.cat((x, y), dim=1)
        x = self.fc3(x)
        
        return x

class MultiRepresentationClassifier(nn.Module):
    """
    A classifier that combines multiple pre-trained models.

    The architecture consists of multiple pre-trained models followed by fully connected layers for classification.

    Takes a List of the model instances. 
    If only the combining MLP has to be trained deactivate respective gradients beforehand.
    Pass fine_tune_models=True to reactivate gradients for all models.
    """
    def __init__(self, *models, output_dim=2, hidden_dim=700, dropout=0.3, num_classes=2, fine_tune_models=True):
        super(MultiRepresentationClassifier, self).__init__()
        self.models = nn.ModuleList(models)
        for model in self.models:
            model.requires_grad_(fine_tune_models)
        self.num_models = len(models)
        self.output = nn.Sequential(
            nn.Linear(output_dim * self.num_models, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, *inputs):

        out = []
        for model, input_data in zip(self.models, inputs):
            x = torch.nn.functional.softmax(model(input_data), dim=1)
            x = model(input_data)
            out.append(x)
        out = torch.cat(out, dim=1)
        out = self.output(out)
        return out

