import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    model = Classifier(2560, 512, 2)
    # torch.save(model.state_dict(), 'model.pth')
    # model.load_state_dict(torch.load('classifier.pth', map_location='cpu'))
    input_tensor = torch.randn(4, 2560)
    output = model(input_tensor)
    print(output)
