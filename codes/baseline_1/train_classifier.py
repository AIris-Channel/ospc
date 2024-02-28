import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


def train_model(train_loader, model, loss_fn, optimizer, epochs):
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            y = y.unsqueeze(-1).float()
            pred = model(X)
            loss = loss_fn(pred,y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = torch.sigmoid(pred).round()
            correct_predictions += (predicted_labels == y).sum().item()
            total_predictions += y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Check if this is the best model (based on loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save the model
            torch.save(model.state_dict(), 'classifier.pth')

    print("Training complete. Best model saved to 'classifier.pth'")
X_train = torch.load('./X_train.pth')
y_train = torch.load('./y_train.pth')
# 创建数据加载器
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)



# 初始化模型、损失函数和优化器
model = SimpleMLP(input_size=8, hidden_size=16, output_size=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# 训练模型
train_model(train_loader, model, loss_fn, optimizer, epochs=100)

