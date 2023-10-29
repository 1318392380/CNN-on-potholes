import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import os
from sklearn.metrics import confusion_matrix
import numpy as np

num_epochs = 5
batch_size = 16
learning_rate = 0.05

# 使用预训练的VGG模型
model = models.vgg16(pretrained=True)

# 替换最后一层全连接层以适应你的分类任务（这里是二分类）
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

# 将模型移动到GPU
#model = model.cuda()

# 图像预处理
min_val = 0
max_val = 255
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # transforms.Normalize(min_val, max_val)
])

# 加载训练和测试数据
train_dataset = datasets.ImageFolder('train_data_path', transform=transform)
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

# 重新加载测试集
#test_dataset = datasets.ImageFolder('test_data_path', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建损失函数和优化器，只优化最后一层
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)

# 检查是否已经训练并加载模型
if os.path.isfile('pkl/cnn.pkl'):
    model.load_state_dict(torch.load('pkl/cnn.pkl'))
else:
    # 训练模型
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            #images = images.cuda();labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data))

    # 保存模型
    torch.save(model.state_dict(), 'pkl/cnn.pkl')

# 计算混淆矩阵
confusion_matrix = np.zeros((2, 2))

# 测试和评估模型
model.eval()
total = 0
correct = 0
for images, labels in test_loader:
    images = Variable(images)
    #images = images.cuda();labels = labels.cuda()

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()

    for i in range(len(predicted)):
        confusion_matrix[labels[i]][predicted[i]] += 1

print('Test Accuracy on Test Images = %f %%' % (100 * correct / total))

print("Confusion Matrix:")
print(confusion_matrix)
