import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# 定义超参
batch_size = 16
num_epochs = 300  # 因为要让他过拟合，epoch数要多一些
# 加载数据
dataset = 'emotion'
train_directory = os.path.join(dataset, 'train')
image_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3), #将图片转化为灰度图，3表示r=g=b
        transforms.RandomResizedCrop(size=45, scale=(0.8, 1.0)),
        transforms.CenterCrop(size=40), #对图片进行中心切割，得到正方形图片
        transforms.ToTensor(), #转化为tensor
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  #给定均值，方差进行正则化
    ])
}

data={
'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train'])
}
train_data_size = len(data['train'])
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
print(data['train'].imgs)


resnet18 = models.resnet18(pretrained=True)
fc_inputs = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),  #设置网络的全连接层
    nn.ReLU(),  #修正线性单元
    nn.Dropout(0.4),
    nn.Linear(256, 99),
    nn.LogSoftmax(dim=1)  #输出概率分布
)

# 定义损失函数和优化器
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet18.parameters())

def train(model,loss_function,optimizer,epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        epoch_start = time.time()  # 返回当前的时间的时间戮
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()
        train_loss = 0.0
        train_acc = 0.0  # 训练精度

        for i, (inputs, labels) in enumerate(train_data):
            optimizer.zero_grad()  #把梯度置零,也就是把loss关于weight的导数变成0
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()  #反向传递，计算梯度
            optimizer.step()   #根据梯度更新神经网络
            train_loss += loss.item() * inputs.size(0)
            #print(labels)
            ret, predictions = torch.max(outputs.data, 1)   #返回最大值及其索引
            #print(ret,predictions)
            correct_counts = predictions.eq(labels.data.view_as(predictions))  #验证是否预测成功
            print(correct_counts)
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            #print(i,'train_acc=',train_acc)


        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size


        history.append([avg_train_loss,  avg_train_acc])

        if best_acc < avg_train_acc:
            best_acc = avg_train_acc
            best_epoch = epoch + 1

        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, train_loss, avg_train_acc * 100, avg_train_loss, avg_train_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        model_path = 'models_resnet18_ep' + str(num_epochs)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model, model_path + '/' + dataset + '_model_' + str(epoch + 1) + '_a'+str(round(avg_train_acc,3))+'.pt')
    return model, history, best_acc, best_epoch




trained_model, history, best_acc, best_epoch = train(resnet18, loss_func, optimizer, num_epochs)

model_path = 'models_resnet18_ep' + str(num_epochs)
torch.save(history, model_path + '/' + dataset + '_history.pt')