import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# test_acc
resnet18_acc = np.load("test_acc/resnet18.npy", allow_pickle=True)
resnet34_acc = np.load("test_acc/resnet34.npy", allow_pickle=True)
resnet50_acc = np.load("test_acc/resnet50.npy", allow_pickle=True)
efficientnetb4_acc = np.load("test_acc/resnet18.npy", allow_pickle=True)

# plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(len(resnet18_acc)), y=resnet18_acc, label='ResNet18')
sns.lineplot(x=np.arange(len(resnet34_acc)), y=resnet34_acc, label='ResNet34')
sns.lineplot(x=np.arange(len(resnet50_acc)), y=resnet50_acc, label='ResNet50')
sns.lineplot(x=np.arange(len(efficientnetb4_acc)), y=resnet18_acc, label='EfficientNetb4')
plt.title('Test Set Accuracy')
plt.xlabel('Training Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("test_acc/test_acc.png")