# DeepFake-Face-Detection

## 代码框架

```markdown=
DeepFake-Face-Detection
├── checkpoint             # 预训练文件
│   ├── efficientnetb4.pth 
│   ├── resnet18.pth       # ResNet18
│   ├── resnet34.pth       # ResNet34
│   ├── resnet50.pth       # ResNet50
│   ├── efficientnetb4.pth # EfficientNetB4
│   ├── svm_color.pkl      # SVM(颜色直方图)
│   ├── svm_hog.pkl        # SVM(方向梯度直方图)
│   ├── svm_lbp.pkl        # SVM(局部二值模式)
│   └── svm_resize.pkl     # SVM(缩放)
├── detection              # 核心代码
│   ├── model
│   │   ├── resnet.py      # ResNet网络
│   │   ├── efficientb4.py # EfficientB4网络
│   │   └── svm.py         # SVM
│   └── train
│       ├── dataset.py     # 数据处理
│       └── trainer.py     # DNN训练器
├── homework_dataset       # 全部数据集
│   ├── deep_face          # 本实验采用的数据集（包括train和test）
│   ├── deep_text
│   └── deep_voice
├── log                    # 训练日志
│   ├── resnet18.log       # ResNet18训练日志（50轮）
│   ├── resnet34.log       # ResNet34训练日志（50轮）
│   ├── resnet50.log       # ResNet50训练日志（50轮）
│   └── efficientnetb4.log # EfficientNetB4训练日志（50轮）
├── test_acc               # 训练时测试集准确率变化
│   ├── resnet18.npy
│   ├── resnet34.npy
│   ├── resnet50.npy
│   ├── efficientnetb4.npy # EfficientNetB4训练日志（50轮）
│   └── test_acc.png       # 测试集准确率变化曲线
├── plot.py                # 作图
├── test1.py               # 传统方法测试
├── test2.py               # DNN测试
├── train1.py              # 传统方法训练
├── train2.py              # DNN训练
└── README.md              # 说明文件
```

## 环境

* 在基础环境上需要安装 ``scikit-image`` 包

## 训练与测试

```bash
# SVM(train)
python train1.py

# SVM(test)
python test1.py

# DNN(train)
CUDA_VISIBLE_DEVICES=0 nohup python train2.py > log/resnet18.log
CUDA_VISIBLE_DEVICES=1 nohup python train2.py > log/resnet34.log
CUDA_VISIBLE_DEVICES=2 nohup python train2.py > log/resnet50.log
CUDA_VISIBLE_DEVICES=3 nohup python train2.py > log/resnet.log

# DNN(test)
CUDA_VISIBLE_DEVICES=0 python test2.py
```