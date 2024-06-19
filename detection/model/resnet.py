from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck



def ResNet18():
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], num_classes=2)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=2)


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=2)


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=2)