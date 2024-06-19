from detection import DeepFaceTrainer, ResNet18, ResNet34, ResNet50, EfficientNetB4


def test_resnet18():
    model = ResNet18()
    trainer = DeepFaceTrainer(
        model, 
        pretrained=True,
        pretrained_path="checkpoint/resnet18.pth",
        device="cuda", 
        batch_size=256
    )
    acc, f1, auc = trainer.test_epoch()
    message = f"ResNet18 Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)


def test_resnet34():
    model = ResNet34()
    trainer = DeepFaceTrainer(
        model, 
        pretrained=True,
        pretrained_path="checkpoint/resnet34.pth",
        device="cuda", 
        batch_size=256
    )
    acc, f1, auc = trainer.test_epoch()
    message = f"ResNet34 Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)
    

def test_resnet50():
    model = ResNet50()
    trainer = DeepFaceTrainer(
        model, 
        pretrained=True,
        pretrained_path="checkpoint/resnet50.pth",
        device="cuda", 
        batch_size=256
    )
    acc, f1, auc = trainer.test_epoch()
    message = f"ResNet50 Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)
    

def test_efficientnetb4():
    model = EfficientNetB4()
    trainer = DeepFaceTrainer(
        model, 
        pretrained=True,
        pretrained_path="checkpoint/efficientnetb4.pth",
        device="cuda", 
        batch_size=16
    )
    acc, f1, auc = trainer.test_epoch()
    message = f"EfficientNetB4 Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)
    
    
if __name__ == "__main__":
    test_resnet18()
    test_resnet34()
    test_resnet50()
    test_efficientnetb4()