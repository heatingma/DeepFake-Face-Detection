from detection import DeepFaceTrainer, EfficientNetB4
from detection import ResNet18, ResNet34, ResNet50


def train_resnet18():
    model = ResNet18()
    trainer = DeepFaceTrainer(
        model=model,
        save_ckpt_name="resnet18.pth",
        save_test_acc_path="test_acc/resnet18.npy",
        device="cuda", 
        batch_size=256,
    )
    trainer.fit()
    

def train_resnet34():
    model = ResNet34()
    trainer = DeepFaceTrainer(
        model=model,
        save_ckpt_name="resnet34.pth",
        save_test_acc_path="test_acc/resnet34.npy",
        device="cuda", 
        batch_size=256
    )
    trainer.fit()


def train_resnet50():
    model = ResNet50()
    trainer = DeepFaceTrainer(
        model=model,
        save_ckpt_name="resnet50.pth",
        save_test_acc_path="test_acc/resnet50.npy",
        device="cuda", 
        batch_size=128
    )
    trainer.fit()
    

def train_efficientnetb4():
    model = EfficientNetB4()
    trainer = DeepFaceTrainer(
        model=model,
        save_ckpt_name="efficientnetb4.pth",
        save_test_acc_path="test_acc/efficientnetb4.npy",
        device="cuda", 
        batch_size=32
    )
    trainer.fit()


if __name__ == "__main__":
    train_resnet18()
    train_resnet34()
    train_resnet50()
    train_efficientnetb4()