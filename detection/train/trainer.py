'''Train DeepFace with PyTorch.'''
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from detection.train.dataset import get_datasets


class DeepFaceTrainer():
    def __init__(
        self,
        model: nn.Module,
        pretrained: bool = False,
        pretrained_path: str = None,
        save_ckpt_dir: str = "checkpoint",
        save_ckpt_name: str = "best.pth",
        save_test_acc_path: str = "test_acc.npy",
        trainset_dir: str = "homework_dataset/deep_face/train",
        testset_dir: str = "homework_dataset/deep_face/test",
        batch_size: int = 16,
        num_workers: int = 4,
        learning_rate = 0.001,
        max_epochs: int = 50,        
        device: str = "cpu"
    ) -> None:
        # datasets & dataloader
        self.trainset, self.testset = get_datasets(
            trainset_dir=trainset_dir, testset_dir=testset_dir
        )
        self.trainloader = DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.testloader = DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.len_trainloader = len(self.trainloader)
        self.len_testloader = len(self.testloader)
        
        # model & pretrained
        self.device = device
        self.model = model.to(device)
        if pretrained:
            checkpoint = torch.load(pretrained_path, map_location=device)
            self.model.load_state_dict(state_dict=checkpoint["state_dict"])
            self.best_acc = checkpoint["acc"]
        else:
            self.best_acc = 0

        # record test acc
        self.test_accs = list()
        self.save_test_acc_path = save_test_acc_path
        
        # save checkpoint
        self.save_ckpt_dir = save_ckpt_dir
        self.save_ckpt_path = os.path.join(save_ckpt_dir, save_ckpt_name)
        
        # training tools
        self.max_epochs = max_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=5e-4
        )
        T_max = len(self.trainset) / batch_size * max_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=1e-6
        )
        
    def train_epoch(self, epoch_idx: int):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(self.trainloader, desc=f"Epoch {epoch_idx}", unit="batch")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # input data and targets
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            # forward
            outputs = self.model(inputs)
            outputs: torch.Tensor

            # loss backward
            loss = self.criterion(outputs, targets)
            loss: torch.Tensor
            loss.backward()
            self.optimizer.step()
            
            # statistic
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # progress bar
            progress_bar.set_postfix(loss=loss.item(), accuracy=100.*correct/total)
    
    def test_epoch(self):       
        self.model.eval()
        # ACC
        test_loss = 0
        correct = 0
        total = 0
        # F1 Score
        tp = 0
        fp = 0
        fn = 0
        # AUC
        y_true = []
        y_score = []
        with torch.no_grad():
            progress_bar = tqdm(self.testloader, desc=f"Test", unit="batch")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                # input data and targets
                inputs: torch.Tensor
                targets: torch.Tensor
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # forward
                outputs = self.model(inputs)
                outputs: torch.Tensor

                # loss
                loss = self.criterion(outputs, targets)
                loss: torch.Tensor
                
                # statistic
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # TP, FP, FN
                tp += ((predicted == 1) & (targets == 1)).sum().item()
                fp += ((predicted == 1) & (targets == 0)).sum().item()
                fn += ((predicted == 0) & (targets == 1)).sum().item()

                # For AUC calculation
                y_true.extend(targets.tolist())
                y_score.extend(outputs[:, 1].tolist())

                # progress bar
                progress_bar.set_postfix(loss=loss.item(), accuracy=100.*correct/total)
        
        # save checkpoint
        acc = 100.0 * correct / total
        self.test_accs.append(acc)
        if acc > self.best_acc:
            state = {
                'state_dict': self.model.state_dict(),
                'acc': acc,
            }
            if not os.path.isdir(self.save_ckpt_dir):
                os.mkdir(self.save_ckpt_dir)
            torch.save(state, self.save_ckpt_path)
            self.best_acc = acc
        
        # F1 score
        epsilon = 1e-6
        precision = (tp + epsilon) / (tp + fp + epsilon)
        recall = (tp + epsilon) / (tp + fn + epsilon)
        f1 = 200 * (precision * recall) / (precision + recall)
        
        # AUC
        auc = 100 * roc_auc_score(y_true, y_score)
        
        return acc, f1, auc
            
    def fit(self):
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            self.test_epoch()
            self.scheduler.step()
        self.test_accs = np.array(self.test_accs)
        if not os.path.exists("test_acc"):
            os.makedirs("test_acc")
        np.save(self.save_test_acc_path, self.test_accs)