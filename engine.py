import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm


class Evaluator:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def run(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc="Validating....")
            for videos, labels in pbar:
                videos, labels = videos.to(self.device), labels.to(self.device)

                with torch.amp.autocast('cuda'):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}',
                                  'acc': f'{correct / total:.4f}'})

        return total_loss / len(self.dataloader), correct / total


class TrainOneEpoch:
    def __init__(self, model, dataloader, optimizer, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()

    def run(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(self.dataloader, desc="Training TSM...")
        for videos, labels in pbar:
            videos, labels = videos.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss/len(self.dataloader), correct/total


def evaluate(model, dataloader, criterion, device):
    return Evaluator(model, dataloader, criterion, device).run()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    return TrainOneEpoch(model, dataloader, optimizer, criterion, device).run()
