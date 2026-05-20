import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATA_ROOT,
    EPOCHS,
    FRAME_STRIDE,
    IMG_SIZE,
    LEARNING_RATE,
    NUM_FRAMES,
    NUM_WORKERS,
    SEED,
    VAL_RATIO,
    WEIGHT_DECAY,
)
from data.dataset import HMDB51Dataset, collate_fn
from engine import Evaluator, TrainOneEpoch
from models.tsm import TSM_Network
from utils import plot_history


class TSMTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_one_epoch = None
        self.evaluator = None
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': [],
                        'train_time': []}
        self.best_acc = 0.0

    def setup_data(self):
        self.train_dataset = HMDB51Dataset(
            root=DATA_ROOT,
            split="train",
            num_frames=NUM_FRAMES,
            frame_stride=FRAME_STRIDE,
            image_size=IMG_SIZE,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )

        self.val_dataset = HMDB51Dataset(
            root=DATA_ROOT,
            split="val",
            num_frames=NUM_FRAMES,
            frame_stride=FRAME_STRIDE,
            image_size=IMG_SIZE,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )

        self.test_dataset = HMDB51Dataset(
            root=DATA_ROOT,
            split="test",
            num_frames=NUM_FRAMES,
            frame_stride=FRAME_STRIDE,
            image_size=IMG_SIZE,
            seed=SEED,
        )

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=NUM_WORKERS,
                                       collate_fn=collate_fn)

        self.val_loader = DataLoader(self.val_dataset,
                                     BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=NUM_WORKERS,
                                     collate_fn=collate_fn)

        self.test_loader = DataLoader(self.test_dataset,
                                      BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=NUM_WORKERS,
                                      collate_fn=collate_fn)

        print(f"Train clips: {len(self.train_dataset)}")
        print(f"Train Class count: {len(self.train_dataset.classes)}")
        print(f"Val clips: {len(self.val_dataset)}")
        print(f"Val Class count: {len(self.val_dataset.classes)}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def setup_model(self):
        self.model = TSM_Network(
            num_classes=len(self.train_dataset.classes),
            n_segment=NUM_FRAMES,
        ).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"   Model created")
        print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   Architecture: TSM ResNet-50")

    def setup_training(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=EPOCHS,
        )
        self.train_one_epoch = TrainOneEpoch(
            self.model,
            self.train_loader,
            self.optimizer,
            self.criterion,
            self.device,
        )
        self.evaluator = Evaluator(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
        )

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Training setup:")
        print(f"   Epochs: {EPOCHS}")
        print(f"   Learning rate: {LEARNING_RATE}")
        print(f"   Trainable parameters: {trainable_params:,} (ALL params)")
        print(f"   Optimizer: Adam")
        print(f"   Scheduler: CosineAnnealing")

    def fit(self):
        print(f"\n{'='*63}")
        print(f"Fine-tuning TSM ResNet-50 for {EPOCHS} epochs")
        print(f"\n{'='*63}")

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            start = time.time()
            train_loss, train_acc = self.train_one_epoch.run()
            duration = time.time() - start

            val_loss, val_acc = self.evaluator.run()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_time'].append(duration)

            print(f"  -Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"  -Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
                print(f"    Best model saved with acc = {self.best_acc:.4f}")

            self.scheduler.step()

        print(f"\n{'='*63}")
        print(f"Training complete!")
        print(f"Best val accuracy: {self.best_acc:.4f}")
        print(f"\n{'='*63}")

    def test(self):
        print("\n--- BAT DAU KIEM THU TREN TAP TEST ---")
        test_model = TSM_Network(
            num_classes=len(self.train_dataset.classes),
            n_segment=NUM_FRAMES,
        ).to(self.device)
        test_model.load_state_dict(torch.load(BEST_MODEL_PATH))

        test_loss, test_acc = Evaluator(
            test_model,
            self.test_loader,
            self.criterion,
            self.device,
        ).run()

        print(f"Diem so thuc te tren tap Test:")
        print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    def run(self):
        self.setup_data()
        self.setup_model()
        self.setup_training()
        self.fit()
        self.test()
        plot_history(self.history)


if __name__ == "__main__":
    trainer = TSMTrainer()
    trainer.run()
