import torch
import matplotlib.pyplot as plt


def denormalize(frames):
    """Denormalize for visualizing"""
    frames = frames.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1) # mean & std of ImageNet
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    frames = frames * std + mean
    return frames.clamp(0,1)


def plot_history(history, save_path="tsm_training_results.png"):
    # CODE VE BIEU DO
    fig, axes = plt.subplots(1, 2, figsize=(15,6))

    # plot loss
    axes[0].plot(history['train_loss'], label='Train', marker = 'o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # plot accuracy
    axes[1].plot(history['train_acc'], label='Train', marker = 'o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0,1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print("Plot saved!")
    plt.show()
