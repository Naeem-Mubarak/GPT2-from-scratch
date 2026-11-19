import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, directory=checkpoint_dir, filename='best_checkpoint.pth'):
    path = os.path.join(directory, filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, path, )
    print(f"Model checkpoint saved successfully to {path}")



def latest_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, directory=checkpoint_dir, filename='complete_checkpoint.pth'):
    path = os.path.join(directory, filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, path)
    print(f"Latest checkpoint saved successfully to {path}")