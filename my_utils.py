import os
import torch
import logging

def save_checkpoint(filename, model_state_dict, optimizer_state_dict,
                    scheduler_state_dict, epoch, iteration, loss, config, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'loss': loss,
        'training_config': config
    }
    path = os.path.join(weights_dir, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    logging.info(f"Saved checkpoint {filename} at epoch {epoch}, iter {iteration}, loss {loss}")

def label_converter(labels, null_idx, unconditional_percentage, batch_size, device):
    masked_labels = labels.clone()
    mask = torch.rand(batch_size,device=device) < unconditional_percentage
    masked_labels[mask] = null_idx
    return masked_labels