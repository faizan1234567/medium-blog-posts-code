import torch
from utils import log_image_table


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def validate(model, valid_dl, loss_func, log_images=False, batch_idx=0):
  """
  Compute performance of the model on the validation dataset and log a wandb.Table

  Args:
    model: torch.nn.Module
    valid_dl: torch.utils.data.DataLoader
    loss_func: torch.nn.BinaryCrossEntropy
    log_images: bool
    batch_idx: int
  
  """
  model.eval()
  val_loss = 0.
  with torch.no_grad():
      correct = 0
      for i, (images, labels) in enumerate(valid_dl):
          images, labels = images.to(device), labels.to(device)

          # Forward propagation
          outputs = model(images)
          val_loss += loss_func(outputs, labels)*labels.size(0)

          # Compute accuracy and accumulate
          _, predicted = torch.max(outputs.data, 1)
          correct += (predicted == labels).sum().item()

          # Log one batch of images to the dashboard, always same batch_idx.
          if i==batch_idx and log_images:
              log_image_table(images, predicted, labels, outputs.softmax(dim=1))
  return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)