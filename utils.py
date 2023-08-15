
import wandb

def log_image_table(images, predicted, labels, probs):
  """
  Create a log table for experiment comparison

  Args:
    images: torch.tensor
    predicted: torch.tensor
    labels: torch.tensor
    probs: torch.tensor
  """

  # Create a wandb Table to log images, labels and predictions 
  table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
  for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
      table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
  wandb.log({"predictions_table":table}, commit=False)