import torch
import torch.nn as nn
from torchvision import transforms
from models.segnet import SegNet
from models.unet import UNet
from models.pspnet import PSPNet
from models.enet import ENet
from losses.losses import cross_entropy_with_aux_loss_pspnet
from losses.class_weighting import enet_style_bounded_log_weighting, median_frequency_balancing
from data.dataset import UPS31Dataset, ImageFolder
from data.tranforms import Resize, ToTensor, PadToSquare
from evaluate import evaluate


# --- Hyperparameters ---
batch_size = 3

num_classes = 32
input_height = 256
input_width = 256
output_height = 256
output_width = 256


# --- Predict data ---
image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images3/train"
label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks3/train"

eval_transforms = transforms.Compose([PadToSquare(),
                                      Resize(input_height,
                                             input_width,
                                             output_height,
                                             output_width),
                                      ToTensor()])

eval_dataset = UPS31Dataset(image_dir=image_dir,
                            label_dir=label_dir,
                            transform=eval_transforms)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Eval images found: {}'.format(len(eval_dataset)))
print('Device: {}'.format(device))


# --- Model and Loss---
model = PSPNet(num_classes)
saved_model_path = "./saved_models/pspnet_test.tar"
criterion = nn.CrossEntropyLoss(weight=None)

# --- Predict ---
evaluate(model,
         saved_model_path,
         eval_dataset,
         criterion,
         batch_size,
         device,
         num_classes)

