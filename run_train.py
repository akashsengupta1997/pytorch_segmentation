import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models.segnet import SegNet
from models.unet import UNet
from models.pspnet import PSPNet
from models.enet import ENet
from losses.losses import cross_entropy_with_aux_loss_pspnet
from losses.class_weighting import enet_style_bounded_log_weighting, median_frequency_balancing
from data.dataset import SegmentationDataset, ImageFolder
from data.tranforms import Resize, ToTensor, PadToSquare, RandomRotate, RandomCrop
from train import train_model

# --- Hyperparameters, Dimensions, Visualisation parameters ---

batch_size = 3
val_batch_size = 1
num_epochs = 501

num_classes = 32
input_height = 256
input_width = 256
output_height = 256
output_width = 256

batches_per_print = 1
epochs_per_visualise = 50
epochs_per_save = 50

random_rotate_range = 40
random_crop_min_height_scale = 0.8
random_crop_min_width_scale = 0.8

# --- Load Dataset ---
image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images3/train"
label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks3/train"
val_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images3/val"
val_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/masks3/val"
monitor_image_dir = "./monitor_dir/images/"

dataset_transforms = {'train': transforms.Compose([RandomCrop(random_crop_min_height_scale,
                                                              random_crop_min_width_scale),
                                                   RandomRotate(random_rotate_range),
                                                   PadToSquare(),
                                                   Resize(input_height,
                                                          input_width,
                                                          output_height,
                                                          output_width),
                                                   ToTensor()]),
                      'val': transforms.Compose([PadToSquare(),
                                                 Resize(input_height,
                                                        input_width,
                                                        output_height,
                                                        output_width),
                                                 ToTensor()])}

train_dataset = SegmentationDataset(image_dir=image_dir,
                                    label_dir=label_dir,
                                    transform=dataset_transforms['train'])
val_dataset = SegmentationDataset(image_dir=val_image_dir,
                                  label_dir=val_label_dir,
                                  transform=dataset_transforms['val'])
monitor_dataset = ImageFolder(monitor_image_dir,
                              transform=dataset_transforms['val'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Training images found: {}'.format(len(train_dataset)))
print('Validation images found: {}'.format(len(val_dataset)))
# print('Monitoring images found: {}'.format(len(monitor_dataset)))
print('Device: {}'.format(device))

# --- Model, loss and optimiser ---

model = ENet(num_classes)
class_weights = enet_style_bounded_log_weighting(label_dir, output_height, output_width,
                                                 num_classes)
# class_weights = median_frequency_balancing(label_dir, output_height, output_width,
#                                                   num_classes)
class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=None)
# criterion = cross_entropy_with_aux_loss_pspnet(aux_weight=0.4, class_weights=class_weights)
optimiser = optim.Adam(model.parameters())

trained_model = train_model(model,
                            train_dataset,
                            val_dataset,
                            monitor_dataset,
                            criterion,
                            optimiser,
                            batch_size,
                            val_batch_size,
                            './saved_models/lol.tar',
                            device,
                            num_epochs=num_epochs,
                            batches_per_print=batches_per_print,
                            epochs_per_visualise=epochs_per_visualise,
                            epochs_per_save=epochs_per_save,
                            visualise_training_data=True)


#TODO device and GPU
#TODO loading model to resume training
