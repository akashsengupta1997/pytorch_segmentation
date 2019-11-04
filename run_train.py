import os
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
from losses.class_weighting import simple_bg_down_weight
from data.dataset import UPS31Dataset, ImageFolder
from data.transforms import Resize, ToTensor, PadToSquare, RandomRotate, RandomCrop, RandomHorizFlip
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

random_rotate_range = 70
random_crop_min_height_scale = 0.4
random_crop_min_width_scale = 0.7
random_horiz_flip_prob = 0.5
horiz_flip_classes_to_swap = [(1, 14), (2, 15), (3, 16), (4, 17), (5, 18), (6, 19), (7, 20),
                              (8, 21), (9, 22), (10, 23), (11, 24), (12, 25), (13, 26),
                              (27, 28), (29, 30)]

# --- Load Dataset ---
image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/images/train"
label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/masks/train"
val_image_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/val_images/val"
val_label_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/val_masks/val"
monitor_dir = "./monitor_dir1/"
monitor_image_dir = os.path.join(monitor_dir, 'images')

dataset_transforms = {'train': transforms.Compose([RandomHorizFlip(horiz_flip_classes_to_swap,
                                                                   random_horiz_flip_prob),
                                                   RandomCrop(random_crop_min_height_scale,
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

train_dataset = UPS31Dataset(image_dir=image_dir,
                             label_dir=label_dir,
                             transform=dataset_transforms['train'])
val_dataset = UPS31Dataset(image_dir=val_image_dir,
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

model = PSPNet(num_classes)
# class_weights = enet_style_bounded_log_weighting(label_dir, output_height, output_width,
#                                                  num_classes)
# class_weights = median_frequency_balancing(label_dir, output_height, output_width,
#                                                   num_classes)
# class_weights = simple_bg_down_weight(0.1, num_classes)
# class_weights = class_weights.to(device)
# criterion = nn.CrossEntropyLoss(weight=None)
criterion = cross_entropy_with_aux_loss_pspnet(aux_weight=0.4, class_weights=None)
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
                            monitor_dir,
                            num_epochs=num_epochs,
                            batches_per_print=batches_per_print,
                            epochs_per_visualise=epochs_per_visualise,
                            epochs_per_save=epochs_per_save,
                            visualise_training_data=True)


#TODO device and GPU
#TODO loading model to resume training
