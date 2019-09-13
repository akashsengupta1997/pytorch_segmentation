import torch
from torchvision import transforms
from models.segnet import SegNet
from models.unet import UNet
from models.pspnet import PSPNet
from models.enet import ENet
from data.dataset import ImageFolder
from data.tranforms import Resize, ToTensor, PadToSquare
from predict import predict


# --- Hyperparameters ---
batch_size = 3

num_classes = 32
input_height = 256
input_width = 256
output_height = 256
output_width = 256


# --- Predict data ---
predict_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/trial/images3/train"
predict_transforms = transforms.Compose([PadToSquare(),
                                         Resize(input_height,
                                                input_width,
                                                output_height,
                                                output_width),
                                         ToTensor()])
predict_dataset = ImageFolder(predict_dir, transform=predict_transforms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Test images found: {}'.format(len(predict_dataset)))
print('Device: {}'.format(device))


# --- Model ---
model = PSPNet(num_classes)
saved_model_path = "./saved_models/pspnet_test.tar"


# --- Predict ---
predict(model,
        saved_model_path,
        predict_dataset,
        batch_size,
        "./tests/",
        device)

