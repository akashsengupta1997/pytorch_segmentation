import copy
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vis_utils import visualise_images_masks, visualise_intermediate_training_outputs


def predict(model, saved_model_path, predict_dataset, batch_size, save_dir):

    predict_dataloader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        for num, images in enumerate(predict_dataloader):
            outputs = model(images)
            seg_maps = np.argmax(np.transpose(outputs.detach().numpy(), [0, 2, 3, 1]), axis=-1)
            images = images.detach().numpy()
            images = np.transpose(images, [0, 2, 3, 1])

            for i in range(batch_size):
                plt.figure(figsize=(10, 10))
                plt.tight_layout()
                plt.axis('off')
                plt.subplot(1, 2, 1)
                plt.imshow(images[i])
                plt.subplot(1, 2, 2)
                plt.imshow(seg_maps[i])
                plt.savefig(os.path.join(save_dir, '{:5d}_seg.png'.format(num*batch_size + i)),
                            bbox_inches='tight')
