import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def predict(model, saved_model_path, predict_dataset, batch_size, save_dir, device):

    predict_dataloader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for num, images in enumerate(predict_dataloader):
            print('Batch:', num)
            images = images.to(device)
            outputs = model(images)

            outputs = outputs.to('cpu')
            images = images.to('cpu')
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
