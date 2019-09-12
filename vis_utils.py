import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualise_images_masks(images, masks):
    plt.figure()
    batch_size = images.size()[0]
    for i in range(batch_size):
        image = images[i]
        mask = masks[i]
        plt.tight_layout()
        ax = plt.subplot(2, batch_size, i + 1)
        plt.imshow(np.transpose(image, [1, 2, 0]))
        ax = plt.subplot(2, batch_size, i + batch_size + 1)
        plt.imshow(mask)
    plt.show()


def visualise_intermediate_training_outputs(model, monitor_dataset, save_dir, epoch):
    with torch.no_grad():
        for i in range(len(monitor_dataset)):
            image = monitor_dataset[i]
            image = torch.unsqueeze(image, dim=0)
            output = model(image)
            output = output.detach().numpy()
            output = np.transpose(output, [0, 2, 3, 1])
            seg_map = np.argmax(output, axis=-1)
            image = image.detach().numpy()
            image = np.transpose(image, [0, 2, 3, 1])

            plt.figure(figsize=(10, 10))
            plt.tight_layout()
            plt.subplot(2, 1, 1)
            plt.imshow(image[0])
            plt.subplot(2, 1, 2)
            plt.imshow(seg_map[0])
            plt.savefig(os.path.join(save_dir, 'epoch{}_seg{}.png'.format(epoch, i)),
                        bbox_inches='tight')
            plt.close()