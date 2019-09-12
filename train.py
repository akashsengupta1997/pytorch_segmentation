import copy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


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


def visualise_intermediate_outputs(model, monitor_dataset, save_dir, epoch):
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
            plt.savefig(os.path.join(save_dir, 'epoch{}_seg{}.png'.format(epoch, i)))
            plt.close()


def train_model(model, train_dataset, val_dataset, monitor_dataset, criterion, optimiser,
                batch_size, val_batch_size, model_save_path, num_epochs=100,
                batches_per_print=10, epochs_per_visualise=10, epochs_per_save=10,
                visualise_training_data=False):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_val_loss = np.inf

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss_sum = 0.0
        val_loss_sum = 0.0

        # --- Training ---
        model.train()
        for batch_num, samples_batch in enumerate(train_dataloader):
            images = samples_batch['image']
            masks = samples_batch['mask']

            # Visualise first batch of training data - useful to check if labels have been
            # loaded correctly + right transforms have been applied.
            if visualise_training_data and epoch == 0 and batch_num == 0:
                visualise_images_masks(images, masks)

            # Backpropagate
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimiser.step()

            train_loss_sum += loss.item() * images.size()[0]  # going from mean to sum for loss

            # Print training loss per batch_per_print in every epoch
            if batch_num % batches_per_print == batches_per_print - 1:
                print('Epoch: {:d}, Batch: {:5d}, Training Loss: {:.3f}'.format(epoch + 1,
                                                                                batch_num + 1,
                                                                                loss.item()))

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            for batch_num, samples_batch in enumerate(val_dataloader):
                images = samples_batch['image']
                masks = samples_batch['mask']
                outputs = model(images)
                val_loss = criterion(outputs, masks)

                val_loss_sum += val_loss.item() * images.size()[0]  # mean -> sum for loss

        epoch_train_loss = train_loss_sum/len(train_dataset)
        epoch_val_loss = val_loss_sum/len(val_dataset)
        print('Finished Epoch: {:d}, Train Loss: {:.3f}, Val Loss: {:.3f}'.format(epoch + 1,
                                                                                  epoch_train_loss,
                                                                                  epoch_val_loss))
        # --- Saving best model ---
        if epoch_val_loss < best_epoch_val_loss:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch_val_loss = epoch_val_loss

        if epoch % epochs_per_save == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': best_model_wts,
                        'optimiser_state_dict': optimiser.state_dict(),
                        'loss': loss},
                       model_save_path)
            print('Model saved! Best Val Loss: {:.3f}'.format(best_epoch_val_loss))

        # --- Visualising outputs during training ---
        if epoch % epochs_per_visualise == 0:
            visualise_intermediate_outputs(model, monitor_dataset, './monitor_dir', epoch)

    print('Training Completed. Best Val Loss: {:.3f}'.format(best_epoch_val_loss))

    model.load_state_dict(best_model_wts)
    return model
