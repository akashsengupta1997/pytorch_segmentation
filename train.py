import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from vis_utils import visualise_images_masks, visualise_intermediate_training_outputs


def train_model(model, train_dataset, val_dataset, monitor_dataset, criterion, optimiser,
                batch_size, val_batch_size, model_save_path, device, num_epochs=100,
                batches_per_print=10, epochs_per_visualise=10, epochs_per_save=10,
                visualise_training_data=False):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_val_loss = np.inf
    best_epoch = 0

    model.to(device)

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
            images, masks = images.to(device), masks.to(device)

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
                images, masks = images.to(device), masks.to(device)
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
            best_epoch = epoch

        if epoch % epochs_per_save == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': best_model_wts,
                        'optimiser_state_dict': optimiser.state_dict(),
                        'loss': loss},
                       model_save_path)
            print('Model saved! Best Val Loss: {:.3f} in epoch {}'.format(best_epoch_val_loss,
                                                                          best_epoch))

        # --- Visualising outputs during training ---
        if epoch % epochs_per_visualise == 0:
            visualise_intermediate_training_outputs(model, monitor_dataset, './monitor_dir',
                                                    epoch, device)

    print('Training Completed. Best Val Loss: {:.3f}'.format(best_epoch_val_loss))

    model.load_state_dict(best_model_wts)
    return model
