import torch
import numpy as np
from torch.utils.data import DataLoader


def intersection_union_gt_pixels_per_class(ground_truth, predict, num_classes):
    """
    Compute number of intersections and unions between ground truth label and predicted label
    for 1 training batch.
    """
    num_intersections_per_class = []
    num_unions_per_class = []
    gt_pixels_per_class = []
    for class_num in range(num_classes):  # not including background class
        ground_truth_binary = np.zeros(ground_truth.shape)
        predict_binary = np.zeros(predict.shape)
        ground_truth_binary[ground_truth == class_num] = 1
        predict_binary[predict == class_num] = 1

        intersection = np.logical_and(ground_truth_binary, predict_binary)
        union = np.logical_or(ground_truth_binary, predict_binary)
        num_intersections = float(np.sum(intersection))
        num_unions = float(np.sum(union))
        gt_pixels = float(np.sum(ground_truth_binary))
        num_intersections_per_class.append(num_intersections)
        num_unions_per_class.append(num_unions)
        gt_pixels_per_class.append(gt_pixels)

    return np.array(num_intersections_per_class), np.array(num_unions_per_class), gt_pixels_per_class


def evaluate(model, saved_model_path, eval_dataset, criterion, batch_size, device,
             num_classes):
    """
    Compute val loss, mean IOU, global accuracy & class accuracy.
    """

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_intersects_per_class = np.zeros(num_classes)
    total_unions_per_class = np.zeros(num_classes)
    total_gt_pixels_per_class = np.zeros(num_classes)
    total_loss_sum = 0.0
    total_num_corrects = 0
    total_pixels = 0

    with torch.no_grad():
        for batch_num, samples_batch in enumerate(eval_dataloader):
            images = samples_batch['image']
            masks = samples_batch['mask']
            images.to(device)
            masks.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, masks)

            total_loss_sum += val_loss.item() * images.size()[0]  # mean -> sum for loss
            total_num_corrects += torch.sum(torch.argmax(outputs, dim=1) == masks).item()
            total_pixels += images.shape[0] * images.shape[2] * images.shape[3]

            outputs.to('cpu')
            images.to('cpu')
            seg_maps = np.argmax(np.transpose(outputs.detach().numpy(), [0, 2, 3, 1]), axis=-1)
            masks = masks.detach().numpy()

            num_intersects, num_unions, num_gt_pixels = intersection_union_gt_pixels_per_class(masks,
                                                                                               seg_maps,
                                                                                               num_classes)
            total_intersects_per_class += num_intersects
            total_unions_per_class += num_unions
            total_gt_pixels_per_class += num_gt_pixels

        class_ious = np.divide(total_intersects_per_class, total_unions_per_class)
        mean_iou = np.mean(class_ious[1:])  # ignoring class 0 (background)

        class_accs = np.divide(total_intersects_per_class, total_gt_pixels_per_class)
        mean_class_acc = 100.0 * np.mean(class_accs)  # equals mean class recall

        global_acc = 100.0 * float(total_num_corrects)/total_pixels

        mean_loss = total_loss_sum/len(eval_dataset)

        print('Val Loss: {:.3f}, GA: {:.2f}, Mean PA: {:.2f}, mIOU: {:.2f}'.format(mean_loss,
                                                                                   global_acc,
                                                                                   mean_class_acc,
                                                                                   mean_iou))

        print('Class accs:', class_accs)
        print('Class ious:', class_ious)




