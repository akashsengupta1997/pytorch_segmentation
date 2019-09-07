import torch
import torch.nn.functional as F


def cross_entropy_with_aux_loss_pspnet(inputs, target, aux_weight=0.4):
    input, aux_input = inputs
    target_height, target_width = target.shape[1:]

    aux_resized = F.interpolate(aux_input, size=(target_height, target_width), mode='bilinear',
                                align_corners=True)

    main_loss = F.cross_entropy(input, target)
    aux_loss = F.cross_entropy(aux_resized, target)

    return main_loss + aux_weight * aux_loss