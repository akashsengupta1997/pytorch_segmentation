import torch.nn.functional as F


def cross_entropy_with_aux_loss_pspnet(aux_weight=0.4, class_weights=None):

    def loss_fixed(inputs, target):
        if isinstance(inputs, list):
            input, aux_input = inputs
            target_height, target_width = target.shape[1:]

            aux_resized = F.interpolate(aux_input, size=(target_height, target_width),
                                        mode='bilinear', align_corners=True)

            main_loss = F.cross_entropy(input, target, weight=class_weights)
            aux_loss = F.cross_entropy(aux_resized, target, weight=class_weights)

            return main_loss + aux_weight * aux_loss
        else:
            return F.cross_entropy(inputs, target, weight=class_weights)

    return loss_fixed
