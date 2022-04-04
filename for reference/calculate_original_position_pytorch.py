def calculate_original_position_pytorch(preds, crop_center, crop_size, hand_side, resized_size):
    """
    checked
    """
    new_preds = preds.clone()
    bs = preds.shape[0]

    hand_side = hand_side.view(bs,-1)

    for i in range(bs):
        if hand_side[i]:
            new_preds[i, :, 0] = resized_size - new_preds[i, :, 0]

    current_center = crop_center.view(bs,1,2).to(preds.device)
    current_scale = (2*crop_size/resized_size).view(bs,1,1).to(preds.device)

    new_preds = (new_preds - resized_size/2) * current_scale + current_center
    return new_preds