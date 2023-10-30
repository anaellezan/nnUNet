import numpy as np

def get_intersection(mask_1, mask_2):
    return (mask_1 & mask_2)

def compute_dice_coefficient(mask_gt, mask_pred):
    """Computes soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum 


def compute_IoU_coefficient(mask_gt, mask_pred):
    """Computes IoU coefficient.

    compute the IoU coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the IoU coeffcient as float. If both masks are empty, the result is NaN.
    """
    volume_union = (mask_gt | mask_pred).sum()
    if volume_union == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return volume_intersect / volume_union 


def compute_contour_volume(mask_gt, mask_pred):
    """Computes Contour Volume.

    compute the absolute volume between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the Contour Volume coeffcient as float. If both masks are empty, the result is NaN.
    """
    volume_union = (mask_gt | mask_pred).sum()
    if volume_union == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return volume_union - volume_intersect