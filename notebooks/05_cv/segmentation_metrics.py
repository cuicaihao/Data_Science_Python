import numpy as np

# import keras.backend as K
from tensorflow import keras as K


def metrics_np(
    y_true,
    y_pred,
    metric_name,
    metric_type="standard",
    drop_last=True,
    mean_per_class=False,
    verbose=False,
):
    """
    Compute mean metrics of two segmentation masks, via numpy.

    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)

    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.

    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    assert (
        y_true.shape == y_pred.shape
    ), "Input masks should be same shape, instead are {}, {}".format(
        y_true.shape, y_pred.shape
    )
    assert (
        len(y_pred.shape) == 4
    ), "Inputs should be B*W*H*N tensors, instead have shape {}".format(y_pred.shape)

    flag_soft = metric_type == "soft"
    flag_naive_mean = metric_type == "naive"

    num_classes = y_pred.shape[-1]
    # if only 1 class, there is no background class and it should never be dropped
    drop_last = drop_last and num_classes > 1

    if not flag_soft:
        if num_classes > 1:
            # get one-hot encoded masks from y_pred (true masks should already be in correct format, do it anyway)
            y_pred = np.array(
                [np.argmax(y_pred, axis=-1) == i for i in range(num_classes)]
            ).transpose(1, 2, 3, 0)
            y_true = np.array(
                [np.argmax(y_true, axis=-1) == i for i in range(num_classes)]
            ).transpose(1, 2, 3, 0)
        else:
            y_pred = (y_pred > 0).astype(int)
            y_true = (y_true > 0).astype(int)

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1, 2)  # W,H axes of each image
    # or, np.logical_and(y_pred, y_true) for one-hot
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    # or, np.logical_or(y_pred, y_true) for one-hot
    union = mask_sum - intersection

    if verbose:
        print(
            "intersection (pred*true), intersection (pred&true), union (pred+true-inters), union (pred|true)"
        )
        print(
            intersection,
            np.sum(np.logical_and(y_pred, y_true), axis=axes),
            union,
            np.sum(np.logical_or(y_pred, y_true), axis=axes),
        )

    smooth = 0.001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    metric = {"iou": iou, "dice": dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask = np.not_equal(union, 0).astype(int)
    # mask = 1 - np.equal(union, 0).astype(int) # True = 1

    if drop_last:
        metric = metric[:, :-1]
        mask = mask[:, :-1]

    # return mean metrics: remaining axes are (batch, classes)
    # if mean_per_class, average over batch axis only
    # if flag_naive_mean, average over absent classes too
    if mean_per_class:
        if flag_naive_mean:
            return np.mean(metric, axis=0)
        else:
            # mean only over non-absent classes in batch (still return 1 if class absent for whole batch)
            return (np.sum(metric * mask, axis=0) + smooth) / (
                np.sum(mask, axis=0) + smooth
            )
    else:
        if flag_naive_mean:
            return np.mean(metric)
        else:
            # mean only over non-absent classes
            class_count = np.sum(mask, axis=0)
            return np.mean(
                np.sum(metric * mask, axis=0)[class_count != 0]
                / (class_count[class_count != 0])
            )


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via numpy.

    Calls metrics_np(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name="iou", **kwargs)


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    Compute mean Dice coefficient of two segmentation masks, via numpy.

    Calls metrics_np(y_true, y_pred, metric_name='dice'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name="dice", **kwargs)


# keras version


# def seg_metrics(y_true, y_pred, metric_name, metric_type='standard', drop_last=True, mean_per_class=False, verbose=False):
#     """
#     Compute mean metrics of two segmentation masks, via Keras.

#     IoU(A,B) = |A & B| / (| A U B|)
#     Dice(A,B) = 2*|A & B| / (|A| + |B|)

#     Args:
#         y_true: true masks, one-hot encoded.
#         y_pred: predicted masks, either softmax outputs, or one-hot encoded.
#         metric_name: metric to be computed, either 'iou' or 'dice'.
#         metric_type: one of 'standard' (default), 'soft', 'naive'.
#           In the standard version, y_pred is one-hot encoded and the mean
#           is taken only over classes that are present (in y_true or y_pred).
#           The 'soft' version of the metrics are computed without one-hot
#           encoding y_pred.
#           The 'naive' version return mean metrics where absent classes contribute
#           to the class mean as 1.0 (instead of being dropped from the mean).
#         drop_last = True: boolean flag to drop last class (usually reserved
#           for background class in semantic segmentation)
#         mean_per_class = False: return mean along batch axis for each class.
#         verbose = False: print intermediate results such as intersection, union
#           (as number of pixels).
#     Returns:
#         IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
#           in which case it returns the per-class metric, averaged over the batch.

#     Inputs are B*W*H*N tensors, with
#         B = batch size,
#         W = width,
#         H = height,
#         N = number of classes
#     """

#     flag_soft = (metric_type == 'soft')
#     flag_naive_mean = (metric_type == 'naive')

#     # always assume one or more classes
#     num_classes = K.shape(y_true)[-1]

#     if not flag_soft:
#         # get one-hot encoded masks from y_pred (true masks should already be one-hot)
#         y_pred = K.one_hot(K.argmax(y_pred), num_classes)
#         y_true = K.one_hot(K.argmax(y_true), num_classes)

#     # if already one-hot, could have skipped above command
#     # keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
#     y_true = K.cast(y_true, 'float32')
#     y_pred = K.cast(y_pred, 'float32')

#     # intersection and union shapes are batch_size * n_classes (values = area in pixels)
#     axes = (1, 2)  # W,H axes of each image
#     intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
#     mask_sum = K.sum(K.abs(y_true), axis=axes) + \
#         K.sum(K.abs(y_pred), axis=axes)
#     # or, np.logical_or(y_pred, y_true) for one-hot
#     union = mask_sum - intersection

#     smooth = .001
#     iou = (intersection + smooth) / (union + smooth)
#     dice = 2 * (intersection + smooth)/(mask_sum + smooth)

#     metric = {'iou': iou, 'dice': dice}[metric_name]

#     # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
#     mask = K.cast(K.not_equal(union, 0), 'float32')

#     if drop_last:
#         metric = metric[:, :-1]
#         mask = mask[:, :-1]

#     if verbose:
#         print('intersection, union')
#         print(K.eval(intersection), K.eval(union))
#         print(K.eval(intersection/union))

#     # return mean metrics: remaining axes are (batch, classes)
#     if flag_naive_mean:
#         return K.mean(metric)

#     # take mean only over non-absent classes
#     class_count = K.sum(mask, axis=0)
#     non_zero = tf.greater(class_count, 0)
#     non_zero_sum = tf.boolean_mask(K.sum(metric * mask, axis=0), non_zero)
#     non_zero_count = tf.boolean_mask(class_count, non_zero)

#     if verbose:
#         print('Counts of inputs with class present, metrics for non-absent classes')
#         print(K.eval(class_count), K.eval(non_zero_sum / non_zero_count))

#     return K.mean(non_zero_sum / non_zero_count)


# def mean_iou(y_true, y_pred, **kwargs):
#     """
#     Compute mean Intersection over Union of two segmentation masks, via Keras.

#     Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
#     """
#     return seg_metrics(y_true, y_pred, metric_name='iou', **kwargs)


# def mean_dice(y_true, y_pred, **kwargs):
#     """
#     Compute mean Dice coefficient of two segmentation masks, via Keras.

#     Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
#     """
#     return seg_metrics(y_true, y_pred, metric_name='dice', **kwargs)
