class ContourLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

def contour_loss(y_true, y_pred):  
    sobelFilters = K.variable([ 
                         [ [ [-1, -2, -1], [-2, -4, -2], [-1, -2, -1] ],
                                [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
                                 [ [1, 2, 1], [2, 4, 2], [1, 2, 1] ] ],
                         [ [ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ],
                                [ [2, 4, 2], [0, 0, 0],  [-2, -4, -2] ],
                                 [ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ] ],
                         [ [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ],
                                [ [-2, 0, 2], [-4, 0, 4], [-2, 0, 2] ],
                                 [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ] ]
                            ])
    sobelFilters = K.expand_dims(sobelFilters, axis=-1)
    sobelFilters = K.expand_dims(sobelFilters, axis=-1)
    contour = K.sum( K.concatenate(
            [K.abs(K.conv3d(y_pred, sobelFilters[0], padding='same', data_format='channels_first')),
                K.abs(K.conv3d(y_pred, sobelFilters[1], padding='same', data_format='channels_first')),
                    K.abs(K.conv3d(y_pred, sobelFilters[2], padding='same', data_format='channels_first'))]
                , axis=0), axis=0)
    contour_f = K.batch_flatten(contour)
    y_true_f = K.batch_flatten( K.abs(y_true) - config["drain"])

    finalChamferDistanceSum = K.sum(contour_f * y_true_f, axis=1, keepdims=True)

    return K.mean(finalChamferDistanceSum)