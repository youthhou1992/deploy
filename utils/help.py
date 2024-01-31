import numpy as np

def pad_rec_im(im, wh_ratio=15):
    """
    pad image to the right side to satify the wh_ratio for crnn recognition in gray, with the average background color
    Args:
        im: cv2 image
        wh_ratio: the expect imwidth / imheight ratio for crnn recognition
    """
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    h, w = im.shape
    if 1. * w / h >= wh_ratio:  # pad h
        return im
    else:  # pad w
        new_w = int(wh_ratio * h)
        # if new_w - w == 0:
        #     return im
        # p_l = randint(0, new_w - w)
        p_r = new_w - w
        color = np.argmax(np.bincount(im.reshape(-1)))
        padding = np.ones((h, p_r), np.uint8) * color
        im = np.hstack((im, padding))
    return im