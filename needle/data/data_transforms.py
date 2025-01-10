import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if not flip_img:
            return img
        
        # h, w, c = img.shape
        # for i in range(h):
        #     for j in range(w//2):
        #         for k in range(c):
        #             img[i,w-j-1,k], img[i,j,k] = img[i,j,k], img[i,w-j-1,k]
        
        return np.flip(img, axis=1)
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        ndim = len(img.shape)
        assert ndim in (2,3), "ndim = %d is not supported" % ndim

        if ndim == 3:
            h, w, _ = img.shape
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
            corner_x = self.padding+shift_x
            corner_y = self.padding+shift_y
            return img[corner_x: corner_x+h, corner_y: corner_y+w, :]
        
        else:
            h, w  = img.shape
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding)), 'constant')
            corner_x = self.padding+shift_x
            corner_y = self.padding+shift_y
            return img[corner_x: corner_x+h, corner_y: corner_y+w]
        ### END YOUR SOLUTION
