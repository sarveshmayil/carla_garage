import numpy as np
from skimage.util import random_noise
from skimage.exposure import adjust_gamma

def add_random_noise(image:np.ndarray, noise_type:str='gaussian', **kwargs):
    """
    Adds noise to the given image.

    Parameters:
    - image: numpy.ndarray \\
        The input image.
    - noise_type: str, optional \\
        The type of noise to add. Default is 'gaussian'. \\
        Possible values: 'gaussian', 'poisson', 'salt',
                         'pepper', 's&p', 'speckle'.
    - kwargs: dict, optional \\
        Additional keyword arguments for the specific noise type.

    Returns:
    - noisy_image: numpy.ndarray \\
        The image with added noise.
    """
    assert noise_type in ['gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle']

    noisy_image = random_noise(image, mode=noise_type, **kwargs)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image

def adjust_exposure(image:np.ndarray, gamma:float=None, gamma_var:float=0.1):
    """
    Adjusts the exposure of the given image using gamma correction.

    Parameters:
    - image: numpy.ndarray
        The input image.
    - gamma: float
        The gamma value for gamma correction. \\
        gamma = 1.0 means no correction, \\
        gamma > 1.0 darkens the image, \\
        gamma < 1.0 brightens the image.

    Returns:
    - adjusted_image: numpy.ndarray \\
        The image with adjusted exposure.
    """
    if gamma is None:
        gamma = np.random.normal(loc=1.0, scale=gamma_var)

    adjusted_image = adjust_gamma(image, gamma)
    return adjusted_image

def add_random_black_patches(image: np.ndarray, max_n_patches:int=5, min_patch_size:int=1):
    """
    Adds random black patches to the given image.

    Parameters:
    - image: numpy.ndarray \\
        The input image.

    - max_n_patches: int, optional \\
        The maximum number of black patches to add. Default is 5.

    Returns:
    - patched_image: numpy.ndarray \\
        The image with random black patches.
    """
    h, w, _ = image.shape
    max_patch_size = [h // 2, w // 2]

    patched_image = np.copy(image)
    num_patches = np.random.randint(1, max_n_patches)  # Randomly choose number of patches

    for _ in range(num_patches):
        patch_size = np.random.randint(size=2, low=min_patch_size, high=max_patch_size)  # Randomly choose patch size
        patch_x = np.random.randint(0, w - patch_size[1])  # Randomly choose patch x-coordinate
        patch_y = np.random.randint(0, h - patch_size[0])  # Randomly choose patch y-coordinate

        patched_image[patch_y:patch_y+patch_size[0], patch_x:patch_x+patch_size[1], :] = 0  # Set patch to black

    return patched_image


