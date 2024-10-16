"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Utilities
File: tools.py
"""
import os, sys
import torch.nn.functional
import numpy as np
import torch
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import defaults
from interface_tools import errorMessage


def is_grayscale(img):
    """
    Checks if loaded image is grayscale. Compares channel
    arrays for equality.

    Parameters
    ------
    img: np.array
        Image data array [HWC].
    """
    r_ch = img[:, :, 0]
    g_ch = img[:, :, 1]
    b_ch = img[:, :, 2]

    return np.array_equal(r_ch, g_ch) and np.array_equal(r_ch, b_ch)


def get_image(img_path, ch=3, scale=None, tile_size=None, interpolate=cv2.INTER_AREA):
    """
    Loads image data into standard Numpy array
    Reads image and reverses channel order.
    Loads image as 8 bit (regardless of original depth)

    Parameters
    ------
    img_path: str
        Image file path.
    ch: int
        Number of input channels (default = 3).
    scale: float
        Scaling factor.
    tile_size: int
        Tile dimension (square).
    interpolate: int
        Interpolation method (OpenCV).

    Returns
    ------
    numpy array
        Image array; formats: grayscale: [HW]; colour: [HWC].
    w: int
        Image width (px).
    h: int
        Image height (px).
    w_resized: int
        Image width resized (px).
    h_resized: int
        Image height resized (px).
     """

    assert ch == 3 or ch == 1, 'Invalid number of input channels:\t{}.'.format(ch)
    assert os.path.exists(img_path), 'Image path {} does not exist.'.format(img_path)

    if not tile_size:
        tile_size = defaults.tile_size

    # verify image channel number
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if is_grayscale(img) and ch == 3:
        errorMessage('\nInput image is grayscale but process expects colour (RGB).\n\tApplication stopped.')
        return
    elif not is_grayscale(img) and ch == 1:
        errorMessage('\nInput image is colour (RGB) but process expects grayscale.\n\tApplication stopped.')
        return

    # load image data
    if ch == 3:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # get dimensions
    height, width = img.shape[:2]
    height_resized = height
    width_resized = width

    # apply scaling
    if scale:
        min_dim = min(height, width)
        # adjust scale to minimum size (tile dimensions)
        if min_dim < tile_size:
            errorMessage("Scale too small. Setting to minimum.")
            scale = tile_size / min_dim
        dim = (int(scale * width), int(scale * height))
        img = cv2.resize(img, dim, interpolation=interpolate)
        height_resized, width_resized = img.shape[:2]

    return img, width, height, width_resized, height_resized


def adjust_to_tile(img, tile_size, stride, ch, interpolate=cv2.INTER_AREA):
    """
    Scales image to n x tile dimensions with stride
    and crops to match input image aspect ratio

    Parameters
    ------
    img: np.array array
        Image array.
    tile_size: int
        Tile dimension.
    stride: int
        Stride of tile extraction.
    ch: int
        Number of input channels.
    interpolate: int
        Interpolation method (OpenCV).

    Returns
    ------
    numpy array
        Adjusted image array.
    int
        Height of adjusted image.
    int
        Width of adjusted image.
    int
        Size of crop to top of the image.
    """

    # Get full-sized dimensions
    w = img.shape[1]
    h = img.shape[0]

    assert tile_size % stride == 0 and stride <= tile_size, "Tile size must be multiple of stride."

    # Get padded width for tiling
    if (w-tile_size)%stride == 0:
        w_scaled = w
    else:
        w_scaled = (w // stride) * stride + tile_size
    if (h-tile_size)%stride == 0:
        h_scaled = h
    else:
        h_scaled = (h // stride) * stride + tile_size

    a = (w_scaled-w)//2
    aa = w_scaled - w - a
    b = (h_scaled - h)//2
    bb = h_scaled - h - b

    if ch == 1:
        img_resized = np.pad(img, pad_width = ((b,bb), (a,aa)), mode = 'symmetric')
    elif ch == 3:
        img_resized = np.pad(img, pad_width = ((b,bb), (a,aa), (0,0)), mode = 'symmetric')

    return img_resized, img_resized.shape[1], img_resized.shape[0]


def reconstruct(logits, meta):
    """
    Reconstruct tiles into full-sized segmentation mask.
    Uses metadata generated from image tiling (adjust_to_tile)

      Returns
      ------
      mask_reconstructed: np.array
         Reconstructed image data.
     """

    # get tiles from tensor outputs
    logits = [t.cpu() for t in logits] if torch.cuda.is_available() else logits
    tiles = np.concatenate(logits, axis=0)

    # load metadata
    w = meta.extract['w_fitted']
    h = meta.extract['h_fitted']
    w_full = meta.extract['w_scaled']
    h_full = meta.extract['h_scaled']
    offset = meta.extract['offset']
    tile_size = meta.tile_size
    stride = meta.stride
    palette = meta.palette_rgb
    n_classes = meta.n_classes

    n_strides_in_row = w // stride - 1 if stride < tile_size else w // stride
    n_strides_in_col = h // stride - 1 if stride < tile_size else h // stride

    # Calculate overlap
    olap_size = tile_size - stride

    # initialize full image numpy array
    mask_fullsized = np.empty((n_classes, h + offset, w), dtype=np.float32)

    # Create empty rows
    r_olap_prev = None
    r_olap_merged = None

    # row index (set to offset height)
    row_idx = offset

    # anti-tile artifact code
    for tile in tiles:
        tileshape = tile.shape[1:]
        indices = np.indices(tileshape)
        center = np.array(tileshape) // 2
        distances = np.sqrt(np.sum((indices - center[:, np.newaxis, np.newaxis])**2, axis=0))


        for lc in range(tile.shape[0]):
            a = tile[lc]/(1+np.exp(0.1*distances-20))
            b = tile[lc]*((1/(1+np.exp(-0.1*distances+20)))+1)
            tile[lc] = np.where(tile[lc]>0, a, b)

    for i in range(n_strides_in_col):
        # Get initial tile in row
        t_current = tiles[i * n_strides_in_row]
        r_current = np.empty((n_classes, tile_size, w), dtype=np.float32)
        col_idx = 0
        # Step 1: Collate column tiles in row
        for j in range(n_strides_in_row):
            t_current_width = t_current.shape[2]
            if j < n_strides_in_row - 1:
                # Get adjacent tile
                t_next = tiles[i * n_strides_in_row + j + 1]
                # Extract right overlap of current tile
                olap_current = t_current[:, :, t_current_width - olap_size:t_current_width]
                # Extract left overlap of next (adjacent) tile
                olap_next = t_next[:, :, 0:olap_size]
                # Average the overlapping segment logits
                olap_current = torch.nn.functional.softmax(torch.tensor(olap_current), dim=0)
                olap_next = torch.nn.functional.softmax(torch.tensor(olap_next), dim=0)
                olap_merged = (olap_current + olap_next) / 2
                # Insert averaged overlap into current tile
                np.copyto(t_current[:, :, t_current_width - olap_size:t_current_width], olap_merged)
                # Insert updated current tile into row
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)
                col_idx += t_current_width
                # Crop next tile and copy to current tile
                t_current = t_next[:, :, olap_size:t_next.shape[2]]

            else:
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)

        # Step 2: Collate row slices into full mask
        r_current_height = r_current.shape[1]
        # Extract overlaps at top and bottom of current row
        r_olap_top = r_current[:, 0:olap_size, :]
        r_olap_bottom = r_current[:, r_current_height - olap_size:r_current_height, :]

        # Average the overlapping segment logits
        if i > 0:
            r_olap_top = torch.nn.functional.softmax(torch.tensor(r_olap_top), dim=0)
            r_olap_prev = torch.nn.functional.softmax(torch.tensor(r_olap_prev), dim=0)
            r_olap_merged = (r_olap_top + r_olap_prev) / 2

        # Top row: crop by bottom overlap (to be averaged)
        if i == 0 and n_strides_in_col >1:
            # Crop current row by bottom overlap size
            r_current = r_current[:, 0:r_current_height - olap_size, :]
        # Otherwise: Merge top overlap with previous
        elif n_strides_in_col > 1:
            # Replace top overlap with averaged overlap in current row
            np.copyto(r_current[:, 0:olap_size, :], r_olap_merged)

        # Crop middle rows by bottom overlap
        if 0 < i < n_strides_in_col - 1:
            r_current = r_current[:, 0:r_current_height - olap_size, :]

        # Copy current row to full mask
        np.copyto(mask_fullsized[:, row_idx:row_idx + r_current.shape[1], :], r_current)
        row_idx += r_current.shape[1]
        r_olap_prev = r_olap_bottom

    # colourize to palette
    mask_fullsized = np.expand_dims(mask_fullsized, axis=0)

    _mask_pred = colourize(np.argmax(mask_fullsized, axis=1), n_classes, palette=palette)

    probs_reconstructed = torch.max(torch.nn.functional.softmax(torch.tensor(mask_fullsized), dim=1), dim=1).values.numpy()

    #crop mask and probabilities to image size
    a = (w - w_full)//2
    aa = w - w_full - a
    b = (h - h_full)//2
    bb = h - h_full - b

    probs_reconstructed = probs_reconstructed[0,b:probs_reconstructed.shape[1]-bb,a:probs_reconstructed.shape[2]-aa].astype('float16')

    mask_reconstructed = _mask_pred[0,b:_mask_pred.shape[1]-bb,a:_mask_pred.shape[2]-aa,:].astype('float32')

    return mask_reconstructed, probs_reconstructed


def colourize(img, n_classes, palette=None):
    """
        Colourize one-hot encoded image by palette
        Input format: NCWH (one-hot class encoded).

        Parameters
        ------
        img: np.array array
            Image array.
        n_classes: int
            Number of classes.
        palette: list
            Colour palette for mask.

        Returns
        ------
        numpy array
            Colourized image array.
    """

    palette = palette if palette is not None else defaults.palette_rgb

    n = img.shape[0]
    w = img.shape[1]
    h = img.shape[2]

    # collapse one-hot encoding to single channel
    # make 3-channel (RGB) image
    img_data = np.moveaxis(np.stack((img,) * 3, axis=1), 1, -1).reshape(n * w * h, 3)

    # map categories to palette colours
    for i in range(n_classes):
        class_bool = img_data == np.array([i, i, i])
        class_idx = np.all(class_bool, axis=1)
        img_data[class_idx] = palette[i]

    return img_data.reshape(n, w, h, 3)


def coshuffle(img_array, mask_array):
    """
        Shuffle image/mask datasets with same indicies.

        Parameters
        ------
        img_array: np.array array
            Image array.
        mask_array: np.array array
            Image array.

        Returns
        ------
        numpy array
            Shuffled image array.
        numpy array
            Shuffled mask array.
    """

    idx_arr = np.arange(len(img_array))
    np.random.shuffle(idx_arr)
    img_array = img_array[idx_arr]
    mask_array = mask_array[idx_arr]

    return img_array, mask_array


def map_palette(img_array, key):
    """
        Map classes for different palettes. The key gives
        the new values to map palette

        Parameters
        ------
        img_array: tensor
            Image array.
        key: np.array array
            Palette mapping key.

        Returns
        ------
        numpy array
            Remapped image.
    """

    palette = range(len(key))
    data = img_array.numpy()
    index = np.digitize(data.ravel(), palette, right=True)
    return torch.tensor(key[index].reshape(img_array.shape))


def class_encode(img_array, palette):
    """
    Convert RGB mask array to class-index encoded values.
    Uses RGB-value encoding, where C = RGB (3). Outputs
    one-hot encoded classes, where C = number of classes
    Palette parameters in form [CC'], where C is the
    number of classes, C' = 3 (RGB)

    Parameters
    ------
    img_array: tensor
        Image array [NCWH].
    palette: list
        Colour palette for mask.

    Returns
    ------
    tensor
        Class-encoded image [NCWH].
    """

    assert img_array.shape[1] == 3, "Input data must be 3 channel (RGB)"

    (n, ch, w, h) = img_array.shape
    input_data = np.moveaxis(img_array.numpy(), 1, -1).reshape(n * w * h, ch)
    encoded_data = np.ones(n * w * h)

    # map mask colours to segmentation classes
    try:
        for idx, c in enumerate(palette):
            bool_idx = input_data == np.array(c)
            bool_idx = np.all(bool_idx, axis=1)
            encoded_data[bool_idx] = idx
    except Exception as inst:
        errorMessage('Mask cannot be encoded by selected palette. Please check schema settings.')
        exit(1)
    return torch.tensor(encoded_data.reshape(n, w, h), dtype=torch.uint8)


def load_files(path, exts):
    """
    Loads file path(s) of given extension(s) from directory path.

      Parameters
      ------
      path: str
         Directory/File path.
      exts: list
         List of file extensions.

      Returns
      ------
      list
         List of file names.
     """
    if not os.path.exists(path):
        errorMessage('File not found:\n\t{} .'.format(path))
        return

    files = []
    if os.path.isfile(path):
        ext = os.path.splitext(os.path.basename(path))[1]
        assert ext in exts, "File {} of type {} is invalid.".format(path, ext)
        files.append(path)
        
    elif os.path.isdir(path):
        files.extend(list(sorted([os.path.join(path, f)
                                  for f in os.listdir(path) if any(ext in f for ext in exts)])))

    return files


def get_fname(path):
    """
    Get file name from path

      Returns
      ------
      path: str
         File path.
    """
    if os.path.isfile(path):
        return os.path.splitext(os.path.basename(path))[0]
    return path


def mk_path(path, check=True):
    """
    Makes directory at path if none exists.

    Parameters
    ------
    path: str
        Directory path.
    check: bool
        Confirm that new directory be created.

    Returns
    ------
    path: str
        Created directory path.
    """

    if os.path.exists(path):
        return path
    elif check or input("\nRequested directory does not exist:\n\t{}"
                        "\n\nCreate?  (Enter \'Y\' or \'y\' for yes): ".format(path)) in ['Y', 'y']:
        os.makedirs(path)
        print('\nDirectory created:\n\t{}.'.format(path))
        return path
    else:
        print('Application stopped.')
        exit(0)

