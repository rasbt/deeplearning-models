import os
import imageio
import numpy as np


def quickdraw_npy_to_imagefile(inpath, outpath, filetype='png', subset=None):
    """
    Creates a folder with subfolders for each image class
    from the Quickdraw dataset (https://quickdraw.withgoogle.com)
    downloaded in .npy format.

    To download the .npy formatted dataset:
      gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy quickdraw-png

    Usage example:
      quickdraw_npy_to_imagefile('quickdraw-npy', 'quickdraw-png')

    Parameters
    ----------

    inpath : str
        string specifying the path to the input directory containing
        the .npy files

    outpath : str
        string specifying the path for the output images

    subset : tuple or list (default=None)
        A subset of categories to consider. E.g.
        `("lollipop", "binoculars", "mouse", "basket")`

    """
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    npy_list = [i for i in os.listdir(inpath) if i.endswith('.npy')]

    if subset:
        npy_list = [i for i in npy_list if i.split('.npy')[0] in subset]

    if not len(npy_list):
        raise ValueError('No .npy files found in %s' % inpath)

    npy_paths = [os.path.join(inpath, i) for i in npy_list]

    for i, j in zip(npy_list, npy_paths):

        label = (i.split('-')[-1]).split('.npy')[0]
        folder = os.path.join(outpath, label)
        if not os.path.exists(folder):
            os.mkdir(folder)
        X = np.load(j)

        cnt = 0
        for row in X:
            img_array = row.reshape(28, 28)
            assert cnt < 1000000
            outfile = os.path.join(folder, '%s_%06d.%s' % (
                label, cnt, filetype))
            imageio.imwrite(outfile,
                            img_array[:, :])
            cnt += 1
