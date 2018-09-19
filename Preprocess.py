import os
import dask
import dask.array
from dask.distributed import Client
from dask import delayed
import multiprocessing
import zarr
import numpy
from PIL import Image
import logging
import time

logger = logging.getLogger('Preprocess')
logging.basicConfig(level=logging.INFO)


class ImagePreprocessor:

    def __init__(self):
        self._size = None

    #def __setattr__(self, key, value):
    #    pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @property
    def collection_size(self):
        return self._size

    def prepare_train_test_data(self, inp: str, out_channels: int, resolution: tuple, batch_size: int, out: str,
                    files: list=None):

        array, labels = ImagePreprocessor._load_collection(self, inp, out_channels, resolution, files)

        ImagePreprocessor._save(self,
                                data=array,
                                labels=labels,
                                batch_size=batch_size,
                                out=out)

    def _load_collection(self, inp: str, out_channels: int, resolution: tuple, files: list=None):

        if files is None:
            data = [delayed(pure=True)(i) for i in os.listdir(inp)]
        else:
            data = [delayed(pure=True)(i) for i in files]

        collection = [ImagePreprocessor._load_image(self,
                                                    name=inp + '/' + name,
                                                    out_channels=out_channels,
                                                    resolution=resolution)
                      for name in data]

        labels = ImagePreprocessor._load_labels(self, data)
        labels = dask.array.from_delayed(labels, shape=(len(data), ), dtype=numpy.int32)

        array = [dask.array.from_delayed(image, shape=(out_channels, resolution[0], resolution[1]),
                                         dtype=numpy.int32) for image in collection]
        array = dask.array.stack(array, axis=0)

        return array, labels

    @delayed(pure=True)
    def _load_image(self, name: str, out_channels: int, resolution: tuple):

        img = Image.open(name)
        if out_channels == 1:
            img = img.convert('L')
        img = img.resize(resolution)
        img = numpy.array(img, dtype=numpy.int32)
        img = img.reshape(out_channels, resolution[0], resolution[1])

        logger.info('Loaded Image {0}, {1}'.format(name, multiprocessing.current_process().name))

        return img

    @delayed(pure=True)
    def _load_labels(self, data):

        labels = []

        for name in data:
            if 'cat' in name:
                labels.append(1)
            elif 'dog' in name:
                labels.append(0)

        labels = numpy.array(labels, dtype=numpy.int32)

        return labels

    def _save(self, data, labels, batch_size, out):

        synchronizer = zarr.ProcessSynchronizer(out + '.sync')

        z = zarr.open(out, 'w', synchronizer=synchronizer)
        z.create_dataset('data', shape=data.shape, chunks=batch_size)
        z.create_dataset('labels', shape=labels.shape, chunks=batch_size)

        dask.array.store([data, labels], [z['data'], z['labels']],
                         lock=False,
                         scheduler='distributed')


if __name__ == '__main__':

    start = time.time()

    files = os.listdir('train')
    train_size = 0.8
    train_data = files[0:int(train_size*(len(files)))]
    test_data = files[int(train_size*(len(files))):]

    client = Client()
    preprocessor = ImagePreprocessor()

    #preprocessor.prepare_train_test_data(inp='train',
    #                        out_channels=1,
    #                        resolution=(64, 64),
    #                        batch_size=256,
    #                        out='train_data',
    #                        files=train_data)

    preprocessor.prepare_train_test_data(inp='train',
                            out_channels=1,
                            resolution=(64, 64),
                            batch_size=256,
                            out='test_data',
                            files=test_data)
    end = time.time()

    elapsed = round((end-start)/60., 2)
    logger.info('Time Elapsed {0} Minutes'.format(elapsed))