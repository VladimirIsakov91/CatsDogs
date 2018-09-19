import torch
from torch import tensor
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy
import logging
import zarr
import time
import random

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
numpy.random.seed(0)

cuda0 = torch.device('cuda:0')

logger = logging.getLogger('Network')
logging.basicConfig(level=logging.INFO)


class Network(nn.Module):

    def __init__(self,
                 input_channels,
                 padding,
                 batch_size,
                 kernel_size,
                 pool_kernel,
                 pool_stride,
                 n_hidden,
                 n_classes
                 ):

        super(Network, self).__init__()

        self.batch_size = batch_size
        self.input_channels = input_channels
        self.n_hidden = n_hidden
        self.padding = padding
        self.kernel_size = kernel_size
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.n_classes = n_classes
        self._resolution = None

        if self.input_channels == 3:
            self._mode = 'rgb'
        elif self.input_channels == 1:
            self._mode = 'gray'

        self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                               out_channels=64,
                               kernel_size=self.kernel_size,
                               padding=self.padding)

        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=32,
                               kernel_size=self.kernel_size,
                               padding=self.padding)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=16,
                               kernel_size=self.kernel_size,
                               padding=self.padding)

        self.normalization = nn.BatchNorm2d(self.input_channels)
        self.drop = nn.Dropout(p=0.4)
        self.linear0 = nn.Linear(1024, self.n_hidden)
        self.linear1 = nn.Linear(self.n_hidden, 10)
        self.linear2 = nn.Linear(10, self.n_classes)

    #0.6168 kernel 7
    #0.6682 kernel 5
    #0.6821 kernel 3

    def forward(self, batch):

        x = self.normalization(batch)
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), kernel_size=self.pool_kernel, stride=self.pool_stride)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), kernel_size=self.pool_kernel, stride=self.pool_stride)
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), kernel_size=self.pool_kernel, stride=self.pool_stride)
        x = x.view(x.size()[0], self.flatten(x))
        print(x.size())
        x = self.drop(x)
        x = F.leaky_relu(self.linear0(x))
        x = self.linear1(x)

        return x

    def flatten(self, x):

        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train_network(self, epochs, train_data,
                      resolution, weight_decay, lr, stepsize, gamma):

        self.cuda()
        self.train(True)

        self._resolution = resolution

        adam = optim.Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
        loss_f = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(adam, step_size=stepsize, gamma=gamma)
        samples = 0

        train_data = zarr.open(train_data, 'r')

        data = train_data['data']
        labels = train_data['labels']

        for epoch in range(epochs):

            start = time.time()
            avg_loss = numpy.zeros((1,))
            sample_size = 1
            scheduler.step()

            n_batches = int(numpy.ceil(data.shape[0] / self.batch_size))

            for idx in range(n_batches):

                data_batch = data[idx * self.batch_size:idx * self.batch_size + self.batch_size, :, :, :]
                label_batch = labels[idx * self.batch_size:idx * self.batch_size + self.batch_size]

                samples += int(data_batch.shape[0])

                data_batch = tensor(data_batch, dtype=torch.float32, requires_grad=True, device=cuda0)
                label_batch = tensor(label_batch, dtype=torch.int64, device=cuda0)

                adam.zero_grad()
                out = self(data_batch)

                loss = loss_f(out, label_batch)
                loss.backward()
                adam.step()
                sample_size += 1
                avg_loss += numpy.round(loss.cpu().detach().numpy(), 4)

            avg_loss /= sample_size
            avg_loss = numpy.round(avg_loss, 4)
            end = time.time()
            logger.info('Epoch {0}, Average Loss {1}, Time Elapsed {2} Minutes'
                        .format(epoch + 1, avg_loss.data[0], round((end - start) / 60.)))

        torch.save(self, 'model' + '.pt')
        logger.info('Number of Samples: {0}'.format(samples))

    def test(self, test_data):

        self.train(False)
        self.cuda()

        test_data = zarr.open(test_data, 'r')

        data = test_data['data']
        labels = test_data['labels']
        labels = numpy.array(labels, dtype=numpy.int32)

        accuracy = 0
        n_batches = int(numpy.ceil(data.shape[0] / self.batch_size))
        samples = 0

        for idx in range(n_batches):

            data_batch = data[idx * self.batch_size:idx * self.batch_size + self.batch_size, :, :, :]
            label_batch = labels[idx * self.batch_size:idx * self.batch_size + self.batch_size]

            samples += int(data_batch.shape[0])

            data_batch = tensor(data_batch, dtype=torch.float32, requires_grad=True, device=cuda0)

            out = self(data_batch)
            out = out.cpu().detach().numpy()
            out = numpy.argmax(out, axis=1)
            accuracy_ = out[out == label_batch]
            accuracy_ = accuracy_.shape[0]/data_batch.shape[0]
            accuracy += accuracy_

        accuracy /= n_batches
        accuracy = round(accuracy, 4)
        logger.info('Test Accuracy: {0}'.format(accuracy))


if __name__ == '__main__':

    train_data = 'train_data'
    test_data = 'test_data'

    model = Network(input_channels=1,
                    padding=1,
                    batch_size=64,
                    kernel_size=3,
                    pool_kernel=2,
                    pool_stride=2,
                    n_hidden=100,
                    n_classes=2)

    start = time.time()

    model.train_network(epochs=20,
                        train_data=train_data,
                        weight_decay=1e-5,
                        lr=1e-3,
                        stepsize=5,
                        gamma=0.5,
                        resolution=(64, 64))

    model.test(test_data=test_data)

    end = time.time()

    logger.info('Time Elapsed {0} Minutes'.format(round((end - start) / 60., 2)))