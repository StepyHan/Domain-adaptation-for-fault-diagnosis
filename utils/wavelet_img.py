import numpy as np
import os

one_hot = {"normal": [1, 0, 0, 0],
           "outter": [0, 1, 0, 0],
           "inner": [0, 0, 1, 0],
           "ball": [0, 0, 0, 1]}


# one_hot = {"normal": [1, 0, 0],
#            "outter": [0, 1, 0],
#            "inner": [0, 0, 1]}

class Wav_img():
    def __init__(self, path='./data/Wavelet_img_0HP', normalization=True):
        data = []
        label = []
        file_list = os.listdir(path)
        for file in file_list:
            file_path = os.path.join(path, file)
            print("loading %s" % file_path)
            file_data = np.load(file_path)
            n_sample = file_data.shape[0]
            file_label = np.tile(one_hot[file.split('_')[0]], [n_sample, 1])
            data.append(file_data)
            label.append(file_label)

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)

        self.filename = path.split('/')[-1].split('.')[0]

        self._num_examples = self.data.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.ranges, self.mean_arr = None, None

        if normalization:
            print("***************normalization******************")
            self.data, self.ranges, self.mean_arr = self.normalization(self.data)
        else:
            print("-------------without normalization-----------------")

        print("初次shuffle")
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.label = self.label[perm]

        self.train_data = self.data[: int(self._num_examples*0.7)][:]
        self.train_label = self.label[: int(self._num_examples*0.7)][:]

        self.test_data = self.data[int(self._num_examples*0.7):]
        self.test_label = self.label[int(self._num_examples*0.7):]

    def normalization(self, data):
        max_arr = data.max(axis=0)
        min_arr = data.min(axis=0)
        mean_arr = data.mean(axis=0)
        ranges = max_arr - min_arr
        m = data.shape[0]
        norDataSet = data - np.tile(mean_arr, (m, 1, 1, 1))
        norDataSet = norDataSet / np.tile(ranges, (m, 1, 1, 1))
        return norDataSet, ranges, mean_arr

    def next_batch(self, batch_size):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.train_data.shape[0]:
            print("---------------dataset message: finish %s" % self._epochs_completed)

            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.train_data.shape[0])
            print("---------------shuffle!!!!!")

            np.random.shuffle(perm)
            self.train_data = self.train_data[perm]
            self.train_label = self.train_label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.train_data[start: end], self.train_label[start: end]

def main():
    A = Wav_img()

    data, label = A.next_batch(64)
    i = 0
    while True:
        data, label = A.next_batch(64)
        print(label.shape)


if __name__ == '__main__':
    main()