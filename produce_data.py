import os

import pywt
import numpy as np
import scipy.signal.spectral as sss
import tensorflow as tf
from PIL import Image
# import seaborn as sns
import matplotlib.pyplot as plt


def batch2image(signal_batch, sampling_rate, pooling_size, process_type='wavelet'):

    num, dim = np.shape(signal_batch)
    batch_set = []
    if process_type == 'wavelet':
        for i in range(num):
            image, _ = wavelet2image(signal_batch[i, :], sampling_rate)
            batch_set.append(image)
    elif process_type == 'stft':
        for i in range(num):
            image, _ = stft2image(signal_batch[i, :], sampling_rate)
            batch_set.append(image)
    else:
        raise KeyError("process_type must be wavelet of stft!")
    batch_set = np.array(batch_set)
    batch_set = image_downsampling(batch_set, pooling_size, form='avg_pooling')

    return batch_set

def wavelet2image(signal, sampling_rate, freq_dim_scale=256, wavelet_name='morl'):

    """
    :param signal: 1D temporal sequence
    :param sampling_rate: sampling rate for the sequence
    :param freq_dim_scale: frequency resolution
    :param wavelet_name: wavelet name for CWT, here we have 'morl', 'gaus', 'cmor',...
    :return: time-freq image and its reciprocal frequencies
    """

    freq_centre = pywt.central_frequency(wavelet_name)            # 所选小波的中心频率
    cparam = 2 * freq_centre * freq_dim_scale
    scales = cparam / np.arange(1, freq_dim_scale + 1, 1)         # 获取小波基函数的尺度参数 a 的倒数
    [cwt_matrix, frequencies] = pywt.cwt(signal, scales, wavelet_name, 1.0 / sampling_rate)

    return abs(cwt_matrix), frequencies

def stft2image(signal, sampling_rate, freq_dim_scale=256, window_name=('gaussian', 3.0)):

    """
    :param signal: signal input for stft
    :param sampling_rate:
    :param window_name: (gaussian,3), hann, hamming, etc.

    Notes
    -----
    Window types:

        `boxcar`, `triang`, `blackman`, `hamming`, `hann`, `bartlett`,
        `flattop`, `parzen`, `bohman`, `blackmanharris`, `nuttall`,
        `barthann`, `kaiser` (needs beta), `gaussian` (needs standard
        deviation), `general_gaussian` (needs power, width), `slepian`
        (needs width), `dpss` (needs normalized half-bandwidth),
        `chebwin` (needs attenuation), `exponential` (needs decay scale),
        `tukey` (needs taper fraction)

    :return: time-freq image and its frequencies
    """

    f, t, Zxx = sss.stft(signal, fs=sampling_rate, window=window_name, nperseg=freq_dim_scale)

    return Zxx, f

def image_downsampling(image_set, pooling_size=2, form='max_pooling', axis=None):

    """
    :param image_set: input image with large size
    :param pooling_size: down-sampling rate
    :param form: 'max_pooling' or 'avg_pooling'
    :param axis: if axis is not None, it means that the image will be down-sampled
                 just within it row(axis=0) or column(axis=1).
    :return: image has been down-sampled
    """

    num, time_dim, freq_dim = np.shape(image_set)[0], np.shape(image_set)[1], np.shape(image_set)[2]
    image_set = image_set.reshape(num, time_dim, freq_dim, 1)
    im_input = tf.placeholder(dtype=tf.float32, shape=[num, time_dim, freq_dim, 1])
    kernel_size = [pooling_size, 2*pooling_size]
    if axis == 0:
        kernel_size = [pooling_size, 1]
    elif axis == 1:
        kernel_size = [1, pooling_size]

    with tf.device('/gpu:0'):
        pooling_max = tf.contrib.slim.max_pool2d(im_input, kernel_size=kernel_size, stride=kernel_size)
        pooling_avg = tf.contrib.slim.avg_pool2d(im_input, kernel_size=kernel_size, stride=kernel_size)

    with tf.Session() as sess:
        down_sampling_im = sess.run(fetches=pooling_max, feed_dict={im_input: image_set})
        if form == 'avg_pooling':
            down_sampling_im = sess.run(fetches=pooling_avg, feed_dict={im_input: image_set})

    return down_sampling_im

def get_batch(filename, window_size=512, batch_size=1000, stride=180):
    data = np.loadtxt(filename)
    start = 0
    cnt = 0
    batch_data = []
    while start + window_size < data.shape[0] and cnt < batch_size:
        batch_data.append(data[start: start + window_size])
        start = start + stride + 1
        cnt += 1
    batch_data = np.array(batch_data)
    return batch_data

def get_multi_condition_data(path='./data/MFPT_data_multi_condition/', name='MFPT'):
    hps = os.listdir(path)
    for hp in hps:
        print("**********************condition: %s" % hp)

        for root, dirs, files in os.walk(os.path.join(path, hp)):
            print(files)

        output_path = './data/%s_wavelet_img_%s' % (name, hp)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i, file in enumerate(files):
            print("processing %s" % file)
            c = "normal" if "normal" in file else file.split('.')[0].split('_')[-1]
            label = "{}_{}".format(c, i)
            file_path = os.path.join(root, file)
            if c == 'normal':
                signal = get_batch(filename=file_path, batch_size=500)
            else:
                signal = get_batch(filename=file_path, batch_size=500)
            batch_image = batch2image(signal, sampling_rate=1, pooling_size=4)
            print("saving %s/%s.npy, shape %s" % (output_path, label, batch_image.shape))
            np.save("%s/%s.npy" % (output_path, label), batch_image)

# def get_MFPT_data():
#     path = './data/MFPT_data'
#     for root, dirs, files in os.walk(path):
#         print(files)
#         output_path = './data/MFPT_wavelet'
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)
#
#         for i, file in enumerate(files):
#             print("processing %s" % file)
#             c = file.split('_')[1] if "normal" in file else file.split('_')[-1]
#             label = "{}_{}".format(c, i)
#             print('label: ', label)
#             file_path = os.path.join(root, file)
#             if c == 'normal':
#                 signal = get_batch(filename=file_path, batch_size=200, stride=512)
#             else:
#                 signal = get_batch(filename=file_path, batch_size=100, stride=512)
#             batch_image = batch2image(signal, sampling_rate=1, pooling_size=4)
#             print("saving %s/%s.npy, shape %s" % (output_path, label, batch_image.shape))
#             np.save("%s/%s_%s.npy" % (output_path, label, batch_image.shape[0]), batch_image)


# get_multi_condition_data()

# a = np.load('./data/MFPT_wavelet/MFPT_wavelet_img_050/inner_0_0.npy')
# from utils.wavelet_img import *
# a, _ = Wav_img(path='.\data\MFPT_wavelet\MFPT_wavelet_img_others/normal', normalization=False).next_batch(8)
#
# for i, gen_img in enumerate(a[0: 8]):
#     # imageio.imsave(os.path.join(self.output_dir, './train_{:02d}_%s.png'.format(counter) % i), gen_img)
#     plt.subplot(1, 8, i + 1)
#     plt.axis('off')
#     print(gen_img.reshape([64, 64]))
#     print(gen_img.reshape([64, 64]).shape)
#     plt.imshow(gen_img.reshape([64, 64]), cmap='gray')
# plt.savefig(os.path.join('normal_杂乱.png'), format='png')

import numpy as np
import os

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=1, hspace=1)

inner = np.loadtxt('./data\MFPT_data_multi_condition/050/MFPT_fault_data_050lbs_inner.txt')[0: 1024]
plt.subplot(3, 1, 1)
plt.plot(inner, color='k')
outter = np.loadtxt('./data\MFPT_data_multi_condition/050/MFPT_fault_data_050lbs_outter.txt')[0: 1024]
plt.subplot(3, 1, 2)
plt.plot(outter, color='k')
normal = np.loadtxt('./data\MFPT_data_multi_condition/050/MFPT_normal_data_270lbs_11.txt')[0: 1024]
plt.subplot(3, 1, 3)
plt.plot(normal, color='k')
plt.savefig(os.path.join('aaaa.png'), format='png')