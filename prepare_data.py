#encoding: utf-8
import os
import numpy as np
import h5py
from PIL import Image
import random
import wget

TRAINING_SAMPLE_PERCENTAGE = 0.9

def convert_nyu(path):
    assert TRAINING_SAMPLE_PERCENTAGE < 1.0, "training example percentage must be < 1.0"

    imgdir = os.path.join("data", "nyu_datasets")
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    nyuurl = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
    file = os.path.join("data", "nyu_depth_v2_labeled.mat")
    if not os.path.exists(file):
        filename = wget.download(nyuurl, out="data")
        print('\n downloaded: ', filename)
        return

    print("load dataset: %s" % (path))
    f = h5py.File(path, 'r')

    trains = []
    file_count = len(f['depths'])
    # for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
    for i in range(file_count):
        image = f['images'][i]
        depth = f['depths'][i]

        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0)

        # normalize image to 0, 255 based on max depth value in image.
        re_depth = (ra_depth/np.max(ra_depth))*255.0
        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        image_name = os.path.join("data", "nyu_datasets", "%05d.jpg" % (i))
        image_pil.save(image_name)
        depth_name = os.path.join("data", "nyu_datasets", "%05d.png" % (i))
        depth_pil.save(depth_name)

        trains.append((image_name, depth_name))

        if i % 100 == 0:
            print ('processed %d out of %d files' % (i, file_count))

    print ('processed %d out of %d files' % (file_count, file_count))

    random.shuffle(trains)

    if os.path.exists('train.csv'):
        os.remove('train.csv')

    train_sample_count = (int)(TRAINING_SAMPLE_PERCENTAGE * len(trains))
    with open('train.csv', 'w') as output:
        for i in range(0, train_sample_count):
            image_name = trains[i][0]
            depth_name = trains[i][1]
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

    with open('test.csv', 'w') as output:
        for i in range(train_sample_count, len(trains)):
            image_name = trains[i][0]
            depth_name = trains[i][1]
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

if __name__ == '__main__':
    current_directory = os.getcwd()
    nyu_path = 'data/nyu_depth_v2_labeled.mat'
    convert_nyu(nyu_path)
