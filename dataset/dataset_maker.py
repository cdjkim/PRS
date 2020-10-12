"""
Collection of classes for making dataset easily
download function in NUSWIDEseqMaker is based on https://github.com/nmhkahn/NUS-WIDE-downloader
"""
import os
import json
import h5py
import tqdm
import skimage.io
import numpy as np
from PIL import Image
from collections import defaultdict
from threading import Thread


def open_image(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.asarray(img)

    if len(img.shape) != 3:  # gray-scale img
        img = np.stack([img, img, img], axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class BaseMaker():
    def __init__(self, source_path, *args):
        self.source_path = source_path
        with open(os.path.join(source_path, 'multihot_map.json'), 'r') as f:
            self.multihot_map = json.load(f)  # XXX multihot_map to be a dict format

    def make(self, ids, dst):
        raise NotImplementedError("make needs make func!")


class COCOseqMaker(BaseMaker):
    def __init__(self, cocosource_path, *args):
        super().__init__(cocosource_path, args)

        with open(os.path.join(self.source_path, 'annotations', 'instances_train2014.json'), 'r') as f:
            _instances_train2014 = json.load(f)

        with open(os.path.join(self.source_path, 'annotations', 'instances_val2014.json'), 'r') as f:
            _instances_val2014 = json.load(f)

        # dict for converting catid to catname
        self.categories = {cat_info['id']: cat_info['name'] for cat_info in _instances_train2014['categories']}

        # map cocoid with image path and cat names
        cocoid_mapped_instances = dict()
        for img in _instances_train2014['images'] + _instances_val2014['images']:
            cocoid_mapped_instances[img['id']] = \
                {'img_pth': os.sep.join(img['coco_url'].split('/')[-2:])}

        for ann in _instances_train2014['annotations'] + _instances_val2014['annotations']:
            if 'cats' not in cocoid_mapped_instances[ann['image_id']]:
                cocoid_mapped_instances[ann['image_id']]['cats'] = list()

            cat = self.categories[ann['category_id']]
            if cat not in self.multihot_map.keys():
                continue

            cocoid_mapped_instances[ann['image_id']]['cats'].append(cat)

        self.cocodata = cocoid_mapped_instances

    def make(self, ids, dst):
        imgs = []
        multihot_labels = []

        desc = '_'.join(dst.split(os.sep)[-1].split('_')[:2])
        for id in tqdm.tqdm(ids, desc=desc, leave=False):
            info = self.cocodata[id]  # keys: img_path, cats
            # get img
            img = open_image(os.path.join(self.source_path, info['img_pth']))
            imgs.append(img)

            # get multihot-label
            multihot_label = np.zeros(len(self.multihot_map), dtype=np.int64)
            for cat in info['cats']:
                multihot_label[self.multihot_map[cat]] = 1
            multihot_labels.append(multihot_label.tolist())

        hf = h5py.File(dst.format(data='imgs', ext='hdf5'), 'w')
        hf.create_dataset('images', data=np.asarray(imgs))
        #hf.create_dataset('labels', data=np.asarray(multihot_labels))
        hf.close()

        with open(dst.format(data='multi_hot_categories', ext='json'), 'w') as f:
            json.dump(multihot_labels, f)


    def save_multihotdict(self, dst):
        multihot_dict_name = dst.format(dataset_name='coco')
        multihot_dict = [None for _ in range(len(self.multihot_map))]
        for cat, idx in self.multihot_map.items():
            multihot_dict[idx] = cat
        with open(multihot_dict_name, 'w') as f:
            json.dump(multihot_dict, f)


class NUSWIDEseqMaker(BaseMaker):
    def __init__(self, nuswidesource_path, download_thread=1, *args):
        super().__init__(nuswidesource_path, args)
        self.download_thread = download_thread

        with open(os.path.join(self.source_path, 'ImageList', 'TrainImagelist.txt'), 'r') as f:
            _TrainImageList = f.readlines()
        with open(os.path.join(self.source_path, 'ImageList', 'TestImagelist.txt'), 'r') as f:
            _TestImageList = f.readlines()

        self.train_img_list = [fname.replace('\\', os.sep).strip() for fname in _TrainImageList]
        self.test_img_list = [fname.replace('\\', os.sep).strip() for fname in _TestImageList]

        train_labels = [_ for _ in range(len(self.multihot_map))]
        test_labels = [_ for _ in range(len(self.multihot_map))]
        for label_file in os.listdir(os.path.join(self.source_path, 'TrainTestLabels')):
            label = label_file.split('_')[1]
            if label not in self.multihot_map:
                continue

            with open(os.path.join(self.source_path, 'TrainTestLabels', label_file), 'r') as f:
                l_indicator = f.readlines()

            if 'Train' in label_file:
                train_labels[self.multihot_map[label]] = [int(l.strip()) for l in l_indicator]
            else:
                test_labels[self.multihot_map[label]] = [int(l.strip()) for l in l_indicator]

        # shape (L, N) L: # labels, N: # data
        train_labels = np.asarray(train_labels).astype(np.int64)
        test_labels = np.asarray(test_labels).astype(np.int64)
        # shape (N, L)
        self.train_label_list = np.transpose(train_labels, (1, 0))
        self.test_label_list = np.transpose(test_labels, (1, 0))


        if not os.path.exists(os.path.join(self.source_path, 'image')):
            self.download_imgs(src=os.path.join(self.source_path, 'NUS-WIDE-urls.txt'),
                               dst=os.path.join(self.source_path, 'image'))

    def make(self, ids, dst):
        is_train = 'train' in dst.lower()

        img_list = self.train_img_list if is_train else self.test_img_list
        label_list = self.train_label_list if is_train else self.test_label_list

        imgs = []
        multihot_labels = []

        not_available = 0
        desc = '_'.join(dst.split(os.sep)[-1].split('_')[:2])
        for id in tqdm.tqdm(ids, desc=desc, leave=False):
            try:
                img = open_image(os.path.join(self.source_path, 'image', img_list[id]))
                imgs.append(img)

                multihot_label = label_list[id]
                multihot_labels.append(multihot_label.tolist())
            except FileNotFoundError as e:
                not_available += 1

        hf = h5py.File(dst.format(data='imgs', ext='hdf5'), 'w')
        hf.create_dataset('images', data=np.asarray(imgs))
        hf.close()

        with open(dst.format(data='multi_hot_categories', ext='json'), 'w') as f:
            json.dump(multihot_labels, f)

        print('# {} imgs are not availbale for {}'.format(not_available, dst))

    def save_multihotdict(self, dst):
        multihot_dict_name = dst.format(dataset_name='nuswide')
        multihot_dict = [None for _ in range(len(self.multihot_map))]
        for cat, idx in self.multihot_map.items():
            multihot_dict[idx] = cat
        with open(multihot_dict_name, 'w') as f:
            json.dump(multihot_dict, f)


    def download_imgs(self, src, dst):
        with open(src, 'r') as f:
            urls = f.readlines()
        urls = urls[1:]  # delete header
        pbar = tqdm.tqdm(total=len(urls), desc="download images from urls")

        def download(urls, counter):
            for url in urls:
                pbar.update(1)
                infos = url.split()
                # weird format
                if len(infos) != 6:
                    counter["weird"] += 1
                    continue

                fname = infos[0]
                url_middle = infos[3]
                # no url
                if "null" == url_middle:
                    counter["no_url"] += 1
                    continue

                for i in range(3):
                    try:
                        im = skimage.io.imread(url_middle)
                    except:
                        im = None
                        continue
                    break
                # can't download from url
                if im is None:
                    counter["not_available"] += 1
                    continue

                dir, fname = fname.split('\\')[-2:]
                if not os.path.exists(os.path.join(dst, dir)):
                    os.makedirs(os.path.join(dst, dir))
                skimage.io.imsave(os.path.join(dst, dir, fname), im)

        total = defaultdict(lambda: 0)
        threads = [None for _ in range(self.download_thread)]
        counters = [defaultdict(lambda: 0) for _ in range(self.download_thread)]
        nchunk = len(urls) // self.download_thread

        for i in range(self.download_thread):
            st = nchunk * i
            en = nchunk * (i + 1) if i < self.download_thread - 1 else len(urls)

            threads[i] = Thread(target=download,
                                args=(urls[st:en], counters[i]))
            threads[i].start()

        for i in range(self.download_thread):
            threads[i].join()
            total["weird"] += counters[i]["weird"]
            total["no_url"] += counters[i]["no_url"]
            total["not_available"] += counters[i]["not_available"]

        print("total urls: \t", len(urls))
        print("weird urls: \t", total["weird"])
        print("no    urls: \t", total["no_url"])
        print("not available urls:\t", total["not_available"])


maker = {'COCOseq': COCOseqMaker,
         'NUSWIDEseq': NUSWIDEseqMaker}
