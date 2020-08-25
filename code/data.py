from abc import ABC, abstractmethod
from collections import Iterator
import os
import torch
import h5py
import json
import colorful
import random
import torchvision
import pickle
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    Subset,
    DataLoader,
    RandomSampler,
)
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter
from PIL import Image
from models.base import Model
from eval import validate, get_features

from utils import AverageMeter, Group_AverageMeter, f1_score_per_class, \
    precision_score_per_class, recall_score_per_class, \
    f1_score_overall, precision_score_overall, recall_score_overall, mean_average_precision

# =====================
# Base Classes and ABCs
# =====================
class DataScheduler(Iterator):
    def __init__(self, config):
        self.config = config
        self.schedule = config['data_schedule']
        self.datasets = {}
        self.eval_datasets = {}
        self.total_step = 0
        self.stage = -1

        # include simpler schedule form to config
        schedule_simple = []
        for i in range(len(self.schedule)):
            schedule_simple.append(self.schedule[i]['subsets'][0][1])

        self.config['schedule_simple'] = schedule_simple

        print("SCHEDULE SIMPLE: ", schedule_simple)

        # Prepare datasets
        for i, stage in enumerate(self.schedule):
            for j, subset in enumerate(stage['subsets']):  # e.g, [['mnist', 0], ['mnist', 1]]
                dataset_name, subset_name = subset

                if dataset_name in self.datasets:
                    continue

                self.datasets[dataset_name] = DATASET[dataset_name](self.config)
                self.eval_datasets[dataset_name] = DATASET[dataset_name](
                    self.config, train=False
                )

            if 'step' in stage:
                self.total_step += stage['step']
            elif 'epoch' in stage:
                self.total_step += stage['epoch'] \
                                   * (len(self.datasets[dataset_name])  # Total num of dataset. e.g., 60000 mnist.
                                      // self.config['batch_size'])
            elif 'steps' in stage:
                self.total_step += sum(stage['steps'])
            else:
                self.total_step += len(self.datasets[dataset_name]) \
                                   // self.config['batch_size']

        self.iterator = None


    def __next__(self):
        try:
            if self.iterator is None:
                raise StopIteration
            data = next(self.iterator)
        except StopIteration:
            # Progress to next stage
            # evaluate before progressing
            self.stage += 1

            # return early to evaluate one last time before exit.
            if self.stage >= len(self.schedule):
                return [1], [0], self.stage

            stage = self.schedule[self.stage]

            collate_fn = list(self.datasets.values())[0].collate_fn
            subsets = []
            if 'epoch' in stage:
                for epoch in range(stage['epoch']):
                    for dataset_name, subset_name in stage['subsets']:
                        subsets.append(
                            self.datasets[dataset_name].subsets[subset_name])
            else:
                for dataset_name, subset_name in stage['subsets']:
                    subsets.append(
                        self.datasets[dataset_name].subsets[subset_name])

            dataset = ConcatDataset(subsets)

            sampler = RandomSampler(dataset)

            self.iterator = iter(DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                collate_fn=collate_fn,
                sampler=sampler,
                drop_last=True,
            ))

            data = next(self.iterator)

        cur_t = self.config['schedule_simple'][self.stage]

        # Get next data
        return data[0], data[1], cur_t

    def __len__(self):
        return self.total_step

    def eval(self, model, writer, step, t, eval_title, results_dict):
        for eval_dataset in self.eval_datasets.values():
            eval_dataset.eval(model, writer, step, t, eval_title, results_dict)
        return results_dict


class BaseDataset(Dataset, ABC):
    name = 'base'

    def __init__(self, config, train=True):
        self.config = config
        self.subsets = {}
        self.train = train

    def eval(self, model: Model, writer: SummaryWriter, step, t, eval_title, results_dict):
        if self.config['eval']:
            return self._eval_model(model, writer, step, t, eval_title, results_dict)

    @abstractmethod
    def _eval_model(
            self,
            model,
            writer: SummaryWriter,
            step, t,eval_title, results_dict):
        raise NotImplementedError

    def collate_fn(self, batch):
        return default_collate(batch)


class CustomSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[self.indices[idx]])


# ================
# Generic Datasets
# ================
class ClassificationDataset(BaseDataset, ABC):
    num_classes = NotImplemented
    targets = NotImplemented

    def __init__(self, config, train=True):
        super().__init__(config, train)

    def __len__(self):
        return self.dataset_size

    def _eval_model(
            self,
            model: Model,
            writer: SummaryWriter,
            step, t, eval_title, results_dict=None):
        training = model.training
        model.eval()

        totals = []
        corrects = []

        # Accuracy of each subset
        for subset_name, subset in self.subsets.items():
            data = DataLoader(
                subset,
                batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'],
                collate_fn=self.collate_fn,
            )
            total = 0.
            correct = 0.

            for x, y in iter(data):
                with torch.no_grad():
                    pred = model(x).view(x.size(0), -1).argmax(dim=1)
                total += x.size(0)
                correct += (pred.cpu() == y).float().sum()
            totals.append(total)
            corrects.append(correct)
            accuracy = correct / total
            writer.add_scalar(
                'accuracy/%s/%s/%s' % (eval_title, self.name, subset_name),
                accuracy, step
            )

        # Overall accuracy
        total = sum(totals)
        correct = sum(corrects)
        accuracy = correct / total
        writer.add_scalar('accuracy/%s/%s/overall' % (eval_title, self.name),
                          accuracy, step)
        model.train(training)

    def special(self):
        pass

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        # sample_idx = np.random.choice(self.special(), size=sample_size, replace=False)

        sample_idx.sort()
        sample_list = []
        for i in sample_idx:
            sample_list.append(self[i])

        return sample_list


class MultiLabelDataset(Dataset, ABC):
    num_tasks = NotImplemented
    targets = NotImplemented

    def __init__(self, config, data_name, task_idx, transform=None, train=True):
        self.config = config
        self.data_name = data_name

        if train:
            self.split = 'train'
        else:
            self.split = str(config['eval_split']) # val or teval for full test and val use.

        assert self.split in {'train', 'val', 'test', 'teval', 'teorg'}

        baseFormat = str(config['data_root']) + "/{split}_task{task_idx}_{data}_{dataset}.{ext}"
        self.x_is_img = False
        # Load features
        if config['e'] == 'none':
            self.h = h5py.File(baseFormat.format(split=self.split, task_idx=task_idx, data='features', dataset=data_name, ext='hdf5'), 'r')
        else:   # Load images
            self.x_is_img = True
            self.h = h5py.File(baseFormat.format(split=self.split, task_idx=task_idx, data='imgs', dataset=data_name, ext='hdf5'), 'r')
        self.xs = self.h['images']
        # Load categories
        with open(baseFormat.format(split=self.split, task_idx=task_idx, data='multi_hot_categories', dataset=data_name, ext='json'), 'r') as j:
            self.categories = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        if train:
            self.dataset_size = len(self.categories)
        else:
            self.dataset_size = len(self.xs)

        print(self.data_name, self.split + '_task' + str(task_idx))

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        if self.split is 'train':
            x = torch.FloatTensor(self.xs[i])
            x = x / (255. if self.x_is_img else 1.)
        else:
            x = torch.FloatTensor(self.xs[i])
            x = x / (255. if self.x_is_img else 1.)

        if self.transform is not None:
            x = self.transform(x)

        category = torch.LongTensor(self.categories[i])

        return x, category

    def __len__(self):
        return self.dataset_size

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        sample_idx.sort()
        sample_list = []
        for i in sample_idx:
            sample_list.append(self[i])

        return sample_list


# =================
# Concrete Datasets
# =================
class MNIST(torchvision.datasets.MNIST, ClassificationDataset):
    name = 'mnist'
    # num_classes = 3
    num_classes = 10

    def __init__(self, config, train=True):
        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.ToTensor(),
        ]
        if config['x_c'] > 1:
            transform_list.append(
                lambda x: x.expand(config['x_c'], -1, -1)
            )
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        torchvision.datasets.MNIST.__init__(
            self, root=os.path.join(config['data_root'], 'mnist'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        normalized_p = np.empty([self.num_classes], dtype=float)
        if self.config['longtail']:
            pareto = np.empty([self.num_classes], dtype=float)
            for c in range(self.num_classes):
                alpha = 0.6
                x_m = 1
                p = alpha * (x_m ** alpha) / ((c + 1) ** (alpha + 1))
                pareto[c] = p

            for c in range(self.num_classes):
                if c == 0:
                    normalized_p[c] = 1.0
                else:
                    normalized_p[c] = pareto[c] / (pareto.max() - pareto.min())
        else:
            for c in range(self.num_classes):
                # normalized_p[c] = 0.20
                # normalized_p[c] = 0.30
                normalized_p[c] = 1.00


        major_cats = [c for c in range(self.num_classes) if normalized_p[c] >= 0.7]
        moder_cats = [c for c in range(self.num_classes) if normalized_p[c] < 0.7 and normalized_p[c] >= 0.1]
        minor_cats = [c for c in range(self.num_classes) if normalized_p[c] < 0.1]
        # for reporting detailed performance of alg at long-tail distribution
        self.split_cats_dict = {'major': major_cats, 'moderate': moder_cats, 'minor': minor_cats}
        self.category_map = np.asarray([i for i in range(self.num_classes)])

        longtail_idcs = pickle.load(open('resources/mnist_idcs.pkl', 'rb'))
        """
        multitask_idcs = list()
        for idcs in longtail_idcs.values():
            multitask_idcs.extend(idcs)
        """
        if train:
            # Create subset for each class
            size = 0
            # idcs_dict = dict()
            self.dataset_size = size
            for y in range(self.num_classes):
                self.subsets[y] = Subset(
                    # self, sub_indices
                    self, longtail_idcs[y]
                )
                size += len(self.subsets[y])

            self.dataset_size += size
        else:
            # Create subset for each class

            for y in range(self.num_classes):
                self.subsets[y] = Subset(
                    self,
                    list((self.targets == y).nonzero().squeeze(1).numpy())
                )

    def special(self):
        # to use in multiclass sanity check.
        self.tmp_lst = list((self.targets == 0).nonzero().squeeze(1).numpy()) + \
            list((self.targets == 1).nonzero().squeeze(1).numpy()) + \
            list((self.targets == 2).nonzero().squeeze(1).numpy())
        return self.tmp_lst

    def __len__(self):
        return self.dataset_size


class SVHN(torchvision.datasets.SVHN, ClassificationDataset):
    name = 'svhn'
    num_classes = 10

    def __init__(self, config, train=True):
        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        split = 'train' if train else 'test'
        torchvision.datasets.SVHN.__init__(
            self, root=os.path.join(config['data_root'], 'svhn'),
            split=split, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        normalized_p = np.empty([self.num_classes], dtype=float)
        if self.config['longtail']:
            pareto = np.empty([self.num_classes], dtype=float)
            for c in range(self.num_classes):
                alpha = self.config['alpha']#0.6
                x_m = 1
                p = alpha*(x_m**alpha) / ((c+1)**(alpha+1))
                pareto[c] = p

            for c in range(self.num_classes):
                if c == 0:
                    normalized_p[c] = 1.0
                else:
                    normalized_p[c] = pareto[c]/(pareto.max() - pareto.min())
        else:
            # the sum of balance has to be == 11k
            for c in range(self.num_classes):
                # normalized_p[c] = 0.20
                normalized_p[c] = 1.0 # using full dataset for sanity check.

        major_cats = [c for c in range(self.num_classes) if normalized_p[c] >= 0.7]
        moder_cats = [c for c in range(self.num_classes) if normalized_p[c] < 0.7 and normalized_p[c] >= 0.1]
        minor_cats = [c for c in range(self.num_classes) if normalized_p[c] < 0.1]
        # for reporting detailed performance of alg at long-tail distribution
        self.split_cats_dict = {'major': major_cats, 'moderate': moder_cats, 'minor': minor_cats}
        self.category_map = np.asarray([i for i in range(self.num_classes)])


        longtail_idcs = pickle.load(open('resources/svhn_idcs_a0.6.pkl', 'rb'))
        self.targets = torch.Tensor(self.labels)
        if train:
            # Create subset for each class
            size = 0
            self.dataset_size = size
            idcs_dict = dict()
            for y in range(self.num_classes):
                self.subsets[y] = Subset(
                    # self, sub_indices
                    self, longtail_idcs[y]
                )
                size += len(self.subsets[y])
        else:
            # Create subset for each class
            for y in range(self.num_classes):
                self.subsets[y] = Subset(
                    self,
                    list((self.targets == y).nonzero().squeeze(1).numpy())
                )

    def __len__(self):
        return self.dataset_size


class NUSWIDE(BaseDataset):
    name = 'nuswide'

    def __init__(self, config, train=True):
        """
        :param data_name: base name of processed datasets
        :param transform: image transform pipeline
        """
        super().__init__(config, train)
        # load major, moderate, minor cats lst
        with open('./resources/major_cats_nuswide.json', 'r') as f:
            major_cats = json.load(f)
        with open('./resources/moderate_cats_nuswide.json', 'r') as f:
            moderate_cats = json.load(f)
        with open('./resources/minor_cats_nuswide.json', 'r') as f:
            minor_cats = json.load(f)
        # for reporting detailed performance of alg at long-tail distribution
        self.split_cats_dict = {'major': major_cats, 'moderate': moderate_cats, 'minor': minor_cats}

        self.num_tasks = int(config['num_tasks'])
        # Should I use x_h and x_w in config yaml file?
        if train:
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

        else:
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

        if config['e'] == 'none':
            transform = None

        # multi_hot dict
        with open(os.path.join(str(config['data_root']),'multi_hot_dict_nuswide.json'), 'r') as j:
            self.category_map = np.asarray(json.load(j))

        # Create subset for each task
        for y in range(self.num_tasks):
            self.subsets[y] = MultiLabelDataset(config, self.name, task_idx=y, transform=transform, train=train)

    def collate_fn(self, batch):
        # num of items
        nitems = len(batch[0])

        results = []   # stack img, categories
        for i in range(0, nitems):
            results.append(torch.stack([item[i] for item in batch]))

        return tuple(results)

    def __len__(self):
        total_len = 0
        for y in range(self.num_tasks):
            total_len += len(self.subsets[y])
        return total_len

    def _eval_model(
            self,
            model: Model,
            writer: SummaryWriter,
            step, t, eval_title, results_dict):

        training = model.training
        model.eval()

        if t in self.config['schedule_simple']:
            t_idx = self.config['schedule_simple'].index(t)
        else:
            t_idx = len(self.config['schedule_simple']) - 1

        # for calculating total performance
        targets_total = []
        probs_total = []

        # Accuracy of each subset
        for order_i , t_i in enumerate(self.config['schedule_simple'][:t_idx+1]):
            subset_name = t_i
            last_id = self.config['schedule_simple'][-1] # should be -1. -2 for debugging.
            subset = self.subsets[t_i]
            data = DataLoader(
                subset,
                batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'],
                collate_fn=self.collate_fn,
            )

            # results is dict. {method: group_averagemeter_object}
            results, targets, probs = validate(subset_name, model, data, self.category_map,
                                                  results_dict, last_id, self.split_cats_dict)

            targets_total.append(targets)
            probs_total.append(probs)

            if subset_name in results_dict:
                results_dict[subset_name].append(results)
            else:
                results_dict[subset_name] = [results]


            for metric in results.keys():
                results[metric].write_to_excel(os.path.join(writer.logdir, 'results_{}.xlsx'.format(metric)),
                                                sheet_name='task {}'.format(subset_name),
                                                column_name='task {}'.format(self.config['schedule_simple'][t_idx]),
                                                info='avg')


        # =================================================================================================================
        # calculate scores for trained tasks.
        prefix = 'tally_'   # prefix for tensorboard plotting and csv filename

        targets_total = torch.cat(targets_total, axis=0)
        probs_total = torch.cat(probs_total, axis=0)
        predicts_total = probs_total > 0.5   # BCE style predicts
        total_metric = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'mAP']
        results = dict()   # reset results

        CP, CR, CF1, OP, OR, OF1, mAP = (AverageMeter() for _ in range(len(total_metric)))

        ncats = targets_total.sum(axis=0)
        # ignore classes in future tasks
        cats_in_task_idx = ncats > 0
        cats_in_task_name = self.category_map[cats_in_task_idx].tolist()
        targets_total = targets_total
        probs_total = probs_total
        predicts_total = predicts_total

        # calculate score
        precision_pc = torch.mean(precision_score_per_class(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0))
        recall_pc = torch.mean(recall_score_per_class(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0))
        # CF1. note that CF1 is not a mean value of categories' f1_score
        f1_pc = ((2*precision_pc*recall_pc)/(precision_pc+recall_pc)) if (precision_pc+recall_pc) > 0 else torch.tensor([0.])
        precision_oa = precision_score_overall(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0)
        recall_oa = recall_score_overall(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0)
        f1_oa = f1_score_overall(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0)
        map_ = mean_average_precision(targets_total[:, cats_in_task_idx], probs_total[:, cats_in_task_idx])
        # save to AverageMeter
        CP.update(precision_pc.item())
        CR.update(recall_pc.item())
        CF1.update(f1_pc.item())
        OP.update(precision_oa.item())
        OR.update(recall_oa.item())
        OF1.update(f1_oa.item())
        mAP.update(map_.item())

        results[prefix + 'CP'] = CP
        results[prefix + 'CR'] = CR
        results[prefix + 'CF1'] = CF1
        results[prefix + 'OP'] = OP
        results[prefix + 'OR'] = OR
        results[prefix + 'OF1'] = OF1
        results[prefix + 'mAP'] = mAP

        # for reporting major, moderate, minor cateogory performances
        for report_name in self.split_cats_dict.keys():
            reporter = Group_AverageMeter()

            # get report category idxes
            all_cats = self.category_map.tolist()
            task_cats = set(cats_in_task_name)
            report_cats = task_cats & set(self.split_cats_dict[report_name])
            report_cats_idx = torch.tensor([all_cats.index(cat) for cat in report_cats], dtype=torch.long)

            # CP, CR, CF1 performance of report_categories.
            _class_precision = precision_score_per_class(targets_total[:, report_cats_idx],
                                                         predicts_total[:, report_cats_idx], zero_division=0)
            _class_recall = recall_score_per_class(targets_total[:, report_cats_idx],
                                                   predicts_total[:, report_cats_idx], zero_division=0)
            _class_precision = torch.mean(_class_precision)
            _class_recall = torch.mean(_class_recall)
            # CF1 bias. note that CF1 is not a mean value of categories' f1_score
            _class_f1 = ((2*_class_precision*_class_recall)/(_class_precision+_class_recall)) \
                if (_class_precision+_class_recall)>0 else torch.tensor([0.])

            # OP, OR, OF1 performance of report_categories.
            _overall_precision = precision_score_overall(targets_total[:, report_cats_idx],
                                                        predicts_total[:, report_cats_idx], zero_division=0)
            _overall_recall = recall_score_overall(targets_total[:, report_cats_idx],
                                                predicts_total[:, report_cats_idx], zero_division=0)
            _overall_f1 = f1_score_overall(targets_total[:, report_cats_idx],
                                        predicts_total[:, report_cats_idx], zero_division=0)

            # mAP performance of report_categories.
            _mAP = mean_average_precision(targets_total[:, report_cats_idx],
                                          probs_total[:, report_cats_idx])

            reporter.update(['CP'], [_class_precision.item()], [1])
            reporter.update(['CR'], [_class_recall.item()], [1])
            reporter.update(['CF1'], [_class_f1.item()], [1])
            reporter.update(['OP'], [_overall_precision.item()], [1])
            reporter.update(['OR'], [_overall_recall.item()], [1])
            reporter.update(['OF1'], [_overall_f1.item()], [1])
            reporter.update(['mAP'], [_mAP.item()], [1])

            reporter.total.reset()

            # add to results
            results[prefix + report_name] = reporter


        # write to tensorboard and csv.
        task_len = t_idx + 1
        for metric in results.keys():
            # XXX for light tensorboard summary
            if not metric in [prefix+'CP', prefix+'CR', prefix+'OP', prefix+'OR']:
                results[metric].write(writer, '%s/%s/%s/task_len(%d)' %
                                    (metric, eval_title, self.name, task_len),
                                    step, info='avg')

            results[metric].write_to_excel(os.path.join(writer.logdir, 'results_{}.xlsx'.format(metric)),
                                            sheet_name=prefix,
                                            column_name='task {}'.format(self.config['schedule_simple'][t_idx]),
                                            info='avg')

        # =================================================================================================================
        # print performances at the end
        if t_idx == len(self.config['schedule_simple'])-1:
            src = writer.logdir
            csv_files = ['major', 'moderate', 'minor', 'OF1', 'CF1', 'mAP', \
                         prefix+'major', prefix+'moderate', prefix+'minor', prefix+'CF1', prefix+'OF1', prefix+'mAP', \
                         'forget']
            for csv_file in csv_files:
                try:
                    csv = pd.read_csv(os.path.join(src, 'results_{}.csv'.format(csv_file)), index_col=0)

                    # print performance after training last task
                    pd.set_option('display.max_rows', None)
                    print(colorful.bold_green('\n{:10} result'.format(csv_file)).styled_string)
                    print(csv.round(4).iloc[:,-1])

                    # save as txt
                    with open(os.path.join(src, 'summary.txt'), 'a') as summary_txt:
                        summary_txt.write('\n')
                        summary_txt.write('{:10} result\n'.format(csv_file))
                        summary_txt.write(csv.round(4).iloc[:,-1].to_string())
                        summary_txt.write('\n')

                except FileNotFoundError:
                    print("This excperiment doesn't have {} file!! continue.".format(csv_file))
                    continue

        model.train(training)

        return results_dict



class MSCOCO(BaseDataset):
    name = 'coco'

    def __init__(self, config, train=True):
        """
        :param data_name: base name of processed datasets
        :param transform: image transform pipeline
        """
        super().__init__(config, train)

        # load major, moderate, minor cats lst
        with open('./resources/major_cats.json', 'r') as f:
            major_cats = json.load(f)
        with open('./resources/moderate_cats.json', 'r') as f:
            moderate_cats = json.load(f)
        with open('./resources/minor_cats.json', 'r') as f:
            minor_cats = json.load(f)
        # for reporting detailed performance of alg at long-tail distribution
        self.split_cats_dict = {'major': major_cats, 'moderate': moderate_cats, 'minor': minor_cats}


        self.num_tasks = int(config['num_tasks'])

        if train:
            if config['e'] in ['vac_encoder', 'exstream_encoder_vac']:
                transform = transforms.Compose([
                    transforms.Resize((288, 288)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
            elif config['e'] == 'rma_encoder':
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
            else:
                transform = transforms.Compose([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

        else:
            if config['e'] in ['vac_encoder', 'exstream_encoder_vac']:
                transform = transforms.Compose([
                    transforms.Resize((288, 288)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
            elif config['e'] == 'rma_encoder':
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
            else:
                transform = transforms.Compose([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

        if config['e'] == 'none':
            transform = None

        # multi_hot dict
        with open(os.path.join(str(config['data_root']),'multi_hot_dict_{name}.json'.format(name=self.name)), 'r') as j:
            self.category_map = np.asarray(json.load(j))

        # Create subset for each task
        for y in range(self.num_tasks):
            self.subsets[y] = MultiLabelDataset(config, self.name, task_idx=y, transform=transform, train=train)

    def collate_fn(self, batch):
        nitems = len(batch[0])

        results = []   # stack img, categories
        for i in range(0, nitems):
            results.append(torch.stack([item[i] for item in batch]))

        return tuple(results)

    def __len__(self):
        total_len = 0
        for y in range(self.num_tasks):
            total_len += len(self.subsets[y])
        return total_len

    def _eval_model(
            self,
            model: Model,
            writer: SummaryWriter,
            step, t, eval_title, results_dict):

        training = model.training
        model.eval()

        if t in self.config['schedule_simple']:
            t_idx = self.config['schedule_simple'].index(t)
        else:
            t_idx = len(self.config['schedule_simple']) - 1

        # for calculating total performance
        targets_total = []
        probs_total = []

        # Accuracy of each subset
        for order_i , t_i in enumerate(self.config['schedule_simple'][:t_idx+1]):
            subset_name = t_i
            last_id = self.config['schedule_simple'][-1] # XXX should be -1. -2 for debugging.
            subset = self.subsets[t_i]
            data = DataLoader(
                subset,
                batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'],
                collate_fn=self.collate_fn,
            )

            # results is dict. {method: group_averagemeter_object}
            results, targets, probs = validate(subset_name, model, data, self.category_map,
                                                  results_dict, last_id, self.split_cats_dict)

            targets_total.append(targets)
            probs_total.append(probs)

            if subset_name in results_dict:
                results_dict[subset_name].append(results)
            else:
                results_dict[subset_name] = [results]



            for metric in results.keys():
                results[metric].write_to_excel(os.path.join(writer.logdir, 'results_{}.xlsx'.format(metric)),
                                                sheet_name='task {}'.format(subset_name),
                                                column_name='task {}'.format(self.config['schedule_simple'][t_idx]),
                                                info='avg')

        # =================================================================================================================
        # calculate scores for trained tasks.
        prefix = 'tally_'   # prefix for tensorboard plotting and csv filename

        targets_total = torch.cat(targets_total, axis=0)
        probs_total = torch.cat(probs_total, axis=0)
        predicts_total = probs_total > 0.5   # BCE style predicts
        total_metric = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1', 'mAP']
        results = dict()   # reset results

        CP, CR, CF1, OP, OR, OF1, mAP = (AverageMeter() for _ in range(len(total_metric)))

        ncats = targets_total.sum(axis=0)
        # ignore classes in future tasks
        cats_in_task_idx = ncats > 0
        cats_in_task_name = self.category_map[cats_in_task_idx].tolist()
        targets_total = targets_total
        probs_total = probs_total
        predicts_total = predicts_total

        # calculate score
        precision_pc = torch.mean(precision_score_per_class(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0))
        recall_pc = torch.mean(recall_score_per_class(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0))
        # CF1. note that CF1 is not a mean value of categories' f1_score
        f1_pc = ((2*precision_pc*recall_pc)/(precision_pc+recall_pc)) if (precision_pc+recall_pc)>0 else torch.tensor([0.])
        precision_oa = precision_score_overall(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0)
        recall_oa = recall_score_overall(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0)
        f1_oa = f1_score_overall(targets_total[:, cats_in_task_idx], predicts_total[:, cats_in_task_idx], zero_division=0)
        map_ = mean_average_precision(targets_total[:, cats_in_task_idx], probs_total[:, cats_in_task_idx])
        # save to AverageMeter
        CP.update(precision_pc.item())
        CR.update(recall_pc.item())
        CF1.update(f1_pc.item())
        OP.update(precision_oa.item())
        OR.update(recall_oa.item())
        OF1.update(f1_oa.item())
        mAP.update(map_.item())

        results[prefix + 'CP'] = CP
        results[prefix + 'CR'] = CR
        results[prefix + 'CF1'] = CF1
        results[prefix + 'OP'] = OP
        results[prefix + 'OR'] = OR
        results[prefix + 'OF1'] = OF1
        results[prefix + 'mAP'] = mAP

        # for reporting major, moderate, minor cateogory performances
        for report_name in self.split_cats_dict.keys():
            reporter = Group_AverageMeter()

            # get report category idxes
            all_cats = self.category_map.tolist()
            task_cats = set(cats_in_task_name)
            report_cats = task_cats & set(self.split_cats_dict[report_name])
            report_cats_idx = torch.tensor([all_cats.index(cat) for cat in report_cats], dtype=torch.long)

            # CP, CR, CF1 performance of report_categories.
            _class_precision = precision_score_per_class(targets_total[:, report_cats_idx],
                                                         predicts_total[:, report_cats_idx], zero_division=0)
            _class_recall = recall_score_per_class(targets_total[:, report_cats_idx],
                                                   predicts_total[:, report_cats_idx], zero_division=0)
            _class_precision = torch.mean(_class_precision)
            _class_recall = torch.mean(_class_recall)
            # CF1 bias. note that CF1 is not a mean value of categories' f1_score
            _class_f1 = ((2*_class_precision*_class_recall)/(_class_precision+_class_recall)) \
                if (_class_precision+_class_recall)>0 else torch.tensor([0.])

            # OP, OR, OF1 performance of report_categories.
            _overall_precision = precision_score_overall(targets_total[:, report_cats_idx],
                                                        predicts_total[:, report_cats_idx], zero_division=0)
            _overall_recall = recall_score_overall(targets_total[:, report_cats_idx],
                                                predicts_total[:, report_cats_idx], zero_division=0)
            _overall_f1 = f1_score_overall(targets_total[:, report_cats_idx],
                                        predicts_total[:, report_cats_idx], zero_division=0)

            # mAP performance of report_categories.
            _mAP = mean_average_precision(targets_total[:, report_cats_idx],
                                          probs_total[:, report_cats_idx])

            reporter.update(['CP'], [_class_precision.item()], [1])
            reporter.update(['CR'], [_class_recall.item()], [1])
            reporter.update(['CF1'], [_class_f1.item()], [1])
            reporter.update(['OP'], [_overall_precision.item()], [1])
            reporter.update(['OR'], [_overall_recall.item()], [1])
            reporter.update(['OF1'], [_overall_f1.item()], [1])
            reporter.update(['mAP'], [_mAP.item()], [1])

            reporter.total.reset()
            results[prefix + report_name] = reporter

        # write to tensorboard and csv.
        task_len = t_idx + 1
        for metric in results.keys():
            if not metric in [prefix+'CP', prefix+'CR', prefix+'OP', prefix+'OR']:
                results[metric].write(writer, '%s/%s/%s/task_len(%d)' %
                                    (metric, eval_title, self.name, task_len),
                                    step, info='avg')

            results[metric].write_to_excel(os.path.join(writer.logdir, 'results_{}.xlsx'.format(metric)),
                                            sheet_name=prefix,
                                            column_name='task {}'.format(self.config['schedule_simple'][t_idx]),
                                            info='avg')

        # =================================================================================================================
        # print performances at the end
        if t_idx == len(self.config['schedule_simple'])-1:
            src = writer.logdir
            csv_files = ['major', 'moderate', 'minor', 'OF1', 'CF1', 'mAP', \
                         prefix+'major', prefix+'moderate', prefix+'minor', prefix+'CF1', prefix+'OF1', prefix+'mAP', \
                         'forget']
            for csv_file in csv_files:
                try:
                    csv = pd.read_csv(os.path.join(src, 'results_{}.csv'.format(csv_file)), index_col=0)

                    # print performance after training last task
                    pd.set_option('display.max_rows', None)
                    print(colorful.bold_green('\n{:10} result'.format(csv_file)).styled_string)
                    print(csv.round(4).iloc[:,-1])

                    # save as txt
                    with open(os.path.join(src, 'summary.txt'), 'a') as summary_txt:
                        summary_txt.write('\n')
                        summary_txt.write('{:10} result\n'.format(csv_file))
                        summary_txt.write(csv.round(4).iloc[:,-1].to_string())
                        summary_txt.write('\n')

                except FileNotFoundError:
                    print("This excperiment doesn't have {} file!! continue.".format(csv_file))
                    continue

        model.train(training)

        return results_dict


DATASET = {
    MNIST.name: MNIST,
    SVHN.name: SVHN,
    MSCOCO.name: MSCOCO,
    NUSWIDE.name: NUSWIDE
}
