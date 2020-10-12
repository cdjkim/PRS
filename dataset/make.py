import os
import json
import tqdm
import dataset_maker
from argparse import ArgumentParser


# TODO
# 1. rename ids
# 2. upload make_dataset code
# 3. write readme.md file for constructing dataset
# 4. erase other stuff

def arg_parse():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='COCOseq')
    parser.add_argument('--phase', type=str, default='all')
    parser.add_argument('--data_source', '-src', type=str, default='data_source')
    parser.add_argument('--dest', '-dst', type=str, default='data')
    parser.add_argument('--download_thread', '-dth', type=int, default=8)

    args = parser.parse_args()
    args.data_source = os.path.join(args.data_source, args.dataset)
    args.dest = os.path.join(args.dest, args.dataset)

    return args

if __name__ == '__main__':
    args = arg_parse()

    maker = dataset_maker.maker[args.dataset](args.data_source, args.download_thread)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    phase_token = args.phase
    if args.phase == 'all':
        phase_token = ''

    id_fnames = [fname for fname in os.listdir(os.path.join(args.data_source, 'ids')) \
                 if phase_token in fname]
    id_fnames.sort()

    for id_fname in tqdm.tqdm(id_fnames, desc='Total'):

        with open(os.path.join(args.data_source, 'ids', id_fname), 'r') as f:
            ids = json.load(f)

        data_fname = id_fname.replace('_id_', '_{data}_').replace('.json', '.{ext}')
        maker.make(ids, os.path.join(args.dest, data_fname))

    maker.save_multihotdict(os.path.join(args.dest, 'multi_hot_dict_{dataset_name}.json'))



