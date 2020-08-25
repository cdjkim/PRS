import os
import torch
import random
import colorful
from tensorboardX import SummaryWriter
from models import MLabReservoir
from data import DataScheduler


def train_model(config, model: MLabReservoir,
                scheduler: DataScheduler,
                writer: SummaryWriter):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')

    os.makedirs(saved_model_path, exist_ok=True)

    prev_t = config['data_schedule'][0]['subsets'][0][1]
    done_t_num = 0

    results_dict = dict()
    for step, (x, y, t) in enumerate(scheduler):

        summarize = step % config['summary_step'] == 0
        # if we want to evaluate based on steps.
        evaluate = (
            step % config['eval_step'] == config['eval_step'] - 1
        )

        # find current task t's id in data_schedule to obtain the data name.
        for data_dict in config['data_schedule']:
            for subset in data_dict['subsets']:
                if subset[1] == t:
                    cur_subset = subset[0]

        # Evaluate the model when task changes
        if t != prev_t:
            done_t_num += 1
            results_dict = scheduler.eval(model, writer, step + 1, prev_t,
                                         eval_title='eval', results_dict=results_dict)
            # Save the model
            torch.save(model.state_dict(), os.path.join(
                saved_model_path, 'ckpt-{}'.format(str(step + 1).zfill(6))
            ))

            print(colorful.bold_green('\nProgressing to Task %d' % t).styled_string)

        if step == 0:
            print(colorful.bold_green('\nProgressing to Task %d' % t).styled_string)

        if done_t_num >= len(scheduler.schedule):
            writer.flush()
            return

        # learn the model
        for i in range(config['batch_iter']):
            if 'slab' in config['model_name']:
                model.learn(x, y, t, step*config['batch_iter'] +i,
                            scheduler.datasets[cur_subset].category_map,
                            scheduler.datasets[cur_subset].split_cats_dict,
                            data_obj=scheduler.datasets[cur_subset])
            else:
                model.learn(x, y, t, step*config['batch_iter'] +i,
                            scheduler.datasets[cur_subset].category_map,
                            scheduler.datasets[cur_subset].split_cats_dict,
                            data_obj=scheduler.datasets[cur_subset].subsets[t])

        prev_t = t
