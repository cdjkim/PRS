import torch
import colorful
import os
import math

import numpy as np

from utils import AverageMeter, Group_AverageMeter, accuracy, summarize_example_wise, average_lst, \
                f1_score_per_class, precision_score_per_class, recall_score_per_class, \
                f1_score_overall, precision_score_overall, recall_score_overall, mean_average_precision

from pprint import pprint


def get_features(task_id, model, data, cat_map):
    print(colorful.bold_yellow('Gathering Features of Task: ' + str(task_id)).styled_string)
    with torch.no_grad():
        features_total = []
        cats_total = []
        for i, (imgs, cats) in enumerate(iter(data)):
            # batch_size
            batch_size = imgs.shape[0]

            # Move to device, if available
            imgs = imgs.to(model.device)
            cats = cats.to(model.device)

            # Forward prop.
            features = model.get_features(imgs)
            cats_name = []
            for cat in cats:
                cats_name.append(cat_map[(cat>0).cpu()])

            features_total.append(features.cpu())
            cats_total.extend(cats_name)

        features_total = torch.cat(features_total, axis=0)

    return features_total, cats_total


def validate(task_id, model, data, cat_map, results_dict, last_id, additional_report_cats):
    """
    :param additional_report_cats: categories list for additional report. \
        dict {report_name: [cat1, cat2, cat3, ...] \
        ex) major: ['train', 'horse', 'bird', 'clock', ...], \
        minor: ['bicycle', 'potted plant', ....], \
        moderate: ['kite', 'bench', 'teddy bear', ...]

    """
    results = {}
    losses = AverageMeter()  # loss (per word decoded)
    accuracies = AverageMeter()
    class_precisions = Group_AverageMeter()
    class_recalls = Group_AverageMeter()
    class_f1s = Group_AverageMeter()

    overall_precisions = AverageMeter()
    overall_recalls = AverageMeter()
    overall_f1s = AverageMeter()

    mAP = AverageMeter()

    criterion = model.criterion

    print(colorful.bold_yellow('Validating Task: ' + str(task_id)).styled_string)

    cpu_targets = []
    cpu_predicts = []
    cpu_probs = []
    with torch.no_grad():
        for i, (imgs, cats) in enumerate(iter(data)):
            # batch_size
            batch_size = imgs.shape[0]

            # Move to device, if available
            imgs = imgs.to(model.device)
            cats = cats.to(model.device)

            # Forward prop.
            predict = model.encoder(imgs)
            targets = cats
            # Calculate loss
            loss = criterion(predict, targets)
            loss =loss.mean()

            predict = torch.sigmoid(predict)

            # for mAP score
            cpu_probs.append(predict.cpu())

            predict = predict > 0.5   # BCE
            total_relevant_slots = targets.sum().data
            relevant_predict = (predict * targets.float()).sum().data

            acc = relevant_predict / total_relevant_slots
            losses.update(loss.item(), batch_size)
            accuracies.update(acc, batch_size)

            cpu_targets.append(targets.cpu())
            cpu_predicts.append(predict.cpu())


        cpu_targets = torch.cat(cpu_targets, axis=0)
        cpu_predicts = torch.cat(cpu_predicts, axis=0)
        cpu_probs = torch.cat(cpu_probs, axis=0)

        ncats = cpu_targets.sum(axis=0)
        # ignore classes in other tasks
        cats_in_task_idx = ncats > 0
        ncats = ncats[cats_in_task_idx].tolist()

        f1_pc = f1_score_per_class(cpu_targets[:, cats_in_task_idx], cpu_predicts[:, cats_in_task_idx], zero_division=0)
        precision_pc = precision_score_per_class(cpu_targets[:, cats_in_task_idx], cpu_predicts[:, cats_in_task_idx], zero_division=0)
        recall_pc = recall_score_per_class(cpu_targets[:, cats_in_task_idx], cpu_predicts[:, cats_in_task_idx], zero_division=0)

        f1_oa = f1_score_overall(cpu_targets[:, cats_in_task_idx], cpu_predicts[:, cats_in_task_idx], zero_division=0)
        precision_oa = precision_score_overall(cpu_targets[:, cats_in_task_idx], cpu_predicts[:, cats_in_task_idx], zero_division=0)
        recall_oa = recall_score_overall(cpu_targets[:, cats_in_task_idx], cpu_predicts[:, cats_in_task_idx], zero_division=0)

        # record performances
        cats_in_task_name = cat_map[cats_in_task_idx].tolist()
        class_f1s.update(cats_in_task_name, f1_pc.tolist(), ncats)
        class_precisions.update(cats_in_task_name, precision_pc.tolist(), ncats)
        class_recalls.update(cats_in_task_name, recall_pc.tolist(), ncats)

        overall_f1s.update(f1_oa.item(), len(cpu_targets))
        overall_precisions.update(precision_oa.item(), len(cpu_targets))
        overall_recalls.update(recall_oa.item(), len(cpu_targets))

        # mAP
        mAP.update(mean_average_precision(cpu_targets[:, cats_in_task_idx], cpu_probs[:, cats_in_task_idx]))

        # for reporting major, moderate, minor cateogory performances
        for report_name in additional_report_cats.keys():
            reporter = Group_AverageMeter()

            # get report category idxes
            all_cats = cat_map.tolist()
            task_cats = set(cats_in_task_name)
            report_cats = task_cats & set(additional_report_cats[report_name])
            cats_idx = []
            for cat in report_cats:
                cats_idx.append(all_cats.index(cat))
            report_cats_idx = torch.tensor(cats_idx, dtype=torch.long)

            # there are tasks where the min/mod/maj are missing.
            if len(report_cats_idx) == 0:
                reporter.update(['CP'], [float('NaN')], [1])
                reporter.update(['CR'], [float('NaN')], [1])
                reporter.update(['CF1'], [float('NaN')], [1])
                reporter.update(['OP'], [float('NaN')], [1])
                reporter.update(['OR'], [float('NaN')], [1])
                reporter.update(['OF1'], [float('NaN')], [1])
                reporter.update(['mAP'], [float('NaN')], [1])

                # for major, moderate and minor report, total is a meaningless metric.
                # mean of CP, CR, CF1, ... is meaningless.
                reporter.total.reset()
                # add to results
                results[report_name] = reporter
                continue

            # CP, CR, CF1 performance of report_categories.
            _class_precision = precision_score_per_class(cpu_targets[:, report_cats_idx],
                                                         cpu_predicts[:, report_cats_idx], zero_division=0)
            _class_recall = recall_score_per_class(cpu_targets[:, report_cats_idx],
                                                   cpu_predicts[:, report_cats_idx], zero_division=0)
            _class_precision = torch.mean(_class_precision)
            _class_recall = torch.mean(_class_recall)
            # CF1 bias. note that CF1 is not a mean value of categories' f1_score
            _class_f1 = ((2*_class_precision*_class_recall)/(_class_precision+_class_recall)) \
                if (_class_precision+_class_recall)>0 else torch.tensor([0.])

            # OP, OR, OF1 performance of report_categories.
            _overall_precision = precision_score_overall(cpu_targets[:, report_cats_idx],
                                                        cpu_predicts[:, report_cats_idx], zero_division=0)
            _overall_recall = recall_score_overall(cpu_targets[:, report_cats_idx],
                                                cpu_predicts[:, report_cats_idx], zero_division=0)
            _overall_f1 = f1_score_overall(cpu_targets[:, report_cats_idx],
                                        cpu_predicts[:, report_cats_idx], zero_division=0)

            # mAP performance of report_categories.
            _mAP = mean_average_precision(cpu_targets[:, report_cats_idx],
                                          cpu_probs[:, report_cats_idx])

            reporter.update(['CP'], [_class_precision.item()], [1])
            reporter.update(['CR'], [_class_recall.item()], [1])
            reporter.update(['CF1'], [_class_f1.item()], [1])
            reporter.update(['OP'], [_overall_precision.item()], [1])
            reporter.update(['OR'], [_overall_recall.item()], [1])
            reporter.update(['OF1'], [_overall_f1.item()], [1])
            reporter.update(['mAP'], [_mAP.item()], [1])

            # for major, moderate and minor report, total is a meaningless metric.
            # mean of CP, CR, CF1, ... is meaningless.
            reporter.total.reset()

            # add to results
            results[report_name] = reporter

        # CF1 bias. note that CF1 is not a mean value of categories' f1_score
        class_f1s.total.reset()
        p_pc, r_pc = torch.mean(precision_pc).item(), torch.mean(recall_pc).item()
        class_f1s.total.update(((2*p_pc*r_pc)/(p_pc+r_pc)) if (p_pc+r_pc)>0 else 0)


        # save performances
        results['OF1'] = overall_f1s
        results['OP'] = overall_precisions
        results['OR'] = overall_recalls
        results['CF1'] = class_f1s
        results['CP'] = class_precisions
        results['CR'] = class_recalls
        results['losses'] = losses
        results['accuracies'] = accuracies

        results['mAP'] = mAP

        # Forgetting Measure
        if int(task_id) == int(last_id) and len(results_dict) > 0:
            forget_metrics = ['mAP', 'OF1', 'CF1']
            forget = Group_AverageMeter()
            Cf1_forget = Group_AverageMeter()
            forget_results = {}
            per_cat_forget_results = {}
            for metric in forget_metrics:
                per_metric_results = {}
                for task_name, per_task_results in results_dict.items():
                    if metric == 'CF1':
                        per_total_lst = []
                        per_cat_dict = {}
                        # only up to the 2nd last are used to find the max.
                        for per_task_result in per_task_results[:-1]:
                            per_total_lst.append(per_task_result[metric].total.avg)
                            for cat, cat_avgmtr in per_task_result[metric].data.items():
                                if cat in per_cat_dict:
                                    per_cat_dict[cat].append(cat_avgmtr.avg)
                                else:
                                    per_cat_dict[cat] = [cat_avgmtr.avg]

                        final_task_result = per_task_results[-1][metric].total.avg
                        max_task_result = max(per_total_lst)
                        # subtract the very last added and max of the tasks before.
                        metric_forgot = None
                        if max_task_result == 0 and  final_task_result == 0:
                            forget_results[metric+'_'+str(task_name)] = 1.0
                            metric_forgot = 1.0
                        elif max_task_result == 0 and  final_task_result != 0:
                            metric_forgot = (max_task_result - final_task_result)/1
                            forget_results[metric+'_'+str(task_name)] = metric_forgot
                        else:
                            metric_forgot = (max_task_result - final_task_result)/abs(max_task_result)
                            forget_results[metric+'_'+str(task_name)] = metric_forgot

                        for cat, catobj in per_task_results[-1][metric].data.items():
                            max_cat_result = max(per_cat_dict[cat])
                            final_cat_result = catobj.avg
                            if max_cat_result == 0 and  final_cat_result == 0:
                                per_cat_forget_results[cat] = 1.0
                            elif max_cat_result == 0 and  final_cat_result != 0:
                                per_cat_forget_results[cat] = max_cat_result - catobj.avg/1
                            else:
                                per_cat_forget_results[cat] = (max_cat_result \
                                        - catobj.avg)/abs(max_cat_result)
                    else:
                        per_metric_lst = []
                        for per_task_result in per_task_results[:-1]:
                            per_metric_lst.append(per_task_result[metric].avg)

                        metric_forgot = None
                        final_task_result = per_task_results[-1][metric].avg
                        max_task_result = max(per_metric_lst)
                        if max_task_result == 0 and  final_task_result == 0:
                            metric_forgot = 1.0
                            forget_results[metric+'_'+str(task_name)] = metric_forgot
                        elif max_task_result == 0 and  final_task_result != 0:
                            metric_forgot = (max_task_result - final_task_result)/1
                            forget_results[metric+'_'+str(task_name)] = metric_forgot
                        else:
                            metric_forgot = (max_task_result - final_task_result)/abs(max_task_result)
                            forget_results[metric+'_'+str(task_name)] = metric_forgot

                    for split in ['major', 'moderate', 'minor']:
                        # check if split results in all NaNs
                        if math.isnan(per_task_results[0][split].data[metric].avg):
                            forget_results[split + '_' + metric+'_'+str(task_name)] = float('NaN')
                            continue

                        per_metric_lst = []
                        for per_task_result in per_task_results[:-1]:
                            per_metric_lst.append(per_task_result[split].data[metric].avg)
                        final_task_result = per_task_results[-1][split].data[metric].avg
                        max_task_result = max(per_metric_lst)
                        split_forgot = None
                        if max_task_result == 0 and  final_task_result == 0:
                            split_forgot = 1.0 # forgotten within the first task by majority dominance.
                            forget_results[split + '_' + metric+'_'+str(task_name)] = split_forgot
                        elif max_task_result == 0 and  final_task_result != 0:
                            split_forgot = (max_task_result - final_task_result)/1
                            forget_results[split + '_' + metric+'_'+str(task_name)] = split_forgot
                        else:
                            split_forgot = (max_task_result - final_task_result)/abs(max_task_result)
                            forget_results[split + '_' + metric+'_'+str(task_name)] = split_forgot

                        if metric + split+'Overall' in per_metric_results.keys():
                            per_metric_results[metric + split + 'Overall'].append(split_forgot)
                        else:
                            per_metric_results[metric + split + 'Overall'] = [split_forgot]


                    if metric+'Overall' in per_metric_results.keys():
                        per_metric_results[metric + 'Overall'].append(metric_forgot)
                    else:
                        per_metric_results[metric + 'Overall'] = [metric_forgot]

                forget_results[metric + 'Overall'] = average_lst(per_metric_results[metric + 'Overall'])
                forget_results[metric + 'majorOverall'] = average_lst(per_metric_results[metric + 'majorOverall'])
                forget_results[metric + 'moderateOverall'] = average_lst(per_metric_results[metric + 'moderateOverall'])
                forget_results[metric + 'minorOverall'] = average_lst(per_metric_results[metric + 'minorOverall'])

            keys = []
            values = []
            n = []
            for k, v in forget_results.items():
                keys.append(k)
                values.append(v)
                n.append(1)

            forget.update(keys, values, n)

            keys = []
            values = []
            n = []
            for k, v in per_cat_forget_results.items():
                keys.append(k)
                values.append(v)
                n.append(1)

            Cf1_forget.update(keys, values, n)

            results['forget'] = forget
            results['class_forget'] = Cf1_forget

        print(colorful.bold_cyan(
            'LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f}, RECALL - {rcl.total.avg:.4f},\
            PRECISION - {prc.total.avg:.4f}, F1 - {f1.total.avg:.4f}'
            .format(loss=losses, acc=accuracies, rcl=class_recalls, prc=class_precisions, f1=class_f1s)).styled_string)

    return results, cpu_targets, cpu_probs


