import copy
import os.path

import torch

import argparse
import gpytorch
import pandas as pd
import logging

import libs.utils as u
from libs.gp_utils import ExactGPModel, setup_gp_dir
from libs.experiment import MAX_NUM_SEG, MAX_NUM_CLS
from libs.grid_search import find_x_maxpoint

mapper_dataset = {'oct': 'OCT', 'pascal': 'VOC', 'cityscapes': 'Cityscapes', 'suim': 'SUIM'}


def main(config):
    assert config['surface_interpolation']['method'] == 'splines', 'Only splines are supported for now'
    assert config['method'] == 'gridmax', 'Only gridmax is supported for now'

    surface = pd.read_csv('surfaces/' + config['surface_file'],
                          names=['Model', 'split_cls', 'split_seg', 'IoU', 'Dice'])

    interpolator = u.SplineInterpolation(config, surface, column='Dice')

    dataset = config['dataset']
    i_split_cls = config['initial_point']['split_cls']
    i_split_seg = config['initial_point']['split_seg']
    config['delta_budget'] = config['target_budget'] / config['T']

    config = setup_gp_dir(config)
    logging.info(f'Target budget: {config["target_budget"]}')
    logging.info(f'Ratio: {config["cost_seg-cls"]}')

    for i_step in range(config['T']):
        logging.info(f'Step: {i_step}')
        remaining_steps = config['T'] - i_step

        # Prepare variables
        cls_splits, seg_splits, y = u.retrieve_points_surface(surface, i_split_cls, i_split_seg, column='Dice')
        cls_space = torch.tensor(MAX_NUM_CLS[dataset] * cls_splits / 100, dtype=torch.long)
        seg_space = torch.tensor(MAX_NUM_SEG[dataset] * seg_splits / 100, dtype=torch.long)
        all_space = torch.stack([cls_space, seg_space], dim=-1).to(torch.float)
        y = torch.tensor(y, dtype=torch.float)

        all_space_cost = all_space.clone().float()
        all_space_cost[:, 0] = all_space_cost[:, 0] / config['cost_seg-cls']

        # Find corner and add performance to list (only if it's not the first step)
        if i_step > 0:
            y_corner = interpolator([i_split_cls, i_split_seg], return_tensor=True)
            corner_point = torch.tensor([[MAX_NUM_CLS[dataset] * i_split_cls / (100 * config['cost_seg-cls']),
                                          MAX_NUM_SEG[dataset] * i_split_seg / 100]], dtype=torch.long)
            all_space_cost = torch.cat([all_space_cost, corner_point], dim=0)
            y = torch.cat([y, y_corner], dim=0)

        idx_corner = all_space_cost.sum(dim=1).argmax()
        i_performance = y[idx_corner].unsqueeze(0)
        i_corner = all_space_cost[idx_corner]

        # Train GP
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.rand(all_space_cost.shape[0]) * 0.1,
                                                                       learn_additional_noise=True)
        model = ExactGPModel(all_space_cost, y.squeeze(), likelihood, standardize=False, lr=config['GP']['lr'])
        model.optimize(all_space_cost, y.squeeze(), training_iter=config['GP']['iterations'],
                       verbose=config['GP']['verbose'])

        # Find trajectory -> point
        new_x = find_x_maxpoint(config, i_corner, i_performance, model, remaining_steps)

        # Transform the point to meaningful units
        i_cls = int(new_x[0] * config['cost_seg-cls'])
        i_seg = int(new_x[1])
        i_split_cls = i_cls / MAX_NUM_CLS[dataset] * 100
        i_split_seg = i_seg / MAX_NUM_SEG[dataset] * 100
        print(f'CLS: {i_split_cls:.2f}, SEG: {i_split_seg:.2f}')
        logging.info(f'CLS: {i_split_cls:.2f}, SEG: {i_split_seg:.2f}\n')

    return i_split_cls, i_split_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        default='gp_configs/gp_config_oct.yml',
                        type=str,
                        help='Path to the config file')

    parser.add_argument('-o', '--output-path',
                        default='final_results/gp_results.csv',
                        type=str,
                        help='Output path')

    FLAGS, unparsed = parser.parse_known_args()
    config = u.read_config(FLAGS.config)

    results = []

    assert len(config['target_budget']) == len(config['cost_seg-cls'])
    assert len(config['T']) == len(config['target_budget'])
    for t, c, s in zip(config['target_budget'], config['cost_seg-cls'], config['T']):
        print(f'Target budget: {t}. Ratio: {c}')
        config_run = copy.deepcopy(config)
        config_run['target_budget'] = t
        config_run['cost_seg-cls'] = c
        config_run['T'] = s

        for i in range(config['num_repetitions']):
            print(f'Repetition {i}')
            final_cls, final_seg = main(config_run)
            try:
                # final_cls, final_seg = main(config_run)
                results.append([config['dataset'], t, s, c, i, final_cls / 100, final_seg / 100])
            except Exception as err:
                print(f"Unexpected {err}, {type(err)}")

    df = pd.DataFrame(results, columns=['Dataset', 'budget', 'steps', 'ratio', 'num_run', 'split_cls', 'split_seg'])
    df.to_csv(FLAGS.output_path, mode='a', header=not os.path.exists(FLAGS.output_path), index=False)
