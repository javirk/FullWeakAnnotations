import argparse
import pandas as pd
import os

import libs.utils as u


def main(config, df_test, output_path='final_results/res_alpha.csv', new_file=True):
    surface = pd.read_csv('surfaces/' + config['surface_file'],
                          names=['Model', 'split_cls', 'split_seg', 'IoU', 'Dice'])

    if config['surface_interpolation']['method'] == 'triangles':
        interpolator_iou = u.TriangleInterpolation(config, surface, column='IoU')
        interpolator_dice = u.TriangleInterpolation(config, surface, column='Dice')
    elif config['surface_interpolation']['method'] == 'splines':
        interpolator_iou = u.SplineInterpolation(config, surface, column='IoU')
        interpolator_dice = u.SplineInterpolation(config, surface, column='Dice')
    else:
        raise NotImplementedError

    dataset = config['dataset']

    results = []

    for i in range(len(df_test)):
        row = df_test.loc[i]
        budget = row['budget']
        type_run = row['type']
        ratio = row['ratio']
        steps = row['steps']
        split_cls = row['split_cls']
        split_seg = row['split_seg']
        split_cls_dataset = row['split_cls_dataset']
        split_seg_dataset = row['split_seg_dataset']

        point = [split_cls_dataset * 100, split_seg_dataset * 100]

        interpolated_iou = interpolator_iou(point, return_tensor=False)
        interpolated_dice = interpolator_dice(point, return_tensor=False)
        if interpolated_iou is None:
            results.append(
                [dataset, type_run, budget, steps, ratio, split_cls, split_seg, split_cls_dataset, split_seg_dataset,
                 "None", "None"])
            continue

        results.append([dataset, type_run, budget, steps, ratio, split_cls, split_seg, split_cls_dataset,
                        split_seg_dataset, interpolated_iou, interpolated_dice])

    df = pd.DataFrame(results,
                      columns=['Dataset', 'Type', 'Budget', 'Steps', 'Ratio', 'split_cls', 'split_seg',
                               'split_cls_dataset', 'split_seg_dataset', 'IoU', 'Dice'])
    if new_file:
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        default='gp_configs/gp_config_oct.yml',
                        type=str,
                        help='Path to the config file')

    parser.add_argument('-f', '--test-file',
                        type=str,
                        help='Path to the csv with the testing budgets and splits')

    FLAGS, unparsed = parser.parse_known_args()
    config = u.read_config(FLAGS.config)
    df_test = pd.read_csv(FLAGS.test_file)

    main(config, df_test, new_file=True)
