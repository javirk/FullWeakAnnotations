import pandas as pd
import argparse
import test_budget
import libs.utils as u
from libs.experiment import MAX_NUM_SEG, MAX_NUM_CLS

starting_points = {
    'oct': [4, 8],
    'pascal': [8, 6],
    'suim': [8, 8],
    'cityscapes': [8, 8],
}


def get_config(dataset):
    return f'gp_configs/gp_config_{dataset.lower()}.yml'


def main(df, policies):
    # Get combinations of dataset, alpha and budget (Dataset, Ratio, Target)
    df_combs = df[['Dataset', 'budget', 'ratio']].copy()
    df_combs.drop_duplicates(inplace=True, ignore_index=True)
    df_combs.sort_values(by=['Dataset', 'ratio', 'budget'], inplace=True)

    # Make a df with all the policies
    df_policies = pd.DataFrame(policies, columns=['split_cls', 'split_seg'])
    df_combs = df_combs.merge(df_policies, how='cross')
    df_combs['max_cls'] = df_combs.Dataset.map(lambda x: MAX_NUM_CLS[x])
    df_combs['max_seg'] = df_combs.Dataset.map(lambda x: MAX_NUM_SEG[x])
    df_combs['c0_p'] = df_combs.Dataset.map(lambda x: starting_points[x][0])
    df_combs['s0_p'] = df_combs.Dataset.map(lambda x: starting_points[x][1])
    df_combs['c0'] = df_combs.c0_p * df_combs.max_cls / 100
    df_combs['s0'] = df_combs.s0_p * df_combs.max_seg / 100
    df_combs['b0'] = df_combs.c0 + df_combs.s0 * df_combs.ratio
    df_combs['num_cls'] = (df_combs.budget - df_combs.b0) * df_combs.split_cls / 100 + df_combs.c0
    df_combs['num_seg'] = ((df_combs.budget - df_combs.b0) * df_combs.split_seg / 100) / df_combs['ratio'] + df_combs.s0

    df_combs['split_cls_dataset'] = df_combs['num_cls'] / df_combs['max_cls']
    df_combs['split_seg_dataset'] = df_combs['num_seg'] / df_combs['max_seg']
    df_combs['budget_new'] = df_combs['num_cls'] + 12 * df_combs['num_seg']
    df_combs['type'] = 'baseline'

    df['type'] = 'GP'
    df.rename(columns={'split_cls': 'split_cls_dataset', 'split_seg': 'split_seg_dataset'}, inplace=True)
    df['split_cls'] = 'Ours'

    df_total = pd.concat([df_combs, df])
    # df_total.rename(columns={'Target': 'budget'}, inplace=True)
    datasets = set(df_total['Dataset'])
    for i, dataset in enumerate(datasets):
        c = get_config(dataset)
        config = u.read_config(c)
        df_dataset = df_total[df_total['Dataset'] == dataset]
        df_dataset.reset_index(drop=True, inplace=True)
        test_budget.main(config, df_dataset, new_file=(i == 0),
                         output_path=FLAGS.output_file)  # Make a new file if it's the first iteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--runs-file',
                        default='final_results/gp_results_alpha.csv',
                        type=str,
                        help='Path to the csv with the gp budgets and splits')

    parser.add_argument('-o', '--output-file',
                        default='final_results/res_alpha.csv',
                        type=str,
                        help='Path to the result')

    baseline_combs = [(5, 95), (10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (35, 65), (40, 60), (45, 55), (50, 50)]

    FLAGS, unparsed = parser.parse_known_args()

    df_runs = pd.read_csv(FLAGS.runs_file)

    main(df_runs, baseline_combs)
