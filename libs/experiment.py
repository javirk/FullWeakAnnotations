import yaml
import math
import pandas as pd
import torch
import libs.utils as u

MAX_NUM_CLS = {'pascal': 5717, 'retouch': 22723, 'spleen': 3150, 'airs': 20480, 'suim': 1525, 'oct': 22723,
               'cityscapes': 2975, 'det_pascal': 5717}
MAX_NUM_SEG = {'pascal': 10582, 'retouch': 1000, 'spleen': 3150, 'airs': 11076, 'suim': 1525, 'oct': 902,
               'cityscapes': 2975, 'det_pascal': 5717}

class Experiment:
    def __init__(self, **kwargs):
        self.parameters = kwargs['args']

    @staticmethod
    def splits_from_budget(config):
        """
        B = Ratio * #seg + #cls
        :param config: dict
        :return:
        """
        if not config['from_pretrained_file']:
            num_cls = 0
        else:
            num_cls = 5717 if config['dataset'] == 'pascal' else 22723
            # TODO: Change this. It should be read from a file
            print(f'The model is pretrained on {num_cls} images')
        config['seg_len'] = math.floor((config['budget'] - num_cls) / config['cost_seg-cls'])
        return config

    def splits_from_fixed(self, fixed_parameters):
        if fixed_parameters is None:
            return self

        p = self.parameters

        assert len(fixed_parameters) == 2, f'Only 2 parameters must be fixed. Given: {len(fixed_parameters)}'
        if 'budget' in fixed_parameters and 'seg_len' in fixed_parameters:
            max_cls = MAX_NUM_CLS[p['dataset']]
            cls_len = min(p['budget'] - p['cost_seg-cls'] * p['seg_len'], max_cls)
            self.parameters['cls_len'] = cls_len
            if cls_len == 0:
                self.parameters['num_iterations_classification'] = 0

        elif 'budget' in fixed_parameters and 'cls_len' in fixed_parameters:
            max_cls = MAX_NUM_CLS[p['dataset']]
            assert 0 <= p['cls_len'] < max_cls, f'Classification length not valid: {p["cls_len"]}'

            seg_len = math.floor((p['budget'] - p['cls_len']) / p['cost_seg-cls'])
            self.parameters['seg_len'] = seg_len

        elif 'cls_len' in fixed_parameters and 'seg_len' in fixed_parameters:
            max_cls = MAX_NUM_CLS[p['dataset']]
            assert 0 <= p['cls_len'] < max_cls, f'Classification length not valid: {p["cls_len"]}'

            budget = p['cls_len'] + p['cost_seg-cls'] * p['seg_len']
            self.parameters['budget'] = budget

        return self

    def splits_from_percentage(self):
        p = self.parameters
        max_cls = MAX_NUM_CLS[p['dataset']]
        max_seg = MAX_NUM_SEG[p['dataset']]

        cls_len = int(max_cls * p['split_cls'] / 100)
        seg_len = int(max_seg * p['split_seg'] / 100)
        budget = cls_len + p['cost_seg-cls'] * seg_len

        self.parameters['cls_len'] = cls_len
        self.parameters['seg_len'] = seg_len
        self.parameters['budget'] = budget
        return self

    @staticmethod
    def retrieve_fixed(pars):
        if 'fixed_parameters' in pars:
            fixed_parameters = pars['fixed_parameters']
            fixed_parameters = fixed_parameters.replace(' ', '').split(',')
            pars.pop('fixed_parameters')
            return pars, fixed_parameters
        else:
            return pars, None

    @classmethod
    def from_yml(cls, file):
        return cls(args=u.read_config(file))

    def modify_parameters(self, par):
        for k, v in par.items():
            if v in ['FALSE', 'TRUE']:
                v = u.str2bool(v)

            if '-' in k:
                parent, child = k.split('-')
                self.parameters[parent][child] = v
            else:
                self.parameters[k] = v

    def to_yml(self, name=None):
        with open(name, 'w') as outfile:
            yaml.dump(self.parameters, outfile, default_flow_style=False)


class Experiments:
    def __init__(self, experiments=[]):
        self.experiments = experiments

    def append(self, exp):
        self.experiments.append(exp)

    def to_yml(self, path):
        for i, exp in enumerate(self.experiments):
            exp.to_yml(name=f'{path}/train_config{i}.yml')

    @classmethod
    def from_table(cls, table_path, example_config, num_seeds=1):
        df = pd.read_csv(table_path)
        exp_parameters = df.to_dict(orient='records')
        experiments = []
        for s in range(num_seeds):
            for parameters in exp_parameters:
                if 'seg_seed' not in parameters.keys() and 'cls_seed' not in parameters.keys():
                    parameters['seg_seed'] = torch.randint(low=0, high=100, size=[1]).item()
                    parameters['cls_seed'] = parameters['seg_seed']
                exp = Experiment.from_yml(example_config)
                parameters, fixed_parameters = exp.retrieve_fixed(parameters.copy())
                exp.modify_parameters(parameters)
                exp.splits_from_fixed(fixed_parameters)
                experiments.append(exp)
        return cls(experiments)

    @classmethod
    def from_percentages(cls, p_cls, p_seg, example_config, num_seeds=1):
        experiments = []
        parameters = {}
        for s in range(num_seeds):
            for s_cls in p_cls:
                for s_seg in p_seg:
                    exp = Experiment.from_yml(example_config)
                    # fixed_parameters = ['cls_len', 'seg_len']
                    # The seed is not used in this case
                    parameters['split_cls'] = s_cls
                    parameters['split_seg'] = s_seg
                    exp.modify_parameters(parameters)
                    exp.splits_from_percentage()
                    experiments.append(exp)
        return cls(experiments)



if __name__ == '__main__':
    dataset = 'DET_PASCAL'
    if dataset == 'PASCAL':
        par_csv = '../configurations/parameters_PA.csv'
        model_config = '../configurations/train_config_model_pascal.yml'
        folder_exp = '../configurations/EXPERIMENTS/PASCAL/'
    elif dataset == 'DET_PASCAL':
        par_csv = '../configurations/parameters_PA.csv'
        model_config = '../configurations/train_config_model_pascal_detection.yml'
        folder_exp = '../configurations/EXPERIMENTS/DET_PASCAL/'
    elif dataset == 'SPLEEN':
        par_csv = '../configurations/parameters_SP.csv'
        model_config = '../configurations/train_config_model_sp.yml'
        folder_exp = '../configurations/EXPERIMENTS/SPLEEN/'
    elif dataset == 'AIRS':
        par_csv = '../configurations/parameters_AI.csv'
        model_config = '../configurations/train_config_model_ai.yml'
        folder_exp = '../configurations/EXPERIMENTS/AIRS/'
    elif dataset == 'SUIM':
        par_csv = '../configurations/parameters_SU.csv'
        model_config = '../configurations/train_config_model_su.yml'
        folder_exp = '../configurations/EXPERIMENTS/SUIM/'
    elif dataset == 'Cityscapes':
        par_csv = '../configurations/parameters_SU.csv' # TODO
        model_config = '../configurations/train_config_model_cityscapes.yml'
        folder_exp = '../configurations/EXPERIMENTS/CITYSCAPES/'
    else:
        par_csv = '../configurations/parameters_RT.csv'
        model_config = '../configurations/train_config_model_rt.yml'
        folder_exp = '../configurations/EXPERIMENTS/RETOUCH/'

    # exp = Experiments.from_table(par_csv, model_config, num_seeds=1)

    splits_cls = [80, 100]
    splits_seg = [80, 100]
    exp = Experiments.from_percentages(splits_cls, splits_seg, model_config, num_seeds=1)

    exp.to_yml(folder_exp)
