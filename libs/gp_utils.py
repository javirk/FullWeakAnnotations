import os
import numpy as np
import logging
import gpytorch
import torch
import gpytorch.kernels as k
from datetime import datetime


class LogMeanPrior(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.rand(*batch_shape, input_size)))
        if bias:
            self.register_parameter(name="c", parameter=torch.nn.Parameter(torch.rand(*batch_shape, input_size)))
        else:
            self.c = torch.ones((*batch_shape, input_size))

        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        w = self.softplus(self.weights)
        c = self.softplus(self.c)
        res = c[0] * torch.log(x[:, 0] * w[0] + 1) + c[1] * torch.log(x[:, 1] * w[1] + 1)
        return res


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, standardize=False, initialization=None, lr=0.1):
        self.standardize = standardize
        self.lr = lr
        if standardize:
            self.y_mean = train_y.mean()
            self.y_std = train_y.std(dim=0)
            train_y = (train_y - self.y_mean) / self.y_std

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = LogMeanPrior(input_size=2, bias=True)
        self.covar_module = k.ProductKernel(k.ScaleKernel(k.RBFKernel(ard_num_dims=2)))
        if initialization:
            self.load_model(initialization)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x_pred):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if isinstance(x_pred, np.ndarray):
                x_pred = torch.from_numpy(x_pred).type(torch.FloatTensor)

            prediction = self.likelihood(self(x_pred))
            mean = prediction.mean.detach()
            var = prediction.variance.detach()

            if self.standardize:
                # Rescale prediction to original training data scale
                original_mean = self.y_mean.detach()
                original_std = self.y_std.detach()
                mean = mean * original_std + original_mean
                var = var * original_std ** 2  # Variance is stationary and is only changed by a factor - https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/gaussian_process/_gpr.py#L355
            return mean, var

    def optimize(self, train_x, train_y, training_iter=1000, verbose=True):
        model = self
        likelihood = self.likelihood
        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if verbose and (i + 1) % 10 == 0:
                print('Iter %d/%d - Loss: %.3f - Noise: %.3f' % (
                    i + 1, training_iter, loss.item(), likelihood.noise.mean().item()))

    def load_model(self, file_path):
        state_dict = torch.load(file_path, map_location='cpu')
        overlap_dict = {k: v for k, v in state_dict.items() if 'mean_module' not in k}  # Don't load the mean module
        self.load_state_dict(overlap_dict, strict=False)


def ei(mu, mu_reference, sigma, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:

    Returns:
        Expected improvements at points X.
    '''

    with np.errstate(divide='warn'):
        imp = mu - mu_reference - xi
        Z = imp / sigma
        dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        ei = imp * dist.cdf(Z) + sigma * dist.log_prob(Z).exp()
        ei[sigma == 0.0] = 0.0
    return ei


def setup_gp_dir(p):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs('gp_runs', exist_ok=True)
    os.makedirs(f'gp_runs/{p["dataset"].upper()}', exist_ok=True)
    # os.makedirs(f'gp_runs/{p["dataset"].upper()}/{current_time}', exist_ok=True)
    file_run = os.path.join(f'gp_runs/{p["dataset"].upper()}/', current_time + '.txt')
    p['current_time'] = current_time

    logging.basicConfig(filename=file_run,
                        filemode='a',
                        format='%(message)s',
                        # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # logger = logging.getLogger()

    return p
