import torch
import numpy as np
from functools import partial
from libs.gp_utils import ei
from libs.experiment import MAX_NUM_SEG, MAX_NUM_CLS


def best_cost(xy, xy_corner, hashmap_bestcost, hashmap_gp):
    # Luckily, dictionaries are mutable
    if xy[0] < xy_corner[0] or xy[1] < xy_corner[1]:
        return 0
    left_point = xy.clone()
    left_point[0] -= 1
    down_point = xy.clone()
    down_point[1] -= 1

    xy_tuple = tuple(xy.int().tolist())

    if xy_tuple not in hashmap_bestcost:
        hashmap_bestcost[xy_tuple] = hashmap_gp[xy_tuple] + \
                                     max(best_cost(left_point, xy_corner, hashmap_bestcost, hashmap_gp),
                                         best_cost(down_point, xy_corner, hashmap_bestcost, hashmap_gp))
    return hashmap_bestcost[xy_tuple]



def gp_region(xy_corner, xy_target, gp_model, f_ei):
    x1 = np.arange(xy_corner[0].int(), xy_target[0] + 1, 1, dtype=int)
    x2 = np.arange(xy_corner[1].int(), xy_target[1] + 1, 1, dtype=int)
    x1, x2 = np.meshgrid(x1, x2)
    xx_space = np.vstack([x1.flatten(), x2.flatten()]).T
    xx_space = torch.from_numpy(xx_space).to(torch.float)
    pred, var = gp_model.predict(xx_space)
    ei_space = f_ei(mu=pred, sigma=torch.sqrt(var))
    hashmap_gp = {tuple(x.ceil().tolist()): y.item() for x, y in zip(xx_space, ei_space)}
    return hashmap_gp


def max_point(config, current_point, current_y, gp_model):
    b = config['target_budget']

    max_seg = int(min(b / config['cost_seg-cls'] - current_point[0], MAX_NUM_SEG[config['dataset']]))

    x2 = np.arange(current_point[1] + 1, max_seg + 1, 1, dtype=int)
    x1 = (b / config['cost_seg-cls'] - x2).astype(int)
    xx_space = torch.tensor([(x, y) for x, y in zip(x1, x2)
                             if x * config['cost_seg-cls'] < MAX_NUM_CLS[config['dataset']] and x >= current_point[0]])
    xx_space = xx_space.to(torch.float)

    y_pred, var = gp_model.predict(xx_space)

    ei_budgetline = ei(y_pred, current_y, torch.sqrt(var))
    ei_max = ei_budgetline.argmax()
    return xx_space[ei_max], (xx_space, y_pred)


def ei_based_step(trajectory, ei_traj, remaining_steps):
    max_delta_ei = ei_traj.max() - ei_traj.min()
    max_delta_ei_step = max_delta_ei / remaining_steps
    ei_mask = (ei_traj > (max_delta_ei_step + ei_traj.min()))
    ei_traj[ei_mask] = 0
    new_x = trajectory[ei_traj.argmax()]
    return new_x


def find_x_maxpoint(config, corner_point, corner_performance, gp_model, remaining_steps):
    # Define hashmap
    # Find final point (max EI) -> xy
    # Get cost of all the points
    # Sort points by budget
    # Drop points that decrease EI -> trajectory to budget
    # Take budget that Bf/num_steps_remaning
    # new_x == point with that budget that we have selected

    # Housekeeping
    i_corner_tuple = tuple(corner_point.int().tolist())
    hashmap = {i_corner_tuple: 0}
    f_ei = partial(ei, mu_reference=corner_performance)

    # Find target point (on the budget line)
    target_point, (budget_line, ei_budgetline) = max_point(config, corner_point, corner_performance, gp_model)

    # Run GP on all the points
    hashmap_gp = gp_region(corner_point, target_point, gp_model, f_ei)  # xx_space is in equal-budget dimensions
    xx_space = torch.tensor(list(hashmap_gp.keys()))
    budgets = xx_space.sum(dim=1)
    eis = torch.tensor(list(hashmap_gp.values()))

    trajectory, budget_traj, ei_traj = best_route_maxpoint_fast(xx_space, budgets, eis)
    new_x = ei_based_step(trajectory, ei_traj, remaining_steps)

    return new_x


def best_route_maxpoint_fast(points, budgets, eis):
    budgets, idx = budgets.sort(stable=True)
    eis = eis[idx]
    points = points[idx]

    eis_max = torch.cummax(eis, dim=0).values
    idx_keep = torch.where(eis - eis_max >= 0)
    eis = eis[idx_keep]
    budgets = budgets[idx_keep]
    points = points[idx_keep]
    return points, budgets, eis


if __name__ == '__main__':
    p = torch.rand((100000, 2))
    b = torch.rand((100000,))
    e = torch.rand((100000,))

    t, b_t, e_t = best_route_maxpoint_fast(p.clone(), b.clone(), e.clone())

