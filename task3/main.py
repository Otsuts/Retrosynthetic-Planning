import numpy as np
import argparse
import torch
import random
import time
import pickle
import os
from utils import *
from interface_1.labs import MLPModel


def prepare_mlp(args,templates, model_dump):
    one_step = MLPModel(model_dump, templates, device=args.gpu)
    return one_step


def get_args():
    parser = argparse.ArgumentParser()

    # ===================== gpu id ===================== #
    parser.add_argument('--gpu', type=int, default=0)

    # =================== random seed ================== #
    parser.add_argument('--seed', type=int, default=42)

    # ==================== dataset ===================== #
    parser.add_argument('--test_routes',
                        default='../../Project for ML/Multi-Step task/target_mol_route.pkl')
    parser.add_argument('--starting_molecules',
                        default='../../Project for ML/Multi-Step task/starting_mols.pkl')

    # ================== value dataset ================= #
    parser.add_argument('--value_root', default='dataset')
    parser.add_argument('--value_train', default='train_mol_fp_value_step')
    parser.add_argument('--value_val', default='val_mol_fp_value_step')

    # ================== one-step model ================ #
    parser.add_argument('--mlp_model_dump',
                        default='./interface_1/saved_rollout_state_1_2048.ckpt')
    parser.add_argument('--mlp_templates',
                        default='./interface_1/template_rules_1.dat')
    parser.add_argument('--tsk1_model_path', type=str)
    parser.add_argument('--tsk2_model_path', type=str,default='/NAS2020/Workspaces/DMGroup/jzchen/AI/Retrosynthetic_Planning/code/task3/interface_2/model.pth')

    # ===================== all algs =================== #
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--expansion_topk', type=int, default=50)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--viz_dir', default='viz')

    # ===================== model ====================== #
    parser.add_argument('--fp_dim', type=int, default=2048)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=500)

    # ==================== training ==================== #
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_epoch_int', type=int, default=1)
    parser.add_argument('--save_folder', default='saved_models')

    # ==================== evaluation =================== #
    parser.add_argument('--use_value_fn', action='store_true')
    parser.add_argument('--value_model', default='best_epoch_final_4.pt')
    parser.add_argument('--result_folder', default='results')

    # ==================== model selection ===============#
    parser.add_argument('--aid_model',action='store_true')

    args = parser.parse_args()
    return args


def retro_plan(args):
    device = torch.device('cpu' if args.gpu < 0 else 'cuda:%i' % args.gpu)
    print(device)
    # load data
    starting_mols = prepare_starting_molecules(args.starting_molecules)
    routes = pickle.load(open(args.test_routes, 'rb'))
    print('%d routes extracted from %s loaded' % (len(routes),
                                                  args.test_routes))
    
    if args.aid_model:
        one_step = prepare_mlp(args,args.mlp_templates, args.mlp_model_dump)
    else:
        one_step = get_task1_model(args)

    # create result folder
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    if args.use_value_fn:
        model = get_task2_model(args)

        def value_fn(mol):
            fp = smiles_to_fp(mol, fp_dim=args.fp_dim).reshape(1, -1)
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        def value_fn(x): return 0.

    plan_handle = prepare_molstar_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir
    )

    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_costs': [],
        'route_lens': []
    }
    num_targets = len(routes)
    t0 = time.time()
    for (i, route) in enumerate(routes):
        target_mol = route[0].split('>')[0]
        succ, msg = plan_handle(target_mol, i)

        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[1])
        result['routes'].append(msg[0])
        if succ:
            result['route_costs'].append(msg[0].total_cost)
            result['route_lens'].append(msg[0].length)
        else:
            result['route_costs'].append(None)
            result['route_lens'].append(None)

        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_iter = np.array(result['iter'], dtype=float).mean()
        print('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f' %
              (tot_succ, tot_num, num_targets, avg_time, avg_iter))

    f = open(args.result_folder + f'/plan_TOPK{args.expansion_topk}_USEFUNC{args.use_value_fn}_USEAID{args.aid_model}.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


if __name__ == '__main__':
    args = get_args()
    # setup device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    retro_plan(args)
