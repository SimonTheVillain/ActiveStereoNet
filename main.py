import os
import argparse
import torch
import numpy as np
import random
import os
from torch.backends import cudnn

from Options import parse_opt
from Solvers import get_solver

def main():
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', type=str, help='Path to the option JSON file.', default='./Options/example.json')
    args = parser.parse_args()
    opt = parse_opt(args.options)
    
    # GPU/CPU Specification.
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_ids']
    os.environ['MKL_NUM_THREADS'] = opt['cpu_threads']
    os.environ['NUMEXPR_NUM_THREADS'] = opt['cpu_threads']
    os.environ['OMP_NUM_THREADS'] = opt['cpu_threads']
    
    # Deterministic Settings.
    if opt['deterministic']:
        torch.manual_seed(712)
        np.random.seed(712)
        random.seed(712)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
    
    # Create solver.
    solver = get_solver(opt)
    
    # Run.
    solver.run()

if __name__ == "__main__":
    main()