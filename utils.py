from typing import Optional
import torch
import os

# from samplers.uni_pc import UniPC
# from samplers.heun import Heun
# from samplers.dpm_solverpp import DPM_SolverPP
# from samplers.dpm_solver import DPM_Solver
# from samplers.euler import Euler
# from samplers.ipndm import iPNDM 
# from noise_schedulers import NoiseScheduleVE
import pickle
import argparse
import time
import yaml
import random
import numpy as np 
import ast

PRIOR_TIMESTEPS = {
    "cifar10": {
        4: [80.0, 5.1092, 1.584, 0.47, 0.002],
        5: [80.0, 5.8389, 2.1632, 0.8119, 0.2107, 0.002],
        6: [80.0, 9.7232, 3.3686, 1.3482, 0.5666, 0.1698, 0.002],
        7: [80.0, 10.9836, 3.8811, 1.8543, 0.8119, 0.3183, 0.1079, 0.002],
        8: [80.0, 10.9836, 3.8811, 1.8543, 0.9654, 0.47, 0.2107, 0.0665, 0.002], 
        9: [80.0, 12.3816, 4.459, 2.1632, 1.1431, 0.5666, 0.2597, 0.1079, 0.03, 0.002],
        10: [80.0, 13.9293, 5.1092, 2.5152, 1.3482, 0.6799, 0.3183, 0.1698, 0.0665, 0.0225, 0.002],
    },
    "ffhq": {
        4 :[80.0, 7.5699, 2.1632, 0.5666, 0.002],
        5 : [80.0, 9.7232, 2.9152, 0.9654, 0.2597, 0.002],
        6 : [80.0, 10.9836, 3.8811, 1.584, 0.5666, 0.1698, 0.002],
        7 : [80.0, 12.3816, 4.459, 1.8543, 0.8119, 0.3183, 0.1079, 0.002],
        8: [80.0, 12.3816, 5.1092, 2.1632, 0.9654, 0.47, 0.2107, 0.0665, 0.002],
        9: [80.0, 13.9293, 5.8389, 2.9152, 1.3482, 0.6799, 0.3183, 0.1359, 0.0515, 0.002],
        10: [80.0, 13.9293, 5.8389, 2.9152, 1.584, 0.8119, 0.3878, 0.2107, 0.0851, 0.03, 0.002],
    },
    "afhqv2": {
        4 : [80.0, 7.5699, 2.1632, 0.3878, 0.002],
        5 : [80.0, 8.5888, 2.9152, 0.9654, 0.2107, 0.002],
        6 : [80.0, 9.7232, 3.8811, 1.584, 0.47, 0.1359, 0.002],
        7 : [80.0, 10.9836, 4.459, 1.8543, 0.6799, 0.2597, 0.0851, 0.002],
        8: [80.0, 12.3816, 5.1092, 2.5152, 1.1431, 0.47, 0.2107, 0.0665, 0.002],
        9: [80.0, 13.9293, 5.8389, 2.9152, 1.3482, 0.6799, 0.3183, 0.1359, 0.0515, 0.002],
        10: [80.0, 13.9293, 5.8389, 2.9152, 1.584, 0.8119, 0.3878, 0.2107, 0.1079, 0.0395, 0.002],
    },
    'lsun': {
        4: [83.8225, 2.1307, 0.9556, 0.425, 0.0388],
        5:[83.8225, 2.4793, 1.1629, 0.5745, 0.2411, 0.0388],
        6: [83.8225, 2.4793, 1.2928, 0.7324, 0.3678, 0.1578, 0.0388],
        7:  [83.8225, 2.9282, 1.4464, 0.8717, 0.4929, 0.2586, 0.109, 0.0388],
        8: [83.8225, 3.5196, 1.854, 1.1629, 0.7324, 0.425, 0.2249, 0.1009, 0.0388],
        9: [83.8225, 3.5196, 1.854, 1.1629, 0.7324, 0.4574, 0.2773, 0.1578, 0.0731, 0.0388],
        10:[83.8225, 4.3198, 2.1307, 1.2928, 0.8717, 0.5745, 0.3678, 0.2411, 0.1365, 0.0672, 0.0388],
    },
    'sd': {
        3: [14.6146, 1.7083, 0.532, 0.0292], 
        4: [14.6146, 3.1131, 1.0421, 0.3811, 0.0292], 
        5: [14.6146, 4.39, 1.5286, 0.6526, 0.2667, 0.0292],
        6: [14.6146, 4.7242, 1.9132, 0.9324, 0.4557, 0.1801, 0.0292],
        7: [14.6146, 6.4477, 2.2797, 1.1629, 0.6114, 0.3058, 0.1258, 0.0292],
        8: [14.6146, 6.4477, 2.7391, 1.4467, 0.8319, 0.4936, 0.2667, 0.1258, 0.0292],
        9: [14.6146, 6.4477, 3.3251, 1.9132, 1.1629, 0.7391, 0.4557, 0.2667, 0.1258, 0.0292],
        10: [14.6146, 5.9489, 3.3251, 2.0267, 1.2969, 0.8319, 0.5712, 0.3811, 0.2255, 0.1258, 0.0292],
        11: [14.6146, 6.4477, 3.8092, 2.2797, 1.5286, 1.0421, 0.7391, 0.4936, 0.3437, 0.2255, 0.1258, 0.0292]   
    }
}

def parse_prior_timesteps(args):
    if args.custom_ts_1 is not None:
        try:
            args.custom_ts_1 = ast.literal_eval(args.custom_ts_1)
        except Exception:
            pass
        else:
            if args.custom_ts_2 is not None:
                try:
                    args.custom_ts_2 = ast.literal_eval(args.custom_ts_2)
                except Exception:
                    pass
            if args.custom_ts_2 is None:
                args.custom_ts_2 = args.custom_ts_1
            return
        
    if args.use_gits:
        dataset = None
        if args.model == 'edm':
            for d in ['cifar10', 'afhqv2', 'ffhq']:
                if d in args.ckp_path:
                    dataset = d
                    break
        elif args.model == 'latent_diff':
            dataset = 'lsun'
        elif args.model == 'conditioned_latent_diff':
            dataset = 'sd'
        
        if args.steps in PRIOR_TIMESTEPS[dataset]:
            args.custom_ts_1 = PRIOR_TIMESTEPS[dataset][args.steps]
            args.custom_ts_2 = args.custom_ts_1
            print("################################")
            print("Successfully load gits timesteps for dataset ", dataset)
            print(args.custom_ts_1)
            print("################################")
        else:
            raise NotImplementedError
        

def get_teacher_timesteps(args,teacher_step):
    dataset = None
    if args.model == 'edm':
        for d in ['cifar10', 'afhqv2', 'ffhq']:
            if d in args.ckp_path:
                dataset = d
                break
    elif args.model == 'latent_diff':
        dataset = 'lsun'
    elif args.model == 'conditioned_latent_diff':
        dataset = 'sd'
    
    if args.steps in PRIOR_TIMESTEPS[dataset]:
        args.teacher_ts_1 = PRIOR_TIMESTEPS[dataset][teacher_step]
        args.teacher_ts_2 = args.teacher_ts_1
        print("################################")
        print("Successfully load gits teacher timesteps for dataset ", dataset)
        print(args.teacher_ts_1)
        print("################################")
    else:
        raise NotImplementedError


def parse_prior_pair_timesteps(args,mode = 'low'):
    if mode == 'low':
        custom_ts_1 = args.custom_ts_low_1
        custom_ts_2 = args.custom_ts_low_2
        steps = args.steps_low
    elif mode == 'high':
        custom_ts_1 = args.custom_ts_high_1
        custom_ts_2 = args.custom_ts_high_2
        steps = args.steps_high

    if custom_ts_1 is not None:
        try:
            custom_ts_1 = ast.literal_eval(custom_ts_1)
        except Exception:
            pass
        else:
            if custom_ts_2 is not None:
                try:
                    custom_ts_2 = ast.literal_eval(custom_ts_2)
                except Exception:
                    pass
            if custom_ts_2 is None:
                custom_ts_2 = custom_ts_1

            if mode == 'low':
                args.custom_ts_low_1 = custom_ts_1
                args.custom_ts_low_2 = custom_ts_2
            elif mode == 'high':
                args.custom_ts_high_1 = custom_ts_1
                args.custom_ts_high_2 = custom_ts_2
            return
        
    if args.use_gits:
        dataset = None
        if args.model == 'edm':
            for d in ['cifar10', 'afhqv2', 'ffhq']:
                if d in args.ckp_path:
                    dataset = d
                    break
        elif args.model == 'latent_diff':
            dataset = 'lsun'
        elif args.model == 'conditioned_latent_diff':
            dataset = 'sd'
        
        if steps in PRIOR_TIMESTEPS[dataset]:
            custom_ts_1 = PRIOR_TIMESTEPS[dataset][steps]
            custom_ts_2 = custom_ts_1
            print("################################")
            print("Successfully load gits timesteps for dataset ", dataset)
            print("MODE: ", mode)
            print(custom_ts_1)
            print("################################")
            if mode == 'low':
                args.custom_ts_low_1 = custom_ts_1
                args.custom_ts_low_2 = custom_ts_2
            elif mode == 'high':
                args.custom_ts_high_1 = custom_ts_1
                args.custom_ts_high_2 = custom_ts_2
        else:
            raise NotImplementedError



def set_seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")

    parser.add_argument('--all_config')
    parser.add_argument('--model', help="edm/latent_diff")
    parser.add_argument('--trainer', type=str, help="trainer_ld3 or trainer_tdpo")
    parser.add_argument('--gen_pair_data', action='store_true', help="If we generate pair data for two different NFE FID calculation")
    parser.add_argument('--custom_ts_low_1', type=str, help="Custom timesteps low 1")
    parser.add_argument('--custom_ts_low_2', type=str, help="Custom timesteps low 2")
    parser.add_argument('--custom_ts_high_1', type=str, help="Custom timesteps high 1")
    parser.add_argument('--custom_ts_high_2', type=str, help="Custom timesteps high 2")
    parser.add_argument('--steps_low', type=int, help="Steps low")
    parser.add_argument('--steps_high', type=int, help="Steps high")



    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument("--ckp_path", type=str, help="Path to the checkpoint file.")
    model_group.add_argument("--solver_name", type=str, help="Method for solving: heun/dpm_solver++/uni_pc.")
    model_group.add_argument("--unipc_variant", type=str, choices=["bh1", "bh2"], help="Variant of UniPC: bh1/bh2.")
    model_group.add_argument("--steps", type=int, help="Number of sampling steps.")
    model_group.add_argument("--order", type=int, help="Order for sampling.")
    model_group.add_argument("--time_mode", type=str, help="Time model: time or lambda.")

    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument("--seed", type=int, help="seed")
    training_group.add_argument("--use_ema",  action="store_true", help="If we use ema for LSUN latent diff")
    training_group.add_argument("--log_path", type=str, help="Folder name for storing evaluation results.")
    training_group.add_argument("--old_log_path", type=str, help="Folder name for storing old evaluation results.")
    training_group.add_argument("--data_dir", type=str, help="Path to data dir.")
    training_group.add_argument("--num_train", type=int, help="Number of training sample.")
    training_group.add_argument("--num_valid", type=int,  help="Number of validation sample.")
    training_group.add_argument("--main_train_batch_size", type=int, help="Batch size for training.")
    training_group.add_argument("--main_valid_batch_size", type=int, help="Batch size for validation.")
    training_group.add_argument("--win_rate", type=float, help="Win rate, should be in (0, 0.5]")
    training_group.add_argument("--prior_bound", type=float, help="Prior bound.")
    training_group.add_argument("--fix_bound", action="store_true", help="fix bound or not")
    training_group.add_argument("--loss_type", type=str, choices=["L1", "L2", "LPIPS", "DISTS", "SCDIST"], help="Type of loss: L1, L2 or LPIPS.")
    training_group.add_argument("--training_rounds_v1", type=int, help="Number of training rounds for phase 1.")
    training_group.add_argument("--training_rounds_v2", type=int, help="Number of training rounds for phase 2.")
    training_group.add_argument("--lr_time_1", type=float, help="Learning rate for the first phase.")
    training_group.add_argument("--lr_time_2", type=float, help="Learning rate for the second phase.")
    training_group.add_argument("--min_lr_time_1", type=float, help="Minimum learning rate for the first phase.")
    training_group.add_argument("--min_lr_time_2", type=float, help="Minimum learning rate for the second phase.")
    training_group.add_argument("--momentum_time_1", type=float, help="Momentum for the first phase.")
    training_group.add_argument("--weight_decay_time_1", type=float, help="Weight decay for the first phase.")
    training_group.add_argument("--shift_lr", type=float, help="Learning rate for moving latents.")
    training_group.add_argument("--shift_lr_decay", type=float, help="Learning rate decay for the shift phase.")
    training_group.add_argument("--lr_time_decay", type=float, help="Learning rate decay for the time phase.")
    training_group.add_argument("--patient", type=int, help="Patient for the time phase.")
    training_group.add_argument("--lr2_patient", type=int, help="Patient for the second phase.")
    training_group.add_argument("--no_v1", action="store_true", help="Skip the first phase.")
    training_group.add_argument("--visualize", type=str, help="Visualize.")
    training_group.add_argument("--low_gpu", action="store_true", help="If we using low-mem gpu, we need to use checkpoint.")
    training_group.add_argument("--scale", type=int, help="Guidance scale")
    training_group.add_argument("--match_prior", action="store_true", help="Whether to initial params by prior timesteps")
    training_group.add_argument("--training_method", type=str, help="ld3/tdpo/tipo")
    training_group.add_argument("--training_spin", action="store_true", help="Whether to use SPIN training strategy for pair dataset")
    training_group.add_argument("--evolving_teacher", action="store_true", help="Whether to use evolving teacher for pair dataset")
    training_group.add_argument("--sample_new", action="store_true", help="Whether to sample new images for evolving teacher")
    training_group.add_argument("--beta", type=float, help="Scaling factor for the reward difference")
    training_group.add_argument("--angle_reg_weight", type=float, help="Weight for angle regularization (if used)")
    training_group.add_argument("--use_pair_dataset", action="store_true", help="Whether to use pair dataset")
    training_group.add_argument("--update_latent", type=bool, default=True, help="Whether to update latents")
    training_group.add_argument("--latter_teacher", action="store_true", help="Whether to use latter image as teacher (for pair dataset)")
    training_group.add_argument("--threshold", type=float, help="Threshold for angle regularization (if used)")
    training_group.add_argument("--loss_power", type=float, default=1.0, help="Power for lpips/l2/l1 loss")
    training_group.add_argument("--clamp_loss", action="store_true", help="Whether to clamp the loss values to be non-negative")
    training_group.add_argument("--losesample_is_policy", action="store_true", help="Whether the losing sample is the policy sample (for pair dataset)")
    training_group.add_argument("--disable_load_checkpoint", action="store_true", help="Whether to enable loading from a checkpoint")
    training_group.add_argument("--rethink_loss_weight", default=1.0, type=float, help="Weight for the rethink loss component")
    training_group.add_argument("--dpo_loss_weight", default=1.0, type=float, help="Weight for the DPO loss component")
    training_group.add_argument("--rethink_loss", type=str, default="sum_all", help="Type of rethink loss: sum_all / only_x0 / only_std")
    training_group.add_argument("--rethink_clamp_input", action="store_true", help="Whether to clamp the input images for rethink loss computation")
    training_group.add_argument("--ddimstd_weight", action="store_true", help="Whether to weight the std prediction loss in DDIM rethink")
    training_group.add_argument("--eta", type=float, default=1.0, help="Eta value for DDIM rethink")
    training_group.add_argument("--detach_x_t", action="store_true", help="Whether to detach x_t in DDIM rethink")
    training_group.add_argument("--detach_ddim_std", action="store_true", help="Whether to detach ddim std in DDIM rethink")
    training_group.add_argument('--lose_teacher', action='store_true', help="If we get teacher timesteps for rethink loss")
    training_group.add_argument('--teacher_steps', type=int, help="Number of teacher steps for rethink loss")
    training_group.add_argument('--lose_teacher_selection', type=str, default="firstk", help="Method to select teacher timesteps: firstk/random")
    training_group.add_argument('--ema_rate', type=float, help="EMA rate for model")
    training_group.add_argument('--t_check_scale', type=float, default=0.1, help="Scale for temporal consistency check")
    training_group.add_argument('--sharpness_weight', type=float, default=1.0, help="Weight for sharpness loss")
    training_group.add_argument('--log_std_init', type=float, default=-1.0, help="Initial log standard deviation")
    training_group.add_argument('--train_logstd', action='store_true', help="Whether to train log standard deviation")
    training_group.add_argument('--sc_only', action='store_true', help="Whether to use score correction only")
    training_group.add_argument('--timestep1_only', action='store_true', help="Whether to use only timestep 1")
    training_group.add_argument('--std_eq_ends', action='store_true', help="Whether to set std at the two ends to be equal")
    training_group.add_argument('--sc_cfg_w', type=float, default=0.0, help="Score correction cfg weight")
    training_group.add_argument('--grad_accumulation', type=int, default=1, help="Gradient accumulation steps")
    training_group.add_argument('--no_eval_during_train', action='store_true', help="Whether to skip evaluation during training")
    training_group.add_argument('--training_cfg_weight', action='store_true', help="Whether to use classifier-free guidance during training")
    training_group.add_argument('--lr_time_3', type=float, default=1e-3, help="Learning rate for the third phase.")
    training_group.add_argument('--interpolation_factor', type=int, default=2, help="Interpolation factor for evolving teacher.")
    training_group.add_argument('--no_cfgw_interpolate', action='store_true', help="Whether to not interpolate cfgw during evolving teacher.")
    training_group.add_argument('--timestep_cfg_iterative', action='store_true', help="Whether to use iterative timestep cfg during training.")
    training_group.add_argument('--min_lr_time_3', type=float, default=1e-5, help="Minimum learning rate for the third phase.")
    training_group.add_argument('--cosine_lr_schedule', action='store_true', help="Whether to use cosine learning rate schedule.")
    training_group.add_argument('--evolve_mode', type=str, default="ref", help="Mode for evolving teacher: policy or ref.")
    training_group.add_argument('--simple_sd', action='store_true', help="Whether to use simple score distillation.")
    training_group.add_argument('--ref_update_mode', type=str, default="ema", help="Reference update mode: ema or replace.")
    training_group.add_argument('--fixed_teacher', action='store_true', help="Whether to use a fixed teacher model during training.")


    testing_group = parser.add_argument_group('Testing Parameters')
    testing_group.add_argument("--load_from_version", type=int, default=2, help="Load from whihc version, default=2")
    testing_group.add_argument("--custom_ts_1", type=str, help="Custom timesteps 1")
    testing_group.add_argument("--custom_ts_2", type=str, help="Custom timesteps 2")
    testing_group.add_argument("--use_gits", action="store_true", help="Use pre-computed gits timesteps")
    testing_group.add_argument("--learn", action="store_true", help="Load from learned timesteps.")
    testing_group.add_argument("--load_from", type=str, help="Ckpt path")
    testing_group.add_argument("--skip_type", type=str, help="Type of skip.")
    testing_group.add_argument("--num_multi_steps_fid", type=int, help="num_multi_steps_fid")
    testing_group.add_argument("--fid_folder", type=str, default=None, help="FID path")
    testing_group.add_argument("--sampling_batch_size", type=int, help="Batch size for FID calculation.")
    testing_group.add_argument("--sampling_seed", type=int, help="Sampling seed for FID calculation")
    testing_group.add_argument("--ref_path", type=str,  help="Path to dataset reference statistics.")
    testing_group.add_argument("--total_samples", type=int, help="Total number of sample for FID calculation.")
    testing_group.add_argument("--save_png", action="store_true", help="Save generated img in png.")
    testing_group.add_argument("--save_pt", action="store_true", help="Save generated img and latent in pt files.")
    testing_group.add_argument("--cfgw", type=str, help="Classifier-free guidance weight")

    other_group = parser.add_argument_group('Other Parameters')
    other_group.add_argument("--prompt_path", type=str, help="Prompt json path for stable diff")
    other_group.add_argument("--num_prompts", type=int, default=999999999, help="Number of prompts we want to use, default 5")
    other_group.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samplers per prompt, default 1")
    other_group.add_argument("--ts_config", type=str, help="Path to timestep config json file for qual comparison")

    parser.add_argument("--eval_hps", action="store_true", help="Enable HPS v2 evaluation.")
    parser.add_argument("--eval_aes", action="store_true", help="Enable Aesthetic Score evaluation.")
    parser.add_argument("--eval_clip", action="store_true", help="Enable CLIP Score evaluation.")
    parser.add_argument("--eval_fid", action="store_true", help="Enable FID evaluation.")
    #other_group.add_argument("--image_dir", type=str, help="Path to image dir for qual comparison")
    args = parser.parse_args()

    # Load the config file if specified
    if args.all_config and os.path.isfile(args.all_config):
        print(f"Loading configuration from {args.all_config}")
        print(f"Loading configuration from {args.all_config}")
        print(f"Loading configuration from {args.all_config}")
        print(f"Loading configuration from {args.all_config}")
        print(f"Loading configuration from {args.all_config}")
        print(f"Loading configuration from {args.all_config}")
        with open(args.all_config, 'r') as f:
            config = yaml.safe_load(f)

        #Override the arguments with config values if they are None
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)


        # Override the arguments with config values always
        # for key, value in config.items():
        #     setattr(args, key, value)

    return args

def compute_distance_between_two(x, y, n_channels=3, resolution=256):
    '''
    x: bs x 3 x 256 x 256
    y: bs x 3 x 256 x 256
    '''
    square_distance = (x - y) ** 2
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance

def compute_distance_between_two_L2_clamp(x, y, n_channels=3, resolution=256):
    '''
    x: bs x 3 x 256 x 256
    y: bs x 3 x 256 x 256
    '''
    x = torch.clamp(x, -1.0, 1.0)
    y = torch.clamp(y, -1.0, 1.0)
    square_distance = (x - y) ** 2
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance

def compute_distance_between_two_L1(x, y, n_channels=3, resolution=256):
    '''
    x: bs x 3 x 256 x 256
    y: bs x 3 x 256 x 256
    '''
    square_distance = torch.abs(x - y)
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance

def get_solvers(solver_name: str, NFEs: int, order:int, noise_schedule: NoiseScheduleVE, unipc_variant: Optional[str] = None):
    solver_extra_params = dict()
    if solver_name == 'euler':
        steps = NFEs
        solver = Euler(noise_schedule)
    elif solver_name == 'heun':
        steps = NFEs // 2
        solver = Heun(noise_schedule)

    elif solver_name == 'dpm_solver':
        solver = DPM_Solver(noise_schedule)
        dpm_steps, dpm_orders = solver.compute_K_and_order(NFEs, order=order)
        solver_extra_params['dpm_orders'] = dpm_orders
        solver_extra_params['NFEs'] = NFEs
        solver_extra_params['dpm_steps'] = dpm_steps
        
        steps = dpm_steps
    elif solver_name == 'dpm_solver++':
        steps = NFEs
        solver = DPM_SolverPP(noise_schedule)
    elif solver_name == 'uni_pc':
        steps = NFEs
        solver = UniPC(noise_schedule, variant=unipc_variant)
    elif solver_name == 'ipndm':
        steps = NFEs
        solver = iPNDM(noise_schedule)
    else:
        raise NotImplementedError
    return solver, steps, solver_extra_params

def save_arguments_to_yaml(args, filename):
    with open(filename, 'w') as file:
        yaml.dump(vars(args), file)


def adjust_hyper(args, resolution=64, channel=3):
    parse_prior_timesteps(args)
    if args.lose_teacher or (args.evolving_teacher and args.fixed_teacher):
        assert(args.teacher_steps is not None)
        get_teacher_timesteps(args, args.teacher_steps)
    else:
        args.teacher_ts_1 = None
        args.teacher_ts_2 = None

    
    if args.gen_pair_data:
        #breakpoint()
        parse_prior_pair_timesteps(args, mode='low')
        parse_prior_pair_timesteps(args, mode='high')
    if args.shift_lr is None:
        args.shift_lr = 3.0 * 4 / args.steps
    if not args.fix_bound:
        args.prior_bound = 0.001 * resolution * resolution * channel / (args.steps ** 2)
    args.lr_time_2 = args.lr_time_2 / args.steps
    
    args.lr_time_2 = round(args.lr_time_2, 8)
    # round prior_bound 
    args.prior_bound = round(args.prior_bound, 8)
    # round shift_lr
    args.shift_lr = round(args.shift_lr, 8)
    return args


def create_desc(args):
    NFEs = args.steps
    method_full = args.solver_name
    training_method = getattr(args, "training_method", None)
    desc = f"{args.trainer if training_method is None else training_method}-{method_full}-N{NFEs}-lr2{args.lr_time_2}"#
    if getattr(args, 'use_different_xT', False):
        desc += "-diffxT"
    if getattr(args, 'lr_time_3', None) is not None:
        desc += f"-lr3{args.lr_time_3}"
    if getattr(args, 'lr_time_4', None) is not None:
        desc += f"-lr4{args.lr_time_4}"
    desc += f"rv1{args.training_rounds_v1}-rv2{args.training_rounds_v2}-seed{args.seed}"
    if args.no_v1:
        desc += "-no_v1_only_v2"
    if args.match_prior:
        desc += "-match_prior"
    if getattr(args, 'timesteps1_mode', None) is not None:
        desc += f"-ts1mode{args.timesteps1_mode}"
    return desc



def prepare_paths(args):
    skip_type=""
    if args.learn:
        if args.load_from is None:
            desc = create_desc(args)
            args.log_path = os.path.join(args.log_path, desc)

            #Added line by Jinkyu
            if args.skip_type != "gits":
                args.load_from = os.path.join(args.log_path, f'best_v{args.load_from_version}.pt')
        else:
            args.log_path = os.path.dirname(args.load_from)
            desc = os.path.basename(args.log_path)
        # if not is_trained(args.log_path):
        #     raise ValueError("Model not trained!")
    else:
        NFEs = args.steps
        solver_name = args.solver_name
        skip_type = args.skip_type
        desc = f"{solver_name}_NFE{NFEs}_{skip_type}_seed{args.seed}"
    
    # create fid folder
    if args.fid_folder:
        os.makedirs(args.fid_folder, exist_ok=True)
        fid_log_path = os.path.join(args.fid_folder, f"{desc}.txt")
    else:
        fid_log_path = None
    return desc, fid_log_path, skip_type

def check_fid_file(fid_log_path):
    if os.path.exists(fid_log_path):
        # check if FID has been computed
        with open(fid_log_path, "r") as f:
            scores = f.read()
        # check if fid is a number
        try:
            scores = [float(_) for _ in scores.strip().split()]
            if len(scores) == 1:
                print(f"FID: {scores[0]}")
            elif len(scores) == 2:
                print(f"FID: {scores[0]}")
                print(f"IS: {scores[1]}")
            else:
                return False
            return True
        except ValueError:
            return False
    return False

def is_trained(path):
    log_path = os.path.join(path, 'log.txt')
    print(log_path)
    if not os.path.isfile(log_path):
        print("log.txt not exist")
        return False 
    
    last_line = ""
    # Open the file in read mode
    with open(log_path, 'r') as f:
        # Read each line in the file
        for line in f:
            # Strip any leading or trailing whitespace
            stripped_line = line.strip()
            # Check if the line is not empty
            if stripped_line:
                last_line = stripped_line  # Update last non-empty line
    return "Training time" in last_line


def move_tensor_to_device(*args, device):
    return [arg.to(device) if arg is not None else arg for arg in args]