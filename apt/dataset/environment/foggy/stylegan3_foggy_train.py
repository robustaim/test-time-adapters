import sys

path_to_remove = "/shared_hdd/annasdfghjkl13/anaconda3/lib/python3.12/site-packages"

if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch.utils.data import DataLoader

from stylegan3.dnnlib.util import EasyDict
from stylegan3.torch_utils import misc, training_states
from stylegan3.torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from stylegan3.training.dataset import ImageFolderDataset
from stylegan3.training.loss import StyleGAN2Loss as StyleGAN_Loss
from stylegan3.training.networks_stylegan2 import Discriminator
from stylegan3.training.networks_stylegan3 import Generator
from stylegan3.training.training_loop import setup_snapshot_image_grid, save_image_grid

import legacy
from metrics import metric_main


# opts = EasyDict({
#     "data":,
#     "cond":,
#     "mirror":,
#     "gpus":,
#     "batch",
#     "cbase":,
#     "cmax":,
#     "cfg":,
#     "map_depth":,
#     "freezed":,
#     "mbstd_group":,
#     "gamma":,
#     "glr":,
#     "dlr":,
#     "metrics":,
#     "kimg":,
#     "tick":,
#     "snap":,
#     "seed":,
#     "workers":,
#     "aug":,
#     "resume":,
#     "fp32":,
#     "nobench":,
#     "outdir":,
#     "dry_run":
# })


# training_set_kwargs = EasyDict({
#     "path": opt.data,
#     "use_labels": True,
#     "max_size": None,
#     "xflip": False,
#     #dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
#     "resolution": dataset_obj.resolution, #dataset의 해상도,
#     "use_labels": dataset_obj.has_labels, #dataset의 라벨 
#     "max_size": len(dataset_obj),
#     "random_seed": opts.seed
# })

# c = EasyDict({
#     "G_kwargs": {
#         "class_name": 'training.networks_stylegan3.Generator',
#         "z_dim": 512,
#         "w_dim": 512,
#         "mapping_kwargs": {
#             "num_layer": (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth  
#         },
#         "channel_base": opts.cbase,
#         "channel_max": opts.cmax,
#         "magnitude_ema_beta": 0.5 ** (opts.batch / (20 * 1e3))
#     },
#     "D_kwargs": {
#         "class_name": "training.networks_stylegan2.Discriminator",
#         "block_kwargs": {
#             "freeze_layers": opts.freezed    
#         },
#         "mapping_kwargs": {},
#         "epilogue_kwargs": {
#             "mbstd_group_size": opts.mbstd_group
#         },
#         "channel_base": opts.cbase,
#         "channel_max": opts.cmax
#     },
#     "G_opt_kwargs": {
#         "class_name": "torch.optim.Adam",
#         "betas": [0,0.99],
#         "eps": 1e-8,
#         "lr": (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
#     },
#     "D_opt_kwargs":{
#         "class_name": "torch.optim.Adam",
#         "betas": [0, 0.9],
#         "eps": 1e-8,
#         "lr": opts.dlr
#     },
#     "loss_kwargs": {
#         "r1_gamma": opts.gamma
#     },
#     "data_loader_kwargs": {
#         "pin_memory": True,
#         "prefetch_factor": 2,
#         "num_workers": opts.workers
#     },
#     "training_set_kwargs": training_set_kwargs,
#     "num_gpus" : opts.gpus,
#     "batch_size": opts.batch,
#     "batch_gpu": opts.batch_gpu or opts.batch // opts.gputs,
#     "metrics": opts.metrics,
#     "total_kimg": opts.kimg,
#     "kimg_per_tick": opts.tick,
#     "image_snapshot_ticks": opts.snap,
#     "network_snapshot_ticks": opts.snap,
#     "random_seed": opts.seed,
#     "ema_kimg": opts.batch * 10 / 32,
#     "G_reg_interval": None, # How often to perform regularization for G? None = disable lazy regularization.
#     "D_reg_interval": 16, # How often to perform regularization for D? None = disable lazy regularization.
#     "ada_interval": 4, # How often to perform ADA adjustment?
#     "resume_kimg": 0, # First kimg to report when resuming training.
#     "abort_fn": None, # Callback function for determining whether to abort training. Must return consistent results across ranks.
#     "progress_fn": None # Callback function for updating training progress. Called for all ranks.
# })    

# # Augmentation.
# if opts.aug != 'noaug':
#     c.augment_kwargs = EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
#     if opts.aug == 'ada':
#         c.ada_target = opts.target
#     if opts.aug == 'fixed':
#         c.augment_p = opts.p


# # Resume.
# if opts.resume is not None:
#     c.resume_pkl = opts.resume
#     c.ada_kimg = 100 # Make ADA react faster at the beginning.
#     c.ema_rampup = None # Disable EMA rampup.
#     c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

# # Performance-related toggles.
# if opts.fp32:
#     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
#     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
# if opts.nobench:
#     c.cudnn_benchmark = False

# def training_loop(rank=rank, **c):
#     #Initialize
#     start_time = time.time()
#     device = torch.device('cuda', rank)
    
#     np.random.seed(random_seed * num_gpus + rank)
#     torch.manual_seed(random_seed * num_gpus + rank)
#     torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
#     torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
#     torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
#     conv2d_gradfix.enabled = True                       # Improves training speed.
#     grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    
#     # data loader
#     if rank == 0:
#         print('Loading training set...')
#     training_set = ImageFolderDataset(**training_set_kwargs)
#     training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
#     training_set_iterator = iter(DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
#     if rank == 0:
#         print()
#         print('Num images: ', len(training_set))
#         print('Image shape:', training_set.image_shape)
#         print('Label shape:', training_set.label_shape)
#         print()

#     # model
#     if rank == 0:
#         print('Constructing networks...')
#     common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
#     G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
#     D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
#     G_ema = copy.deepcopy(G).eval()

#     # Resume from existing pickle.
#     if (resume_pkl is not None) and (rank == 0):
#         print(f'Resuming from "{resume_pkl}"')
#         with dnnlib.util.open_url(resume_pkl) as f:
#             resume_data = legacy.load_network_pkl(f)
#         for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
#             misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

#     # augment
#     if rank == 0:
#         print('Setting up augmentation...')
#     augment_pipe = None
#     ada_stats = None
#     if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
#         augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
#         augment_pipe.p.copy_(torch.as_tensor(augment_p))
#         if ada_target is not None:
#             ada_stats = training_stats.Collector(regex='Loss/signs/real')

#     # Distribute across GPUs.
#     if rank == 0:
#         print(f'Distributing across {num_gpus} GPUs...')
#     for module in [G, D, G_ema, augment_pipe]:
#         if module is not None and num_gpus > 1:
#             for param in misc.params_and_buffers(module):
#                 torch.distributed.broadcast(param, src=0)

#     #loss
#     loss = StyleGAN_Loss(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs)

#     # setup training phases
#     if rank == 0:
#         print('Setting up training phases...')
#     phases = []
#     for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
#         if reg_interval is None:
#             opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
#             phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
#         else: # Lazy regularization.
#             mb_ratio = reg_interval / (reg_interval + 1)
#             opt_kwargs = dnnlib.EasyDict(opt_kwargs)
#             opt_kwargs.lr = opt_kwargs.lr * mb_ratio
#             opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
#             opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
#             phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
#             phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
#     for phase in phases:
#         phase.start_event = None
#         phase.end_event = None
#         if rank == 0:
#             phase.start_event = torch.cuda.Event(enable_timing=True)
#             phase.end_event = torch.cuda.Event(enable_timing=True)
            
#     # sample image
#     if rank == 0:
#         print('Exporting sample images...')
#         grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
#         save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
#         grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
#         grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
#         images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
#         save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

#     # Initialize logs.
#     if rank == 0:
#         print('Initializing logs...')
#     stats_collector = training_stats.Collector(regex='.*')
#     stats_metrics = dict()
#     stats_jsonl = None
#     stats_tfevents = None
#     if rank == 0:
#         stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
#         try:
#             import torch.utils.tensorboard as tensorboard
#             stats_tfevents = tensorboard.SummaryWriter(run_dir)
#         except ImportError as err:
#             print('Skipping tfevents export:', err)

#     # Train.
#     if rank == 0:
#         print(f'Training for {total_kimg} kimg...')
#         print()
#     cur_nimg = resume_kimg * 1000
#     cur_tick = 0
#     tick_start_nimg = cur_nimg
#     tick_start_time = time.time()
#     maintenance_time = tick_start_time - start_time
#     batch_idx = 0
#     if progress_fn is not None:
#         progress_fn(0, total_kimg)
#     while True:

#         # Fetch training data.
#         with torch.autograd.profiler.record_function('data_fetch'):
#             phase_real_img, phase_real_c = next(training_set_iterator)
#             phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
#             phase_real_c = phase_real_c.to(device).split(batch_gpu)
#             all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
#             all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
#             all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
#             all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
#             all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

#         # Execute training phases.
#         for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
#             if batch_idx % phase.interval != 0:
#                 continue
#             if phase.start_event is not None:
#                 phase.start_event.record(torch.cuda.current_stream(device))

#             # Accumulate gradients.
#             phase.opt.zero_grad(set_to_none=True)
#             phase.module.requires_grad_(True)
#             for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
#                 loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
#             phase.module.requires_grad_(False)

#             # Update weights.
#             with torch.autograd.profiler.record_function(phase.name + '_opt'):
#                 params = [param for param in phase.module.parameters() if param.grad is not None]
#                 if len(params) > 0:
#                     flat = torch.cat([param.grad.flatten() for param in params])
#                     if num_gpus > 1:
#                         torch.distributed.all_reduce(flat)
#                         flat /= num_gpus
#                     misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
#                     grads = flat.split([param.numel() for param in params])
#                     for param, grad in zip(params, grads):
#                         param.grad = grad.reshape(param.shape)
#                 phase.opt.step()

#             # Phase done.
#             if phase.end_event is not None:
#                 phase.end_event.record(torch.cuda.current_stream(device))

#         # Update G_ema.
#         with torch.autograd.profiler.record_function('Gema'):
#             ema_nimg = ema_kimg * 1000
#             if ema_rampup is not None:
#                 ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
#             ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
#             for p_ema, p in zip(G_ema.parameters(), G.parameters()):
#                 p_ema.copy_(p.lerp(p_ema, ema_beta))
#             for b_ema, b in zip(G_ema.buffers(), G.buffers()):
#                 b_ema.copy_(b)

#         # Update state.
#         cur_nimg += batch_size
#         batch_idx += 1

#         # Execute ADA heuristic.
#         if (ada_stats is not None) and (batch_idx % ada_interval == 0):
#             ada_stats.update()
#             adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
#             augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

#         # Perform maintenance tasks once per tick.
#         done = (cur_nimg >= total_kimg * 1000)
#         if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
#             continue

#         # Print status line, accumulating the same information in training_stats.
#         tick_end_time = time.time()
#         fields = []
#         fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
#         fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
#         fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
#         fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
#         fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
#         fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
#         fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
#         fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
#         fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
#         torch.cuda.reset_peak_memory_stats()
#         fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
#         training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
#         training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
#         if rank == 0:
#             print(' '.join(fields))

#         # Check for abort.
#         if (not done) and (abort_fn is not None) and abort_fn():
#             done = True
#             if rank == 0:
#                 print()
#                 print('Aborting...')

#         # Save image snapshot.
#         if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
#             images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
#             save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

#         # Save network snapshot.
#         snapshot_pkl = None
#         snapshot_data = None
#         if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
#             snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
#             for key, value in snapshot_data.items():
#                 if isinstance(value, torch.nn.Module):
#                     value = copy.deepcopy(value).eval().requires_grad_(False)
#                     if num_gpus > 1:
#                         misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
#                         for param in misc.params_and_buffers(value):
#                             torch.distributed.broadcast(param, src=0)
#                     snapshot_data[key] = value.cpu()
#                 del value # conserve memory
#             snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
#             if rank == 0:
#                 with open(snapshot_pkl, 'wb') as f:
#                     pickle.dump(snapshot_data, f)

#         # Evaluate metrics.
#         if (snapshot_data is not None) and (len(metrics) > 0):
#             if rank == 0:
#                 print('Evaluating metrics...')
#             for metric in metrics:
#                 result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
#                     dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
#                 if rank == 0:
#                     metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
#                 stats_metrics.update(result_dict.results)
#         del snapshot_data # conserve memory

#         # Collect statistics.
#         for phase in phases:
#             value = []
#             if (phase.start_event is not None) and (phase.end_event is not None):
#                 phase.end_event.synchronize()
#                 value = phase.start_event.elapsed_time(phase.end_event)
#             training_stats.report0('Timing/' + phase.name, value)
#         stats_collector.update()
#         stats_dict = stats_collector.as_dict()

#         # Update logs.
#         timestamp = time.time()
#         if stats_jsonl is not None:
#             fields = dict(stats_dict, timestamp=timestamp)
#             stats_jsonl.write(json.dumps(fields) + '\n')
#             stats_jsonl.flush()
#         if stats_tfevents is not None:
#             global_step = int(cur_nimg / 1e3)
#             walltime = timestamp - start_time
#             for name, value in stats_dict.items():
#                 stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
#             for name, value in stats_metrics.items():
#                 stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
#             stats_tfevents.flush()
#         if progress_fn is not None:
#             progress_fn(cur_nimg // 1000, total_kimg)

#         # Update state.
#         cur_tick += 1
#         tick_start_nimg = cur_nimg
#         tick_start_time = time.time()
#         maintenance_time = tick_start_time - tick_end_time
#         if done:
#             break

#     # Done.
#     if rank == 0:
#         print()
#         print('Exiting...')

# import os
# import stylegan3

# print(os.path.dirname(stylegan3.__file__))
