import logging
import dataclasses
import pprint

import torch
from torch import optim
import torch.cuda
import torch.utils.data
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter
from pyhocon import ConfigTree

import src.models
import src.data
import src.loss
import src.optimizer
import src.scheduler

from src.config import Args, get_args
from src.registry import DATA, MODEL, OPTIMIZER, SCHEDULER, LOSS
from src.surrogate_model_training.engine import train_one_epoch, evaluate_one_epoch

from src.torchutils.common import set_cudnn_auto_tune, set_reproducible, generate_random_seed, disable_debug_api
from src.torchutils.common import set_proper_device, get_device
from src.torchutils.common import compute_nparam
from src.torchutils.common import StateCheckPoint
from src.torchutils.common import MetricsStore
from src.torchutils.metrics import EstimatedTimeArrival
from src.torchutils.logging import init_logger, create_code_snapshot


_logger = logging.getLogger(__name__)


def excute_pipeline(
    only_evaluate: bool,
    start_epoch: int,
    max_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    metric_store: MetricsStore,
    state_ckpt: StateCheckPoint,
    states: dict,
    **kwargs
):
    if only_evaluate:
        metric_store += evaluate_one_epoch(
            epoch=0,
            loader=val_loader,
            **kwargs
        )
        return

    eta = EstimatedTimeArrival(max_epochs)

    for epoch in range(start_epoch+1, max_epochs+1):

        metric_store += train_one_epoch(
            epoch=epoch,
            loader=train_loader,
            **kwargs
        )

        metric_store += evaluate_one_epoch(
            epoch=epoch,
            loader=val_loader,
            **kwargs
        )

        for name, metric in metric_store.get_last_metrics().items():
            if "point_wise_l1_distance_list" in  name:
                if len(metric) > 0:
                    point_wise_l1_distance = torch.stack(metric).mean(dim=0).tolist()
                    for i, value in enumerate(point_wise_l1_distance):
                        writer.add_scalar(f"eval/point_wise_l1_distance/epoch_{i}", value, i)
            else:
                writer.add_scalar(name, metric, epoch)
            
        state_ckpt.save(metric_store=metric_store, states=states)

        eta.step()

        best_metrics = metric_store.get_best_metrics()
        _logger.info(f"Epoch={epoch:04d} complete, best val loss={best_metrics['eval/loss']:.8f} "
                     f"(epoch={metric_store.best_epoch+1}), {eta}")


def prepare_for_training(conf: ConfigTree, output_dir: str, local_rank: int):
    model: nn.Module = MODEL.build_from(conf.get("model"))
    _logger.info(f"Model Parameters: {compute_nparam(model)/1e6:.2f}M")

    train_loader, val_loader = DATA.build_from(conf.get("data"))

    criterion = LOSS.build_from(conf.get("loss"))

    optimizer_config: dict = conf.get("optimizer")
    
    optimizer = OPTIMIZER.build_from(optimizer_config, dict(params=model.parameters()))
    
    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    if torch.cuda.is_available():
        model = model.to(device=get_device())
        criterion = criterion.to(device=get_device())

    writer = SummaryWriter(output_dir)

    metric_store = MetricsStore(dominant_metric_name="eval/loss", max_is_best=False)
    
    states = dict(model=model, optimizer=optimizer, scheduler=scheduler)
    state_ckpt = StateCheckPoint(output_dir)

    state_ckpt.restore(metric_store, states, device=get_device())

    return model, train_loader, val_loader, criterion, optimizer, scheduler, \
        writer, metric_store, state_ckpt, states


def _init(local_rank: int, ngpus_per_node: int, args: Args):
    set_proper_device(local_rank)
    rank = args.node_rank*ngpus_per_node+local_rank
    init_logger(rank=rank, filenmae=args.output_dir/"default.log")


    if set_reproducible:
        set_reproducible(generate_random_seed())
    else:
        set_cudnn_auto_tune()
        disable_debug_api()

    create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
                         source_directory=".", store_directory=args.output_dir)

    _logger.info("Collect envs from system:\n" + get_pretty_env_info())
    _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))



def main_worker(local_rank: int,
                ngpus_per_node: int,
                args: Args,
                conf: ConfigTree):

    _init(local_rank=local_rank, ngpus_per_node=ngpus_per_node, args=args)

    model, train_loader, val_loader, criterion, optimizer, scheduler, \
        writer, metric_store, state_ckpt, states = \
        prepare_for_training(conf, args.output_dir, local_rank)

    excute_pipeline(
        only_evaluate=conf.get_bool("only_evaluate"),
        start_epoch=metric_store.total_epoch,
        max_epochs=conf.get_int("max_epochs"),
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        metric_store=metric_store,
        state_ckpt=state_ckpt,
        states=states,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=get_device(),
        log_interval=conf.get_int("log_interval"),
    )


def main(args: Args):
    ngpus_per_node = torch.cuda.device_count()
    local_rank = 0
    main_worker(local_rank, ngpus_per_node, args, args.conf)

if __name__ == "__main__":
    main(get_args())