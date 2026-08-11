import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset, SpecifiedLengthDataset, LengthBucketDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb

transformers.logging.set_verbosity_error()

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # DEP parameters
    parser.add_argument("--dense_stage_ratio", type=float, default=0.5)
    parser.add_argument("--dense_length", type=int, default=256)
    parser.add_argument("--bucket_num", type=int, default=3)
    parser.add_argument("--cali_freq", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=512)

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)

    from datetime import datetime
    args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-DeBLAS-dense_length{args.dense_length}-cali_freq{args.cali_freq}_bucket_num{args.bucket_num}-lr{args.lr}-bs-{args.total_batch_size}-sq_length-{args.max_length}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 0
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens

@torch.no_grad()
def exam_model(model, dev_dataloader, pad_idx, device, split_ranges):
    evaluated_on_tokens = 0
    length_list = []
    loss_list = []
    tmp_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    for batch_id, batch in enumerate(dev_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        sequence_length = batch['attention_mask'].sum(dim=1)
        length_list.append(sequence_length.cpu())

        logits, _ = tmp_model.logit_feature_forward(**batch)
        shift_logits = logits[..., :-1, :].contiguous()
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        shift_labels = labels[..., 1:].contiguous()
        tmp_loss = torch.nn.functional.cross_entropy(shift_logits.permute((0, 2, 1)), shift_labels, reduction='none')

        tmp_loss_sum = tmp_loss.sum(dim=1)
        mask = (tmp_loss != 0.0)
        tmp_loss_nonzero_num = mask.sum(dim=1)
        reduced_tmp_loss = tmp_loss_sum / tmp_loss_nonzero_num
        loss_list.append(reduced_tmp_loss.detach().float().cpu())

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item()

    max_length = split_ranges[-1][0]
    num_bucket = len(split_ranges)
    bin_edges = np.arange(0, max_length + 1, (max_length) // (num_bucket - 1))
    length_list = torch.cat(length_list).numpy()
    loss_list = torch.cat(loss_list).numpy()
    bin_indices = np.digitize(length_list, bin_edges)
    percentage = [np.mean(bin_indices == i) for i in range(1, num_bucket + 1)]
    bucket_loss = [np.mean(loss_list[bin_indices == i]) for i in range(1, num_bucket + 1)]
    total_loss = np.mean(loss_list)

    return percentage, bucket_loss, total_loss


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    model_name = args.model_config.split('/')[1].split('.')[0]
    if global_rank == 0:
        wandb.init(project="C4",
                       name=f"{model_name}_DeBLAS_dense_length{args.dense_length}_bucket-num{args.bucket_num}_cali_freq{args.cali_freq}_total-bs_{args.total_batch_size}_sq-length_{args.max_length}_tr_steps_{args.num_training_steps}_lr{str(args.lr).replace('.', '_')}")

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 42

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    skip_data_num = args.total_batch_size * args.num_training_steps * 2
    second_phase_data = data.skip(skip_data_num)
    dev_data = second_phase_data.take(10000)
    second_phase_data = second_phase_data.skip(10000)
    if not args.single_gpu:
        first_phase_data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )
        second_phase_data = datasets.distributed.split_dataset_by_node(
            second_phase_data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    first_phase_bs = int(args.batch_size * args.max_length / args.dense_length)
    print(f'The first phase batch size is {first_phase_bs}')
    unlimited_tokenizer = AutoTokenizer.from_pretrained("t5-base")
    first_phase_dataset = SpecifiedLengthDataset(first_phase_data, unlimited_tokenizer,
                                                 batch_size=first_phase_bs,
                                                 specified_length=args.dense_length)

    bucket_size = args.max_length // (args.bucket_num - 1)
    second_phase_split_ranges = [(bucket_size * i, bucket_size * (i + 1)) for i in range(args.bucket_num - 1)]
    second_phase_split_ranges.append((args.max_length, args.max_length + 1))
    second_phase_split_ratio = [0.0] * args.bucket_num
    second_phase_split_ratio[-1] = 1.0
    logger.info(f'The split ranges of the second phase is {second_phase_split_ranges} for split ratio {second_phase_split_ratio}')
    second_phase_dataset = LengthBucketDataset(second_phase_data, tokenizer, batch_size=args.batch_size,
                                               max_length=args.max_length,
                                               split_ranges=second_phase_split_ranges,
                                               initial_ratio=second_phase_split_ratio)

    first_phase_dataloader = torch.utils.data.DataLoader(first_phase_dataset, batch_size=None, num_workers=args.workers)
    second_phase_dataloader = torch.utils.data.DataLoader(second_phase_dataset, batch_size=None,num_workers=args.workers)
    dev_dataset = PreprocessedIterableDataset(dev_data, tokenizer, batch_size=args.eval_batch_size, max_length=args.max_length)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        optimizer_checkpoint = torch.load(os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu")
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        # wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    update_time_list = []

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    iter_first_phase_dataloader = iter(first_phase_dataloader)
    iter_second_phase_dataloader = iter(second_phase_dataloader)
    first_phase_iter_num = int(args.num_training_steps * args.dense_stage_ratio)

    for batch_idx in range(args.num_training_steps * 20):

        if update_step >= args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        if update_step < first_phase_iter_num:
            batch = next(iter_first_phase_dataloader)
            if update_step == 0 and global_step % args.gradient_accumulation == 0:
                sequence_length_mean = batch['attention_mask'].sum(dim=1).float().mean()
                logger.info(f'The averaged sequence length for update step {update_step} is {sequence_length_mean}')
        else:
            if update_step == first_phase_iter_num and global_step % args.gradient_accumulation == 0:
                logger.info(f'The second phase begins at update step {update_step}')
            if update_step % args.cali_freq == 0 and global_step == update_step * args.gradient_accumulation:
                percentage, bucket_loss, total_loss = exam_model(model, dev_dataloader, pad_idx, device,
                                                                 second_phase_split_ranges)
                percentage = torch.tensor(percentage)
                bucket_loss = torch.tensor(bucket_loss)
                if global_rank == 0:
                    logger.info(f'The total loss on dev set at update step {update_step}: {total_loss}')
                    logger.info(f'The composition of dev set at update step {update_step}: {percentage} for {second_phase_split_ranges}')
                    logger.info(f'The bucket_loss at update step {update_step}: {bucket_loss}')
                weighted_loss = percentage * bucket_loss
                logger.info(f'The weighted loss is {weighted_loss}')
                weighted_loss_sum = weighted_loss.sum()
                new_split_ratio = [(w_loss / weighted_loss_sum).item() for w_loss in weighted_loss]
                logger.info(
                    f'The split ratio outside for global_rank {global_rank} at update step {update_step}: {["{:.3f}".format(num) for num in new_split_ratio]}')
                second_phase_dataset.set_ratio(new_split_ratio)
            batch = next(iter_second_phase_dataloader)
            if update_step % args.cali_freq == 50 and global_step % args.gradient_accumulation == 0:
                sequence_length = batch['attention_mask'].sum(dim=1).float()
                bin_edges = np.arange(0, args.max_length + 1, (args.max_length) // (args.bucket_num - 1))
                bin_indices = np.digitize(sequence_length.numpy(), bin_edges)
                current_train_length_ratio = [np.mean(bin_indices == i) for i in range(1, args.bucket_num + 1)]
                logger.info(
                    f'The training sequence length ratio for update step {update_step} is {current_train_length_ratio} for {second_phase_split_ranges}')

        global_step += 1
        local_step += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue


        # The below code is only executed during the update step

        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        grad_norm = sum([torch.norm(p.grad.clone().detach().cpu()) for p in model.parameters() if p.grad is not None])

        if global_rank == 0: pbar.update(1)

        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time
        update_time_list.append(update_time)

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(current_model_directory, max_shard_size='100GB')

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.eval_batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_perplexity": np.exp(total_loss),
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "gradnorm": grad_norm,
                "max_memory": max_memory,
                "update_time": update_time,
                },
                step=global_step,
            )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc;
    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.eval_batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_perplexity": np.exp(total_loss),
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss and perplexity: {total_loss} {np.exp(total_loss)}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
