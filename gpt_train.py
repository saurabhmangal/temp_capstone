# import libraries
import glob
import math
import json
import sys
import numpy as np
import os
import time
import torch
from pathlib import Path
from typing import Optional, Tuple, Union
import lightning as L
import torch.nn as nn
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from tsai_gpt.model import GPT, Block, Config
from tsai_gpt.packed_dataset import CombinedDataset, PackedDataset,LLMGeneratedDataset
from tsai_gpt.speed_monitor import SpeedMonitorBase, estimate_flops, measure_flops
from tsai_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from tsai_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, load_checkpoint
from tsai_gpt.prepared_data import *
#from tsai_gpt.packed_dataset import LLMGeneratedDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["https_proxy"] = "http://185.46.212.90:80"
os.environ["http_proxy"] = "http://185.46.212.90:80"
# Configuration for running the model
model_name = "phi-2"
name = "redpajama"
out_dir = Path("out") / name
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-3
batch_size = 8
micro_batch_size = 8
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 60000000000000  # num_epochs * (epoch_size // micro_batch_size) // devices
#max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-6

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

vocab_words = list(tokenizer.get_vocab().keys())

# Data configurations
data_config = [
    ("arxiv_sample_00000000", 33.0),
    ("book_sample_00000000", 33.0),
    ("c4_sample_00000000", 24.0), 
    ("generated_data", 10.0)
]

## setup - main function
hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = CSVLogger("out", name, flush_logs_every_n_steps=log_interval)


def setup(
    devices: int = 4,
    train_data_dir: Path = Path("data/sample"),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    resume: Union[bool, Path] = False,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 
    precision = precision or get_default_supported_precision(training=True)

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, train_data_dir, val_data_dir, resume)

## setup launches main function
def main(fabric: L.Fabric, train_data_dir: Path, val_data_dir: Path, resume: Union[bool, Path]) -> None:
    global model_copy
    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    print(train_data_dir)
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed = (1337 + fabric.global_rank),
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()

    def _init_weights(module: nn.Module) -> None:
            """Meant to be used with `gpt.apply(gpt._init_weights)`."""
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    with fabric.init_module(empty_init=True):
        model = GPT(config)
        model.apply(_init_weights)
    model.apply(_init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)
    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=lambda p: int(p.name.split("-")[1]))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, speed_monitor)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

## train
def train(
    fabric: L.Fabric,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    speed_monitor: SpeedMonitorBase,
) -> None:
    model     = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.max_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    for state["iter_num"], train_data in enumerate(train_dataloader, state["iter_num"]):
        #print("kya yahan hai panga? ")
        #print(state["iter_num"])
        if state["iter_num"] >= max_iters:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.max_seq_length].contiguous()
        targets = train_data[:, 1 : model.max_seq_length + 1].contiguous()

        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (state["iter_num"] + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        # if loss.item()<8:
        #     checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
        #     fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        #     fabric.save(checkpoint_path, state)
        #     break
        if True:#state["iter_num"] % log_interval == 0:
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, LR: {lr:.6f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )
        #print(f'Ab Yahan hai panga: {val_dataloader is not None and not is_accumulating and state["step_count"] % eval_interval == 0}')

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and state["step_count"] % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


## inference
@torch.inference_mode()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0 : model.max_seq_length].contiguous()
        targets = val_data[:, 1 : model.max_seq_length + 1].contiguous()
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits, targets, chunk_size=0)
    out = losses.mean()

    model.train()
    return out


## dataloaders
def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric: L.Fabric, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    print(f'Data Config: {data_config}')
    # for prefix, _ in data_config:
    #     filenames = glob.glob(f'{data_dir}/{prefix}*.bin')
    #     print(f'{data_dir}/{prefix}*.bin')
    #     print(f'yahan 4 hone chaheye: {filenames}')
    #     print(str(data_dir / f"{prefix}"))
    #     dataset = PackedDataset(
    #         filenames,
    #         n_chunks=2,
    #         block_size=block_size,
    #         shuffle=shuffle,
    #         seed=seed,
    #         num_processes=fabric.world_size,
    #         process_rank=fabric.global_rank,
    #     )
    #     datasets.append(dataset)

    llm_dataset = LLMGeneratedDataset(tokenizer, model, vocab_words, block_size, num_samples=10)
    # save as bin
    #print(llm_dataset)
    def save_generated_data(llm_dataset, output_dir, prefix='generated_data', chunk_size=1024):
        # Ensure the output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over the LLMGeneratedDataset to get generated sequences
        for i, sequence in enumerate(llm_dataset):
            # Convert sequence tensor to numpy array and ensure it's an integer type
            sequence_np = sequence

            # Define the filename for this chunk of generated data
            filename = output_dir / f"{prefix}_{i:010d}.json"

            data = {
                'text': sequence_np,
                'meta': {
                    'timestamp': 'gen',
                    'url': prefix,
                    'language': 'en',
                    'source': prefix
                }
            }


            # Writing the dictionary to a .json file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4) 
            f.close()

    # Generate data and save it
    save_generated_data(llm_dataset, output_dir='data/data_gneration')
    prepare(source_path=Path('data/data_gneration'),checkpoint_dir=Path('checkpoints/microsoft/phi-2'),destination_path=Path('data/redpajama_sample'),sample=True)
            # Write the sequence to a binary file (you might need to add a header based on your PackedDataset format)


    for prefix, _ in data_config:
        filenames = glob.glob(f'{data_dir}/{prefix}*.bin')
        # print(f'{data_dir}/{prefix}*.bin')
        # print(f'yahan 4 hone chaheye: {filenames}')
        # print(str(data_dir / f"{prefix}"))
        
        dataset = PackedDataset(
            filenames,
            n_chunks=2,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    # # Create a PackedDataset instance using the generated files
    # generated_dataset = PackedDataset(
    #     filenames=glob.glob('path/to/generated_data/generated_data*.bin'),
    #     n_chunks=2,  # Adjust based on your requirements
    #     block_size=block_size,  # Make sure this matches the block size used for generation
    #     shuffle=True,  # Typically you'd shuffle training data
    #     seed=seed,  # Ensure reproducibility if needed
    #     num_processes=fabric.world_size,
    #     process_rank=fabric.global_rank,
    # )

    # # Now you can add `generated_dataset` to your datasets list for the CombinedDataset
    # datasets = [packed_dataset1, packed_dataset2, packed_dataset3, generated_dataset]





















    # datasets.append(llm_dataset) #+ [PackedDataset(filenames, n_chunks=2, block_size=config.block_size, shuffle=True, seed=seed, num_processes=fabric.world_size, process_rank=fabric.global_rank) for prefix, _ in data_config]
    # weights =  [0.3, 0.4, 0.2, 0.1]#[0.7 / len(data_config) for _ in data_config] + [0.3]

    #print(f'\n\nWeights : {weights}\n\n')
    # print(datasets)
    # print(f'final Data sets: {datasets}')
    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )
    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: L.Fabric,
    train_data_dir: Path = Path("data/sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    
    # print(f'dir inside fuction: create_dataloaders: {train_data_dir}')
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader

## Learning rate scheduler
def get_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


## MAIN CALL - TO TRAIN
torch.set_float32_matmul_precision("medium")
setup(
    devices=3,
    train_data_dir=Path("data/redpajama_sample")
)