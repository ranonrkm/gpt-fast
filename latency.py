# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

from datasets import load_dataset
from tqdm import tqdm

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, **sampling_kwargs):
    '''
    cur_token: [B, 1]
    input_pos: [1]
    '''
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()  #.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    decode_length: int,
    **sampling_kwargs
) -> torch.Tensor:
    B, T = prompt.shape
    T_new = T + decode_length
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=B, max_seq_length=max_seq_length)

    empty = torch.empty((B, T_new), device=device, dtype=dtype) 
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt, input_pos, **sampling_kwargs).clone()
    seq[:, T:T+1] = next_token  # TODO: check shape

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, decode_length - 1, **sampling_kwargs)
    seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    return seq

@torch.no_grad()
def verify(
    model: Transformer,
    prompt: torch.Tensor,
    decode_length: int,
    **sampling_kwargs
):
    B, T = prompt.shape
    prefill_length = T - decode_length
    input_ids = prompt[:, :prefill_length]

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=B, max_seq_length=T)

    input_pos = torch.arange(0, prefill_length, device=device)
    prefill(model, input_ids, input_pos, **sampling_kwargs)

    # verify
    input_pos = torch.arange(prefill_length, T, device=device)
    input_ids = prompt[:, prefill_length:]
    logits = model_forward(model, input_ids, input_pos)

    return sample(logits, **sampling_kwargs)


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size

def main(
    num_samples: int,
    bsz: int,
    prefill_length: int,
    decode_length: int,
    temperature: float = 0.8,
    top_k: int = 200,
    checkpoint_path: Path="/home/rsadhukh/vashisth/gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth",
    compile: bool = True,
    compile_prefill: bool = False,
    device=default_device,
    mode: str = "decode",
    profile: Optional[Path] = None,
) -> None:
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device) # MKG
    print(f"Loaded model in {time.time() - t0:.1f}s")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    # load dataset
    datasetparent = "data/pg19/"
    d_files = os.listdir(datasetparent)
    dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
      
    tokenized_prompts = []
    doc_id = 0
    i = 0
    for _ in range(10):
        cur_batch = []
        tokenized_prompt = encode_tokens(tokenizer, dataset[doc_id]['text'])
        for _ in range(args.bsz):
            if i + prefill_length > tokenized_prompt.size(0):
                i = 0
                doc_id += 1
            assert doc_id < len(dataset)
            cur_batch.append(tokenized_prompt[i : i + prefill_length])
            i += prefill_length
        cur_batch = torch.stack(cur_batch, dim=0)
        tokenized_prompts.append(cur_batch)

    torch.manual_seed(1234)
    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    
    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0

    avg_latency = 0.

    for i in tqdm(range(start, num_samples)):
        device_sync(device=device) # MKG
        encoded = tokenized_prompts[i]
        encoded = encoded.to(device=device)

        if mode == "decode":
            t0 = time.perf_counter()
            import contextlib
            if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            with prof:
                y = generate(
                    model,
                    encoded,
                    decode_length,
                    temperature=temperature,
                    top_k=top_k,
                )        
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue   

            device_sync(device=device) # MKG
            t = time.perf_counter() - t0

            avg_latency += t / decode_length

        else:
            t0 = time.perf_counter()
            import contextlib
            if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            with prof:
                y = verify(
                    model,
                    encoded,
                    decode_length,
                    temperature=temperature,
                    top_k=top_k,
                )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue   

            device_sync(device=device) # MKG
            t = time.perf_counter() - t0

            avg_latency += t            

    print("Avg latency: ", avg_latency / num_samples) 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--prefill-length", type=int, default=2000)
    parser.add_argument("--decode-length", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--checkpoint-path", type=Path, default="/home/rsadhukh/vashisth/gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-prefill", action="store_true")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--mode", type=str, choices=["decode", "verify"], default="decode")

    args = parser.parse_args()

    # pretty print the arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    main(**vars(args))