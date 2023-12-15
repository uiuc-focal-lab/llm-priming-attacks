# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

import numpy as np

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from tqdm import tqdm

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        manual: bool = False,
        interactive: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        rank = torch.distributed.get_rank()

        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        if not manual or not interactive:
            prev_pos = 0
            eos_reached = torch.tensor([False] * bsz, device="cuda")
            input_text_mask = tokens != pad_id
            if min_prompt_len == total_len:
                logits = self.model.forward(tokens, prev_pos)
                token_logprobs = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens,
                    reduction="none",
                    ignore_index=pad_id,
                )
            
            prompt_range = range(min_prompt_len, total_len)

            if rank == 0:
                prompt_range = tqdm(prompt_range, desc="Tokens", leave=False)

            for cur_pos in prompt_range:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p, self.tokenizer)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
                if logprobs:
                    token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                        input=logits.transpose(1, 2),
                        target=tokens[:, prev_pos + 1 : cur_pos + 1],
                        reduction="none",
                        ignore_index=pad_id,
                    )
                eos_reached |= (~input_text_mask[:, cur_pos]) & (
                    next_token == self.tokenizer.eos_id
                )
                prev_pos = cur_pos
                if all(eos_reached):
                    break
        else:
            for b in range(bsz):
                prev_pos = 0
                eos_reached = torch.tensor([False] * bsz, device="cuda")
                input_text_mask = tokens != pad_id

                cur_pos = len(prompt_tokens[b])
                skip_counter = 0
                no_pending_skip = True
                skip_pos = cur_pos
                skip_prev_pos = prev_pos
                skip_eos_reached = None
                num_choices = 10
                preview = 10
                option_preview = 3
                log_cond_probs = []
                prev_log_cond_probs_length = 0
                
                while cur_pos < total_len:
                    logits = self.model.forward(tokens[b:b+1, prev_pos:cur_pos], prev_pos)
                    if temperature > 0:
                        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                        probs_sort, probs_idx = torch.sort(
                            probs, dim=-1, descending=True)
                        probs_sort = probs_sort.squeeze().tolist()
                        probs_idx = probs_idx.squeeze().tolist()

                        if (skip_counter == 0 or cur_pos == total_len - 1):
                            if no_pending_skip == False:
                                print(">  \"{}".format(
                                    "..." if skip_pos > preview else "") + 
                                    str(self.tokenizer.decode(
                                        tokens[b, max(0, skip_pos - preview):cur_pos].tolist())) + "\"")
                                
                                if rank == 0:
                                    accept = input("Accept? (y/n): ")
                                    accept = torch.tensor([ord(c) for c in accept], dtype=torch.uint8)
                                    accept_len = torch.tensor([len(accept)], dtype=torch.int)
                                else:
                                    accept_len = torch.zeros(1, dtype=torch.int)
                                
                                torch.distributed.broadcast(accept_len, src=0)

                                if rank != 0:
                                    accept = torch.zeros(accept_len.item(), dtype=torch.uint8)

                                torch.distributed.broadcast(accept, src=0)
                                accept = [chr(c) for c in accept.tolist()]
                                accept = "".join(accept)

                                no_pending_skip = True

                                if accept.lower() == "y":
                                    print("Accepted.")

                                    if all(eos_reached):
                                        break
                                else:
                                    print("Rejected. Reverting...")

                                    cur_pos = skip_pos
                                    prev_pos = skip_prev_pos
                                    skip_counter = 0
                                    eos_reached = skip_eos_reached
                                    log_cond_probs = log_cond_probs[:prev_log_cond_probs_length]
                                    continue

                            if len(log_cond_probs) > 0:
                                perplexity = np.exp(-np.mean(log_cond_probs))
                            
                            print("\n> \"{}".format(
                                "..." if cur_pos > preview else "") + str(
                                self.tokenizer.decode(
                                    tokens[b, max(0, cur_pos - preview):cur_pos]\
                                        .tolist())) + "\"" + 
                                        ("" if len(log_cond_probs) == 0 else " (PPL=" + 
                                            ("{:.3e}" if abs(perplexity) < 0.1 else \
                                            "{:.3f}").format(perplexity) + ")"))
                            
                            for i, (prob, idx) in \
                                enumerate(
                                    zip(
                                        probs_sort[:num_choices],
                                        probs_idx[:num_choices])):

                                print("{}) ".format(i) + "[{}]".format(idx).ljust(7) + 
                                    (" (P=" + 
                                    ("{:.3e}" if abs(prob) < 0.1 else "{:.3f}")\
                                        .format(prob) + "):").rjust(15) + \
                                    " \"{}{}\"".format(
                                        "..." if cur_pos > option_preview else "",
                                        str(self.tokenizer.decode(
                                            tokens[b, max(0, cur_pos - option_preview):cur_pos].tolist() + [idx]))))
                            
                            # Ask user to select token
                            if rank == 0:
                                action = input("Select token (0-{}, gN=generate N tokens, cN=show N choices, r=restart, q=quit): ".format(num_choices - 1))
                                action = torch.tensor([ord(c) for c in action], dtype=torch.uint8)
                                action_len = torch.tensor([len(action)], dtype=torch.int)
                            else:
                                action_len = torch.zeros(1, dtype=torch.int)
                            
                            torch.distributed.broadcast(action_len, src=0)

                            if rank != 0:
                                action = torch.zeros(action_len.item(), dtype=torch.uint8)

                            torch.distributed.broadcast(action, src=0)
                            action = [chr(c) for c in action.tolist()]
                            action = "".join(action)

                            try:
                                selected = int(action)

                                if selected >= 0 and selected < num_choices:
                                    selected_idx = probs_idx[selected]
                                    next_token = torch.tensor([[selected_idx]])
                                else:
                                    print("Invalid number; please enter an integer between 0 and {}.".format(num_choices - 1))
                                        
                                    continue
                            except ValueError:
                                if len(action) > 0:
                                    if action[0] == "g":
                                        requested = int(action[1:])

                                        if requested > 0:
                                            skip_counter = requested
                                            skip_pos = cur_pos
                                            skip_prev_pos = prev_pos
                                            skip_eos_reached = eos_reached.clone()
                                            no_pending_skip = False
                                            prev_log_cond_probs_length = len(log_cond_probs)
                                        else:
                                            print("Invalid number; please enter an integer greater than 0.")
                                        
                                        continue
                                    elif action[0] == "c":
                                        requested = int(action[1:])

                                        if requested > 0 and \
                                            requested <= self.tokenizer.n_words:
                                            num_choices = requested
                                        else:
                                            print("Invalid number; please enter an integer from 1-{}.".format(self.tokenizer.n_words))

                                        continue
                                    elif action[0] == "r":
                                        print("Restarting...")
                                        
                                        cur_pos = min_prompt_len
                                        prev_pos = 0
                                        eos_reached = torch.tensor([False] * bsz, device="cuda")
                                        skip_counter = 0
                                        no_pending_skip = True
                                        skip_pos = cur_pos
                                        skip_prev_pos = prev_pos
                                        skip_eos_reached = None

                                        continue
                                    elif action[0] == "q":
                                        print()
                                        return None
                                    else:
                                        continue
                                else:
                                    continue
                        else:
                            next_token = sample_top_p(probs, top_p, self.tokenizer)
                            skip_counter -= 1
                    else:
                        next_token = torch.argmax(logits[:, -1], dim=-1)

                    next_token = next_token.reshape(-1)

                    if next_token[0] != pad_id:
                        log_cond_probs.append(
                            torch.log(probs[0, next_token[0]]).item())

                    tokens[b:b+1, cur_pos] = next_token
                    if logprobs:
                        token_logprobs[b:b+1, prev_pos + 1 : cur_pos + 1] = \
                            -F.cross_entropy(
                                input=logits.transpose(1, 2),
                                target=tokens[b:b+1, prev_pos + 1 : cur_pos + 1],
                                reduction="none",
                                ignore_index=pad_id,
                            )
                    eos_reached |= (~input_text_mask[:, cur_pos]) & (
                        next_token == self.tokenizer.eos_id
                    )
                    prev_pos = cur_pos

                    if all(eos_reached):
                        if skip_counter > 0:
                            skip_counter = 0
                        else:
                            break
                
                    cur_pos += 1

                if len(log_cond_probs) > 0:
                    perplexity = np.exp(-np.mean(log_cond_probs))

                    print("\nFinal PPL:", 
                        ("{:.3e}" if abs(perplexity) < 0.1 else "{:.3f}")\
                            .format(perplexity))
        
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        manual: bool = False,
        interactive: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            
            convo_length = len(list(zip(dialog[::2], dialog[1::2])))

            dialog_strings = [
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()}{' ' if i != convo_length - 1 or dialog[-1]['role'] == 'user' else ''}"
                for i, (prompt, answer) in enumerate(zip(dialog[::2], dialog[1::2]))
            ]

            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        s,
                        bos=True,
                        eos=(i != convo_length - 1 or dialog[-1]["role"] == "user"),
                    )
                    for i, s in enumerate(dialog_strings)
                ],
                [],
            )
            
            if dialog[-1]["role"] == "user":
                dialog_tokens += self.tokenizer.encode(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                    bos=True,
                    eos=False,
                )

            prompt_tokens.append(dialog_tokens)

        generation_result = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            manual=manual,
            interactive=interactive,
        )

        if generation_result is not None:
            generation_tokens, generation_logprobs = generation_result
        else:
            return None

        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]

def sample_top_p(probs, p, tokenizer=None):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
