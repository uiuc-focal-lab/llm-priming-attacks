# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog
from llama.generation_attack import Llama
from tqdm import tqdm
import torch

import csv

PROMPT_MSG = "Prompt (h for help): "
HELP_MSG = """Commands:
- h: Print this help message.
- i: Toggle interactive mode.
- q: Quit.
- r <file>: Read a CSV file of prompts and prefixes.
- w <file>: Write to a CSV file (off by default).
- w off: Stop writing to a CSV file.
"""

def get_prompt_message(
    interactive: bool,
    write_file: str = None,
) -> str:
    """
    Returns the prompt message to display.

    Args:
        interactive (bool): Whether interactive mode is enabled.
        write_file (str, optional): The path to the CSV file to write to. Defaults to None.

    Returns:
        str: The prompt message.
    """

    if interactive or write_file is not None:
        flags = []

        if interactive:
            flags.append("I")
        
        if write_file is not None:
            flags.append("W " + write_file)
        
        PREFIX = "[" + ", ".join(flags) + "] "
    else:
        PREFIX = ""

    return PREFIX + PROMPT_MSG


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    batch_size = 8,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for performing priming attacks on Llama 2 models.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        batch_size (int, optional): The batch size for generating sequences.
        max_batch_size (int, optional): The maximum batch size for generating sequences.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    interactive = True
    write_file = None

    while True:
        rank = torch.distributed.get_rank()

        if rank == 0:
            prompt = input(get_prompt_message(interactive, write_file))
            prompt = torch.tensor([ord(c) for c in prompt], dtype=torch.uint8)
            prompt_len = torch.tensor([len(prompt)], dtype=torch.int)
        else:
            prompt_len = torch.zeros(1, dtype=torch.int)
        
        torch.distributed.broadcast(prompt_len, src=0)

        if rank != 0:
            prompt = torch.zeros(prompt_len.item(), dtype=torch.uint8)

        torch.distributed.broadcast(prompt, src=0)
        prompt = [chr(c) for c in prompt.tolist()]
        prompt = "".join(prompt)

        read_data = None

        if prompt == "h":
            print(HELP_MSG)
            continue
        elif prompt == "i":
            interactive = not interactive
            continue
        elif prompt.strip() == "":
            continue
        elif prompt == "q":
            break
        elif prompt[0:2] == "w " and \
            len(w_args := prompt.split(" ")) == 2:

            if w_args[1] == "off":
                write_file = None
            else:
                write_file = w_args[1]
            
            continue
        elif prompt[0:2] == "r " and \
            len(r_args := prompt.split(" ")) == 2:

            read_file = r_args[1]
            dialogs: List[Dialog] = []

            try:
                with open(read_file, "r") as f:
                    reader = csv.reader(f, delimiter=",")
                    read_data = list(reader)
                
                    for row in read_data:
                        prompt = row[0]
                        prefix = "".join(row[1:])

                        dialogs.append([
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": prefix},
                        ])
            except FileNotFoundError:
                print("File {} not found.\n".format(read_file))
                continue
        else:
            dialogs: List[Dialog] = [
                [
                    {"role": "user", "content": prompt},
                ],
            ]

        num_batches = len(dialogs) // batch_size

        if len(dialogs) % batch_size != 0:
            num_batches += 1

        batch_range = range(num_batches)

        if rank == 0 and len(dialogs) > 1:
            batch_range = tqdm(batch_range, desc="Batch")

        for i in batch_range:
            batch = dialogs[i*batch_size:(i+1)*batch_size]

            results = generator.chat_completion(
                batch,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                manual=True,
                interactive=interactive,
            )
            
            if rank == 0 and results is not None:
                for j, (dialog, result) in enumerate(zip(batch, results)):
                    assistant_reply = dialog[-1]["role"] == "assistant"

                    if assistant_reply:
                        full_decoding = generator.tokenizer.decode(
                            generator.tokenizer.encode(
                                dialog[-1]["content"], bos=False, eos=False) + 
                            generator.tokenizer.encode(
                                result["generation"]["content"], bos=False, eos=True))
                    
                        result["generation"]["content"] = full_decoding
                        dialog = dialog[:-1]

                    for msg in dialog:
                        print(f"{msg['role'].capitalize()}: {msg['content']}")

                    print(
                        f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}", end='')
                    print("\n==================================\n")

                    if write_file is not None:
                        with open(write_file, "a") as f:
                            writer = csv.writer(f, delimiter=",")

                            generated = result["generation"]["content"]

                            if read_data is None:
                                context = " ".join([
                                    (msg['role'].capitalize() + ": " if len(dialog) > 1 else "") + msg["content"] for msg in dialog
                                ])

                                row = [context]
                            else:
                                row = read_data[i*batch_size + j]
                                prefix = "".join(row[1:])

                                if len(prefix) > 0:
                                    generated = generated.split(prefix)[1]
                            
                            generated = generated.replace("\n", " ")
                            row.append(generated)
                            
                            writer.writerow(row)

if __name__ == "__main__":
    fire.Fire(main)