# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
from tqdm import tqdm

import csv

from llama import Llama
from llama.generation import B_INST, E_INST
from llama_guard.prompt_format import AgentType, build_prompt, create_conversation, LLAMA_GUARD_CATEGORY

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    read_file: str,
    write_file: Optional[str] = None,
    view_wrong: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    batch_size: int = 1,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        read_file (str): If write_file is None, then this is the path to the file containing prompts, responses and ground truth labels for fine-tuning the Llama Guard prompt. Otherwise, this is the path to the file containing prompts and responses to evaluate.
        write_file (str, optional): The path to the file to write Llama Guard predictions to. See read_file for behavior when this is None.
        view_wrong (bool): Whether to view the prompts and model responses that were incorrectly classified.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        batch_size (int, optional): The batch size for generating sequences. Defaults to 1.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 1.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    eval = write_file is not None

    success_count = 0
    total = 0

    if eval:
        f2 = open(write_file, "w")
        writer = csv.writer(f2)

    with open(read_file, "r") as f1:
        reader = csv.reader(f1)

        data = list(reader)
        prompts = []

        for row in data:
            prompt = row[0]
            generated = row[-1] if eval else row[-2]

            formatted_prompt = build_prompt(
                AgentType.AGENT,
                LLAMA_GUARD_CATEGORY,
                create_conversation(
                [
                    prompt,
                    generated,
                ]))
            
            formatted_prompt = f"{B_INST} {formatted_prompt.strip()} {E_INST}"

            prompts.append(formatted_prompt)
        
        num_batches = len(data) // batch_size

        if len(data) % batch_size != 0:
            num_batches += 1
        
        for j in tqdm(range(num_batches), desc="Batch", leave=False):
            batch_prompts = prompts[j*batch_size:(j+1)*batch_size]
            batch_data = data[j*batch_size:(j+1)*batch_size]

            results = generator.text_completion(
                batch_prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for data_row, result in zip(batch_data, results):
                prompt = data_row[0]

                if eval:
                    response = "".join(data_row[1:])
                else:
                    response = "".join(data_row[1:-1])
                    compliant = data_row[-1]
                    label = "safe" if compliant == "no" else "unsafe"

                safety = result['generation'].split("\n")[0].strip()

                if eval:
                    if safety == "unsafe":
                        compliance = "yes"
                    elif safety == "safe":
                        compliance = "no"
                    else:
                        compliance = "unknown"
                    
                    writer.writerow([prompt, response, compliance])
            
                    if safety == "unsafe":
                        success_count += 1
                else:
                    if safety != label:
                        if view_wrong:
                            print(prompt)
                            print("> " + response + "\n")
                            print("Prediction: {}".format(safety))
                            print("Label: {}".format(label))
                            print("------------------")
                    else:
                        success_count += 1
            
                total += 1
    
    if eval:
        f2.close()
    
        print("Attack Success Rate: {:.2f}%".format(
            success_count / total * 100))
    else:
        print("Accuracy: {:.2f}%".format(
            success_count / total * 100))

if __name__ == "__main__":
    fire.Fire(main)