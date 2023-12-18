from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import csv
from tqdm import tqdm
import fire

def main(
    read_file: str,
    write_file: str,
    model_name: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for performing priming attacks on Vicuna models.

    Args:
        read_file (str): The path to the file containing the priming attacks.
        write_file (str): The path to the file to write the generated text to.
        model_name (str): The name of the model to use for generation.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        batch_size (int, optional): The batch size for generating sequences.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to None.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    prompts, prefixes = [], []

    with open(read_file, mode ='r') as f:
        reader = csv.reader(f)
        data = list(reader)

        for row in data:
            prompt = row[0]
            prefix = "".join(row[1:])

            prompts.append(prompt)
            prefixes.append(prefix)
            
    dialogs = []

    for prompt, prefix in zip(prompts, prefixes):
        dialog = "USER: " + prompt + " ASSISTANT:"

        if len(prefix) > 0:
            dialog += " "

        dialog += prefix

        dialogs.append(dialog)

    num_batches = len(dialogs) // batch_size

    if len(dialogs) % batch_size != 0:
        num_batches += 1

    with open(write_file, "w") as f:
        writer = csv.writer(f, delimiter=",")

        for i in tqdm(range(num_batches), desc="Batch"):
            batch = tokenizer(
                dialogs[i*batch_size:(i+1)*batch_size], return_tensors="pt", 
                padding=True).to('cuda')

            results = model.generate(
                **batch,
                top_p=top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_gen_len,
                temperature=temperature
            )

            batch_prompts = prompts[i*batch_size:(i+1) * batch_size]
            batch_prefixes = prefixes[i*batch_size:(i+1) * batch_size]

            decoded_results = [tokenizer.decode(results[i]).split('ASSISTANT: ', 1)[-1] for i in range(results.size()[0])]
            
            for (prompt, prefix, result) in zip(batch_prompts, batch_prefixes, decoded_results):
                result = result.split("</s>")[0]

                print(prompt)
                print("> " + result)
                print("\n----------------------------------\n")

                if len(prefix) > 0:
                    result = result.split(prefix)[1]

                writer.writerow([
                    prompt,
                    prefix,
                    result.replace("\n", " ")
                ])

if __name__ == "__main__":
    fire.Fire(main)
