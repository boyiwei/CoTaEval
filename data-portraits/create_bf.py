from dataportraits import datasketch
import dataportraits.utils as utils
import subprocess
import pandas as pd
import os
from transformers import AutoTokenizer


# Connect to Redis
subprocess.run(f"python easy_redis.py --shutdown", shell=True, check=True, capture_output=True)
subprocess.run(f"python easy_redis.py --just-start", shell=True, check=True, capture_output=True)
def count_tokens(string_list):
    total_tokens = 0
    for string in string_list:
        tokens = string.split()  # Split the string into tokens by whitespace
        total_tokens += len(tokens)
    return total_tokens

def main(args):
    datatype = args.datatype
    n=args.n
    width=n
    stride=n
    if args.tokenized:
        bf_name = f'{datatype}_tokenized.{6*width}-{6*stride}.bf'
    else:
        bf_name = f'{datatype}.{width}-{stride}.bf'
    tokenizer_path = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    r = datasketch.RedisBFSketch(host='localhost', port=8899, key=bf_name, width=int(n))
    r.bf_client.create(bf_name, 0.001, 1500000000) # bfname, error_rate, max_entries


    if datatype == 'newsqa':
        path = '../eval_data/newsqa/newsqa_blocklisted_infringement.csv'
    elif datatype == 'booksum':
        path = '../eval_data/booksum/booksum_blocklisted_infringement.csv' 


    # List of lyrics

    if datatype in ['newsqa']:
        newsqa_dataset = pd.read_csv(path)
        input_sequences  = newsqa_dataset['story_text']
    elif datatype in ['booksum']:
        booksum_dataset = pd.read_csv(path)
        input_sequences = booksum_dataset['document'].to_list()
   
    print(f"successfully loaded testing chunks from {path}")

    if args.tokenized:  
        text_pipeline=datasketch.build_text_pipeline_fn(width=6*n, stride=6, apply_code_processor=True) # Here we transfer all token_id into a 6-digit number
        tokenized_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in input_sequences]
        tokenized_sequences_tostring=[''.join(f"{num:06d}" for num in sublist) for sublist in tokenized_sequences]
        # tokenized_sequences_tostring = [np.array(seq).tobytes() for seq in tokenized_sequences]
        grams=utils.flatten_batched(text_pipeline(batches_of_text=tokenized_sequences_tostring))
        r.redis_client.execute_command('BF.MADD', bf_name, *grams[1])
        # for gram in tqdm(grams[1]):
        #     token_ids = tokenizer.encode(gram, return_tensors='pt', add_special_tokens=False).squeeze().tolist()
        #     token_ids_hash = hash_token_ids(token_ids)
        
    else:
        text_pipeline=datasketch.build_text_pipeline_fn(width=n, stride=n, apply_code_processor=True)
        grams=utils.flatten_batched(text_pipeline(batches_of_text=input_sequences))
        r.redis_client.execute_command('BF.MADD', bf_name, *grams[1])


    # check whether the path exists
    if args.tokenized:
        if not os.path.exists(f'/your/path/bloom_filters/{datatype}_tokenized/{n}'):
            os.makedirs(f'/your/path/bloom_filters/{datatype}_tokenized/{n}')
        path = f'/your/path/bloom_filters/{datatype}_tokenized/{n}/{bf_name}'
        r.to_file(path=path, verbose=False) #verbose: Show the progress.
    else:
        if not os.path.exists(f'/your/path/bloom_filters/{datatype}/{n}'):
            os.makedirs(f'/your/path/bloom_filters/{datatype}/{n}')
        path = f'/your/path/bloom_filters/{datatype}/{n}/{bf_name}'
        r.to_file(path=path, verbose=False) #verbose: Show the progress.


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a bloom filter for a dataset')
    parser.add_argument('--datatype', type=str, default='lyrics_popular', help='Type of the dataset')
    parser.add_argument('--n', type=int, default=50, help='Width of the bloom filter')
    parser.add_argument('--tokenized', action='store_true', help='Whether the input is tokenized')
    args = parser.parse_args()
    main(args)