#!/bin/bash

cd data-portraits
python easy_redis.py --shutdown
python easy_redis.py --start-from-dir /scratch/gpfs/bw1822/bloom_filters/booksum_tokenized/6
cd ..
