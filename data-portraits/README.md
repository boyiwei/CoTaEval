# Data Portraits

<img width="363" alt="image" src="https://github.com/ruyimarone/data-portraits/assets/10734779/3951d2ee-2560-4fd2-90f2-ec840f2dfbee">

<img width="450" alt="image" src="https://github.com/ruyimarone/data-portraits/assets/10734779/f3fec35c-9879-46b0-a4aa-e264dd06bf01">


This is the code for [Data Portraits: Recording Foundation Model Training Data](https://dataportraits.org/) by Marc Marone and Ben Van Durme.

Large models are trained on increasingly immense and opaque datasets, but it can be very difficult to answer a fundamental question: **Was this in a model's training set?**

We call for documentation artifacts that can answer this membership question and term these artifacts **Data Portraits.**

This repo implements one tool that can answer this question -- based on efficient hash storage with Bloom filters.
Of course, many other dataset documentation tools exist.
See our paper for details about this method, other tools, and properties that make ours unique.

For more details, see [our paper](https://openreview.net/pdf?id=ZrNRBmOzwE).

> [!WARNING]
> Consider this an alpha code release - other portrait files, **MUCH** easier redis installation, and the interface will be added soon!

## Installation

This system uses several components: a python library, an bitarray server (redis), and binary files containing data hashes.

1. Install the library
```shell
pip install -e . #install editable package
```

2. Install redis and RedisBlooom

    If you're familiar with redis already and want to use an existing server or a system installation,
    you can skip this section. Just make sure you know your redis connection details and have RedisBloom loaded into 
    your server (with config files or command line args)

    Included launch scripts assume this repo structure but it's easily changed:
    ```
    ├── RedisBloom
    ├── redis_configs
    ├── redis-stable
    ├── scripts
    └── src
        └── dataportraits
    ```

    1. [Install and build Redis](https://redis.io/docs/install/install-redis/install-redis-from-source/)

    Build with `make install PREFIX="$(pwd)" # install to redis-stable/bin` to put binaries in the expected places.

    After instiall the redis, please include the location where you install the redis to your system path in ```.bashrc```
    
    2. [Install and build RedisBloom](https://github.com/RedisBloom/RedisBloom/)

        Our code was tested against version 2.4.3. *We assume a certain binary header structure in serialized Bloom filter files, other redis versions may change this!*

        You can use the instructions in the repo or these:
    ```shell
    git clone https://github.com/RedisBloom/RedisBloom.git
    cd RedisBloom
    git checkout tags/v2.4.3 
    git submodule update --init --recursive
    make -j 4
    cd ..
    ```

3. Start Redis

    We include a launch script helper. If you're familiar with redis already, you can freely launch servers as you would typically or use an existing server.

    ```shell
    python easy_redis.py --just-start
    ```

    Note that this starts a persistent, daemon server on your system.
    Loading sketches will consume ram until you shut it down: `python easy_redis.py --shutdown`

4. Fetch Data Files

    Fetch data from [https://huggingface.co/mmarone/portraits-wikipedia](mmarone/portraits-wikipedia) on HuggingFace:
    ```
    git lfs install
    git clone https://huggingface.co/mmarone/portraits-wikipedia
    ```
    **This takes about 2.5GB local space**

5. Load Files
    ```
    python easy_redis.py --start-from-dir portraits-wikipedia/
    ```
    Note: if redis wasn't already started (e.g. you skipped step 3, this will attempt to start a server for you).



## Usage

More examples coming soon!

```shell
import dataportraits

# localhost:8899 is the default for the redis server started above
# wikipedia.50-50.bf is the name of the system - see the easy_redis.py script for more
# change as necessary!
portrait = dataportraits.RedisBFSketch('localhost', 8899, 'wiki-demo.50-50.bf', 50)

text = """
Test sentence about Data Portraits - NOT IN WIKIPEDIA!
Bloom proposed the technique for applications where the amount of source data would require an impractically large amount of memory if "conventional" error-free hashing techniques were applied
"""
report = portrait.contains_from_text([text])
print(report[0]['chains'])

# [['cations where the amount of source data would requ', 'ire an impractically large amount of memory if "co', 'nventional" error-free hashing techniques were app']]
```


## Citing
If you find this repo or our web demo useful, please cite [our paper](https://openreview.net/pdf?id=ZrNRBmOzwE).
```
@inproceedings{
    marone2023dataportraits,
    title={Data Portraits: Recording Foundation Model Training Data},
    author={Marc Marone and Benjamin {Van Durme}},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/pdf?id=ZrNRBmOzwE}
}
```
