# MemFree Implementation with DataPortraits

## Data Portraits Initailization

[Data Portraits](https://github.com/ruyimarone/data-portraits) is a Bloom Filter based membership inference system. We utilize it for n-gram detection. Below are the steps on how to initialize Data Portraits and how to create the bloom filter. 
1. Go to ``data-portraits/``, install the library:
    ```sh
    pip install -e . #install editable package
    ```

2. Install Redis and Redis-Bloom
   1. Install Redis: https://redis.io/docs/install/install-redis/install-redis-from-source/
        ```sh
        wget https://download.redis.io/redis-stable.tar.gz

        tar -xzvf redis-stable.tar.gz
        cd redis-stable
        make
        ```
        ```sh
        make install PREFIX="$(pwd)" # install to redis-stable/bin
        ```
   2. Install Redis-Bloom (For successful compilation, the GNU version should >4.0)
        ```
        git clone https://github.com/RedisBloom/RedisBloom.git
        cd RedisBloom
        git checkout tags/v2.4.3 
        git submodule update --init --recursive
        make -j 4
        cd ..
        ```
   3. Add the path where your redis is intalled to your ``.bashrc`` file. Add ``export PATH=$PATH:/your/path/to/data-portraits/redis-stable/bin`` at the end of your ``.bashrc`` file.
   4. Restart bash, try to run command ```redis-server``` on bash to see whether it will work.
## Create the Bloom Filter (.bf file)
   1. go to ``data-portraits`` folder
   2. use ``create_bf.py`` to create the .bf file, specify the dataset and the number of example you want to feed into bloom filter. We also provide a script in ``scripts/create_bf.slurm`` for reference.
