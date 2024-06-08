port=${1:-8899}

base_dir="./instances"
datestamp=$(date +"%Y-%m-%d_%H-%M-%S")
run_name="${datestamp}"_"$RANDOM"
re_bloom="RedisBloom/bin/linux-x64-release/redisbloom.so"
conf_file="../../redis.conf"

#echo $run_name
run_dir=$base_dir/$run_name
#echo $run_dir

# copy config and start redis
mkdir -p $run_dir
# cp $conf_file $run_dir
cd $run_dir

redis-server $conf_file --port $port --daemonize yes --loadmodule ../../$re_bloom
sleep 2 # give redis a chance to start

if [ -e "redis.pid" ]
then
    echo "pid file exists, redis was started!" 1>&2
    echo $HOSTNAME:$default_port
    exit 0
fi

echo "Failed to start, check log in $run_dir"
exit 1
