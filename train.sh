#!/bin/bash

set -x
DATA_DIR=$PWD/demo
LOG_DIR=$PWD/demo
MODEL_DIR=$PWD/cls
CFG=NONE
DATAFORM='zip'

mkdir /tmp/cache/python/lib/
mkdir /tmp/cache/python/lib/python3.6/
mkdir /tmp/cache/python/lib/python3.6/site-packages/
python setup.py install --user

# parsing command line arguments:
while [[ $# > 0 ]]
do
key="$1"

case $key in
    -h|--help)
    echo "Usage: toolkit-execute [run_options]"
    echo "Options:"
    echo "  -c|--cfg <config> - which configuration file to use (default NONE)"
    exit 1
    ;;
    -c|--cfg)
    CFG="$2"
    shift # pass argument
    ;;
    *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    ;;
esac
shift # past argument or value
done

echo "==> train"
echo $PWD
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py ${CFG} --launcher pytorch
