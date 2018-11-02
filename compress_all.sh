#!/bin/sh

BASEDIR=$(dirname "$0")

for example_dir in ${BASEDIR}/*/
do
    pushd ${example_dir}
    zip -r ../`basename ${example_dir}` * -x "main_dev.py" -x "train.py" -x "predict.py" -x "compress_all.sh" -x "*/__pycache__/*"
    popd
done
