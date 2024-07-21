#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=4 ./scripts/test/eval.sh $version: str $test_year: str $error_tol: int


version=$1
test_year=$2
error_tol=$3
dir=lightning_logs/version_$version/

# clean out
rm -rf test_temp/
rm -rf Results_pred_symlg/

# generate predictions
python scripts/test/test.py $version $test_year

# copy predictions to target folder
cp result.zip $dir/$test_year.zip
