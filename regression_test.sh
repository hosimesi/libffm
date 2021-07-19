#!/bin/bash

set +ex

mkdir -p model
rm ./model/*.model

set -ex

make clean
make USEOMP=OFF
python setup.py develop

./ffm-train -p ./bigdata.te.txt -W ./bigdata.iw.txt --auto-stop --auto-stop-threshold 3 ./bigdata.tr.txt ./model/dummy-1.model
pyffm-train -p ./bigdata.te.txt -W ./bigdata.iw.txt --auto-stop --auto-stop-threshold 3 ./bigdata.tr.txt ./model/dummy-2.model

diff ./model/dummy-1.model ./model/dummy-2.model

./ffm-predict ./bigdata.te.txt ./model/dummy-1.model ./model/predicted-1.txt
pyffm-predict ./bigdata.te.txt ./model/dummy-2.model ./model/predicted-2.txt

diff ./model/predicted-1.txt ./model/predicted-2.txt

echo "ok"
