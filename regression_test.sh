#!/bin/bash

set +ex

mkdir -p model
rm ./model/*.model

set -ex

python setup.py develop

./ffm-train -p ./bigdata.te.txt -W ./bigdata.iw.txt -f ./model/dummy-1.model -m key --auto-stop --auto-stop-threshold 3 ./bigdata.tr.txt
pyffm-train -p ./bigdata.te.txt -W ./bigdata.iw.txt -f ./model/dummy-2.model -m key --auto-stop --auto-stop-threshold 3 ./bigdata.tr.txt

diff ./model/dummy-1.model ./model/dummy-2.model

./ffm-train -p ./bigdata.te.txt -W ./bigdata.iw.txt -m key --auto-stop --auto-stop-threshold 3 ./bigdata.tr.txt ./model/dummy-3.model
pyffm-train -p ./bigdata.te.txt -W ./bigdata.iw.txt -m key --auto-stop --auto-stop-threshold 3 ./bigdata.tr.txt ./model/dummy-4.model

diff ./model/dummy-3.model ./model/dummy-4.model

echo "ok"
