#!/bin/bash
# For more info about the data visit
# https://physionet.org/physiobank/database/mitdb/
# and 
# https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

mkdir data && cd data
url=https://physionet.org/physiobank/database/mitdb/
for i in {100..234}
do
    for ext in 'hea' 'dat' 'atr'
    do
        curl -O $url/$i.$ext
    done
done
