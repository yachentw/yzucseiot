#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1E4Dzn3RLwLNtQt1D90DAMJ7u72Es96Ha" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1E4Dzn3RLwLNtQt1D90DAMJ7u72Es96Ha" -o tensorflow-1.15.0-cp37-cp37m-linux_armv7l.whl
echo Download finished.