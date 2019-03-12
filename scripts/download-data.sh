#!/bin/bash
apt-get unzip
wget https://competitions.codalab.org/my/datasets/download/149e003d-775e-478c-8998-7a6b458582e9
mv 149e003d-775e-478c-8998-7a6b458582e9 AutoML3_input_data.zip
mkdir AutoML3_input_data
mv AutoML3_input_data.zip AutoML3_input_data
cd ./AutoML3_input_data
unzip AutoML3_input_data.zip
rm AutoML3_input_data.zip
cd ../