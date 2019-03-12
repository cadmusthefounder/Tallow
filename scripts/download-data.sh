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

mv AutoML3_input_data AutoML3_AA 
mkdir AutoML3_B
mkdir AutoML3_C
mkdir AutoML3_D
mkdir AutoML3_E

mv AutoML3_AA/B AutoML3_B
mv AutoML3_AA/C AutoML3_C
mv AutoML3_AA/D AutoML3_D
mv AutoML3_AA/E AutoML3_E