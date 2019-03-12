#!/bin/bash
apt-get install unzip
wget https://competitions.codalab.org/my/datasets/download/6dcd864f-10f6-421c-84ce-f97c8635d2bd
mv 6dcd864f-10f6-421c-84ce-f97c8635d2bd AutoML3.zip
unzip AutoML3.zip
rm metadata
rm README.md
rm AutoML3.zip
rm AutoML3_sample_code_submission.zip
rm AutoML3_sample_ref.zip  

mkdir AutoML3_ingestion_program
mv AutoML3_ingestion_program.zip AutoML3_ingestion_program
cd ./AutoML3_ingestion_program
unzip AutoML3_ingestion_program.zip
rm AutoML3_ingestion_program.zip
cd ../

mkdir AutoML3_scoring_program
mv AutoML3_scoring_program.zip AutoML3_scoring_program
cd ./AutoML3_scoring_program
unzip AutoML3_scoring_program.zip
rm AutoML3_scoring_program.zip
cd ../

unzip AutoML3_sample_data.zip
rm AutoML3_sample_data.zip