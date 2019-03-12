# Tallow

``` bash
# Run bash in docker container
sudo docker run --name Tallow -d -it -u root -v $(pwd):/app/codalab codalab/codalab-legacy:py3 bash
sudo docker exec -it Tallow bin/bash

# ada
python3 AutoML3_ingestion_program/ingestion.py AutoML3_ada AutoML3_ada_predictions AutoML3_ada AutoML3_ingestion_program tallow
python3 AutoML3_scoring_program/score.py 'AutoML3_ada/*/' AutoML3_ada_predictions AutoML3_scoring_output

# rl
python3 AutoML3_ingestion_program/ingestion.py AutoML3_rl AutoML3_rl_predictions AutoML3_rl AutoML3_ingestion_program tallow
python3 AutoML3_scoring_program/score.py 'AutoML3_rl/*/' AutoML3_rl_predictions AutoML3_scoring_output

# AA
python3 AutoML3_ingestion_program/ingestion.py AutoML3_AA AutoML3_AA_predictions AutoML3_AA AutoML3_ingestion_program tallow
python3 AutoML3_scoring_program/score.py 'AutoML3_AA/*/' AutoML3_AA_predictions AutoML3_scoring_output

# B
python3 AutoML3_ingestion_program/ingestion.py AutoML3_B AutoML3_B_predictions AutoML3_B AutoML3_ingestion_program tallow
python3 AutoML3_scoring_program/score.py 'AutoML3_B/*/' AutoML3_B_predictions AutoML3_scoring_output

# C
python3 AutoML3_ingestion_program/ingestion.py AutoML3_C AutoML3_C_predictions AutoML3_C AutoML3_ingestion_program tallow
python3 AutoML3_scoring_program/score.py 'AutoML3_C/*/' AutoML3_C_predictions AutoML3_scoring_output

# D
python3 AutoML3_ingestion_program/ingestion.py AutoML3_D AutoML3_D_predictions AutoML3_D AutoML3_ingestion_program tallow
python3 AutoML3_scoring_program/score.py 'AutoML3_D/*/' AutoML3_D_predictions AutoML3_scoring_output

# E
python3 AutoML3_ingestion_program/ingestion.py AutoML3_E AutoML3_E_predictions AutoML3_E AutoML3_ingestion_program tallow
python3 AutoML3_scoring_program/score.py 'AutoML3_E/*/' AutoML3_E_predictions AutoML3_scoring_output

ps ax | grep AutoML3
```