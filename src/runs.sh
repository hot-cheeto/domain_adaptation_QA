################# Experiments Commands ####################

### 1. out the box 
export CUDA_VISIBLE_DEVICES=0
python -u main.py --evaluate \
                  --batch_size 64 \
                  --experiment_id out_the_box_test_dev \
                  --create_prediction_file \
                  --sanity \
                  --batch_size 5 \



### 2. finetune on tiny training dataset



### 3. oracle : or cheating bert 



### 4. train jointly with squad with oversample bioasq data




### 5. train jointly with squad with oversample bioasq data and dann




