export GLUE_DIR="data/Amazon_few_shot"

python3 run_classifier.py \
  --task_name Amazon \
  --do_train \
  --do_lower_case \
  --data_dir $GLUE_DIR \
  --bert_model bert-base-uncased-file \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/Amazon_output/