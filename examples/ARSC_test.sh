export GLUE_DIR="data/glue_data"
CUDA_VISIBLE_DEVICES=1

python3 run_classifier.py \
  --task_name COLA \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/CoLA/ \
  --bert_model bert-base-uncased-file \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output1/