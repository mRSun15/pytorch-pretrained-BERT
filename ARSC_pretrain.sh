
python3 ARSC_lm_finetuning.py \
--train_corpus data/Amazon_corpus.txt \
--bert_model bert-base-uncased-file \
--do_lower_case \
--output_dir /tmp/finetuned_lm/ \
--train_batch_size 64 \
--do_train