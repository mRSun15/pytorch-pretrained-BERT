
python3 ARSC_lm_finetuning.py \
--train_corpus samples/sample_text.txt \
--bert_model bert-base-uncased \
--do_lower_case \
--output_dir /tmp/finetuned_lm/ \
--do_train