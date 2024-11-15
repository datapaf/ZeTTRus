CUDA_VISIBLE_DEVICES=0 python3 scripts/transfer.py \
--target_model=meta-llama/Meta-Llama-3-8B \
--revision=refs/pr/129 \
--tokenizer_name=ikkiren/TokenSubstitution_tokenizer \
--output=llama3-8b-hn-rus \
--model_class=AutoModelForCausalLM \
--checkpoint_path=llama3_8b \
--save_pt