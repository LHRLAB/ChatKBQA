# ChatKBQA with ChatGLM2-6b

## Fine-tuning, Retrieval and Evaluation

The following is an example of ChatGLM2-6b fine-tuning and retrieval.

(1) **Train and test LLM model for Logical Form Generation**

- WebQSP: 

Train LLMs for Logical Form Generation:

```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path THUDM/chatglm2-6b --do_train  --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_train --template chatglm2  --finetuning_type lora --lora_target query_key_value --output_dir Reading/ChatGLM2-6b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 100.0 --plot_loss  --fp16 >> train_ChatGLM2-6b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path THUDM/chatglm2-6b --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_test --template chatglm2  --finetuning_type lora --checkpoint_dir Reading/ChatGLM2-6b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --num_beams 10 >> predbeam_ChatGLM2-6b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/ChatGLM2-6b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/generated_predictions.jsonl
```

- CWQ: 

Train LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path THUDM/chatglm2-6b --do_train  --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_train --template chatglm2   --finetuning_type lora --lora_target query_key_value --output_dir Reading/ChatGLM2-6b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 10.0 --plot_loss  --fp16 >> train_ChatGLM2-6b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path THUDM/chatglm2-6b --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_test --template chatglm2  --finetuning_type lora --checkpoint_dir Reading/ChatGLM2-6b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --num_beams 8 >> predbeam_ChatGLM2-6b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/ChatGLM2-6b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/generated_predictions.jsonl
```

(2) **Evaluate KBQA result with Retrieval**

- WebQSP: 

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/ChatGLM2-6b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_ChatGLM2-6b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=4 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/ChatGLM2-6b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_ChatGLM2-6b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

- CWQ: 

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/ChatGLM2-6b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_ChatGLM2-6b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/ChatGLM2-6b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_ChatGLM2-6b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```