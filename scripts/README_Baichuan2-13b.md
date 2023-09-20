# ChatKBQA with Baichuan2-13b

## Fine-tuning, Retrieval and Evaluation

The following is an example of Baichuan2-13b fine-tuning and retrieval.

Download [Baichuan2-13b](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) model to ```LLMs/models/Baichuan2-13B-Base```.

(1) **Train and test LLM model for Logical Form Generation**

- WebQSP: 

Train LLMs for Logical Form Generation:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path baichuan-inc/Baichuan2-13B-Base --do_train  --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_train --template baichuan2  --finetuning_type lora --lora_target W_pack --output_dir Reading/Baichuan2-13b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 100.0 --plot_loss  --fp16 >> train_Baichuan2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path baichuan-inc/Baichuan2-13B-Base --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_test --template baichuan2  --finetuning_type lora --checkpoint_dir Reading/Baichuan2-13b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --num_beams 10 >> predbeam_Baichuan2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/Baichuan2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/generated_predictions.jsonl
```

- CWQ: 

Train LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=4 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path baichuan-inc/Baichuan2-13B-Base --do_train  --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_train --template baichuan2   --finetuning_type lora --lora_target W_pack --output_dir Reading/Baichuan2-13b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 10.0 --plot_loss  --fp16 >> train_Baichuan2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path baichuan-inc/Baichuan2-13B-Base --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_test --template baichuan2  --finetuning_type lora --checkpoint_dir Reading/Baichuan2-13b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --num_beams 8 >> predbeam_Baichuan2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/Baichuan2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/generated_predictions.jsonl
```

(2) **Evaluate KBQA result with Retrieval**

- WebQSP: 

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/Baichuan2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_Baichuan2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=4 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/Baichuan2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_Baichuan2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

- CWQ: 

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/Baichuan2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_Baichuan2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/Baichuan2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_Baichuan2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```