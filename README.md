# ChatKBQA

##  General Setup 

### Environment Setup
```
conda create -n chatkbqa python=3.8
conda activate chatkbqa
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirement.txt
```

###  Freebase KG Setup

Below steps are according to [Freebase Virtuoso Setup](https://github.com/dki-lab/Freebase-Setup). 
#### How to install virtuoso backend for Freebase KG.

1. Clone from `dki-lab/Freebase-Setup`:
```
cd Freebase-Setup
```

2. Processed [Freebase](https://developers.google.com/freebase) Virtuoso DB file can be downloaded from [here](https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip) or via wget (WARNING: 53G+ disk space is needed):
```
tar -zxvf virtuoso_db.zip
```

3. Managing the Virtuoso service:

To start service:
```
python3 virtuoso.py start 3001 -d virtuoso_db
```

and to stop a currently running service at the same port:
```
python3 virtuoso.py stop 3001
```

A server with at least 100 GB RAM is recommended.

#### Download FACC1 mentions for Entity Retrieval.

- Download the mention information (including processed FACC1 mentions and all entity alias in Freebase) from [FACC1](https://1drv.ms/u/s!AuJiG47gLqTznjl7VbnOESK6qPW2?e=HDy2Ye) to `data/common_data/facc1/`.

```
ChatKBQA/
└── data/
    ├── common_data/                  
        ├── facc1/   
            ├── entity_list_file_freebase_complete_all_mention
            └── surface_map_file_freebase_complete_all_mention                                           
```

## Dataset

Experiments are conducted on 2 KBQA benchmarks WebQSP, CWQ.

### WebQSP

Download the WebQSP dataset from [here](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) and put them under `data/WebQSP/origin`. The dataset files should be named as `WebQSP.test[train].json`.

```
ChatKBQA/
└── data/
    ├── WebQSP                  
        ├── origin                    
            ├── WebQSP.train.json                    
            └── WebQSP.test.json                                       
```

### CWQ

Download the CWQ dataset [here](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) and put them under `data/CWQ/origin`. The dataset files should be named as `ComplexWebQuestions_test[train,dev].json`.

```
ChatKBQA/
└── data/
    ├── CWQ                 
        ├── origin                    
            ├── ComplexWebQuestions_train.json                   
            ├── ComplexWebQuestions_dev.json      
            └── ComplexWebQuestions_test.json                              
```


## Data Processing

(1) **Parse SPARQL queries to S-expressions** 

- WebQSP: 

Run `python parse_sparql_webqsp.py` and the augmented dataset files are saved as `data/WebQSP/sexpr/WebQSP.test[train].json`. 

- CWQ: 

Run `python parse_sparql_cwq.py` and the augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train].json`.
 

(2) **Prepare data for training and evaluation**

- WebQSP: 

Run `python data_process.py --action merge_all --dataset WebQSP --split test[train]`. The merged data file will be saved as `data/WebQSP/generation/merged/WebQSP_test[train].json`.

Run `python data_process.py --action get_type_label_map --dataset WebQSP --split train`. The merged data file will be saved as `data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json`.

- CWQ: 

Run `python data_process.py --action merge_all --dataset CWQ --split test[train]` The merged data file will be saved as `data/CWQ/generation/merged/CWQ_test[train].json`.

Run `python data_process.py --action get_type_label_map --dataset CWQ --split train`. The merged data file will be saved as `data/CWQ/generation/label_maps/CWQ_train_type_label_map.json`.


(3) **Prepare data for LLM model**

- WebQSP: 

Run `python process_NQ.py --dataset_type WebQSP`. The merged data file will be saved as `LLMs/data/WebQSP_Freebase_NQ_test[train]/examples.json`.

- CWQ: 

Run `python process_NQ.py --dataset_type CWQ` The merged data file will be saved as `LLMs/data/CWQ_Freebase_NQ_test[train]/examples.json`.


## Fine-tuning, Retrieval and Evaluation

The following is an example of [LLaMa2-13b](README.md) fine-tuning and retrieval, we also provide a variety of LLMs (including [LLaMa2-7b](scripts/README_LLaMa2-7b.md), [ChatGLM2-6b](scripts/README_ChatGLM2-6b.md)) fine-tuning instructions.

(1) **Train and test LLM model for Logical Form Generation**

- WebQSP: 

Train LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=4 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-13b-hf --do_train  --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_train --template default  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 100.0 --plot_loss  --fp16 >> train_LLaMA2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_test --template default --finetuning_type lora --checkpoint_dir Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --num_beams 10 >> predbeam_LLaMA2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/generated_predictions.jsonl
```

- CWQ: 

Train LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-13b-hf --do_train  --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_train --template default  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 10.0 --plot_loss  --fp16 >> train_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_test --template default --finetuning_type lora --checkpoint_dir Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --num_beams 8 >> predbeam_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/generated_predictions.jsonl
```

(2) **Evaluate KBQA result with Retrieval**

- WebQSP: 

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_LLaMA2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_LLaMA2-13b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

- CWQ: 

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=4 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```