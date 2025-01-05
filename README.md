# Sentiment-oriented Sarcasm Integration for Video Sentiment Analysis Enhancement with Sarcasm Assistance
This repo contains the code for the PS2RI research project, which focuses on improving video sentiment analysis by incorparting the sarcasm assistance. This code is based on Pytorch.

## How to run!
The training strategy of our proposed PS2RI involves two step. In the first step, we initially train the sarcasm feature encoder with the supervision of sarcasm labels. With the well-trained sarcasm feature encoder, we train the sentiment feature encoder and the sarcasm-aware sentiment learning module with the supervision of sentiment labels.

### 
First, train the sarcasm feature encoder by running the below code:

`python our_main_sar.py --dataset="sarcasm" --max_seq_length=85 --batch_size=64 --learning_rate=1e-5 --device=cuda:0 --save_path=save_file_path --file_path=file_path`

Then, you can run the below code to train PS2RI (the "weight" parameter is the path of ".pth" file saved in the first step):

`python our_main.py --dataset="sarcasm" --max_seq_length=85 --batch_size=64 --learning_rate=1e-5 --device=cuda:0 --save_path=save_file_path --file_path=file_path --weight=sar_weight`

## Citation
If you find this project useful in your research, please consider citing:

```

@inproceedings{DBLP:conf/mm/FangWLL24,
  author       = {Junlin Fang and
                  Wenya Wang and
                  Guosheng Lin and
                  Fengmao Lv},
  title        = {Sentiment-oriented Sarcasm Integration for Video Sentiment Analysis
                  Enhancement with Sarcasm Assistance},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Multimedia,
                  {MM} 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November
                  2024},
  pages        = {5810--5819},
  publisher    = {{ACM}},
  year         = {2024},
}
```
