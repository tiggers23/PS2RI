# Sentiment-oriented Sarcasm Integration for Video Sentiment Analysis Enhancement with Sarcasm Assistance
## Training

First, train the sarcasm feature encoder by running the below code:

`python our_main_sar.py --dataset="sarcasm" --max_seq_length=85 --batch_size=64 --learning_rate=1e-5 --device=cuda:0 --save_path=save_file_path --file_path=file_path`

Then, you can run the below code to train PS2RI (the "weight" parameter is the path of ".pth" file saved in the first step):

`python our_main.py --dataset="sarcasm" --max_seq_length=85 --batch_size=64 --learning_rate=1e-5 --device=cuda:0 --save_path=save_file_path --file_path=file_path --weight=sar_weight`



