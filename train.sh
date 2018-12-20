#python script/train.py
#python script/train.py --hidden-size 25
#python script/train.py --batch-size 10 --test-batch-size 10 --sentence-sewon True --max-context-len 2000
#python script/train.py --gpu 1 --batch-size 10 --test-batch-size 10 --max-context-len 2000 &
#python script/train.py --gpu 0 --sentence-attention False --batch-size 10 --test-batch-size 10 --max-context-len 2000 &
#python script/train.py --gpu 0 --batch-size 10 --test-batch-size 10 --max-context-len 2000 --answer-hop 1 --align-hop 3 &
python script/train.py --gpu 1 --pretrained data/model_backup/20181213-b8d8d89e.mdl --sentence-only True &



#python script/train.py --batch-size 15 --sentence-sewon False --max-context-len 2000

#python script/train.py --sentence-attention False 
