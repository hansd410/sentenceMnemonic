#v2
#python script/train.py --test True --pretrained data/model_backup/20181127-13621bf4.mdl
#v3
#python script/train.py --test True --pretrained data/model_backup/20181203-7f98de34.mdl
python script/train.py --test True --pretrained data/model_backup/20181218-ca1305c6.mdl --batch-size 10 --test-batch-size 10 --sentence_sewon True --max_context_len 2000
#no sentence attention
#python script/train.py --pretrained data/model_backup/20181213-b8d8d89e.mdl --sentence_attention False --test True
#python script/train.py --pretrained data/model_backup/20181213-b8d8d89e.mdl --sentence_attention False --test True

#sentence selection
#python script/train.py --pretrained data/model_backup/20181204-e91b56ca.mdl --test True
