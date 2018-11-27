#python script/preprocess.py data/datasets data/datasets --split SQuAD-dev-v1.1 --senSplitRules 2
#mv sentCount_SQuAD-dev-v1.1.txt sentCount_SQuAD-dev-v1.1_rule2.txt 
#python script/preprocess.py data/datasets data/datasets --split SQuAD-dev-v1.1 --senSplitRules 3
#mv sentCount_SQuAD-dev-v1.1.txt sentCount_SQuAD-dev-v1.1_rule3.txt 
python script/preprocess.py data/datasets data/datasets --split SQuAD-dev-v1.1 --senSplitRules 4
mv sentCount_SQuAD-dev-v1.1.txt sentCount_SQuAD-dev-v1.1_rule4.txt
mv sentBoundError_SQuAD-dev-v1.1.txt sentBoundError_SQuAD-dev-v1.1_rule4.txt 
python script/preprocess.py data/datasets data/datasets --split SQuAD-dev-v1.1 --senSplitRules 5
mv sentCount_SQuAD-dev-v1.1.txt sentCount_SQuAD-dev-v1.1_rule5.txt
mv sentBoundError_SQuAD-dev-v1.1.txt sentBoundError_SQuAD-dev-v1.1_rule5.txt 

#python script/preprocess.py data/datasets data/datasets --split SQuAD-train-v1.1 --senSplitRules 5
#mv sentCount_SQuAD-train-v1.1.txt sentCount_SQuAD-train-v1.1_rule5.txt 
