# Open-ARC
Open-ARC(Abstraction and Reasoning Corpus) Public Open  Model : AI systems to efficiently learn new skills and solve open-ended problems, rather than depend exclusively on systems trained with extensive datasets

**Ongoing progress – next update coming soon.

# Dim Prediction 
Now use can predict the dimantion ([row], [colums]) in your test set is simple using MLPRegration using sklearn libraries and using the following code
you can predict the output dim accurantly 

## use defualt 
```python # !python test/dim_prediction.py  ```

## Use Custom
```python # !python test/dim_prediction.py  --path demo.json ```

In `demo.json` contain your test data file :

## Test Data Must like 
```json {"train":[{"input":[[0,3... ],[0,3...]],"output":1},{"input":[[0,3...],[0,3...]],"output":1}, "test":[{"input":[[00,0,3,0,3,0,0.... ,2,0,0,3,0,0,0,0,0,0]]]}]} ```

# Datset preparetion 
Using Your local File To prepare Just Data:
```python !python test/process_task.py --train_path /kaggle/input/arc2025-dataset/train.json --solution_path /kaggle/input/arc2025-dataset/solution.json ```

This simple testing for your dataset is valid and can be proceded or not ! 
If you see this like output you good to goo!
```python 
Processing tasks: 100%|█████████████████| 7757/7757 [00:00<00:00, 696879.56it/s]
train shape: 6981 val Shape: 776
{'id': 'port_rev_94_65_87_20_31', 'train': [{'input': [[9], [4]], 'output': [[4], [9]]}, {'input': [[6], [5]], 'output': [[5], [6]]}, {'input': [[8], [7]], 'output': [[7], [8]]}, {'input': [[2], [0]], 'output': [[0], [2]]}], 'test': [{'input': [[3], [1]]}]} ```

