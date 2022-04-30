# dlh_jkchang2_final
final project for dlh

The original paper "Automatic assignment of ICD codes based on clinical notes" can be found here: https://www.sciencedirect.com/science/article/pii/S1532046419300322#b0110

Following steps:

1. Data processing
Initial data processing step using MIMIC notes file (not included here due to access rules).
```
SELECT
  SUBJECT_ID,
  HADM_ID,
  REGEXP_REPLACE(REPLACE(SUBSTR(TEXT, STRPOS(TEXT, 'Discharge Diagnosis')),"\n", " "), "\[.*?\]", "") AS diagnosis_note
FROM
  `prefab-mile-346920.samples.parsed_notes` a
WHERE
  category = 'Discharge summary'
  AND CONTAINS_SUBSTR(TEXT,
    'Discharge Diagnosis')
```
This cleans up the notes to just target the actual diagnosis per visit, making it simpler to look for keywords.
With the cleaned notes, use word2vec to capture associations of keywords.
The keywords are then used to train/test a bi-directional LSTM model to predict the target ICD codes.

![Screen Shot 2022-04-29 at 7 44 54 AM](https://user-images.githubusercontent.com/87827828/165938562-de0f26cb-78dc-47e3-9ecb-9c566741613b.png)

```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 2980)]       0           []                               
                                                                                                  
 embedding (Embedding)          (None, 2980, 100)    3357300     ['input_1[0][0]']                
                                                                                                  
 permute (Permute)              (None, 100, 2980)    0           ['embedding[0][0]']              
                                                                                                  
 dense (Dense)                  (None, 100, 2980)    8883380     ['permute[0][0]']                
                                                                                                  
 attention (Permute)            (None, 2980, 100)    0           ['dense[0][0]']                  
                                                                                                  
 multiply (Multiply)            (None, 2980, 100)    0           ['embedding[0][0]',              
                                                                  'attention[0][0]']              
                                                                                                  
 bidirectional (Bidirectional)  (None, 2980, 10)     4240        ['multiply[0][0]']               
                                                                                                  
 bidirectional_1 (Bidirectional  (None, 10)          640         ['bidirectional[0][0]']          
 )                                                                                                
                                                                                                  
 dense_1 (Dense)                (None, 64)           704         ['bidirectional_1[0][0]']        
                                                                                                  
 dense_2 (Dense)                (None, 5)            325         ['dense_1[0][0]']                
                                                                                                  
==================================================================================================
Total params: 12,246,589
Trainable params: 8,889,289
Non-trainable params: 3,357,300
```

```
Epoch 1/10
9/9 - 307s - loss: 1.4799 - accuracy: 0.6270 - val_loss: 1.1442 - val_accuracy: 0.7125 - 307s/epoch - 34s/step
Epoch 2/10
9/9 - 279s - loss: 1.0599 - accuracy: 0.6956 - val_loss: 0.9866 - val_accuracy: 0.7125 - 279s/epoch - 31s/step
Epoch 3/10
9/9 - 276s - loss: 0.9938 - accuracy: 0.6956 - val_loss: 0.9654 - val_accuracy: 0.7125 - 276s/epoch - 31s/step
Epoch 4/10
9/9 - 275s - loss: 0.9809 - accuracy: 0.6956 - val_loss: 0.9341 - val_accuracy: 0.7125 - 275s/epoch - 31s/step
Epoch 5/10
9/9 - 274s - loss: 0.9671 - accuracy: 0.6956 - val_loss: 0.9179 - val_accuracy: 0.7125 - 274s/epoch - 30s/step
Epoch 6/10
9/9 - 272s - loss: 0.9310 - accuracy: 0.6956 - val_loss: 0.8589 - val_accuracy: 0.7125 - 272s/epoch - 30s/step
Epoch 7/10
9/9 - 276s - loss: 0.8069 - accuracy: 0.6970 - val_loss: 0.8002 - val_accuracy: 0.6890 - 276s/epoch - 31s/step
Epoch 8/10
9/9 - 272s - loss: 0.7324 - accuracy: 0.7311 - val_loss: 0.7853 - val_accuracy: 0.7081 - 272s/epoch - 30s/step
Epoch 9/10
9/9 - 273s - loss: 0.7142 - accuracy: 0.7349 - val_loss: 0.7396 - val_accuracy: 0.7181 - 273s/epoch - 30s/step
Epoch 10/10
9/9 - 272s - loss: 0.6902 - accuracy: 0.7421 - val_loss: 0.7331 - val_accuracy: 0.7192 - 272s/epoch - 30s/step
peak memory: 6131.57 MiB, increment: 5288.42 MiB
```
![Screen Shot 2022-04-29 at 12 03 32 PM](https://user-images.githubusercontent.com/87827828/165981870-a2d8e1d4-d64f-451e-bc62-99f8b754c3cd.png)
---
Ablation

1. Not using attention (removed layer)
```
Epoch 1/10
9/9 - 94s - loss: 1.4787 - accuracy: 0.5714 - val_loss: 1.1250 - val_accuracy: 0.7069 - 94s/epoch - 10s/step
Epoch 2/10
9/9 - 78s - loss: 1.0782 - accuracy: 0.7037 - val_loss: 1.0075 - val_accuracy: 0.7069 - 78s/epoch - 9s/step
Epoch 3/10
9/9 - 78s - loss: 0.9777 - accuracy: 0.7037 - val_loss: 0.9652 - val_accuracy: 0.7069 - 78s/epoch - 9s/step
Epoch 4/10
9/9 - 78s - loss: 0.9566 - accuracy: 0.7037 - val_loss: 0.9435 - val_accuracy: 0.7069 - 78s/epoch - 9s/step
Epoch 5/10
9/9 - 78s - loss: 0.9426 - accuracy: 0.7037 - val_loss: 0.9409 - val_accuracy: 0.7069 - 78s/epoch - 9s/step
Epoch 6/10
9/9 - 78s - loss: 0.9306 - accuracy: 0.7037 - val_loss: 0.9254 - val_accuracy: 0.7069 - 78s/epoch - 9s/step
Epoch 7/10
9/9 - 78s - loss: 0.8942 - accuracy: 0.7037 - val_loss: 0.8489 - val_accuracy: 0.7069 - 78s/epoch - 9s/step
Epoch 8/10
9/9 - 77s - loss: 0.8293 - accuracy: 0.7037 - val_loss: 0.7760 - val_accuracy: 0.7069 - 77s/epoch - 9s/step
Epoch 9/10
9/9 - 77s - loss: 0.7820 - accuracy: 0.7037 - val_loss: 0.7259 - val_accuracy: 0.7069 - 77s/epoch - 9s/step
Epoch 10/10
9/9 - 78s - loss: 0.7634 - accuracy: 0.7037 - val_loss: 0.7729 - val_accuracy: 0.7069 - 78s/epoch - 9s/step
```
![Screen Shot 2022-04-29 at 1 25 48 PM](https://user-images.githubusercontent.com/87827828/165993462-682b9dfa-3c2b-4b60-9ca7-cd2208e6dd5f.png)

Student results:
| Model       | Loss        | Percent Improved |
| ------------| ------------|------------------|
| BiDirectional LSTM without attention      | 0.7729       | 0
| BiDirectional LSTM with attention   | 0.7331        | 5.1

Paper results:
| Model       | Loss        | Percent Improved |
| ------------| ------------|------------------|
| BiDirectional LSTM without attention      | 0.00272      | 0
| BiDirectional LSTM with attention   | 0.00204        | 25

2. CNN instead of RNN

```
Model: "sequential_11"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_10 (Embedding)    (None, 2980, 100)         3384600   
                                                                 
 conv1d_4 (Conv1D)           (None, 2976, 128)         64128     
                                                                 
 global_max_pooling1d_4 (Glo  (None, 128)              0         
 balMaxPooling1D)                                                
                                                                 
 dense_8 (Dense)             (None, 5)                 645       
                                                                 
=================================================================
Total params: 3,449,373
Trainable params: 64,773
Non-trainable params: 3,384,600
_________________________________________________________________
```

---
Citation

Ying Yu, Min Li, Liangliang Liu, Zhihui Fei, Fang-Xiang Wu, Jianxin Wang,
Automatic ICD code assignment of clinical notes based on multilayer attention BiRNN,
Journal of Biomedical Informatics,
Volume 91,
2019,
103114,
ISSN 1532-0464,
https://doi.org/10.1016/j.jbi.2019.103114.
(https://www.sciencedirect.com/science/article/pii/S1532046419300322)
