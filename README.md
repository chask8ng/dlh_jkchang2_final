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
