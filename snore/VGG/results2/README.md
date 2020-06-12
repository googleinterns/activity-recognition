# Model based on VGG, Second Trial

```python3
model = keras.Sequential([
      keras.Input(shape=(128,)),
      keras.layers.Dense(128, activation=activation),
      keras.layers.Dense(1)
  ])
  
model.compile(optimizer=optimizer,
         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
         metrics=metrics)
         
history = model.fit(train_x, train_y,
               epochs=epochs,
               validation_data=(val_x, val_y),
                verbose=1)
```

Specital thanks to Youngmin Cho for advices.

## Description:  
<br>
This folder stores SECOND training results based on VGG features with different configurations.  
This second trial increased ratio of labeled and unlabeled data trying to mediate overfit. By sacrificing in data balance, we introduce more data into this model.  


If ratio is 1, we would train 2213 'snoring' with 2213 'nonsnoring'.  
If ratio is 10, we would train 2213 'snoring' with 2213 * 10 'nonsnoring'.  
Same ratio appied to validation and test data.

<br> 

Train : val : test = 8 : 1 : 1.

<br>

__Name follows the pattern__:   
\<ratio of label and unlabel data\> _ \<activation\> _ \<optimizer\> _ \<metrics\> _ \<epochs\>

_**For a closer observation, use find_pic_with_keyword() function in Model_on_VGG.ipynb.**_

<br>

## Observations