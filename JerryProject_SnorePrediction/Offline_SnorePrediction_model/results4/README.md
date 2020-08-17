# Model based on VGG, Fourth Trial

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

## Description:  

__A bug is fixed, and now we have 10 times more data to train!!!__  

<br>
This folder stores FOURTH training results based on VGG features with different configurations.  
This fourth trial, before feed embeddings into model for training, performs normalization to reform the original feature ranges from [0, 255] to [0, 1].  


**The metrics used 'accuracy'.**  

If ratio is 1, we would train 2213 'snoring' with 2213 'nonsnoring'.  
If ratio is 10, we would train 2213 'snoring' with 2213 * 10 'nonsnoring'.  
Same ratio appied to validation and test data.  
**We use BALANCED data for training in this trail.**  

<br> 

__Train : val : test = 7 : 2 : 1__  
__Train: 35850, val: 10228, test: 5105__  

<br>

__Name follows the pattern__:   
\<trial num\> _ \<ratio of label and unlabel data\> _ \<activation\> _ \<optimizer\> _ \<metrics\> _ \<epochs\>

_**For a closer observation, use find_pic_with_keyword() function in Model_on_VGG.ipynb.**_

<br>

## Observations

### Activations and optimizers:  

Activations: **elu, exponential, relu, selu, sigmoid, softplus, softsign**  
Optimiaers: **adagrad, adam, adamax, sgd**  


## Summary:
1. **Relu, adam** does not give good results.
2. In trial3, val_acc reached ceilling 90%, the same applied to trial4, the highest accuracy is < 90%. Perhaps add more data won't do any good. That is the limitation of VGG feature.




