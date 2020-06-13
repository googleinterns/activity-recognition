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

**The metrics used 'accuracy'.**  

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

### Activations and optimizers:  

Activations: **elu, relu, selu, sigmoid, softsign, tanh**  
Optimiaers: **adagrad, adam, adamax, ftrl, nadam, rmsprop**  

1. **Elu with adam / adamax / nadam / rmsprop** combinations give fluctuate, nonincreasing results, seems to still doing random guessing for many times. **Elu** turning point appears before **30 epochs**.
2. **Relu with adam / adamax / nadam / rmsprop** combinations give fluctuate, nonincreasing results, seems to still doing random guessing for many times. **Relu** turning point appears before **30 epochs**.
3. **Selu with adamax / nadam / rmsprop** combinations give fluctuate, nonincreasing results, seems to still doing random guessing for many times. **Relu** turning point appears before **30 epochs**.
4. **Sigmoid with adam / nadam** combinations give fluctuate, nonincreasing results. Recommend for further testing. **Sigmoid** turning point appears around **10 epochs**. But when combined with **adagrad**, it kept going beyond 100 epochs.
5. **Softsign witn adam / nadam** combinations give fluctuate, nonincreasing results. Recommend for further testing. **Softsign** turning point appears around **10 epochs**. But when combined with **adagrad**, it kept going beyond 100 epochs.
6. **Tanh witn adam / nadam** combinations give fluctuate, nonincreasing results. Recommend for further testing. **Tanh** turning point appears around **30 epochs**. But when combined with **adagrad**, it kept going beyond 100 epochs.
7. **Adagrad** gives relatively low acc.
8. **Adamax** has a big overfit problem even with increased ratio.
8. **Rmsprop with elu, relu, selu** seems giving more overfitting results, even after increasing the ratio of label:unlabel. Mostly over 10% acc on ratio 1, and over 5% acc on ratio 10.


## Summary:
1. Increasing ratio of labeled:unlabeled indeed seems to increase accuracy and mediate overfitting. But need to be suspicious and cautious about imbalanced data problem.
2. Activations **sigmoid, softsign, and tanh** are more stable, and seems to be better activation choices.
3. Optimizers **adagrad, ftrl** would be better choices. Especially **ftrl** works fine with almost all activations.
4. Fluctuation of val_acc still exists.

## Further work:
1. Try to do normalization on feature sets.
2. Focus on better activation and optimizers.
3. While trying to increase size of data by increasing ratio of label:unlabel, should keep an eye on imbalance data problem. Could probably consider using other metrics, such as precision, recall, or F1 score.
4. Consider increasing size of labeled data by synthesizing or oversampling.
5. Modify structure of neural network, for example, add one more layer.
6. Epochs could be at range 10 - 40.




