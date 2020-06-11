# Model based on VGG, First Trial

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
__Max epochs: 50__  
<br>
This folder stores training results based on VGG features with different configurations.  
This first trial focus on balanced data input, where ratio is set to 1.

<br>

Training data labelled as 'snoring': 2213  
Validation data labelled as 'snoring': 60  
Test data labelled as 'snoring': 60

<br>

If ratio is 1, we would train 2213 'snoring' with 2213 'nonsnoring'.  
If ratio is 10, we would train 2213 'snoring' with 2213 * 10 'nonsnoring'.  
Same ratio appied to validation and test data.

<br>

__Name follows the pattern__:   
\<ratio of label and unlabel data\> _ \<activation\> _ \<optimizer\> _ \<metrics\> _ \<epochs\>

_**For a closer observation, use find_pic_with_keyword() function in Model_on_VGG.ipynb.**_

<br>

## Observations

|              	| elu                                            	| exponential     	| relu                     	| selu                               	| sigmoid                            	| softmax               	| softplus                        	| softsign                        	| tanh                            	|
|--------------	|------------------------------------------------	|-----------------	|--------------------------	|------------------------------------	|------------------------------------	|-----------------------	|---------------------------------	|---------------------------------	|---------------------------------	|
| __adadelta__ 	| Not reach turning point. Val_acc<60%.          	| Acc always 50%. 	| Stable, but Val_acc<55%. 	| Not reach turn point. Val_acc<60%. 	| Not reach turn point. Val_acc<55%. 	| Acc=50%.              	| Val_acc<65%.                    	| Val_acc<60%.                    	| Val_acc<50%.                    	|
| __adagrad__  	| Converge, overfit. Val_acc<80%.                	| Acc always 50%. 	| Overfit. Val_acc<80%.    	| Overfit. Val_acc<75%.              	| Overfit. Val_acc<75%.              	| Acc=50%.              	| Unstable. Overfit. Val_acc<80%. 	| Overfit. Val_acc<80%.           	| Overfit. Val_acc<75%.           	|
| __adam__     	| Overfit. Val_acc<80%.                          	| Acc always 50%. 	| Overfit. Val_acc<80%.    	| Overfit. Val_acc<80%.              	| Overfit. Val_acc<85%.              	| Overfit. Val_acc<80%. 	| Unstable. Overfit. Val_acc<80%. 	| Overfit. Val_acc<80%.           	| Overfit. Val_acc<80%.           	|
| __adamax__   	| Not reach turning point, overfig. Val_acc<85%. 	| Acc always 50%. 	| Overfit. Val_acc<80%.    	| Overfit. Val_acc<80%.              	| Overfit. Val_acc<85%.              	| Overfit. Val_acc<75%. 	| Unstable. Overfit. Val_acc<80%. 	| Overfit. Val_acc<80%.           	| Overfit. Val_acc<85%.           	|
| __ftrl__     	| Converge, overfit. Val_acc<83%.                	| Acc always 50%. 	| Overfit. Val_acc<80%.    	| Overfit. Val_acc<80%.              	| Stable. Overfit. Val_acc<85%.      	| Acc=50%               	| Unstable. Overfit. Val_acc<80%. 	| Overfit. Val_acc<80%.           	| Overfit. Val_acc<85%.           	|
| __nadam__    	| Overfit. Val_acc<80%.                          	| Acc always 50%. 	| Overfit. Val_acc<85%.    	| Overfit. Val_acc<80%.              	| Overfit. Val_acc<85%.              	| Overfit. Val_acc<80%. 	| Unstable. Overfit. Val_acc<80%. 	| Overfit. Val_acc<80%.           	| Unstable. Overfit. Val_acc<80%. 	|
| __rmsprop__  	| Overfit. Val_acc<80%.                          	| Acc always 50%. 	| Overfit. Val_acc<80%.    	| Overfit. Val_acc<85%.              	| Overfit. Val_acc<80%.              	| Overfit. Val_acc<80%. 	| Unstable. Overfit. Val_acc<80%. 	| Overfit. Val_acc<85%.           	| Overfit. Val_acc<80%.           	|
| __sgd__      	| Horizontal loss and acc. Acc maintain on 50%.  	| Acc always 50%. 	| Unstable. Most acc=50%.  	| Unstable. Most acc=50%.            	| Overfit. Val_acc<85%.              	| Very unstable.        	| Most acc=50%.                   	| Unstable. Overfit. Val_acc<85%. 	| Very unstable.                  	|


## Summary:

1. Activation **exponential** should be dropped, that it is no better than random guess.
2. Activation **softmax** should be considered dropped, for its weird and unstable outcomes.
3. Activation **softplus** gives very unstable results.
4. Activation **elu, relu, selu, sigmoid, softsign, tanh** could be used for further test.
5. Optimizer **adadelta** gives poor accuracy, should be dropped.
6. Optimizer **sgd** most time is no better than random guess. should be dropped.
7. Optimezer **adagrad, adam, adamax, ftrl, nadam, rmsprop** generally works better. Specifically, **ftrl** gives very stable results.
8. Overall, all training appear to be overfitted and unstable. Should consider to increase the size of validation set by reshuffling, and increase the total data for training by increase ratio.
9. **Adam with relu** could be considered as the **baseline**.

## Future work:
1. Increase the size of validation set, comparing to the size of training set. Should mediate the unstable val_acc.
2. Increase the total data input, by inputting more data not labelled as 'snoring', sacrifing the balance. Should mediate overfit.
3. Train on cloudtop.

