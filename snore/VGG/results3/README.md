# Model based on VGG, Third Trial

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
<br>
This folder stores THIRD training results based on VGG features with different configurations.  
This third trial, before feed embeddings into model for training, performs normalization to reform the original feature ranges from [0, 255] to [0, 1].

**The metrics used 'accuracy'.**  

If ratio is 1, we would train 2213 'snoring' with 2213 'nonsnoring'.  
If ratio is 10, we would train 2213 'snoring' with 2213 * 10 'nonsnoring'.  
Same ratio appied to validation and test data.  
**We use BALANCED data for training in this trail.**  

<br> 

Train : val : test = 8 : 1 : 1.

<br>

__Name follows the pattern__:   
\<trial num\> _ \<ratio of label and unlabel data\> _ \<activation\> _ \<optimizer\> _ \<metrics\> _ \<epochs\>

_**For a closer observation, use find_pic_with_keyword() function in Model_on_VGG.ipynb.**_

<br>

## Observations

### Activations and optimizers:  

Activations: **elu, exponential, relu, selu, sigmoid, softmax, softplus, softsign, tanh**  
Optimiaers: **adadelta, adagrad, adam, adamax, ftrl, nadam, rmsprop, sgd**  

_Considering the possible differences in training before/after normalization, we use all options as we did in trial1, and gradually remove unwanted options from this stage._    

|              	| elu                       	| exponential                	| relu                       	| selu                       	| sigmoid                   	| softmax             	| softplus            	| softsign            	| tanh             	|
|--------------	|---------------------------	|----------------------------	|----------------------------	|----------------------------	|---------------------------	|---------------------	|---------------------	|---------------------	|------------------	|
| __adadelta__ 	| Half ok, need more epochs 	| Half ok, need more epochs  	| Half ok, need more epochs  	| Half ok, need more epochs  	| Not ok                    	| Not ok, val_acc=50% 	| Not ok              	| Not ok              	| Not ok           	|
| __adagrad__  	| Ok                        	| Ok                         	| Ok                         	| Ok                         	| Half ok, need more epochs 	| Not ok, val_acc=50% 	| Ok                  	| Ok                  	| Half ok, overfit 	|
| __adam__     	| Half ok, unstable         	| Half ok, unstable, overfit 	| Half ok, unstable, overfit 	| Half ok, unstable          	| Ok                        	| Ok                  	| Half ok, unstable   	| Half ok, unstable   	| Not ok           	|
| __adamax__   	| Ok                        	| Ok                         	| Ok                         	| Ok, fluctuate a bit        	| Ok                        	| Ok                  	| Ok                  	| Ok                  	| Half ok, overfit 	|
| __ftrl__     	| Not ok, val_acc=50%       	| Weird output, check it out 	| Not ok, val_acc=50%        	| Weird output, check it out 	| Not ok, val_acc=50%       	| Not ok, val_acc=50% 	| Not ok, val_acc=50% 	| Not ok, val_acc=50% 	| Half ok, overfit 	|
| __nadam__    	| Half ok, unstable         	| Half ok, unstable, overfit 	| Half ok, overfit           	| Half ok, unstable          	| Ok                        	| Ok                  	| Ok                  	| Half ok, unstable   	| Half ok, overfit 	|
| __rmsprop__  	| Half ok, unstable         	| Half ok, unstable          	| Half ok, unstable, overfit 	| Half ok, unstable!         	| Ok                        	| Ok                  	| Half ok, unstable!  	| Half ok, unstable!  	| Half ok, overfit 	|
| __sgd__      	| Ok                        	| Ok                         	| Ok                         	| Ok                         	| Ok, better if more epochs 	| Not ok, val_acc=50% 	| Ok                  	| Ok                  	| Half ok, overfit 	|


## Summary:
1. Activations **elu, exponential, relu, selu, sigmoid, softplus, softsign** seems better.
2. Optimizers **adagrad, adamax, sgd** would be better choices.




