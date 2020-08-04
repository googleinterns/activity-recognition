# Model based on VGG  

**Input data**: 128-dimensional audio features extracted at 1Hz. 
The audio features were extracted using a VGG-inspired acoustic model described in Hershey et. al., 
trained on a preliminary version of YouTube-8M. 
The features were PCA-ed and quantized to be compatible with the audio features provided with YouTube-8M. 
They are stored as TensorFlow Record files.

Please check Model_on_VGG.ipynb for detialed training information.  
Please check output results/ README.md for observation and summaries.
