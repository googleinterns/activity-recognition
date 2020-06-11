This folder stores training results based on VGG features.

Training data labelled as 'snoring': 2213
Validation data labelled as 'snoring': 60
Test data labelled as 'snoring': 60

If ratio is 1, we would train 2213 'snoring' with 2213 'nonsnoring'.
If ratio is 10, we would train 2213 'snoring' with 2213 * 10 'nonsnoring'.
Same ratio appied to validation and test data.

Name follows the pattern: 
\<ratio of label and unlabel data\> _ \<activation\> _ \<optimizer\> _ \<metrics\> _ \<epochs\>

