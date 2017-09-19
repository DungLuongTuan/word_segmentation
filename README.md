# Vietnamese word segentation

## Copus
VLSP_SP73
training set: 53204 sentences
testing set : 14993 sentences

## Models
### B-LSTM + relu + softmax
> python word_segmentation_all_word2vec.py

### B-LSTM + CRF
> python word_segmentation_all_word2vec_crf.py
models save into ./models/crf

## Evaluate
Evaluate on training data
> python evaluate_train.py path_to_model epoch
model save into models/word2vec

Evaluate on test data
> python evaluate_test.py path_to_model epoch

eg: evaluate model0.ckpt of CRF version
> python evaluate_test.py ./models/crf 0