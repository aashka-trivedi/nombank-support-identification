# Nombank Support Identification

**Collaborators: Aashka Trivedi, Raksha Hegde, Sarvani Nadiminty**

## Data

- `all_nombank.clean.dev`, `all_nombank.clean.test`, `all_nombank.clean.training` are the data files with sentences containing all sentences from Nombank.
- `support-dev`, `support-test`, `support-training` are data files that contain only sentences with a support for the argument/predicate.
- `train-features`, `test-features`, `dev-features` are the feature files for the case where the model has information about only the predicates
- `arg-train-features`, `arg-test-features`, `arg-dev-features` are the feature files for the case where the model has information about the predicates and all the arguments

## Code

- `get_features.py`is used to get features from the data files. Use flag `--test_features` to produce test features (without training labels), and the flag `--arguments_known` if the training should be done with complete information of predicates and all arguments.

## Implementation Details

### Creating Features

The following features are obtained:

1. Word related Features: word, POS, BIO, position (in the sentence), stem, has_capital, is_noun, is_pred (whether word is the )
2. Previous Word (upto 2 words) Related Features: prev_word, prev_POS, prev_BIO, prev_2_word, prev_2_POS, prev_2_BIO
3. Next word (upto 2 word) Related Features: next_word, next_POS, next_BIO, next_2_word, next_2_POS, next_2_BIO
4. Distance from predicated: pred_forward_distance (forward distance from predicate), pred_backward_distance (backward distance from predicate)
5. If all the arguments is known then we also include argument related features: Argument related features (argx can be arg0,arg1,arg2,arg3,arg4): is_argx, argx_forward_distance (forward distance from argx), argx_backward_distance (backward distance from argx)

Labels: for training data, the labels are "SUPPORT" or None
