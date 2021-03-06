# Nombank Support Identification

**Collaborators: Aashka Trivedi, Raksha Hegde, Sarvani Nadiminty**

## Data

- `all_nombank.clean.dev`, `all_nombank.clean.test`, `all_nombank.clean.training` are the data files with sentences containing all sentences from Nombank.
- `support-dev`, `support-test`, `support-training` are data files that contain only sentences with a support for the argument/predicate.
- `train-features`, `test-features`, `dev-features` are the feature files for the case where the model has information about only the predicates
- `arg-train-features`, `arg-test-features`, `arg-dev-features` are the feature files for the case where the model has information about the predicates and all the arguments

Notes:

1. Please note- any data with the prefix "modelN" contains the features as described by the models below
2. Any data with the suffix "arg" has knowledge of all the arguments along with the predicate, and if the suffix is missing, it has only the knowledge of the predicates only

## Code

- `get_features.py`is used to get features from the data files. Run as `python3 get_features.py --inputfile $INPUT_FILE [--distance_features][--test_features] [--arguments_known] [--transparent_noun --transparent_noun_path $TRANSPARENT_NOUN_LIST] [--support_verb --support_verb_path $SUPPORT_VERB_LIST] [--prev_tag]`

Flags:

1. `--test_features`: to produce test features (without training labels)
2. `--arguments_known` if the training should be done with complete information of predicates and all arguments
3. `--distance_features`: to produce features with distance-related features (Model 1)
4. `--transparent_noun`: to produce features with tranparent-noun-related features (Model 2)
5. `--transparent_noun_path`: the list of transparent nouns (needed if --transparent_noun is used) (Model 2)
6. `--support_verb`: to produce features with support-verb-related features (Model 3)
7. `--support_verb_path`: the list of support verbs(needed if --support_verb is used) (Model 3)
8. `--prev_tag`: whether to use the previous tag as a feature (Model 4)

## Implementation Details

### Creating Features

The following features are obtained:

1. Word related Features: word, POS, BIO, position (in the sentence), stem, has_capital, is_noun, is_pred (whether word is the )
2. Previous Word (upto 2 words) Related Features: prev_word, prev_POS, prev_BIO, prev_2_word, prev_2_POS, prev_2_BIO
3. Next word (upto 2 word) Related Features: next_word, next_POS, next_BIO, next_2_word, next_2_POS, next_2_BIO
4. Distance from predicated: pred_forward_distance (forward distance from predicate), pred_backward_distance (backward distance from predicate)
5. If all the arguments is known then we also include argument related features: Argument related features (argx can be arg0,arg1,arg2,arg3,arg4): is_argx, argx_forward_distance (forward distance from argx), argx_backward_distance (backward distance from argx)
6. Transparent noun related features: is_transparent_noun, 3_before_transparent (if word occurs within a window of three words before a transparent noun), 2_before_transparent, 1_before_transparent, 1_after_transparent, 2_after_transparent, 3_after_transparent (if word occurs within a window of three words after a transparent noun)
7. Previous Tag

Labels: for training data, the labels are "SUPPORT" or None

### Models

We train a total of 6 models: with 3 feature sets and 2 information levels.

1. Model 0: The baseline model, containing only word related features, and no distance related features. The model only has knowledge of the predicate
2. Model 0 arg: Model trained with only word-related features, which has knowledge of the predicate and all arguments
3. Model 1: Trained with word-related features, and distance related features. The model only has knowledge of the predicate
4. Model 1 arg: Model trained with word-related and distance-related features, which has knowledge of the predicate and all arguments
5. Model 2: Trained with word-related features, distance related features, and transparent-noun related features. The model only has knowledge of the predicate
6. Model 2 arg: Model trained with word-related and distance-related features, and transparent-noun related features which has knowledge of the predicate and all arguments
7. Model 3: Trained with word-related features, distance related features, transparent-noun related features, and support verb related features. The model only has knowledge of the predicate
8. Model 3 arg: Model trained with word-related and distance-related features, transparent-noun related features and support verb related features which has knowledge of the predicate and all arguments
9. Model 4: Trained with word-related features, distance related features, transparent-noun related features, support verb related features, and previous word tag. The model only has knowledge of the predicate
10. Model 4 arg: Trained with word-related features, distance related features, transparent-noun related features, support verb related features, and previous word tag. The model has knowledge of the predicate and all arguments
11. Model 5: Trained with word-related features, distance related features, and previous word tag. The model only has knowledge of the predicate
12. Model 5 arg: Trained with word-related features, distance related features, and previous word tag. The model has knowledge of the predicate and all arguments
