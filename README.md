# Telos: Deep Learning for Long Distance Dependencies

Telos is a program that trains a deep learning model to learn long
distance dependencies in a synthetic data set.

The data consists of two equal-length sequences.
The feature sequence is a random ordering of ten digits from 0
to 9, in which one of the digits is repeated in two random places and
the rest of the digits are unique.
The label sequence is a list of ten 0s and 1s, where 1s appear in
positions corresponding to the repeated digits and 0s appear in
positions corresponding to the unique ones.

     8 9 5 2 4 7 3 0 3 1
     0 0 0 0 0 0 1 0 1 0

     6 6 5 4 8 3 9 2 1 7
     1 1 0 0 0 0 0 0 0 0

     8 1 7 2 0 5 6 4 8 3
     1 0 0 0 0 0 0 0 1 0

Telos trains a recursive neural network to predict the 0 and 1 labels
given a sequence of digits.
This requires a model to learn a simple long-distance dependency.

See `telos --help` for details on training and evaluating models.
