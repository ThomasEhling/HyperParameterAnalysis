# HyperParameterAnalysis
Hyper parameters impact on a classifer and a regression model


1) Simple classifer on the CIFAR10 dataset.

Experiment the impact on the loss and accuracy by epoch for :

- loss functions : “hinge”, “squared_hinge”, “kullback_leibler_divergence”, “categorical_crossentropy”
- optimizers : SGD, RMSProp, AdaGrad, Adam
- Regularization :
  - L2 : 0.1, 0.01, 0.001, 0.0001
  - Dropout : 0.2, 0.3, 0.4, 0.5
  - Batch Normalization
  
  
2) Simple Regression model on the UCI Crime dataset.

Experiment the impact on the loss and accuracy by epoch for :

- loss functions : “L1”, “L2”, “log-cosh”, “hubert”
- optimizers : SGD, RMSProp, AdaGrad, Adam
- Regularization :
  - L2 : 0.1, 0.01, 0.001, 0.0001
  - Dropout : 0.2, 0.3, 0.4, 0.5
  - Batch Normalization

Please see the "src/hyper_parameter_codes.pdf" or ".html" for the implementation and the results.
