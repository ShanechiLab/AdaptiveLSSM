# Adaptive LSSM algorithm
This repository contains the code for adaptive linear state-space model fitting algorithm (adaptive LSSM algorithm).
This algorithm is used for the paper ["Adaptive tracking of human ECoG network dynamics"](https://iopscience.iop.org/article/10.1088/1741-2552/abae42/meta).
The mathematical derivation of the algorithm is explained in more details in the paper ["Adaptive latent state modeling of brain network dynamics with real-time learning rate optimization"](https://iopscience.iop.org/article/10.1088/1741-2552/abcefd/meta).
## Installation guide
You just need to download the current repository or clone it using git. You can run the codes on Matlab provided that all the folders are added to the path.
## Dependencies
This repository does not need any extra dependencies apart from the built-in MATLAB functions.
## User guide
The main functions are [AdaptiveLSSMFittingAlgorithm_wholeTrial.m](./functions/AdaptiveLSSMFittingAlgorithm_wholeTrial.m) which fits the model parameters and [prediction_performance.m](./functions/prediction_performance.m) which computes the prediction performance using the fitted model parameters. There is a test script [testScript_adaptiveLSSM.m](./testScript_adaptiveLSSM.m), to get more familiar with the algorithm. It simulates a time-varying LSSM and generates time-series of brain network activity from it. Then it adaptively learns the time-varying LSSM parameters for each given forgetting factor (learning rate) and computes the prediction performance based on it. Finally, the prediction performance is plotted as a function of the forgetting factor. To test on your own data, you just need to provide the brain network activity time-series, the forgetting factor and a few more setting parameters.
## License
Copyright (c) 2020 University of Southern California<br/>
See full notice in [LICENSE.md](./LICENSE.md)<br/>
Parima Ahmadipour, Yuxiao Yang and Maryam M. Shanechi<br/>
Shanechi Lab, University of Southern California<br/>



