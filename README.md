# Gradient Prediction

Repository for the code used in generating, training and evaluating temperature
navigation ANNs in
([Haesemeyer, Schier & Engert, 2018](https://www.biorxiv.org/content/10.1101/390435v2)).


## Main code modules
### core.py
Includes generally required classes and functions including classes to set up,
save and load all ANN models.
### analysis.py
Contains routines for the analysis of behavioral and neural data.
### *_simulators.py
Contain "species specific" implementations of arena simulations and white noise
simulations as well as simulations for training data generation for predictive
models. Simulator for reinforcement learning models is contained in the
RL_trainingGround.py file since this model learns during the simulation.
### *_trainingGround.py
Contain code for training various predictive and reinforcement learning models.
### data_stores.py
Interface classes that allow storing the runs of ANN simulations since these
take a considerable amount of time.
### Testing scripts
The following files were generated for testing purposes only and have not been
used in the actual publication:
* activationMovie.py
* analyzeTempResponses.py
* ce_retrain.py
* gradientSimulation.py
* mixedInputModel.py
* plot_data.py
* singleInputModel.py

## Notes
### Cluster assignments
While spectral clustering will essentially draw the same cluster boundaries on
every invocation, the assignment of units to individual cluster indices is not
deterministic (in other words, the activity profile belonging to cluster 1 on
one invocation may be found in cluster 5 on the next). Therefore any
correspondence between zebrafish / _C. elegans_ types and ANN cluster centroids
has to be adjusted accordingly.

For zebrafish the assignments are found by calling zfish_ann_correspondence.py
whereas they are visual in the case of _C. elegans_, in other words here the
ANN clusters are assigned based on the closest match in the literature.

### Figure panel arrangements
Since Figure panels were moved around during the publication process, the
assignment of panels to Figures in the Figure*.py scripts is only approximate.
Also, all plots related to reinforcement learning networks are in rl_plots.py
irrespective of their paper figure location.

## Dependencies
The code depends on the libraries listed below. Note that version information
merely refers to the version that was used and likely does not indicate a
strict requirement for that particular library version.
* [python](https://www.python.org/) - 3.6
* [tensorflow](https://www.tensorflow.org/) - 1.10
* [numpy](http://www.numpy.org/) - 1.15
* [scipy](https://www.scipy.org/) - 1.1
* [scikit-learn](https://scikit-learn.org/stable/) - 0.19
* [matplotlib](https://matplotlib.org/) - 3.0
* [seaborn](https://seaborn.pydata.org/) - 0.9
* [pandas](https://pandas.pydata.org/) - 0.23
* [h5py](http://docs.h5py.org/) - 2.8

---
All code is licensed under the MIT license. See LICENSE for details.  
&copy; Martin Haesemeyer, 2017-2019