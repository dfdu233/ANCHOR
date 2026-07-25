## Conformal Decision Trees - An example of Conformal Generation



### Datasets

All five datasets in the experiments are from [OpenML](https://www.openml.org). The details of
the datasets are as follows.

|                                   | #Test | #Features | #Classes | AUC    |
|-----------------------------------|-------|-----------|----------|--------|
| GesturePhaseSegmentationProcessed | 988   | 32        | 5        | 0.9027 |
| Click_prediction_small            | 3995  | 11        | 2        | 0.6484 |
| adult                             | 4885  | 14        | 2        | 0.9104 |
| Census-Income                     | 29929 | 41        | 2        | 0.9478 |
| MiniBooNE                         | 13007 | 50        | 2        | 0.9790 |

We train a simple random forest model with 100 trees for all the datasets using the training set. 
The AUC on the test set is reported in the above table. Note that the train-test split was done
on OpenML. We did not do any hyperparameter tuning on the model, thus we did not split the training
set into validation set.

We chose the five datasets because of the number of features, size of the datasets, the model
performances and the number of classes.

- **GesturePhaseSegmentationProcessed** _Madeo, R., Wagner, P., & Peres, S. (2013). Gesture Phase Segmentation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5Z32C._
  
  The dataset is composed by features extracted from 7 videos with people gesticulating, 
  aiming at studying Gesture Phase Segmentation. Each video is represented by two files:
  a raw file, which contains the position of hands, wrists, head and spine of the user in
  each frame; and a processed file, which contains velocity and acceleration of hands and 
  wrists. More details can be found [here](https://openml.org/search?type=data&id=4538).

- **Click_prediction_small** _Felipe Coutinho. Click prediction. https://kaggle.com/competitions/click-prediction-cds, 2022. Kaggle._
 
  This data is derived from the 2012 KDD Cup. The data is subsampled to 0.1% of the original
  number of instances, downsampling the majority class (click=0) so that the target feature 
  is reasonably balanced (5 to 1). The data is about advertisements shown alongside search results 
  in a search engine, and whether or not people clicked on these ads. The task is to build the
  best possible model to predict whether a user will click on a given ad. More details can be found
  [here](https://openml.org/search?type=data&id=41434).
 
- **adult** _Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20._
 
  Prediction task is to determine whether a person makes over 50K a year. Extraction was done
  by Barry Becker from the 1994 Census database. More details can be found
  [here](https://openml.org/search?type=data&id=1590).
 
- **Census-Income** _Census-Income (KDD) [Dataset]. (2000). UCI Machine Learning Repository. https://doi.org/10.24432/C5N30T._
 
  This dataset contains weighted census data extracted from the 1994 and 1995 Current Population
  Surveys conducted by the U.S. Census Bureau. The data contains 41 demographic and employment
  related variables. More details can be found [here](https://openml.org/search?type=data&id=4535).
 
- **MiniBooNE** _Roe, B. (2005). MiniBooNE particle identification [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5QC87._
 
  This dataset is taken from the MiniBooNE experiment and is used to distinguish electron
  neutrinos (signal) from muon neutrinos (background). More details can be found
  [here](https://openml.org/search?type=data&id=41150).


### Random Forest
Our base model is a scikit-learn random forest model. Each leaf of the tree contains a probability
vector that indicates the probability of each class. The final prediction of the random forest is
the average of the probability vector of the predicted leaves over all trees. Note that in random
forest, they never work on the logit space. Thus the averaging is of the probabilities, instead
of the logits.

Each leaf of the tree also contains how many training samples are falling into the leaf. The number
of training samples are weighted by their class imbalance as well. Details can be found in the
documentation of scikit-learn "Understanding the Tree Structure" page.

### Sequence selector and Score function
Our score function of each tree will be the number of weighted samples falling into the leaf. Note
that this can roughly translate into how confident the model is to predict that node, as the model
has seen this amount of samples during training. Our sequence selector is to select the smallest
subset of trees in which the sum of the weighted samples of the predicted leaves are greater than
a specific value, &#955;.

### Admissibility function
Admissibility function should reflect whether some confidence on whether the prediction is correct.
It also relates to the conformal guarantee we are using. Let K be fixed. We say that the model
is confident in the prediction if at least K trees predict the same class as the ground truth class.
Note that we are only taking the predicted class, not their probability for each tree. Thus, the
conformal guarantee is that at least 100&#947;% of the predictions has at least K trees predicted
the correct class.

Note that as the trees are already trained, it may not be possible for each sample to have at least
K trees predicting the correct class. In our conformal generation framework, this corresponds to
the conformal threshold being infinity. Thus, we hope to choose a high enough value of K but not too
great so that the conformal threshold will be infinity.

In our experiments, we chose K=30. Thus, our framework guarantees that at least 30 out of 100
outputted trees by the sequence selector are generating the correct prediction.

### Performance using Conformal Generation
While the conformal generation is mainly used for having a nice and meaningful conformal guarantee,
our experiments also report the difference in performance as an extra. Our new predictions are
generated using the sequence selector function on the conformal threshold. Thus, our new predictions
will use fewer trees, while guaranteeing at least K trees are predicting the correct class.

We note that our framework would not degrade the performance much for a suitable value of K. 
Interestingly, we see a very minor performance boost in some cases. Note that this is what we
could get for free after the model has been trained.

### Experiments

To run the experiments, 

```shell
python -m explore
```

This will download the dataset from OpenML, run the conformal generation and save the results to
`all_results.csv`

For the plotting, one can use this [notebook](DecisionTreeDemo.ipynb).