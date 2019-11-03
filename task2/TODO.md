
Deadlines:
* Nov 3: Get past step 2
* Nov 10: Have a selection of models

Standardize?

Handle class imbalance

Performance metrics: confusion matrix, F1 and BMAC

**Step 0:**

Read up on Bootstrap


**Step 1:**

Create a function in a pipeline fashion with a very simple model, that can be used for testing

Separately for bootstrapping and cv?


**Step 2:**

Visualize data
* TSNE
* PCA
* Correlation
* Kernel PCA

=> Determine whether we should eliminate outliers and/or do feature selection


**Step 3:**

Fit different models with different parameters in the bootstrap and cv pipelines

Evaluate with our different scores


**Step 4 (?):**

Ensemble

**Notes**

10-fold CV on undersampling method (300 estimators, contamination = 0.35, learning rate = 0.5, max_depth = 10): mean expected error:  0.6618747970313829 std:  0.034925522970548
resampling with replacement from minority classes gives overall worse results compared to sampling without replacement from class 1 and keeping the other 2 constant
considering reinverting and putting undersampling before again
