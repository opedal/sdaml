Task 1 Goals
=========

1. Preprocessing - preprocessing.py

 * Remove missing values
 * Select top X features (https://scikit-learn.org/stable/modules/feature_selection.html)
 * Normalize features
 * Manage outliers
 * Visualize (normalized) top X features (scatter plot different columns, PCA, t-SNE)
 * Do linear regression fit (with and without regularization) and visualize (answers whether we need a nonlinear model)
 * Submit initial model and see how it performs

2. Model - model.py
 * Model selection
     - Change features used
     - Square some features etc
     - Try some built in scipy models that are fast (random forests, polynomial reg)
 * Cross-validation (run our models and test)


---------------------

Goals part 2:

1. Try inverting outlier detection and feature selection
   * Do PCA and get the first 400 components, and do outlier detection on those
2. Look at the normality of the data and maybe try a transform to make it normal
   * Do PCA on the first 400 components and check normality of those
   * Try log-transform
3. After the normality and outliers, we do feature selection
   * Using random forest, and other methods
4. Fit Model
   * Try SGD training on some models we have
   * Then if all else doesnt work, try a neural net with SGD
