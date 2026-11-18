## Maritime Traffic Slowdowns
### Using PortWatch datasets to model traffic slowdowns for maritime commercial shipping. 
#### Headline Result

Logistic regression fit on three ratio features achieved a **Recall@3 score of 0.824**, outperforming the persistence baseline of **0.527**.

| Model | Recall@3 | Average Precision (AP) |
|---|---:|---:|
| Logistic Regression (3 ratio features) | **0.824** | 0.438 |
| Persistence Baseline | 0.527 | 0.278 |

The logistic regression model improves Recall@3 by **0.297** over the persistence baseline.

#### To Run
```bash
git clone https://github.com/samAmabile/Maritime.git 
cd Maritime 

# the easiest way to run is to press run all in an IDE that supports ipynb, or run cells individually top to bottom. Running all takes about 5 minutes to complete. 
```

* Complete analysis in ```report.ipynb```
* Final feature vector in ```data/final_feature_vector.csv```
* press run all, or run manually, final cell shows the headline result. 

#### Jupyter Notebooks

* report.ipynb: the complete final notebook
* anlysis.ipynb: initial modelling and analysis 

##### HTML renderings of notebooks 

[complete final notebook with all models, results](https://samAmabile.github.io/Maritime/report.html)

[notebook with exploratory analysis, proof of concept](https://samAmabile.github.io/Maritime/analysis.html)


### Modeling Time-Series Data

According to Faouzi (2025), the general assumption of conditional independence of observations required by many standard machine learning algorithms such as the Naive Bayes is not suited for time series data, as the order of the values in the dataset is itself an important feature of observations, and correlation is expected. A recognized technique for adapting time-series for classification is to extract features from the chronological data to use as inputs for a standard classification model. If one variable is used from the time series the model is considered univariate, multiple variables is called multivariate. 

For this analysis a single variable was used, the total number of ships transiting a chokepoint on a given day, meaning the underlying dataset is univariate. Multiple features are calculated from the single variable, so the final feature matrix is multivariate, but the overall model should be considered univariate since all features are derived from the same variable.

[(Faouzi, 2025)](https://inria.hal.science/hal-03558165v2)

There are 28 chokepoints in the data, and in order to predict traffic slowdowns at each chokepoint they should be modeled separately, so the complete dataset is a panel of 28 univariate time series. 

[(Alam, 2020)](https://medium.com/data-science/panel-data-regression-a-powerful-time-series-modeling-technique-7509ce043fa8) 

## Chokepoint Day
### The dataset is chronological, so a unit of observation is a chokepoint day

For dataset $\mathbb{D}$, let $\mathcal{C}$ be the set of chokepoints ($|\mathcal{C}| = 28$) and $\mathcal{T}$ be the set of calendar dates. Then let $x_{c, t}$ be the number of ships ('n_total' in the csv) transiting chokepoint $c \in \mathcal{C}$ on date $t \in \mathcal{T}$. And let $\Omega$ be the set of observed chokepoint-date pairs. 

$$
\mathbb{D} = \{(c,\,\, t, \,\, x_{c, t}) : (c, t) \in \Omega\}, \qquad \Omega \subseteq \mathcal{C} \times \mathcal{T}
$$

*Observations are not independent. Each chokepoint is its own time series, so what happens on one day is dependent on previous days and affects subsequent days. Additionally, chokepoints can be affected by the same shocks, or a slowdown at one chokepoint may cause a traffic increase at another. All sampling, data splitting, predicting, and validating must be designed specifically for temporal data to avoid data leakage.*

#### Ports and Chokepoints

![ports and chokepoints](plots/ports_chokepoints.png)

#### Maritime Traffic

![plot of maritime traffic through selected chokepoints](plots/traffic.png)

#### Results

![plot of Recall@K and Precision@K against precision baseline](plots/headline.png)

#### Datasets: 

- ports 

https://portwatch.imf.org/datasets/acc668d199d1472abaaf2467133d4ca4/about

- chokepoints

https://portwatch.imf.org/datasets/fa9a5800b0ee4855af8b2944ab1e07af/about 

- disruptions

https://portwatch.imf.org/datasets/d9b37bf4b2104c85aebdcc0c1d8a2ab7/about

- daily chokepoint data 

https://portwatch.imf.org/datasets/3da2b9ca97684916b75c4013f95d18ab/about

#### Unsupervised analysis: K-Means clusters 

![plots of k-means clustering results](plots/k_means.png)


#### Metrics and Baselines 

Evaluation metric was updated from **Average Precision** only to include **Recall@K** for K=3 chokepoints. This was done after the modelling during evaluation, so it is not part of the original experiment, but rather part of the exploratory analysis of the model after the fact. 

Similarly, in addition to the original baseline of the **Prior Predictor**, during exploratory evaluation the model was also tested against the **Persistence Rule**, which predicts traffic slowdown based on already being in a period of slowdown. 

## Splitting Time Series into Train and Test sets Chronologically 

Time series data has a meaningful chronological order and therefore cannot be shuffled and split like more standard datasets for regression or classification. Instead, we have to split the data chronologically, such that all Training data is taken from before the cutoff date $T_0$, and all Testing is done on the data from after $T_0$. We include a 7 day gap buffer to ensure that none of the training forecast targets use data from the testing period. 

$$
\text{Train} : t < T_0 - 7 
$$
$$
\text{Test} : t \geq T_0 
$$

I picked the date January 1st, 2025 (2025-01-01) for the cutoff, resulting in

Training Rows:  59892

Test Rows:  18156

## Standardization

Before fitting a model, we standardize the features using Scikit-Learn's Standard Scaler. The intention of this is to put all numerical features on the same scale. The general formula 

For feature j

$$
x_{i, j}^* = \frac{x_{i,j} - \bar{x}_j}{s_j} 
$$

where $s_j$ is the standard deviation of feature j
## Logistic Regression 

Using logistic regression on the scaled features, the model learns a linear combination of regression coefficients $\beta$ for $p$ features

$$
z_i = \beta_0 + \sum_{j=1}^p \beta_j x_{i,j}^*
$$

### Sigmoid function 

The prediction is converted to a value between 0 and 1 representing a probability by the sigmoid function $\sigma(z)$ 

$$
\sigma(z_i) = \frac{1}{1 + e^{-z}}
$$
$$
P(y_i == 1 | X_i) = \sigma(z_i)
$$

For this classification we start with the default classifcation threshold of 0.5 

$$
\hat{y_i} = 
\begin{cases}
1 : &P(y_i==1|X_i) \geq 0.5\\
0 : &P(y_i==1|X_i) < 0.5
\end{cases}
$$

Where a prediction of 1 means the model predicts that traffic will decrease by more than 30% and a predicion of 0 means the model predicts it will not. 

Scaler and Logistic Regression model in a pipeline, fit it to the pre $T_0 - 7$ training data, and test on the post $T_0$ test set.

# Addressing the class imbalance 

The dataset only has about 3\% positive labels, so testing against a 50% baseline would not work, since a model that predicts 0 100\% of the time would achieve 97\% accuracy. Instead, we test against the prior, which predicts each class according to its proportion in the dataset. 

While this is enough to show that the model is detecting a signal, after further evaluation a stronger benchmark was selected. **The Persisntence Rule** addresses the issues of temporal auto-correlations by predicting a slowdown based on whether a chokepoint is already in a period of decreased traffic, based on the expectation that that will continue. Since autoregressive predictors that use a single variable against time can most easily predict a trend by simply following the slope of the data, beating this is a more significant achievement. 
