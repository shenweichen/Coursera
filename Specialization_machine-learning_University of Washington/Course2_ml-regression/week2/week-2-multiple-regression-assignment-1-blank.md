
# Regression Week 2: Multiple Regression (Interpretation)

The goal of this first notebook is to explore multiple regression and feature engineering with existing graphlab functions.

In this notebook you will use data on house sales in King County to predict prices using multiple regression. You will:
* Use SFrames to do some feature engineering
* Use built-in graphlab functions to compute the regression weights (coefficients/parameters)
* Given the regression weights, predictors and outcome write a function to compute the Residual Sum of Squares
* Look at coefficients and interpret their meanings
* Evaluate multiple models via RSS

# Fire up graphlab create


```python
import graphlab
```

# Load in house sales data

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


```python
sales = graphlab.SFrame('kc_house_data.gl/')
```

    This trial license of GraphLab Create is assigned to 519589356@qq.com and will expire on July 24, 2016. Please contact trial@dato.com for licensing options or to request a free non-commercial license for personal or academic use.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v1.10.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1467543608.log.0
    

# Split data into training and testing.
We use seed=0 so that everyone running this notebook gets the same results.  In practice, you may set a random seed (or let GraphLab Create pick a random seed for you).  


```python
train_data,test_data = sales.random_split(.8,seed=0)
```

# Learning a multiple regression model

Recall we can use the following code to learn a multiple regression model predicting 'price' based on the following features:
example_features = ['sqft_living', 'bedrooms', 'bathrooms'] on training data with the following code:

(Aside: We set validation_set = None to ensure that the results are always the same)


```python
example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = graphlab.linear_regression.create(train_data, target = 'price', features = example_features, 
                                                  validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 3</pre>



<pre>Number of unpacked features : 3</pre>



<pre>Number of coefficients    : 4</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 1.012521     | 4146407.600631     | 258679.804477 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


Now that we have fitted the model we can extract the regression weights (coefficients) as an SFrame as follows:


```python
example_weight_summary = example_model.get("coefficients")
print example_weight_summary
```

    +-------------+-------+----------------+---------------+
    |     name    | index |     value      |     stderr    |
    +-------------+-------+----------------+---------------+
    | (intercept) |  None | 87910.0724924  |  7873.3381434 |
    | sqft_living |  None | 315.403440552  | 3.45570032585 |
    |   bedrooms  |  None | -65080.2155528 | 2717.45685442 |
    |  bathrooms  |  None | 6944.02019265  | 3923.11493144 |
    +-------------+-------+----------------+---------------+
    [4 rows x 4 columns]
    
    

# Making Predictions

In the gradient descent notebook we use numpy to do our regression. In this book we will use existing graphlab create functions to analyze multiple regressions. 

Recall that once a model is built we can use the .predict() function to find the predicted values for data we pass. For example using the example model above:


```python
example_predictions = example_model.predict(train_data)
print example_predictions[0] # should be 271789.505878
```

    271789.505878
    

# Compute RSS

Now that we can make predictions given the model, let's write a function to compute the RSS of the model. Complete the function below to calculate RSS given the model, data, and the outcome.


```python
def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predict = model.predict(data);
    # Then compute the residuals/errors
    residuals = outcome - predict;
    # Then square and add them up
    RSS = (residuals*residuals).sum()
    return(RSS)    
```

Test your function by computing the RSS on TEST data for the example model:


```python
rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price'])
print rss_example_train # should be 2.7376153833e+14
```

    2.7376153833e+14
    

# Create some new features

Although we often think of multiple regression as including multiple different features (e.g. # of bedrooms, squarefeet, and # of bathrooms) but we can also consider transformations of existing features e.g. the log of the squarefeet or even "interaction" features such as the product of bedrooms and bathrooms.

You will use the logarithm function to create a new feature. so first you should import it from the math library.


```python
from math import log
```

Next create the following 4 new features as column in both TEST and TRAIN data:
* bedrooms_squared = bedrooms\*bedrooms
* bed_bath_rooms = bedrooms\*bathrooms
* log_sqft_living = log(sqft_living)
* lat_plus_long = lat + long 
As an example here's the first one:


```python
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)
```


```python
# create the remaining 3 features in both TEST and TRAIN data
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']
train_data['log_sqft_living'] = train_data['sqft_living'].apply(log)
test_data['log_sqft_living'] = test_data['sqft_living'].apply(log)
train_data['lat_plus_long'] = train_data['lat']+train_data['long']
test_data['lat_plus_long'] = test_data['lat']+test_data['long']
```

* Squaring bedrooms will increase the separation between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2 = 16. Consequently this feature will mostly affect houses with many bedrooms.
* bedrooms times bathrooms gives what's called an "interaction" feature. It is large when *both* of them are large.
* Taking the log of squarefeet has the effect of bringing large values closer together and spreading out small values.
* Adding latitude to longitude is totally non-sensical but we will do it anyway (you'll see why)

**Quiz Question: What is the mean (arithmetic average) value of your 4 new features on TEST data? (round to 2 digits)**


```python
#test_data['bedrooms_squared'].mean()
#test_data['bed_bath_rooms'].mean()
#test_data['log_sqft_living'].mean()
test_data['lat_plus_long'].mean()

```




    -74.65333497217306



# Learning Multiple Models

Now we will learn the weights for three (nested) models for predicting house prices. The first model will have the fewest features the second model will add one more feature and the third will add a few more:
* Model 1: squarefeet, # bedrooms, # bathrooms, latitude & longitude
* Model 2: add bedrooms\*bathrooms
* Model 3: Add log squarefeet, bedrooms squared, and the (nonsensical) latitude + longitude


```python
model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']
```

Now that you have the features, learn the weights for the three different models for predicting target = 'price' using graphlab.linear_regression.create() and look at the value of the weights/coefficients:


```python
# Learn the three models: (don't forget to set validation_set = None)
model_1 = graphlab.linear_regression.create(train_data,
                                          features=['sqft_living', 'bedrooms', 'bathrooms', 'lat','long'],
                                          target='price',
                                          validation_set = None)
model_2 = graphlab.linear_regression.create(train_data,
                                          features=['sqft_living', 'bedrooms', 'bathrooms', 'lat','long','bed_bath_rooms'],
                                          target='price',
                                          validation_set = None)
model_3 = graphlab.linear_regression.create(train_data,
                                          features=['sqft_living', 'bedrooms', 'bathrooms', 'lat','long','bedrooms_squared', 'log_sqft_living', 'lat_plus_long'],
                                          target='price',
                                          validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 5</pre>



<pre>Number of unpacked features : 5</pre>



<pre>Number of coefficients    : 6</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.015696     | 4074878.213096     | 236378.596455 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 6</pre>



<pre>Number of unpacked features : 6</pre>



<pre>Number of coefficients    : 7</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.027879     | 4014170.932927     | 235190.935428 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 17384</pre>



<pre>Number of features          : 8</pre>



<pre>Number of unpacked features : 8</pre>



<pre>Number of coefficients    : 9</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.007634     | 3220897.833490     | 228259.060773 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
# Examine/extract each model's coefficients:
#model_1.get("coefficients")
model_2.get("coefficients")
#model_3.get("coefficients")
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-54410676.1152</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1650405.16541</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sqft_living</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">304.449298057</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.20217535637</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bedrooms</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-116366.043231</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4805.54966546</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bathrooms</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-77972.3305135</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7565.05991091</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">lat</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">625433.834953</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13058.3530972</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">long</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-203958.60296</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13268.1283711</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bed_bath_rooms</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">26961.6249092</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1956.36561555</td>
    </tr>
</table>
[7 rows x 4 columns]<br/>
</div>



**Quiz Question: What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 1?**

**Quiz Question: What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 2?**

Think about what this means.

# Comparing multiple models

Now that you've learned three models and extracted the model weights we want to evaluate which model is best.

First use your functions from earlier to compute the RSS on TRAINING Data for each of the three models.


```python
# Compute the RSS on TRAINING data for each of the three models and record the values:
RSS_1 =get_residual_sum_of_squares(model_1,train_data,train_data['price'])
RSS_2 =get_residual_sum_of_squares(model_2,train_data,train_data['price'])
RSS_3 =get_residual_sum_of_squares(model_3,train_data,train_data['price'])
RSS_1,RSS_2,RSS_3
```




    (971328233543667.1, 961592067855751.0, 905744624371633.8)



**Quiz Question: Which model (1, 2 or 3) has lowest RSS on TRAINING Data?** Is this what you expected?

Now compute the RSS on on TEST data for each of the three models.


```python
# Compute the RSS on TESTING data for each of the three models and record the values:
RSS_1 =get_residual_sum_of_squares(model_1,test_data,test_data['price'])
RSS_2 =get_residual_sum_of_squares(model_2,test_data,test_data['price'])
RSS_3 =get_residual_sum_of_squares(model_3,test_data,test_data['price'])
RSS_1,RSS_2,RSS_3
```




    (226568089092795.38, 224368799993615.4, 287197673034428.0)



**Quiz Question: Which model (1, 2 or 3) has lowest RSS on TESTING Data?** Is this what you expected?Think about the features that were added to each model from the previous.
