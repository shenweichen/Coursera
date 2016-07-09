
# Regression Week 4: Ridge Regression (interpretation)

In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
* Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression
* Use matplotlib to visualize polynomial regressions
* Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression, this time with L2 penalty
* Use matplotlib to visualize polynomial regressions under L2 regularization
* Choose best L2 penalty using cross-validation.
* Assess the final fit using test data.

We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

# Fire up graphlab create


```python
import graphlab
```

# Polynomial regression, revisited

We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:


```python
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1']=feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name]=poly_sframe['power_1'].apply(lambda x:x**power)
    return poly_sframe
```

Let's use matplotlib to visualize what a polynomial regression looks like on the house data.


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
sales = graphlab.SFrame('kc_house_data.gl/')
```

    This trial license of GraphLab Create is assigned to 519589356@qq.com and will expire on July 24, 2016. Please contact trial@dato.com for licensing options or to request a free non-commercial license for personal or academic use.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v1.10.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1467801947.log.0
    

As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.


```python
sales = sales.sort(['sqft_living','price'])
```

Let us revisit the 15th-order polynomial model using the 'sqft_living' input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. When fitting the model, use an L2 penalty of `1e-5`:


```python
l2_small_penalty = 1e-5
```

Note: When we have so many features and so few data points, the solution can become highly numerically unstable, which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will introduce a tiny amount of regularization (`l2_penalty=1e-5`) to make the solution numerically stable.  (In lecture, we discussed the fact that regularization can also help with numerical stability, and here we are seeing a practical example.)

With the L2 penalty specified above, fit the model and print out the learned weights.

Hint: make sure to add 'price' column to the new SFrame before calling `graphlab.linear_regression.create()`. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set=None` in this call.


```python
poly_data = polynomial_sframe(sales['sqft_living'],12)
my_features = poly_data.column_names()
poly_data['price']=sales['price']
poly_model = graphlab.linear_regression.create(poly_data,
                                               target='price',
                                              validation_set = None,
                                              features=my_features,
                                              l2_penalty = l2_small_penalty)
poly_model.get('coefficients')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 12</pre>



<pre>Number of unpacked features : 12</pre>



<pre>Number of coefficients    : 13</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.023545     | 2662032.986676     | 245657.789286 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





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
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">164314.572643</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">341473.474397</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.327299576</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1495.54023476</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.134893196562</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.74705282073</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.000135089853992</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0028030600101</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.62273923999e-08</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.7770161343e-06</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-9.04925981069e-12</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.41084535575e-10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.23740102359e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.09560421027e-13</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.55696188821e-20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.06901405848e-17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1.6976668529e-24</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.41009046711e-21</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.57603316052e-28</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.82506357285e-25</td>
    </tr>
</table>
[13 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



***QUIZ QUESTION:  What's the learned value for the coefficient of feature `power_1`?***

# Observe overfitting

Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. But first, we must reproduce the experiment we did in Week 3.

First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use `.random_split` function and make sure you set `seed=0`. 


```python
(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)
```

Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.

Hint: When calling `graphlab.linear_regression.create()`, use the same L2 penalty as before (i.e. `l2_small_penalty`).  Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
(semi_split1, semi_split2) = poly15_data.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)
```


```python
model15_set1 = graphlab.linear_regression.create(set_1, target = 'price', features = my_features, validation_set = None,l2_penalty=l2_small_penalty)
model15_set1.get("coefficients")
plt.plot(set_1['power_1'],set_1['price'],'.',
        set_1['power_1'], model15_set1.predict(set_1),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5404</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.015630     | 2191984.900939     | 248699.117254 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x165ff518>,
     <matplotlib.lines.Line2D at 0x165ff5f8>]




![png](output_23_16.png)



```python
model15_set2 = graphlab.linear_regression.create(set_2, target = 'price', features = my_features, validation_set = None,l2_penalty=l2_small_penalty)
model15_set2.get("coefficients")
plt.plot(set_2['power_1'],set_2['price'],'.',
        set_2['power_1'], model15_set2.predict(set_2),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5398</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.015628     | 1975178.190682     | 234533.610645 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x258c7240>,
     <matplotlib.lines.Line2D at 0x258c7320>]




![png](output_24_16.png)



```python
model15_set3 = graphlab.linear_regression.create(set_3, target = 'price', features = my_features, validation_set = None,l2_penalty=l2_small_penalty)
model15_set3.get("coefficients")
plt.plot(set_3['power_1'],set_3['price'],'.',
        set_3['power_1'], model15_set3.predict(set_3),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5409</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.003008     | 2283722.683933     | 251097.728054 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x25aa7780>,
     <matplotlib.lines.Line2D at 0x25aa7860>]




![png](output_25_16.png)



```python
model15_set4 = graphlab.linear_regression.create(set_4, target = 'price', features = my_features, validation_set = None,l2_penalty=l2_small_penalty)
model15_set4.get("coefficients")
plt.plot(set_4['power_1'],set_4['price'],'.',
        set_4['power_1'], model15_set4.predict(set_4),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5402</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.000000     | 2378292.371612     | 244341.293203 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x25d7bdd8>,
     <matplotlib.lines.Line2D at 0x25d7beb8>]




![png](output_26_16.png)


The four curves should differ from one another a lot, as should the coefficients you learned.

***QUIZ QUESTION:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?***  (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)


```python
print model15_set1.get('coefficients')[1]['value']
print model15_set2.get('coefficients')[1]['value']
print model15_set3.get('coefficients')[1]['value']
print model15_set4.get('coefficients')[1]['value']
```

    585.86581347
    783.493802508
    -759.251842854
    1247.59035088
    

# Ridge regression comes to rescue

Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)

With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
model15_set1 = graphlab.linear_regression.create(set_1, target = 'price', features = my_features, validation_set = None,l2_penalty=1e5)
model15_set1.get("coefficients")
plt.plot(set_1['power_1'],set_1['price'],'.',
        set_1['power_1'], model15_set1.predict(set_1),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5404</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.000000     | 5978778.434729     | 374261.720860 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x260fdc88>,
     <matplotlib.lines.Line2D at 0x260fdd68>]




![png](output_31_16.png)



```python
model15_set2 = graphlab.linear_regression.create(set_2, target = 'price', features = my_features, validation_set = None,l2_penalty=1e5)
model15_set2.get("coefficients")
plt.plot(set_2['power_1'],set_2['price'],'.',
        set_2['power_1'], model15_set2.predict(set_2),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5398</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.015592     | 2984894.541944     | 323238.809634 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x16285cf8>,
     <matplotlib.lines.Line2D at 0x16285dd8>]




![png](output_32_16.png)



```python
model15_set3 = graphlab.linear_regression.create(set_3, target = 'price', features = my_features, validation_set = None,l2_penalty=1e5)
model15_set3.get("coefficients")
plt.plot(set_3['power_1'],set_3['price'],'.',
        set_3['power_1'], model15_set3.predict(set_3),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5409</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.015625     | 3695342.767093     | 350033.521294 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x263d1278>,
     <matplotlib.lines.Line2D at 0x263d1358>]




![png](output_33_16.png)



```python
model15_set4 = graphlab.linear_regression.create(set_4, target = 'price', features = my_features, validation_set = None,l2_penalty=1e5)
model15_set4.get("coefficients")
plt.plot(set_4['power_1'],set_4['price'],'.',
        set_4['power_1'], model15_set4.predict(set_4),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5402</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.022149     | 3601895.280124     | 323111.582889 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x266c88d0>,
     <matplotlib.lines.Line2D at 0x266c89e8>]




![png](output_34_16.png)


These curves should vary a lot less, now that you applied a high degree of regularization.

***QUIZ QUESTION:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?*** (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)


```python
print model15_set1.get('coefficients')[1]['value']
print model15_set2.get('coefficients')[1]['value']
print model15_set3.get('coefficients')[1]['value']
print model15_set4.get('coefficients')[1]['value']
```

    2.58738875673
    2.04470474182
    2.26890421877
    1.91040938244
    

# Selecting an L2 penalty via cross-validation

Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.

We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:

Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
...<br>
Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set

After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 

To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. GraphLab Create has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder. (Make sure to use `seed=1` to get consistent answer.)


```python
(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)
```

Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.


```python
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)
```

    0 (0, 1938)
    1 (1939, 3878)
    2 (3879, 5817)
    3 (5818, 7757)
    4 (7758, 9697)
    5 (9698, 11636)
    6 (11637, 13576)
    7 (13577, 15515)
    8 (15516, 17455)
    9 (17456, 19395)
    

Let us familiarize ourselves with array slicing with SFrame. To extract a continuous slice from an SFrame, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.


```python
train_valid_shuffled[0:10] # rows 0 to 9
```

Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.


```python
validation4 = train_valid_shuffled[5818:7758]
```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the fourth segment. When rounded to nearest whole number, the average should be $536,234.


```python
print int(round(validation4['price'].mean(), 0))
```

    536234
    

After designating one of the k segments as the validation set, we train a model using the rest of the data. To choose the remainder, we slice (0:start) and (end+1:n) of the data and paste them together. SFrame has `append()` method that pastes together two disjoint sets of rows originating from a common dataset. For instance, the following cell pastes together the first and last two rows of the `train_valid_shuffled` dataframe.


```python
n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print first_two.append(last_two)
```

Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.


```python
train4 = train_valid_shuffled[0:5818].append(train_valid_shuffled[7758:19396])
```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the data with fourth segment excluded. When rounded to nearest whole number, the average should be $539,450.


```python
print int(round(train4['price'].mean(), 0))
```

    539450
    

Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.

* For each i in [0, 1, ..., k-1]:
  * Compute starting and ending indices of segment i and call 'start' and 'end'
  * Form validation set by taking a slice (start:end+1) from the data.
  * Form training set by appending slice (end+1:n) to the end of slice (0:start).
  * Train a linear model using training set just formed, with a given l2_penalty
  * Compute validation error using validation set just formed


```python
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    avg = 0;
    for i in xrange(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation_data = data[start:end+1]
        training_data = data[0:start-1].append(data[end+1:len(data)])
        model = graphlab.linear_regression.create(training_data,target=output_name,features=features_list,l2_penalty = l2_penalty,validation_set=None,verbose=False)
        avg =  avg + ((model.predict(validation_data)-validation_data[output_name])**2).sum()
    return avg/k
```

    
    

Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
* We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
* For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
    * Run 10-fold cross-validation with `l2_penalty`
* Report which L2 penalty produced the lowest average validation error.

Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!


```python
import numpy as np
l2_penalty_values = np.logspace(1, 7, num=13)
train_valid_shuffled
poly_shuffled = polynomial_sframe(train_valid_shuffled['sqft_living'],15)
my_features = poly_shuffled.column_names()
poly_shuffled['price']=train_valid_shuffled['price']
cross_validation_error = list()
for p in penalty:
    cvrss = k_fold_cross_validation(10, p, poly_shuffled,'price',my_features )
    print p,cvrss
    cross_validation_error.append(cvrss)
```

    10.0 4.91890589201e+14
    31.6227766017 2.87539512117e+14
    100.0 1.60896975039e+14
    316.227766017 1.22072329342e+14
    1000.0 1.21137227369e+14
    3162.27766017 1.23679107421e+14
    10000.0 1.35834622474e+14
    31622.7766017 1.69381389833e+14
    100000.0 2.26676260834e+14
    316227.766017 2.51266158066e+14
    1000000.0 2.57927893456e+14
    3162277.66017 2.62536140291e+14
    10000000.0 2.64791307332e+14
    

***QUIZ QUESTIONS:  What is the best value for the L2 penalty according to 10-fold validation?***

You may find it useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  


```python
# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.
plt.xscale('log')
plt.plot(l2_penalty_values,cross_validation_error,'.',)

```




    [<matplotlib.lines.Line2D at 0x26412198>]




![png](output_61_1.png)


Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of `l2_penalty`.  This way, your final model will be trained on the entire dataset.


```python
model_best = graphlab.linear_regression.create(poly_shuffled,target='price',features=my_features,l2_penalty =l2_penalty_values[4] ,validation_set=None,verbose=False)
```

***QUIZ QUESTION: Using the best L2 penalty found above, train a model using all training data. What is the RSS on the TEST data of the model you learn with this L2 penalty? ***


```python
poly_test = polynomial_sframe(test['sqft_living'],15)
poly_test['price']= test['price']
RSS = ((model_best.predict(poly_test)-poly_test['price'])**2).sum()
print RSS
```

    1.28780855058e+14
    
