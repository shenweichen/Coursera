
# Regression Week 3: Assessing Fit (polynomial regression)

In this notebook you will compare different regression models in order to assess which model fits best. We will be using polynomial regression as a means to examine this topic. In particular you will:
* Write a function to take an SArray and a degree and return an SFrame where each column is the SArray to a polynomial value up to the total degree e.g. degree = 3 then column 1 is the SArray column 2 is the SArray squared and column 3 is the SArray cubed
* Use matplotlib to visualize polynomial regressions
* Use matplotlib to visualize the same polynomial degree on different subsets of the data
* Use a validation set to select a polynomial degree
* Assess the final fit using test data

We will continue to use the House data from previous notebooks.

# Fire up graphlab create


```python
import graphlab
```

Next we're going to write a polynomial function that takes an SArray and a maximal degree and returns an SFrame with columns containing the SArray to all the powers up to the maximal degree.

The easiest way to apply a power to an SArray is to use the .apply() and lambda x: functions. 
For example to take the example array and compute the third power we can do as follows: (note running this cell the first time may take longer than expected since it loads graphlab)


```python
tmp = graphlab.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print tmp
print tmp_cubed
```

    This trial license of GraphLab Create is assigned to 519589356@qq.com and will expire on July 24, 2016. Please contact trial@dato.com for licensing options or to request a free non-commercial license for personal or academic use.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v1.10.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1467617071.log.0
    

    [1.0, 2.0, 3.0]
    [1.0, 8.0, 27.0]
    

We can create an empty SFrame using graphlab.SFrame() and then add any columns to it with ex_sframe['column_name'] = value. For example we create an empty SFrame and make the column 'power_1' to be the first power of tmp (i.e. tmp itself).


```python
ex_sframe = graphlab.SFrame()
ex_sframe['power_1'] = tmp
print ex_sframe
```

    +---------+
    | power_1 |
    +---------+
    |   1.0   |
    |   2.0   |
    |   3.0   |
    +---------+
    [3 rows x 1 columns]
    
    

# Polynomial_sframe function

Using the hints above complete the following function to create an SFrame consisting of the powers of an SArray up to a specific degree:


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

To test your function consider the smaller tmp variable and what you would expect the outcome of the following call:


```python
print polynomial_sframe(tmp, 3)
```

    +---------+---------+---------+
    | power_1 | power_2 | power_3 |
    +---------+---------+---------+
    |   1.0   |   1.0   |   1.0   |
    |   2.0   |   4.0   |   8.0   |
    |   3.0   |   9.0   |   27.0  |
    +---------+---------+---------+
    [3 rows x 3 columns]
    
    

# Visualizing polynomial regression

Let's use matplotlib to visualize what a polynomial regression looks like on some real data.

Download Link:https://s3.amazonaws.com/static.dato.com/files/coursera/course-2/kc_house_data.gl.zip


```python
sales = graphlab.SFrame('kc_house_data.gl/')
```

As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.


```python
sales = sales.sort(['sqft_living', 'price'])
```

Let's start with a degree 1 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.


```python
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target
```

NOTE: for all the models in this notebook use validation_set = None to ensure that all results are consistent across users.


```python
model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 1</pre>



<pre>Number of unpacked features : 1</pre>



<pre>Number of coefficients    : 2</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 1.016782     | 4362074.696077     | 261440.790724 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
#let's take a look at the weights before we plot
model1.get("coefficients")
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
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-43579.0852515</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4402.68969743</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">280.622770886</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.93639855513</td>
    </tr>
</table>
[2 rows x 4 columns]<br/>
</div>




```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')
```




    [<matplotlib.lines.Line2D at 0x21d5c3c8>,
     <matplotlib.lines.Line2D at 0x21d5c4e0>]




![png](output_24_1.png)


Let's unpack that plt.plot() command. The first pair of SArrays we passed are the 1st power of sqft and the actual price we then ask it to print these as dots '.'. The next pair we pass is the 1st power of sqft and the predicted values from the linear model. We ask these to be plotted as a line '-'. 

We can see, not surprisingly, that the predicted values all fall on a line, specifically the one with slope 280 and intercept -43579. What if we wanted to plot a second degree polynomial?


```python
poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 2</pre>



<pre>Number of unpacked features : 2</pre>



<pre>Number of coefficients    : 3</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.012032     | 5913020.984255     | 250948.368758 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
model2.get("coefficients")
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
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">199222.496445</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7058.00483552</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">67.9940640677</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.28787201316</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0385812312789</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.000898246547032</td>
    </tr>
</table>
[3 rows x 4 columns]<br/>
</div>




```python
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')
```




    [<matplotlib.lines.Line2D at 0x2223ea90>,
     <matplotlib.lines.Line2D at 0x2223eb70>]




![png](output_28_1.png)


The resulting model looks like half a parabola. Try on your own to see what the cubic looks like:


```python
poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features = poly3_data.column_names() # get the name of the features
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features, validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 3</pre>



<pre>Number of unpacked features : 3</pre>



<pre>Number of coefficients    : 4</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.011031     | 3261066.736007     | 249261.286346 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')
```




    [<matplotlib.lines.Line2D at 0x22c94ef0>,
     <matplotlib.lines.Line2D at 0x22c94fd0>]




![png](output_31_1.png)


Now try a 15th degree polynomial:


```python
poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>| 1         | 2        | 0.036597     | 2662308.584338     | 245690.511190 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
        poly15_data['power_1'], model15.predict(poly15_data),'-')
```




    [<matplotlib.lines.Line2D at 0x233208d0>,
     <matplotlib.lines.Line2D at 0x23320940>]




![png](output_34_1.png)


What do you think of the 15th degree polynomial? Do you think this is appropriate? If we were to change the data do you think you'd get pretty much the same curve? Let's take a look.

# Changing the data and re-learning

We're going to split the sales data into four subsets of roughly equal size. Then you will estimate a 15th degree polynomial model on all four subsets of the data. Print the coefficients (you should use .print_rows(num_rows = 16) to view all of them) and plot the resulting fit (as we did above). The quiz will ask you some questions about these results.

To split the sales data into four subsets, we perform the following steps:
* First split sales into 2 subsets with `.random_split(0.5, seed=0)`. 
* Next split the resulting subsets into 2 more subsets each. Use `.random_split(0.5, seed=0)`.

We set `seed=0` in these steps so that different users get consistent results.
You should end up with 4 subsets (`set_1`, `set_2`, `set_3`, `set_4`) of approximately equal size. 


```python
set_1,set_3=poly15_data.random_split(0.5,seed=0)
set_1,set_2=set_1.random_split(0.5,seed=0)
set_3,set_4=set_3.random_split(0.5,seed=0)
```

Fit a 15th degree polynomial on set_1, set_2, set_3, and set_4 using sqft_living to predict prices. Print the coefficients and make a plot of the resulting model.


```python
model15_set1 = graphlab.linear_regression.create(set_1, target = 'price', features = my_features, validation_set = None)
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



<pre>| 1         | 2        | 0.009025     | 2195218.932304     | 248858.822200 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x2383a470>,
     <matplotlib.lines.Line2D at 0x2383a550>]




![png](output_40_16.png)



```python
model15_set2 = graphlab.linear_regression.create(set_2, target = 'price', features = my_features, validation_set = None)
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



<pre>| 1         | 2        | 0.008524     | 2069212.978547     | 234840.067186 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x25c12f60>,
     <matplotlib.lines.Line2D at 0x25c22080>]




![png](output_41_16.png)



```python
model15_set3 = graphlab.linear_regression.create(set_3, target = 'price', features = my_features, validation_set = None)
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



<pre>| 1         | 2        | 0.006516     | 2269769.506521     | 251460.072754 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x25e5d438>,
     <matplotlib.lines.Line2D at 0x25e5d518>]




![png](output_42_16.png)



```python
model15_set4 = graphlab.linear_regression.create(set_4, target = 'price', features = my_features, validation_set = None)
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



<pre>| 1         | 2        | 0.006517     | 2314893.173827     | 244563.136754 |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    [<matplotlib.lines.Line2D at 0x260329b0>,
     <matplotlib.lines.Line2D at 0x26032a90>]




![png](output_43_16.png)


Some questions you will be asked on your quiz:

**Quiz Question: Is the sign (positive or negative) for power_15 the same in all four models?**

**Quiz Question: (True/False) the plotted fitted lines look the same in all four plots**


```python
 model15_set1.get("coefficients").print_rows(num_rows=16)
print model15_set2.get("coefficients").print_rows(num_rows=16)
print model15_set3.get("coefficients").print_rows(num_rows=16)
print model15_set4.get("coefficients").print_rows(num_rows=16)
```

    +-------------+-------+--------------------+-------------------+
    |     name    | index |       value        |       stderr      |
    +-------------+-------+--------------------+-------------------+
    | (intercept) |  None |   223312.750249    |   835599.663129   |
    |   power_1   |  None |   118.086127587    |   3486.76167789   |
    |   power_2   |  None |  -0.0473482011345  |   5.85700993834   |
    |   power_3   |  None |  3.2531034247e-05  |  0.00510891365748 |
    |   power_4   |  None | -3.32372152563e-09 | 2.39840457452e-06 |
    |   power_5   |  None | -9.75830457749e-14 |  3.4383369873e-10 |
    |   power_6   |  None | 1.15440303427e-17  |        nan        |
    |   power_7   |  None | 1.05145869404e-21  |        nan        |
    |   power_8   |  None | 3.46049616534e-26  |        nan        |
    |   power_9   |  None | -1.0965445417e-30  | 3.72369281471e-25 |
    |   power_10  |  None | -2.42031812013e-34 |        nan        |
    |   power_11  |  None | -1.99601206822e-38 |        nan        |
    |   power_12  |  None | -1.0770990383e-42  |        nan        |
    |   power_13  |  None | -2.72862818005e-47 |        nan        |
    |   power_14  |  None | 2.44782693056e-51  |        nan        |
    |   power_15  |  None | 5.01975232909e-55  |        nan        |
    +-------------+-------+--------------------+-------------------+
    [16 rows x 4 columns]
    
    +-------------+-------+--------------------+-------------------+
    |     name    | index |       value        |       stderr      |
    +-------------+-------+--------------------+-------------------+
    | (intercept) |  None |   89836.5077336    |    1677961.1901   |
    |   power_1   |  None |   319.806946762    |    9997.9019222   |
    |   power_2   |  None |  -0.103315397041   |   25.2922349935   |
    |   power_3   |  None | 1.06682476068e-05  |  0.0358798007061  |
    |   power_4   |  None | 5.75577097709e-09  | 3.18207878802e-05 |
    |   power_5   |  None | -2.54663464754e-13 |  1.8608338365e-08 |
    |   power_6   |  None | -1.09641345055e-16 | 7.35876527801e-12 |
    |   power_7   |  None | -6.36458441789e-21 | 1.97896850554e-15 |
    |   power_8   |  None | 5.52560416916e-25  | 3.56555717807e-19 |
    |   power_9   |  None | 1.35082038973e-28  | 3.98876030791e-23 |
    |   power_10  |  None | 1.18408188259e-32  |        nan        |
    |   power_11  |  None | 1.98348000471e-37  |        nan        |
    |   power_12  |  None | -9.92533590368e-41 |        nan        |
    |   power_13  |  None | -1.60834847057e-44 |  8.5064756305e-40 |
    |   power_14  |  None | -9.12006024271e-49 | 2.51954686947e-43 |
    |   power_15  |  None | 1.68636658332e-52  | 7.12456701345e-48 |
    +-------------+-------+--------------------+-------------------+
    [16 rows x 4 columns]
    
    None
    +-------------+-------+--------------------+-------------------+
    |     name    | index |       value        |       stderr      |
    +-------------+-------+--------------------+-------------------+
    | (intercept) |  None |   87317.9795547    |        nan        |
    |   power_1   |  None |   356.304911045    |        nan        |
    |   power_2   |  None |  -0.164817442809   |        nan        |
    |   power_3   |  None | 4.40424992697e-05  |        nan        |
    |   power_4   |  None | 6.48234876179e-10  |        nan        |
    |   power_5   |  None | -6.75253226587e-13 |        nan        |
    |   power_6   |  None | -3.36842592661e-17 |        nan        |
    |   power_7   |  None | 3.60999704242e-21  |        nan        |
    |   power_8   |  None | 6.46999725625e-25  |        nan        |
    |   power_9   |  None | 4.23639388865e-29  |        nan        |
    |   power_10  |  None | -3.62149427043e-34 |        nan        |
    |   power_11  |  None | -4.27119527274e-37 |        nan        |
    |   power_12  |  None | -5.61445971705e-41 | 3.22504679661e-36 |
    |   power_13  |  None | -3.87452772861e-45 | 3.81940189807e-40 |
    |   power_14  |  None | 4.69430359483e-50  | 1.00042633659e-44 |
    |   power_15  |  None | 6.39045885992e-53  |        nan        |
    +-------------+-------+--------------------+-------------------+
    [16 rows x 4 columns]
    
    None
    +-------------+-------+--------------------+-------------------+
    |     name    | index |       value        |       stderr      |
    +-------------+-------+--------------------+-------------------+
    | (intercept) |  None |   259020.879455    |   1418633.01576   |
    |   power_1   |  None |   -31.7277162089   |   8986.43219604   |
    |   power_2   |  None |   0.10970276962    |   23.6372627005   |
    |   power_3   |  None | -1.58383847342e-05 |  0.0340870703714  |
    |   power_4   |  None | -4.4766062378e-09  |  2.9922677913e-05 |
    |   power_5   |  None | 1.13976573483e-12  | 1.65942118384e-08 |
    |   power_6   |  None | 1.97669120543e-16  | 5.64257457465e-12 |
    |   power_7   |  None | -6.15783678625e-21 | 8.28262236121e-16 |
    |   power_8   |  None | -4.88012304078e-24 |        nan        |
    |   power_9   |  None | -6.62186781367e-28 |        nan        |
    |   power_10  |  None | -2.70631583096e-32 | 5.03468199063e-27 |
    |   power_11  |  None |  6.7237041138e-36  | 8.34356427948e-31 |
    |   power_12  |  None | 1.74115646277e-39  |        nan        |
    |   power_13  |  None | 2.09188375718e-43  |        nan        |
    |   power_14  |  None | 4.78015566127e-48  | 3.26026625058e-43 |
    |   power_15  |  None | -4.74535333103e-51 | 1.22114805265e-47 |
    +-------------+-------+--------------------+-------------------+
    [16 rows x 4 columns]
    
    None
    

# Selecting a Polynomial Degree

Whenever we have a "magic" parameter like the degree of the polynomial there is one well-known way to select these parameters: validation set. (We will explore another approach in week 4).

We split the sales dataset 3-way into training set, test set, and validation set as follows:

* Split our sales data into 2 sets: `training_and_validation` and `testing`. Use `random_split(0.9, seed=1)`.
* Further split our training data into two sets: `training` and `validation`. Use `random_split(0.5, seed=1)`.

Again, we set `seed=1` to obtain consistent results for different users.


```python
training_and_validation,testing = sales.random_split(0.9,seed=1)
training,validation = training_and_validation.random_split(0.5,seed=1)
```

Next you should write a loop that does the following:
* For degree in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (to get this in python type range(1, 15+1))
    * Build an SFrame of polynomial data of train_data['sqft_living'] at the current degree
    * hint: my_features = poly_data.column_names() gives you a list e.g. ['power_1', 'power_2', 'power_3'] which you might find useful for graphlab.linear_regression.create( features = my_features)
    * Add train_data['price'] to the polynomial SFrame
    * Learn a polynomial regression model to sqft vs price with that degree on TRAIN data
    * Compute the RSS on VALIDATION data (here you will want to use .predict()) for that degree and you will need to make a polynmial SFrame using validation data.
* Report which degree had the lowest RSS on validation data (remember python indexes from 0)

(Note you can turn off the print out of linear_regression.create() with verbose = False)


```python
for degree in range(1,16):
    poly_data = polynomial_sframe(training['sqft_living'],degree)
    vali_data = polynomial_sframe(validation['sqft_living'],degree)
    my_features = poly_data.column_names()
    poly_data['price'] = training['price']
    model = graphlab.linear_regression.create(poly_data,target='price',
                                             features = my_features,
                                             validation_set=None,
                                             verbose=False)
    predict = model.predict(vali_data)
    residual = predict - validation['price']
    rss = (residual**2).sum()
    print degree,rss
```

    1 6.76709775198e+14
    2 6.07090530698e+14
    3 6.16714574533e+14
    4 6.09129230654e+14
    5 5.99177138584e+14
    6 5.89182477809e+14
    7 5.91717038418e+14
    8 6.01558237777e+14
    9 6.12563853988e+14
    10 6.21744288936e+14
    11 6.27012012704e+14
    12 6.27757914772e+14
    13 6.24738503262e+14
    14 6.19369705905e+14
    15 6.13089202414e+14
    

**Quiz Question: Which degree (1, 2, …, 15) had the lowest RSS on Validation data?**

Now that you have chosen the degree of your polynomial using validation data, compute the RSS of this model on TEST data. Report the RSS on your quiz.


```python
    poly_data_6 = polynomial_sframe(training['sqft_living'],6)
    test_data = polynomial_sframe(testing['sqft_living'],6)
    my_features = poly_data_6.column_names()
    poly_data_6['price'] = training['price']
    model_6 = graphlab.linear_regression.create(poly_data_6,target='price',
                                             features = my_features,
                                             validation_set=None,
                                             verbose=False)
    predict_6 = model_6.predict(test_data)
    residual = predict_6 - testing['price']
    rss = (residual**2).sum()
    print rss
```

    1.25529337848e+14
    

**Quiz Question: what is the RSS on TEST data for the model with the degree selected from Validation data?**


```python

```
