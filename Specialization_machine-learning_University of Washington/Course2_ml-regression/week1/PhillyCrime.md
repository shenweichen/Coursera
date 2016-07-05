
#Fire up graphlab create


```python
import graphlab
```

#Load some house value vs. crime rate data

Dataset is from Philadelphia, PA and includes average house sales price in a number of neighborhoods.  The attributes of each neighborhood we have include the crime rate ('CrimeRate'), miles from Center City ('MilesPhila'), town name ('Name'), and county name ('County').


```python
sales = graphlab.SFrame.read_csv('Philadelphia_Crime_Rate_noNA.csv/')
```

    This trial license of GraphLab Create is assigned to 519589356@qq.com and will expire on July 24, 2016. Please contact trial@dato.com for licensing options or to request a free non-commercial license for personal or academic use.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v1.10.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1467420731.log.0
    


<pre>Finished parsing file C:\Users\51958\Desktop\Coursera\Specialization_machine-learning_University of Washington\Course2_ml-regression\week1\Philadelphia_Crime_Rate_noNA.csv</pre>



<pre>Parsing completed. Parsed 99 lines in 0.017545 secs.</pre>


    ------------------------------------------------------
    Inferred types from first 100 line(s) of file as 
    column_type_hints=[long,float,float,float,float,str,str]
    If parsing fails due to incorrect types, you can correct
    the inferred type list above and pass it to read_csv in
    the column_type_hints argument
    ------------------------------------------------------
    


<pre>Finished parsing file C:\Users\51958\Desktop\Coursera\Specialization_machine-learning_University of Washington\Course2_ml-regression\week1\Philadelphia_Crime_Rate_noNA.csv</pre>



<pre>Parsing completed. Parsed 99 lines in 0.017581 secs.</pre>



```python
sales
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">HousePrice</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">HsPrc ($10,000)</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">CrimeRate</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">MilesPhila</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">PopChg</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">County</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">140463</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.0463</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">29.7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Abington</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Montgome</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">113033</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11.3033</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24.1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Ambler</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Montgome</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">124186</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.4186</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">19.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">25.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aston</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Delaware</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110490</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11.049</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">49.4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">25.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bensalem</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bucks</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">79124</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.9124</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">54.1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">19.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bristol B.</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bucks</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">92634</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.2634</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">48.6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bristol T.</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bucks</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">89246</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.9246</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30.8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">15.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Brookhaven</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Delaware</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">195145</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">19.5145</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-3.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bryn Athyn</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Montgome</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">297342</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">29.7342</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20.2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bryn Mawr</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Montgome</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">264298</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">26.4298</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20.4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">26.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Buckingham</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bucks</td>
    </tr>
</table>
[99 rows x 7 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



#Exploring the data 

The house price in a town is correlated with the crime rate of that town. Low crime towns tend to be associated with higher house prices and vice versa.


```python
graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="CrimeRate", y="HousePrice")
```



#Fit the regression model using crime as the feature


```python
crime_model = graphlab.linear_regression.create(sales, target='HousePrice', features=['CrimeRate'],validation_set=None,verbose=False)
```

#Let's see what our fit looks like

Matplotlib is a Python plotting library that is also useful for plotting.  You can install it with:

'pip install matplotlib'


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.plot(sales['CrimeRate'],sales['HousePrice'],'.',
        sales['CrimeRate'],crime_model.predict(sales),'-')
```




    [<matplotlib.lines.Line2D at 0x1b4aa898>,
     <matplotlib.lines.Line2D at 0x1b4aa978>]




![png](output_13_1.png)


Above: blue dots are original data, green line is the fit from the simple regression.

# Remove Center City and redo the analysis

Center City is the one observation with an extremely high crime rate, yet house prices are not very low.  This point does not follow the trend of the rest of the data very well.  A question is how much including Center City is influencing our fit on the other datapoints.  Let's remove this datapoint and see what happens.


```python
sales_noCC = sales[sales['MilesPhila'] != 0.0] 
```


```python
sales_noCC.show(view="Scatter Plot", x="CrimeRate", y="HousePrice")
```



### Refit our simple regression model on this modified dataset:


```python
crime_model_noCC = graphlab.linear_regression.create(sales_noCC, target='HousePrice', features=['CrimeRate'],validation_set=None, verbose=False)
```

### Look at the fit:


```python
plt.plot(sales_noCC['CrimeRate'],sales_noCC['HousePrice'],'.',
        sales_noCC['CrimeRate'],crime_model.predict(sales_noCC),'-')
```




    [<matplotlib.lines.Line2D at 0x1b723940>,
     <matplotlib.lines.Line2D at 0x1b723a20>]




![png](output_22_1.png)


# Compare coefficients for full-data fit versus no-Center-City fit

Visually, the fit seems different, but let's quantify this by examining the estimated coefficients of our original fit and that of the modified dataset with Center City removed.


```python
crime_model.get('coefficients')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">176626.046881</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">CrimeRate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-576.804949058</td>
    </tr>
</table>
[2 rows x 3 columns]<br/>
</div>




```python
crime_model_noCC.get('coefficients')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">225204.604303</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">CrimeRate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2287.69717443</td>
    </tr>
</table>
[2 rows x 3 columns]<br/>
</div>



Above: We see that for the "no Center City" version, per unit increase in crime, the predicted decrease in house prices is 2,287.  In contrast, for the original dataset, the drop is only 576 per unit increase in crime.  This is significantly different!

###High leverage points: 
Center City is said to be a "high leverage" point because it is at an extreme x value where there are not other observations.  As a result, recalling the closed-form solution for simple regression, this point has the *potential* to dramatically change the least squares line since the center of x mass is heavily influenced by this one point and the least squares line will try to fit close to that outlying (in x) point.  If a high leverage point follows the trend of the other data, this might not have much effect.  On the other hand, if this point somehow differs, it can be strongly influential in the resulting fit.

###Influential observations:  
An influential observation is one where the removal of the point significantly changes the fit.  As discussed above, high leverage points are good candidates for being influential observations, but need not be.  Other observations that are *not* leverage points can also be influential observations (e.g., strongly outlying in y even if x is a typical value).

# Remove high-value outlier neighborhoods and redo analysis

Based on the discussion above, a question is whether the outlying high-value towns are strongly influencing the fit.  Let's remove them and see what happens.


```python
sales_nohighend = sales_noCC[sales_noCC['HousePrice'] < 350000] 
crime_model_nohighend = graphlab.linear_regression.create(sales_nohighend, target='HousePrice', features=['CrimeRate'],validation_set=None, verbose=False)
```

### Do the coefficients change much?


```python
crime_model_noCC.get('coefficients')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">225204.604303</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">CrimeRate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2287.69717443</td>
    </tr>
</table>
[2 rows x 3 columns]<br/>
</div>




```python
crime_model_nohighend.get('coefficients')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">199073.589615</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">CrimeRate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1837.71280989</td>
    </tr>
</table>
[2 rows x 3 columns]<br/>
</div>



Above: We see that removing the outlying high-value neighborhoods has *some* effect on the fit, but not nearly as much as our high-leverage Center City datapoint.


```python

```
