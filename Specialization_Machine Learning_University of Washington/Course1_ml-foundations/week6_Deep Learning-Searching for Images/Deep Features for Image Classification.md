

```python
import graphlab
```


```python
image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')
```

    This trial license of GraphLab Create is assigned to 519589356@qq.com and will expire on July 24, 2016. Please contact trial@dato.com for licensing options or to request a free non-commercial license for personal or academic use.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v1.10.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1467248148.log.0
    


```python
graphlab.canvas.set_target('ipynb')
```


```python
image_train['image'].show()
```



# 训练一个原始图像像素的分类器


```python
raw_pixel_model = graphlab.logistic_classifier.create(image_train,
                                                     target='label',
                                                     features=['image_array'])
```

    PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
              You can set ``validation_set=None`` to disable validation tracking.
    
    


<pre>WARNING: The number of feature dimensions in this problem is very large in comparison with the number of examples. Unless an appropriate regularization value is set, this model may not provide accurate predictions for a validation/test set.</pre>



<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 1908</pre>



<pre>Number of classes           : 4</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 3072</pre>



<pre>Number of coefficients    : 9219</pre>



<pre>Starting L-BFGS</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training-accuracy | Validation-accuracy |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| 1         | 6        | 0.000016  | 2.711136     | 0.331237          | 0.340206            |</pre>



<pre>| 2         | 8        | 1.000000  | 3.380917     | 0.389937          | 0.402062            |</pre>



<pre>| 3         | 9        | 1.000000  | 3.786495     | 0.403040          | 0.412371            |</pre>



<pre>| 4         | 10       | 1.000000  | 4.197599     | 0.436583          | 0.412371            |</pre>



<pre>| 5         | 11       | 1.000000  | 4.644276     | 0.446541          | 0.453608            |</pre>



<pre>| 6         | 12       | 1.000000  | 5.036824     | 0.457023          | 0.443299            |</pre>



<pre>| 10        | 16       | 1.000000  | 6.693225     | 0.518868          | 0.474227            |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>TERMINATED: Iteration limit reached.</pre>



<pre>This model may not be optimal. To improve it, consider increasing `max_iterations`.</pre>


# 用简单模型做一个预测


```python
image_test[0:3]['image'].show()
```




```python
image_test[0:3]['label']
```




    dtype: str
    Rows: 3
    ['cat', 'automobile', 'cat']




```python
raw_pixel_model.predict(image_test[0:3])
```




    dtype: str
    Rows: 3
    ['bird', 'cat', 'bird']



# 评估原始图片模型


```python
raw_pixel_model.evaluate(image_test)
```




    {'accuracy': 0.4745, 'auc': 0.7205338750000017, 'confusion_matrix': Columns:
     	target_label	str
     	predicted_label	str
     	count	int
     
     Rows: 16
     
     Data:
     +--------------+-----------------+-------+
     | target_label | predicted_label | count |
     +--------------+-----------------+-------+
     |     dog      |    automobile   |  121  |
     |     bird     |       dog       |  191  |
     |     dog      |       cat       |  235  |
     |     dog      |       bird      |  227  |
     |     cat      |    automobile   |  181  |
     |     cat      |       cat       |  343  |
     |     dog      |       dog       |  417  |
     |     cat      |       dog       |  296  |
     |     bird     |       cat       |  160  |
     |  automobile  |    automobile   |  651  |
     +--------------+-----------------+-------+
     [16 rows x 3 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns., 'f1_score': 0.4709172420498797, 'log_loss': 1.210051435299983, 'precision': 0.46901690768969195, 'recall': 0.47450000000000003, 'roc_curve': Columns:
     	threshold	float
     	fpr	float
     	tpr	float
     	p	int
     	n	int
     	class	int
     
     Rows: 400004
     
     Data:
     +-----------+-----+-----+------+------+-------+
     | threshold | fpr | tpr |  p   |  n   | class |
     +-----------+-----+-----+------+------+-------+
     |    0.0    | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   1e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   2e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   3e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   4e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   5e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   6e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   7e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   8e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   9e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     +-----------+-----+-----+------+------+-------+
     [400004 rows x 6 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.}



# 用深度特征提升模型


```python
len(image_train)
```




    2005




```python
deep_learning_model = graphlab.load_model('imagenet_model')
image_train['deep_features'] = deep_learning_model.extract_features(image_train)
```


```python
image_train.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">deep_features</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image_array</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.242871761322,<br>1.09545373917, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[73.0, 77.0, 58.0, 71.0,<br>68.0, 50.0, 77.0, 69.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cat</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.525087952614, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[7.0, 5.0, 8.0, 7.0, 5.0,<br>8.0, 5.0, 4.0, 6.0, 7.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cat</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.566015958786, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[169.0, 122.0, 65.0,<br>131.0, 108.0, 75.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">70</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">dog</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.12979578972, 0.0, 0.0,<br>0.778194487095, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[154.0, 179.0, 152.0,<br>159.0, 183.0, 157.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">90</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.71786928177, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[216.0, 195.0, 180.0,<br>201.0, 178.0, 160.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">97</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.57818555832, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[33.0, 44.0, 27.0, 29.0,<br>44.0, 31.0, 32.0, 45.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">107</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">dog</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0,<br>0.220677852631, 0.0,  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[97.0, 51.0, 31.0, 104.0,<br>58.0, 38.0, 107.0, 61.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">121</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.23753464222, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[93.0, 96.0, 88.0, 102.0,<br>106.0, 97.0, 117.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">136</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0, 0.0, 0.0,<br>0.0, 7.5737862587, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[35.0, 59.0, 53.0, 36.0,<br>56.0, 56.0, 42.0, 62.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">138</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.658935725689, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[205.0, 193.0, 195.0,<br>200.0, 187.0, 193.0, ...</td>
    </tr>
</table>
[10 rows x 5 columns]<br/>
</div>



# 用深度特征训练一个分类器


```python
deep_features_model = graphlab.logistic_classifier.create(image_train,
                                                         features=['deep_features'],
                                                         target='label')
```

    PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
              You can set ``validation_set=None`` to disable validation tracking.
    
    


<pre>WARNING: The number of feature dimensions in this problem is very large in comparison with the number of examples. Unless an appropriate regularization value is set, this model may not provide accurate predictions for a validation/test set.</pre>



<pre>WARNING: Detected extremely low variance for feature(s) 'deep_features' because all entries are nearly the same.
Proceeding with model training using all features. If the model does not provide results of adequate quality, exclude the above mentioned feature(s) from the input dataset.</pre>



<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 1899</pre>



<pre>Number of classes           : 4</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 4096</pre>



<pre>Number of coefficients    : 12291</pre>



<pre>Starting L-BFGS</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training-accuracy | Validation-accuracy |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| 1         | 5        | 0.000132  | 2.371826     | 0.754081          | 0.679245            |</pre>



<pre>| 2         | 9        | 0.250000  | 4.313088     | 0.770405          | 0.622642            |</pre>



<pre>| 3         | 10       | 0.250000  | 5.078141     | 0.774618          | 0.632075            |</pre>



<pre>| 4         | 11       | 0.250000  | 5.710824     | 0.781464          | 0.641509            |</pre>



<pre>| 5         | 12       | 0.250000  | 6.439439     | 0.786203          | 0.650943            |</pre>



<pre>| 6         | 13       | 0.250000  | 7.318300     | 0.793049          | 0.650943            |</pre>



<pre>| 7         | 14       | 0.250000  | 8.028710     | 0.803581          | 0.698113            |</pre>



<pre>| 8         | 15       | 0.250000  | 8.852419     | 0.824118          | 0.698113            |</pre>



<pre>| 9         | 16       | 0.250000  | 9.655053     | 0.849921          | 0.698113            |</pre>



<pre>| 10        | 17       | 0.250000  | 10.418116    | 0.863086          | 0.716981            |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>TERMINATED: Iteration limit reached.</pre>



<pre>This model may not be optimal. To improve it, consider increasing `max_iterations`.</pre>


# 应用深度特征模型来预测


```python
image_test[0:3]['image'].show()
```




```python
deep_features_model.predict(image_test[0:3])
```




    dtype: str
    Rows: 3
    ['cat', 'automobile', 'cat']



# 计算深度特征在测试集得准确度


```python
deep_features_model.evaluate(image_test)
```




    {'accuracy': 0.792, 'auc': 0.9405419583333324, 'confusion_matrix': Columns:
     	target_label	str
     	predicted_label	str
     	count	int
     
     Rows: 16
     
     Data:
     +--------------+-----------------+-------+
     | target_label | predicted_label | count |
     +--------------+-----------------+-------+
     |     dog      |    automobile   |   17  |
     |     bird     |       bird      |  826  |
     |  automobile  |       bird      |   29  |
     |     dog      |       bird      |   57  |
     |     cat      |    automobile   |   38  |
     |     cat      |       cat       |  686  |
     |     bird     |       cat       |  105  |
     |     cat      |       dog       |  188  |
     |     dog      |       dog       |  701  |
     |  automobile  |    automobile   |  955  |
     +--------------+-----------------+-------+
     [16 rows x 3 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns., 'f1_score': 0.7913284121277684, 'log_loss': 0.557603100802054, 'precision': 0.7911304071990589, 'recall': 0.792, 'roc_curve': Columns:
     	threshold	float
     	fpr	float
     	tpr	float
     	p	int
     	n	int
     	class	int
     
     Rows: 400004
     
     Data:
     +-----------+----------------+-----+------+------+-------+
     | threshold |      fpr       | tpr |  p   |  n   | class |
     +-----------+----------------+-----+------+------+-------+
     |    0.0    |      1.0       | 1.0 | 1000 | 3000 |   0   |
     |   1e-05   | 0.976333333333 | 1.0 | 1000 | 3000 |   0   |
     |   2e-05   |     0.972      | 1.0 | 1000 | 3000 |   0   |
     |   3e-05   | 0.964666666667 | 1.0 | 1000 | 3000 |   0   |
     |   4e-05   | 0.959333333333 | 1.0 | 1000 | 3000 |   0   |
     |   5e-05   |     0.955      | 1.0 | 1000 | 3000 |   0   |
     |   6e-05   |     0.951      | 1.0 | 1000 | 3000 |   0   |
     |   7e-05   | 0.948333333333 | 1.0 | 1000 | 3000 |   0   |
     |   8e-05   | 0.944666666667 | 1.0 | 1000 | 3000 |   0   |
     |   9e-05   | 0.941333333333 | 1.0 | 1000 | 3000 |   0   |
     +-----------+----------------+-----+------+------+-------+
     [400004 rows x 6 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.}




```python

```
