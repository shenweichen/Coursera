

```python
import graphlab
```

# 加载CIFAR-10数据集


```python
image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')
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



# 训练一个最近邻模型来获取图像用深度特征


```python
knn_model = graphlab.nearest_neighbors.create(image_train,
                                             features=['deep_features'],
                                             label='id')
```


<pre>Starting brute force nearest neighbors model training.</pre>


# 使用模型来获取相似的图片


```python
graphlab.canvas.set_target('ipynb')
cat = image_train[18:19]
cat['image'].show()
```




```python
knn_model.query(cat)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 17.545ms     |</pre>



<pre>| Done         |         | 100         | 219.022ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">384</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6910</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.9403137951</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39777</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.4634888975</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36870</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.7559623119</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.7866014148</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')
```


```python
cat_neighbors = get_images_from_ids(knn_model.query(cat))
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 11.534ms     |</pre>



<pre>| Done         |         | 100         | 214.57ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
cat_neighbors['image'].show()
```




```python
car = image_train[8:9]
car['image'].show()
```




```python
get_images_from_ids(knn_model.query(car))['image'].show()
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 9.024ms      |</pre>



<pre>| Done         |         | 100         | 183.488ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>




# 创建一个lambda来展示最近邻图像


```python
show_neighbors = lambda i:get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()
```


```python
show_neighbors(8)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 12.534ms     |</pre>



<pre>| Done         |         | 100         | 221.599ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





```python
show_neighbors(26)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 13.536ms     |</pre>



<pre>| Done         |         | 100         | 244.149ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





```python
show_neighbors(2000)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 22.559ms     |</pre>



<pre>| Done         |         | 100         | 222.591ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>




# 作业1


```python
image_train['label'].sketch_summary()
```




    
    +------------------+-------+----------+
    |       item       | value | is exact |
    +------------------+-------+----------+
    |      Length      |  2005 |   Yes    |
    | # Missing Values |   0   |   Yes    |
    | # unique values  |   4   |    No    |
    +------------------+-------+----------+
    
    Most frequent items:
    +-------+------------+-----+-----+------+
    | value | automobile | cat | dog | bird |
    +-------+------------+-----+-----+------+
    | count |    509     | 509 | 509 | 478  |
    +-------+------------+-----+-----+------+
    



# 作业2


```python
dog = image_train.filter_by(['dog'],'label')
cat = image_train.filter_by(['cat'],'label')
automobile = image_train.filter_by(['automobile'],'label')
bird = image_train.filter_by(['bird'],'label')
```


```python
dog_model = graphlab.nearest_neighbors.create(dog,
                                             features=['deep_features'],
                                             label='id')
cat_model = graphlab.nearest_neighbors.create(cat,
                                             features=['deep_features'],
                                             label='id')
automobile_model = graphlab.nearest_neighbors.create(automobile,
                                             features=['deep_features'],
                                             label='id')
bird_model = graphlab.nearest_neighbors.create(bird,
                                             features=['deep_features'],
                                             label='id')
```


<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



```python
image_test[0:1]['image'].show()
```




```python
query_cat = cat_model.query(image_test[0:1])
get_images_from_ids(query_cat[0])['image'].show()
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 8.024ms      |</pre>



<pre>| Done         |         | 100         | 74.874ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





```python
query_dog = dog_model.query(image_test[0:1])
get_images_from_ids(query_dog[0])['image'].show()
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 23.868ms     |</pre>



<pre>| Done         |         | 100         | 94.315ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>




# 作业3


```python
query_cat
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">16289</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.623719208</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45646</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.0068799284</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">32139</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.5200813436</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">25713</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.7548502521</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">331</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.8731228168</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
query_cat['distance'].mean()
```




    36.15573070978294




```python
query_dog['distance'].mean()
```




    37.77071136184156



# 作业4


```python
image_test_dog = image_test.filter_by(['dog'],'label')
image_test_cat = image_test.filter_by(['cat'],'label')
image_test_automobile = image_test.filter_by(['automobile'],'label')
image_test_bird = image_test.filter_by(['bird'],'label')
```


```python
dog_cat_neighbors = cat_model.query(image_test_dog, k=1)
dog_dog_neighbors = dog_model.query(image_test_dog, k=1)
dog_bird_neighbors = bird_model.query(image_test_dog, k=1)
dog_automobile_neighbors = automobile_model.query(image_test_dog, k=1)

```


<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 336.575ms    |</pre>



<pre>| Done         | 509000  | 100         | 366.155ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 320.751ms    |</pre>



<pre>| Done         | 509000  | 100         | 371.887ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 119000  | 24.8954     | 311.933ms    |</pre>



<pre>| Done         | 478000  | 100         | 355.047ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 321.192ms    |</pre>



<pre>| Done         | 509000  | 100         | 366.813ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
dog_distances = graphlab.SFrame({'dog-dog':dog_dog_neighbors['distance'],
                                'dog-cat':dog_cat_neighbors['distance'],
                                'dog-automobile':dog_automobile_neighbors['distance'],
                                'dog-bird':dog_bird_neighbors['distance']})
```


```python
dog_distances
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-automobile</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-bird</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-cat</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-dog</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.9579761457</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.7538647304</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.4196077068</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.4773590373</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">46.0021331807</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.3382958925</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.8353268874</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">32.8458495684</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.9462290692</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.6157590853</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.9763410854</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.0397073189</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.6866060048</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.0892269954</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.5750072914</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.9010327697</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.2269664935</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.272288694</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.778824791</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.4849250909</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.5845117698</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.1462089236</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.1171578292</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.945165344</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.1067352961</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.523040106</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.6095830913</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.0957278345</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.3221140974</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.1947918393</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.9036867306</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.7696131032</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.8244654995</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.1567131661</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.0674700168</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.1089144603</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.4976929401</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.5597962603</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.7258732951</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.2422832585</td>
    </tr>
</table>
[1000 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
dog_distances[0]
```




    {'dog-automobile': 41.957976145712024,
     'dog-bird': 41.753864730351246,
     'dog-cat': 36.41960770675437,
     'dog-dog': 33.47735903726335}




```python
def is_dog_correct(row):
    for key,value in row.items():
        if value < row['dog-dog']:
            return 0
    return 1
```


```python
accuracy = float(dog_distances.apply(is_dog_correct).sum())/len(image_test_dog)
```


```python
accuracy
```




    0.678


