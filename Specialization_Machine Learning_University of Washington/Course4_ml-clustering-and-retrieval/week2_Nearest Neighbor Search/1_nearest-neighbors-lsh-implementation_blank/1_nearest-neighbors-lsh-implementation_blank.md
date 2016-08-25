
# Locality Sensitive Hashing

Locality Sensitive Hashing (LSH) provides for a fast, efficient approximate nearest neighbor search. The algorithm scales well with respect to the number of data points as well as dimensions.

In this assignment, you will
* Implement the LSH algorithm for approximate nearest neighbor search
* Examine the accuracy for different documents by comparing against brute force search, and also contrast runtimes
* Explore the role of the algorithm’s tuning parameters in the accuracy of the method

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import necessary packages

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
import numpy as np
import graphlab
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt
%matplotlib inline

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'

'''compute norm of a sparse vector
   Thanks to: Jaiyam Sharma'''
def norm(x):
    sum_sq=x.dot(x.T)
    norm=np.sqrt(sum_sq)
    return(norm)
```

## Load in the Wikipedia dataset
#### Download Link:https://static.dato.com/files/coursera/course-4/people_wiki.gl.zip


```python
wiki = graphlab.SFrame('people_wiki.gl/')
```

    [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1469622671.log.0
    

    This non-commercial license of GraphLab Create for academic use is assigned to last.fantasy@qq.com and will expire on July 25, 2017.
    

For this assignment, let us assign a unique ID to each document.


```python
wiki = wiki.add_row_number()
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
    </tr>
</table>
[59071 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Extract TF-IDF matrix

We first use GraphLab Create to compute a TF-IDF representation for each document.


```python
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'since':<br>1.455376717308041, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'precise':<br>6.44320060695519, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'just':<br>2.7007299687108643, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all':<br>1.6431112434912472, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'legendary':<br>4.280856294365192, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'now': 1.96695239252401,<br>'currently': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'exclusive':<br>10.455187230695827, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'taxi':<br>6.0520214560945025, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'houston':<br>3.935505942157149, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'phenomenon':<br>5.750053426395245, ...</td>
    </tr>
</table>
[59071 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



For the remainder of the assignment, we will use sparse matrices. Sparse matrices are [matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics%29 ) that have a small number of nonzero entries. A good data structure for sparse matrices would only store the nonzero entries to save space and speed up computation. SciPy provides a highly-optimized library for sparse matrices. Many matrix operations available for NumPy arrays are also available for SciPy sparse matrices.

We first convert the TF-IDF column (in dictionary format) into the SciPy sparse matrix format.


```python
def sframe_to_scipy(column):
    """ 
    Convert a dict-typed SArray into a SciPy sparse matrix.
    
    Returns
    -------
        mat : a SciPy sparse matrix where mat[i, j] is the value of word j for document i.
        mapping : a dictionary where mapping[j] is the word whose values are in column j.
    """
    # Create triples of (row_id, feature_id, count).
    x = graphlab.SFrame({'X1':column})
    
    # 1. Add a row number.
    x = x.add_row_number()
    # 2. Stack will transform x to have a row for each unique (row, key) pair.
    x = x.stack('X1', ['feature', 'value'])

    # Map words into integers using a OneHotEncoder feature transformation.
    f = graphlab.feature_engineering.OneHotEncoder(features=['feature'])

    # We first fit the transformer using the above data.
    f.fit(x)

    # The transform method will add a new column that is the transformed version
    # of the 'word' column.
    x = f.transform(x)

    # Get the feature mapping.
    mapping = f['feature_encoding']

    # Get the actual word id.
    x['feature_id'] = x['encoded_features'].dict_keys().apply(lambda x: x[0])

    # Create numpy arrays that contain the data for the sparse matrix.
    i = np.array(x['id'])
    j = np.array(x['feature_id'])
    v = np.array(x['value'])
    width = x['id'].max() + 1
    height = x['feature_id'].max() + 1

    # Create a sparse matrix.
    mat = csr_matrix((v, (i, j)), shape=(width, height))

    return mat, mapping
```

The conversion should take a few minutes to complete.


```python
start=time.time()
corpus, mapping = sframe_to_scipy(wiki['tf_idf'])
end=time.time()
print end-start
```

    70.2720000744
    

**Checkpoint**: The following code block should return 'Check passed correctly', indicating that your matrix contains TF-IDF values for 59071 documents and 547979 unique words.  Otherwise, it will return Error.


```python
assert corpus.shape == (59071, 547979)
print 'Check passed correctly!'
```

    Check passed correctly!
    

## Train an LSH model

LSH performs an efficient neighbor search by randomly partitioning all reference data points into different bins. Today we will build a popular variant of LSH known as random binary projection, which approximates cosine distance. There are other variants we could use for other choices of distance metrics.

The first step is to generate a collection of random vectors from the standard Gaussian distribution.


```python
def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)
```

To visualize these Gaussian random vectors, let's look at an example in low-dimensions.  Below, we generate 3 random vectors each of dimension 5.


```python
# Generate 3 random vectors of dimension 5, arranged into a single 5 x 3 matrix.
np.random.seed(0) # set seed=0 for consistent results
generate_random_vectors(num_vector=3, dim=5)
```




    array([[ 1.76405235,  0.40015721,  0.97873798],
           [ 2.2408932 ,  1.86755799, -0.97727788],
           [ 0.95008842, -0.15135721, -0.10321885],
           [ 0.4105985 ,  0.14404357,  1.45427351],
           [ 0.76103773,  0.12167502,  0.44386323]])



We now generate random vectors of the same dimensionality as our vocubulary size (547979).  Each vector can be used to compute one bit in the bin encoding.  We generate 16 vectors, leading to a 16-bit encoding of the bin index for each document.


```python
# Generate 16 random vectors of dimension 547979
np.random.seed(0)
random_vectors = generate_random_vectors(num_vector=16, dim=547979)
random_vectors.shape
```




    (547979L, 16L)



Next, we partition data points into bins. Instead of using explicit loops, we'd like to utilize matrix operations for greater efficiency. Let's walk through the construction step by step.

We'd like to decide which bin document 0 should go. Since 16 random vectors were generated in the previous cell, we have 16 bits to represent the bin index. The first bit is given by the sign of the dot product between the first random vector and the document's TF-IDF vector.


```python
doc = corpus[0, :] # vector of tf-idf values for document 0
doc.dot(random_vectors[:, 0]) >= 0 # True if positive sign; False if negative sign
```




    array([ True], dtype=bool)



Similarly, the second bit is computed as the sign of the dot product between the second random vector and the document vector.


```python
doc.dot(random_vectors[:, 1]) >= 0 # True if positive sign; False if negative sign
```




    array([ True], dtype=bool)



We can compute all of the bin index bits at once as follows. Note the absence of the explicit `for` loop over the 16 vectors. Matrix operations let us batch dot-product computation in a highly efficent manner, unlike the `for` loop construction. Given the relative inefficiency of loops in Python, the advantage of matrix operations is even greater.


```python
doc.dot(random_vectors) >= 0 # should return an array of 16 True/False bits
```




    array([[ True,  True, False, False, False,  True,  True, False,  True,
             True,  True, False, False,  True, False,  True]], dtype=bool)




```python
np.array(doc.dot(random_vectors) >= 0, dtype=int) # display index bits in 0/1's
```




    array([[1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]])



All documents that obtain exactly this vector will be assigned to the same bin. We'd like to repeat the identical operation on all documents in the Wikipedia dataset and compute the corresponding bin indices. Again, we use matrix operations  so that no explicit loop is needed.


```python
corpus[0:2].dot(random_vectors) >= 0 # compute bit indices of first two documents
```




    array([[ True,  True, False, False, False,  True,  True, False,  True,
             True,  True, False, False,  True, False,  True],
           [ True, False, False, False,  True,  True, False,  True,  True,
            False,  True, False,  True, False, False,  True]], dtype=bool)




```python
corpus.dot(random_vectors) >= 0 # compute bit indices of ALL documents
```




    array([[ True,  True, False, ...,  True, False,  True],
           [ True, False, False, ..., False, False,  True],
           [False,  True, False, ...,  True, False,  True],
           ..., 
           [ True,  True, False, ...,  True,  True,  True],
           [False,  True,  True, ...,  True, False,  True],
           [ True, False,  True, ..., False, False,  True]], dtype=bool)



We're almost done! To make it convenient to refer to individual bins, we convert each binary bin index into a single integer: 
```
Bin index                      integer
[0,0,0,0,0,0,0,0,0,0,0,0]   => 0
[0,0,0,0,0,0,0,0,0,0,0,1]   => 1
[0,0,0,0,0,0,0,0,0,0,1,0]   => 2
[0,0,0,0,0,0,0,0,0,0,1,1]   => 3
...
[1,1,1,1,1,1,1,1,1,1,0,0]   => 65532
[1,1,1,1,1,1,1,1,1,1,0,1]   => 65533
[1,1,1,1,1,1,1,1,1,1,1,0]   => 65534
[1,1,1,1,1,1,1,1,1,1,1,1]   => 65535 (= 2^16-1)
```
By the [rules of binary number representation](https://en.wikipedia.org/wiki/Binary_number#Decimal), we just need to compute the dot product between the document vector and the vector consisting of powers of 2:


```python
doc = corpus[0, :]  # first document
index_bits = (doc.dot(random_vectors) >= 0)
powers_of_two = (1 << np.arange(15, -1, -1))
print index_bits
print powers_of_two
print index_bits.dot(powers_of_two)
```

    [[ True  True False False False  True  True False  True  True  True False
      False  True False  True]]
    [32768 16384  8192  4096  2048  1024   512   256   128    64    32    16
         8     4     2     1]
    [50917]
    

Since it's the dot product again, we batch it with a matrix operation:


```python
index_bits = corpus.dot(random_vectors) >= 0
index_bits.dot(powers_of_two)
```




    array([50917, 36265, 19365, ..., 52983, 27589, 41449])



This array gives us the integer index of the bins for all documents.

Now we are ready to complete the following function. Given the integer bin indices for the documents, you should compile a list of document IDs that belong to each bin. Since a list is to be maintained for each unique bin index, a dictionary of lists is used.

1. Compute the integer bin indices. This step is already completed.
2. For each document in the dataset, do the following:
   * Get the integer bin index for the document.
   * Fetch the list of document ids associated with the bin; if no list yet exists for this bin, assign the bin an empty list.
   * Add the document id to the end of the list.



```python
def train_lsh(data, num_vector=16, seed=None):
    
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
  
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
  
    table = {}
    
    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)
  
    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)
    
    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = list() # YOUR CODE HERE
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(data_index)# YOUR CODE HERE

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}
    
    return model
```

**Checkpoint**. 


```python
model = train_lsh(corpus, num_vector=16, seed=143)
table = model['table']
if   0 in table and table[0]   == [39583] and \
   143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print 'Passed!'
else:
    print 'Check your code.'
```

    Passed!
    

**Note.** We will be using the model trained here in the following sections, unless otherwise indicated.

## Inspect bins

Let us look at some documents and see which bins they fall into.


```python
wiki[wiki['name'] == 'Barack Obama']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35817</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Barack_Obama&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">barack hussein obama ii<br>brk husen bm born august ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations':<br>3.811771079388818, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>



**Quiz Question**. What is the document `id` of Barack Obama's article?

**Quiz Question**. Which bin contains Barack Obama's article? Enter its integer index.


```python
print model['bin_indices'][35817] # integer format
```

    50194
    

Recall from the previous assignment that Joe Biden was a close neighbor of Barack Obama.


```python
wiki[wiki['name'] == 'Joe Biden']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24478</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Joe_Biden&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">joseph robinette joe<br>biden jr dosf rbnt badn ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'delaware':<br>11.396456717061318, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>



**Quiz Question**. Examine the bit representations of the bins containing Barack Obama and Joe Biden. In how many places do they agree?

1. 16 out of 16 places (Barack Obama and Joe Biden fall into the same bin)
2. 14 out of 16 places
3. 12 out of 16 places
4. 10 out of 16 places
5. 8 out of 16 places


```python
print np.array(model['bin_index_bits'][24478], dtype=int) # list of 0/1's
print model['bin_indices'][24478] # integer format
sum(model['bin_index_bits'][24478] == model['bin_index_bits'][35817])
```

    [1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0]
    33794
    




    14



Compare the result with a former British diplomat, whose bin representation agrees with Obama's in only 8 out of 16 places.


```python
wiki[wiki['name']=='Wynn Normington Hugh-Jones']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">22745</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Wynn_Normington_H ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Wynn Normington Hugh-<br>Jones ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sir wynn normington<br>hughjones kb sometimes ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'forced':<br>3.919175540571719, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>




```python
print np.array(model['bin_index_bits'][22745], dtype=int) # list of 0/1's
print model['bin_indices'][22745] # integer format
model['bin_index_bits'][35817] == model['bin_index_bits'][22745]
```

    [0 0 0 1 0 0 1 0 0 0 1 1 0 1 0 0]
    4660
    




    array([False, False,  True, False,  True, False, False,  True,  True,
            True, False,  True,  True, False, False,  True], dtype=bool)



How about the documents in the same bin as Barack Obama? Are they necessarily more similar to Obama than Biden?  Let's look at which documents are in the same bin as the Barack Obama article.


```python
model['table'][model['bin_indices'][35817]]
```




    [21426, 35817, 39426, 50261, 53937]



There are four other documents that belong to the same bin. Which documents are they?


```python
doc_ids = list(model['table'][model['bin_indices'][35817]])
doc_ids.remove(35817) # display documents other than Obama

docs = wiki.filter_by(values=doc_ids, column_name='id') # filter by id column
docs
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21426</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Mark_Boulware&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mark Boulware</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">mark boulware born 1948<br>is an american diplomat ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'ambassador':<br>15.90834582606623, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39426</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/John_Wells_(polit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">John Wells (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sir john julius wells<br>born 30 march 1925 is a ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'when':<br>1.3806055739282235, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">50261</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Francis_Longstaff&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Francis Longstaff</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">francis a longstaff born<br>august 3 1956 is an ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all':<br>1.6431112434912472, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">53937</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Madurai_T._Sriniv ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Madurai T. Srinivasan</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">maduraitsrinivasan is a<br>wellknown figure in the ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'alarmelu':<br>21.972990778450388, ...</td>
    </tr>
</table>
[4 rows x 5 columns]<br/>
</div>



It turns out that Joe Biden is much closer to Barack Obama than any of the four documents, even though Biden's bin representation differs from Obama's by 2 bits.


```python
def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy/(norm(x)*norm(y))
    return 1-dist[0,0]

obama_tf_idf = corpus[35817,:]
biden_tf_idf = corpus[24478,:]

print '================= Cosine distance from Barack Obama'
print 'Barack Obama - {0:24s}: {1:f}'.format('Joe Biden',
                                             cosine_distance(obama_tf_idf, biden_tf_idf))
for doc_id in doc_ids:
    doc_tf_idf = corpus[doc_id,:]
    print 'Barack Obama - {0:24s}: {1:f}'.format(wiki[doc_id]['name'],
                                                 cosine_distance(obama_tf_idf, doc_tf_idf))
```

    ================= Cosine distance from Barack Obama
    Barack Obama - Joe Biden               : 0.703139
    Barack Obama - Mark Boulware           : 0.950867
    Barack Obama - John Wells (politician) : 0.975966
    Barack Obama - Francis Longstaff       : 0.978256
    Barack Obama - Madurai T. Srinivasan   : 0.993092
    

**Moral of the story**. Similar data points will in general _tend to_ fall into _nearby_ bins, but that's all we can say about LSH. In a high-dimensional space such as text features, we often get unlucky with our selection of only a few random vectors such that dissimilar data points go into the same bin while similar data points fall into different bins. **Given a query document, we must consider all documents in the nearby bins and sort them according to their actual distances from the query.**

## Query the LSH model

Let us first implement the logic for searching nearby neighbors, which goes like this:
```
1. Let L be the bit representation of the bin that contains the query documents.
2. Consider all documents in bin L.
3. Consider documents in the bins whose bit representation differs from L by 1 bit.
4. Consider documents in the bins whose bit representation differs from L by 2 bits.
...
```

To obtain candidate bins that differ from the query bin by some number of bits, we use `itertools.combinations`, which produces all possible subsets of a given list. See [this documentation](https://docs.python.org/3/library/itertools.html#itertools.combinations) for details.
```
1. Decide on the search radius r. This will determine the number of different bits between the two vectors.
2. For each subset (n_1, n_2, ..., n_r) of the list [0, 1, 2, ..., num_vector-1], do the following:
   * Flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector.
   * Fetch the list of documents belonging to the bin indexed by the new bit vector.
   * Add those documents to the candidate set.
```

Each line of output from the following cell is a 3-tuple indicating where the candidate bin would differ from the query bin. For instance,
```
(0, 1, 3)
```
indicates that the candiate bin differs from the query bin in first, second, and fourth bits.


```python
from itertools import combinations
```


```python
num_vector = 16
search_radius = 3

for diff in combinations(range(num_vector), search_radius):
    print diff
```

    (0, 1, 2)
    (0, 1, 3)
    (0, 1, 4)
    (0, 1, 5)
    (0, 1, 6)
    (0, 1, 7)
    (0, 1, 8)
    (0, 1, 9)
    (0, 1, 10)
    (0, 1, 11)
    (0, 1, 12)
    (0, 1, 13)
    (0, 1, 14)
    (0, 1, 15)
    (0, 2, 3)
    (0, 2, 4)
    (0, 2, 5)
    (0, 2, 6)
    (0, 2, 7)
    (0, 2, 8)
    (0, 2, 9)
    (0, 2, 10)
    (0, 2, 11)
    (0, 2, 12)
    (0, 2, 13)
    (0, 2, 14)
    (0, 2, 15)
    (0, 3, 4)
    (0, 3, 5)
    (0, 3, 6)
    (0, 3, 7)
    (0, 3, 8)
    (0, 3, 9)
    (0, 3, 10)
    (0, 3, 11)
    (0, 3, 12)
    (0, 3, 13)
    (0, 3, 14)
    (0, 3, 15)
    (0, 4, 5)
    (0, 4, 6)
    (0, 4, 7)
    (0, 4, 8)
    (0, 4, 9)
    (0, 4, 10)
    (0, 4, 11)
    (0, 4, 12)
    (0, 4, 13)
    (0, 4, 14)
    (0, 4, 15)
    (0, 5, 6)
    (0, 5, 7)
    (0, 5, 8)
    (0, 5, 9)
    (0, 5, 10)
    (0, 5, 11)
    (0, 5, 12)
    (0, 5, 13)
    (0, 5, 14)
    (0, 5, 15)
    (0, 6, 7)
    (0, 6, 8)
    (0, 6, 9)
    (0, 6, 10)
    (0, 6, 11)
    (0, 6, 12)
    (0, 6, 13)
    (0, 6, 14)
    (0, 6, 15)
    (0, 7, 8)
    (0, 7, 9)
    (0, 7, 10)
    (0, 7, 11)
    (0, 7, 12)
    (0, 7, 13)
    (0, 7, 14)
    (0, 7, 15)
    (0, 8, 9)
    (0, 8, 10)
    (0, 8, 11)
    (0, 8, 12)
    (0, 8, 13)
    (0, 8, 14)
    (0, 8, 15)
    (0, 9, 10)
    (0, 9, 11)
    (0, 9, 12)
    (0, 9, 13)
    (0, 9, 14)
    (0, 9, 15)
    (0, 10, 11)
    (0, 10, 12)
    (0, 10, 13)
    (0, 10, 14)
    (0, 10, 15)
    (0, 11, 12)
    (0, 11, 13)
    (0, 11, 14)
    (0, 11, 15)
    (0, 12, 13)
    (0, 12, 14)
    (0, 12, 15)
    (0, 13, 14)
    (0, 13, 15)
    (0, 14, 15)
    (1, 2, 3)
    (1, 2, 4)
    (1, 2, 5)
    (1, 2, 6)
    (1, 2, 7)
    (1, 2, 8)
    (1, 2, 9)
    (1, 2, 10)
    (1, 2, 11)
    (1, 2, 12)
    (1, 2, 13)
    (1, 2, 14)
    (1, 2, 15)
    (1, 3, 4)
    (1, 3, 5)
    (1, 3, 6)
    (1, 3, 7)
    (1, 3, 8)
    (1, 3, 9)
    (1, 3, 10)
    (1, 3, 11)
    (1, 3, 12)
    (1, 3, 13)
    (1, 3, 14)
    (1, 3, 15)
    (1, 4, 5)
    (1, 4, 6)
    (1, 4, 7)
    (1, 4, 8)
    (1, 4, 9)
    (1, 4, 10)
    (1, 4, 11)
    (1, 4, 12)
    (1, 4, 13)
    (1, 4, 14)
    (1, 4, 15)
    (1, 5, 6)
    (1, 5, 7)
    (1, 5, 8)
    (1, 5, 9)
    (1, 5, 10)
    (1, 5, 11)
    (1, 5, 12)
    (1, 5, 13)
    (1, 5, 14)
    (1, 5, 15)
    (1, 6, 7)
    (1, 6, 8)
    (1, 6, 9)
    (1, 6, 10)
    (1, 6, 11)
    (1, 6, 12)
    (1, 6, 13)
    (1, 6, 14)
    (1, 6, 15)
    (1, 7, 8)
    (1, 7, 9)
    (1, 7, 10)
    (1, 7, 11)
    (1, 7, 12)
    (1, 7, 13)
    (1, 7, 14)
    (1, 7, 15)
    (1, 8, 9)
    (1, 8, 10)
    (1, 8, 11)
    (1, 8, 12)
    (1, 8, 13)
    (1, 8, 14)
    (1, 8, 15)
    (1, 9, 10)
    (1, 9, 11)
    (1, 9, 12)
    (1, 9, 13)
    (1, 9, 14)
    (1, 9, 15)
    (1, 10, 11)
    (1, 10, 12)
    (1, 10, 13)
    (1, 10, 14)
    (1, 10, 15)
    (1, 11, 12)
    (1, 11, 13)
    (1, 11, 14)
    (1, 11, 15)
    (1, 12, 13)
    (1, 12, 14)
    (1, 12, 15)
    (1, 13, 14)
    (1, 13, 15)
    (1, 14, 15)
    (2, 3, 4)
    (2, 3, 5)
    (2, 3, 6)
    (2, 3, 7)
    (2, 3, 8)
    (2, 3, 9)
    (2, 3, 10)
    (2, 3, 11)
    (2, 3, 12)
    (2, 3, 13)
    (2, 3, 14)
    (2, 3, 15)
    (2, 4, 5)
    (2, 4, 6)
    (2, 4, 7)
    (2, 4, 8)
    (2, 4, 9)
    (2, 4, 10)
    (2, 4, 11)
    (2, 4, 12)
    (2, 4, 13)
    (2, 4, 14)
    (2, 4, 15)
    (2, 5, 6)
    (2, 5, 7)
    (2, 5, 8)
    (2, 5, 9)
    (2, 5, 10)
    (2, 5, 11)
    (2, 5, 12)
    (2, 5, 13)
    (2, 5, 14)
    (2, 5, 15)
    (2, 6, 7)
    (2, 6, 8)
    (2, 6, 9)
    (2, 6, 10)
    (2, 6, 11)
    (2, 6, 12)
    (2, 6, 13)
    (2, 6, 14)
    (2, 6, 15)
    (2, 7, 8)
    (2, 7, 9)
    (2, 7, 10)
    (2, 7, 11)
    (2, 7, 12)
    (2, 7, 13)
    (2, 7, 14)
    (2, 7, 15)
    (2, 8, 9)
    (2, 8, 10)
    (2, 8, 11)
    (2, 8, 12)
    (2, 8, 13)
    (2, 8, 14)
    (2, 8, 15)
    (2, 9, 10)
    (2, 9, 11)
    (2, 9, 12)
    (2, 9, 13)
    (2, 9, 14)
    (2, 9, 15)
    (2, 10, 11)
    (2, 10, 12)
    (2, 10, 13)
    (2, 10, 14)
    (2, 10, 15)
    (2, 11, 12)
    (2, 11, 13)
    (2, 11, 14)
    (2, 11, 15)
    (2, 12, 13)
    (2, 12, 14)
    (2, 12, 15)
    (2, 13, 14)
    (2, 13, 15)
    (2, 14, 15)
    (3, 4, 5)
    (3, 4, 6)
    (3, 4, 7)
    (3, 4, 8)
    (3, 4, 9)
    (3, 4, 10)
    (3, 4, 11)
    (3, 4, 12)
    (3, 4, 13)
    (3, 4, 14)
    (3, 4, 15)
    (3, 5, 6)
    (3, 5, 7)
    (3, 5, 8)
    (3, 5, 9)
    (3, 5, 10)
    (3, 5, 11)
    (3, 5, 12)
    (3, 5, 13)
    (3, 5, 14)
    (3, 5, 15)
    (3, 6, 7)
    (3, 6, 8)
    (3, 6, 9)
    (3, 6, 10)
    (3, 6, 11)
    (3, 6, 12)
    (3, 6, 13)
    (3, 6, 14)
    (3, 6, 15)
    (3, 7, 8)
    (3, 7, 9)
    (3, 7, 10)
    (3, 7, 11)
    (3, 7, 12)
    (3, 7, 13)
    (3, 7, 14)
    (3, 7, 15)
    (3, 8, 9)
    (3, 8, 10)
    (3, 8, 11)
    (3, 8, 12)
    (3, 8, 13)
    (3, 8, 14)
    (3, 8, 15)
    (3, 9, 10)
    (3, 9, 11)
    (3, 9, 12)
    (3, 9, 13)
    (3, 9, 14)
    (3, 9, 15)
    (3, 10, 11)
    (3, 10, 12)
    (3, 10, 13)
    (3, 10, 14)
    (3, 10, 15)
    (3, 11, 12)
    (3, 11, 13)
    (3, 11, 14)
    (3, 11, 15)
    (3, 12, 13)
    (3, 12, 14)
    (3, 12, 15)
    (3, 13, 14)
    (3, 13, 15)
    (3, 14, 15)
    (4, 5, 6)
    (4, 5, 7)
    (4, 5, 8)
    (4, 5, 9)
    (4, 5, 10)
    (4, 5, 11)
    (4, 5, 12)
    (4, 5, 13)
    (4, 5, 14)
    (4, 5, 15)
    (4, 6, 7)
    (4, 6, 8)
    (4, 6, 9)
    (4, 6, 10)
    (4, 6, 11)
    (4, 6, 12)
    (4, 6, 13)
    (4, 6, 14)
    (4, 6, 15)
    (4, 7, 8)
    (4, 7, 9)
    (4, 7, 10)
    (4, 7, 11)
    (4, 7, 12)
    (4, 7, 13)
    (4, 7, 14)
    (4, 7, 15)
    (4, 8, 9)
    (4, 8, 10)
    (4, 8, 11)
    (4, 8, 12)
    (4, 8, 13)
    (4, 8, 14)
    (4, 8, 15)
    (4, 9, 10)
    (4, 9, 11)
    (4, 9, 12)
    (4, 9, 13)
    (4, 9, 14)
    (4, 9, 15)
    (4, 10, 11)
    (4, 10, 12)
    (4, 10, 13)
    (4, 10, 14)
    (4, 10, 15)
    (4, 11, 12)
    (4, 11, 13)
    (4, 11, 14)
    (4, 11, 15)
    (4, 12, 13)
    (4, 12, 14)
    (4, 12, 15)
    (4, 13, 14)
    (4, 13, 15)
    (4, 14, 15)
    (5, 6, 7)
    (5, 6, 8)
    (5, 6, 9)
    (5, 6, 10)
    (5, 6, 11)
    (5, 6, 12)
    (5, 6, 13)
    (5, 6, 14)
    (5, 6, 15)
    (5, 7, 8)
    (5, 7, 9)
    (5, 7, 10)
    (5, 7, 11)
    (5, 7, 12)
    (5, 7, 13)
    (5, 7, 14)
    (5, 7, 15)
    (5, 8, 9)
    (5, 8, 10)
    (5, 8, 11)
    (5, 8, 12)
    (5, 8, 13)
    (5, 8, 14)
    (5, 8, 15)
    (5, 9, 10)
    (5, 9, 11)
    (5, 9, 12)
    (5, 9, 13)
    (5, 9, 14)
    (5, 9, 15)
    (5, 10, 11)
    (5, 10, 12)
    (5, 10, 13)
    (5, 10, 14)
    (5, 10, 15)
    (5, 11, 12)
    (5, 11, 13)
    (5, 11, 14)
    (5, 11, 15)
    (5, 12, 13)
    (5, 12, 14)
    (5, 12, 15)
    (5, 13, 14)
    (5, 13, 15)
    (5, 14, 15)
    (6, 7, 8)
    (6, 7, 9)
    (6, 7, 10)
    (6, 7, 11)
    (6, 7, 12)
    (6, 7, 13)
    (6, 7, 14)
    (6, 7, 15)
    (6, 8, 9)
    (6, 8, 10)
    (6, 8, 11)
    (6, 8, 12)
    (6, 8, 13)
    (6, 8, 14)
    (6, 8, 15)
    (6, 9, 10)
    (6, 9, 11)
    (6, 9, 12)
    (6, 9, 13)
    (6, 9, 14)
    (6, 9, 15)
    (6, 10, 11)
    (6, 10, 12)
    (6, 10, 13)
    (6, 10, 14)
    (6, 10, 15)
    (6, 11, 12)
    (6, 11, 13)
    (6, 11, 14)
    (6, 11, 15)
    (6, 12, 13)
    (6, 12, 14)
    (6, 12, 15)
    (6, 13, 14)
    (6, 13, 15)
    (6, 14, 15)
    (7, 8, 9)
    (7, 8, 10)
    (7, 8, 11)
    (7, 8, 12)
    (7, 8, 13)
    (7, 8, 14)
    (7, 8, 15)
    (7, 9, 10)
    (7, 9, 11)
    (7, 9, 12)
    (7, 9, 13)
    (7, 9, 14)
    (7, 9, 15)
    (7, 10, 11)
    (7, 10, 12)
    (7, 10, 13)
    (7, 10, 14)
    (7, 10, 15)
    (7, 11, 12)
    (7, 11, 13)
    (7, 11, 14)
    (7, 11, 15)
    (7, 12, 13)
    (7, 12, 14)
    (7, 12, 15)
    (7, 13, 14)
    (7, 13, 15)
    (7, 14, 15)
    (8, 9, 10)
    (8, 9, 11)
    (8, 9, 12)
    (8, 9, 13)
    (8, 9, 14)
    (8, 9, 15)
    (8, 10, 11)
    (8, 10, 12)
    (8, 10, 13)
    (8, 10, 14)
    (8, 10, 15)
    (8, 11, 12)
    (8, 11, 13)
    (8, 11, 14)
    (8, 11, 15)
    (8, 12, 13)
    (8, 12, 14)
    (8, 12, 15)
    (8, 13, 14)
    (8, 13, 15)
    (8, 14, 15)
    (9, 10, 11)
    (9, 10, 12)
    (9, 10, 13)
    (9, 10, 14)
    (9, 10, 15)
    (9, 11, 12)
    (9, 11, 13)
    (9, 11, 14)
    (9, 11, 15)
    (9, 12, 13)
    (9, 12, 14)
    (9, 12, 15)
    (9, 13, 14)
    (9, 13, 15)
    (9, 14, 15)
    (10, 11, 12)
    (10, 11, 13)
    (10, 11, 14)
    (10, 11, 15)
    (10, 12, 13)
    (10, 12, 14)
    (10, 12, 15)
    (10, 13, 14)
    (10, 13, 15)
    (10, 14, 15)
    (11, 12, 13)
    (11, 12, 14)
    (11, 12, 15)
    (11, 13, 14)
    (11, 13, 15)
    (11, 14, 15)
    (12, 13, 14)
    (12, 13, 15)
    (12, 14, 15)
    (13, 14, 15)
    

With this output in mind, implement the logic for nearby bin search:


```python
def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.
    
    Example usage
    -------------
    >>> model = train_lsh(corpus, num_vector=16, seed=143)
    >>> q = model['bin_index_bits'][0]  # vector for the first document
  
    >>> candidates = search_nearby_bins(q, model['table'])
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
    
    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)
    
    for different_bits in combinations(range(num_vector), search_radius):       
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        ## Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = 1-alternate_bits[0] # YOUR CODE HERE 
        
        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)
        
        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])# YOUR CODE HERE: Update candidate_set with the documents in this bin.
            
    return candidate_set
```

**Checkpoint**. Running the function with `search_radius=0` should yield the list of documents belonging to the same bin as the query.


```python
obama_bin_index = model['bin_index_bits'][35817] # bin index of Barack Obama
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0)
if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
    print 'Passed test'
else:
    print 'Check your code'
print 'List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261'
```

    Passed test
    List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261
    

**Checkpoint**. Running the function with `search_radius=1` adds more documents to the fore.


```python
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1, initial_candidates=candidate_set)
if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                         23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                         19699, 2804, 20347]):
    print 'Passed test'
else:
    print 'Check your code'
```

    Check your code
    

**Note**. Don't be surprised if few of the candidates look similar to Obama. This is why we add as many candidates as our computational budget allows and sort them by their distance to the query.

Now we have a function that can return all the candidates from neighboring bins. Next we write a function to collect all candidates and compute their true distance to the query.


```python
def query(vec, model, k, max_search_radius):
  
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]
    
    
    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
    
    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in xrange(max_search_radius+1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
    
    # Sort candidates by their true distances from the query
    nearest_neighbors = graphlab.SFrame({'id':candidate_set})
    candidates = data[np.array(list(candidate_set)),:]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()
    
    return nearest_neighbors.topk('distance', k, reverse=True), len(candidate_set)
```

Let's try it out with Obama:


```python
query(corpus[35817,:], model, k=10, max_search_radius=3)
```




    (Columns:
     	id	int
     	distance	float
     
     Rows: 10
     
     Data:
     +-------+--------------------+
     |   id  |      distance      |
     +-------+--------------------+
     | 35817 | -6.66133814775e-16 |
     | 24478 |   0.703138676734   |
     | 10121 |   0.936896394645   |
     | 41631 |   0.947459482005   |
     | 25877 |   0.94941844741    |
     | 30761 |   0.949852430281   |
     | 21426 |   0.950866757525   |
     |  3179 |   0.951113382767   |
     | 56558 |   0.951551450997   |
     | 33243 |   0.951765770113   |
     +-------+--------------------+
     [10 rows x 2 columns], 70)



To identify the documents, it's helpful to join this table with the Wikipedia table:


```python
query(corpus[35817,:], model, k=10, max_search_radius=3)[0].join(wiki[['id', 'name']], on='id').sort('distance')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35817</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-6.66133814775e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24478</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.703138676734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10121</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.936896394645</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bill Bradley</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41631</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.947459482005</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Binayak Sen</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">25877</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.94941844741</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Kris Hanna</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30761</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.949852430281</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">John Poda</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21426</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.950866757525</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mark Boulware</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3179</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.951113382767</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cyril Keeper</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">56558</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.951551450997</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Elizabeth Weir</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33243</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.951765770113</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Janice Lachance</td>
    </tr>
</table>
[10 rows x 3 columns]<br/>
</div>



We have shown that we have a working LSH implementation!

# Experimenting with your LSH implementation

In the following sections we have implemented a few experiments so that you can gain intuition for how your LSH implementation behaves in different situations. This will help you understand the effect of searching nearby bins and the performance of LSH versus computing nearest neighbors using a brute force search.

## Effect of nearby bin search

How does nearby bin search affect the outcome of LSH? There are three variables that are affected by the search radius:
* Number of candidate documents considered
* Query time
* Distance of approximate neighbors from the query

Let us run LSH multiple times, each with different radii for nearby bin search. We will measure the three variables as discussed above.


```python
wiki[wiki['name']=='Barack Obama']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35817</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Barack_Obama&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">barack hussein obama ii<br>brk husen bm born august ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations':<br>3.811771079388818, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>




```python
num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in xrange(17):
    start=time.time()
    result, num_candidates = query(corpus[35817,:], model, k=10,
                                   max_search_radius=max_search_radius)
    end=time.time()
    query_time = end-start
    
    print 'Radius:', max_search_radius
    print result.join(wiki[['id', 'name']], on='id').sort('distance')
    
    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()
    
    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)
```

    Radius: 0
    +-------+--------------------+-------------------------+
    |   id  |      distance      |           name          |
    +-------+--------------------+-------------------------+
    | 35817 | -6.66133814775e-16 |       Barack Obama      |
    | 21426 |   0.950866757525   |      Mark Boulware      |
    | 39426 |   0.97596600411    | John Wells (politician) |
    | 50261 |   0.978256163041   |    Francis Longstaff    |
    | 53937 |   0.993092148424   |  Madurai T. Srinivasan  |
    +-------+--------------------+-------------------------+
    [5 rows x 3 columns]
    
    Radius: 1
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 41631 |   0.947459482005   |          Binayak Sen          |
    | 21426 |   0.950866757525   |         Mark Boulware         |
    | 33243 |   0.951765770113   |        Janice Lachance        |
    | 39426 |   0.97596600411    |    John Wells (politician)    |
    | 34547 |   0.978214931987   | Nathan Murphy (Australian ... |
    | 50261 |   0.978256163041   |       Francis Longstaff       |
    | 27404 |   0.985798641463   |      Michael T. Ringwood      |
    | 21715 |   0.992012225904   |     Alexander Ruud Tveter     |
    | 53937 |   0.993092148424   |     Madurai T. Srinivasan     |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 2
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 10121 |   0.936896394645   |          Bill Bradley         |
    | 41631 |   0.947459482005   |          Binayak Sen          |
    | 21426 |   0.950866757525   |         Mark Boulware         |
    | 33243 |   0.951765770113   |        Janice Lachance        |
    | 14986 |   0.953127798343   |          Glenn Sterle         |
    | 15099 |   0.963442597024   | Scott Barrett (political s... |
    | 12171 |   0.970076463118   |      Daniel Karam Toumeh      |
    | 39426 |   0.97596600411    |    John Wells (politician)    |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 3
    +-------+--------------------+-----------------+
    |   id  |      distance      |       name      |
    +-------+--------------------+-----------------+
    | 35817 | -6.66133814775e-16 |   Barack Obama  |
    | 24478 |   0.703138676734   |    Joe Biden    |
    | 10121 |   0.936896394645   |   Bill Bradley  |
    | 41631 |   0.947459482005   |   Binayak Sen   |
    | 25877 |   0.94941844741    |    Kris Hanna   |
    | 30761 |   0.949852430281   |    John Poda    |
    | 21426 |   0.950866757525   |  Mark Boulware  |
    |  3179 |   0.951113382767   |   Cyril Keeper  |
    | 56558 |   0.951551450997   |  Elizabeth Weir |
    | 33243 |   0.951765770113   | Janice Lachance |
    +-------+--------------------+-----------------+
    [10 rows x 3 columns]
    
    Radius: 4
    +-------+--------------------+--------------------------+
    |   id  |      distance      |           name           |
    +-------+--------------------+--------------------------+
    | 35817 | -6.66133814775e-16 |       Barack Obama       |
    | 24478 |   0.703138676734   |        Joe Biden         |
    | 56044 |   0.897604308082   |         Jon Kyl          |
    |  2516 |   0.908683175249   |      John Kitzhaber      |
    | 10121 |   0.936896394645   |       Bill Bradley       |
    | 50471 |   0.940035687215   |   Dick Clark (senator)   |
    | 40045 |   0.940391971267   |       Ernesto Sanz       |
    | 14526 |   0.945366391287   |       Dan Morrish        |
    | 21127 |   0.945522013021   | Thomas Demetrios Lambros |
    | 45715 |   0.947378941769   |       Joseph Bruno       |
    +-------+--------------------+--------------------------+
    [10 rows x 3 columns]
    
    Radius: 5
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    | 56044 |   0.897604308082   |            Jon Kyl            |
    |  2516 |   0.908683175249   |         John Kitzhaber        |
    | 32196 |   0.911385398406   |   Scott Fraser (politician)   |
    | 37399 |   0.919222822987   |         Maggie Hassan         |
    | 44927 |   0.919571596777   |         Donald Lively         |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 6
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 7
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 8
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 9
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 10
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 11
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 12
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 13
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 14
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 15
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 16
    +-------+--------------------+-------------------------------+
    |   id  |      distance      |              name             |
    +-------+--------------------+-------------------------------+
    | 35817 | -6.66133814775e-16 |          Barack Obama         |
    | 24478 |   0.703138676734   |           Joe Biden           |
    | 16155 |   0.849858405926   | Bob Smith (American politi... |
    | 19656 |   0.871604079645   |          Cleo Fields          |
    | 13509 |   0.884400857817   |         James S. Bubar        |
    | 16989 |   0.886460907515   |            Ted Cruz           |
    | 44157 |   0.888341185152   |        Joan Fitz-Gerald       |
    | 46179 |   0.893085421534   |          Mike Gravel          |
    | 31307 |   0.895249425764   |           Dave Kopel          |
    |  4274 |   0.897229911127   |       Stephen Murphy III      |
    +-------+--------------------+-------------------------------+
    [10 rows x 3 columns]
    
    

Notice that the top 10 query results become more relevant as the search radius grows. Let's plot the three variables:


```python
plt.figure(figsize=(7,4.5))
plt.plot(num_candidates_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('# of documents searched')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(query_time_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output_91_0.png)



![png](output_91_1.png)



![png](output_91_2.png)


Some observations:
* As we increase the search radius, we find more neighbors that are a smaller distance away.
* With increased search radius comes a greater number documents that have to be searched. Query time is higher as a consequence.
* With sufficiently high search radius, the results of LSH begin to resemble the results of brute-force search.

**Quiz Question**. What was the smallest search radius that yielded the correct nearest neighbor, namely Joe Biden?


**Quiz Question**. Suppose our goal was to produce 10 approximate nearest neighbors whose average distance from the query document is within 0.01 of the average for the true 10 nearest neighbors. For Barack Obama, the true 10 nearest neighbors are on average about 0.77. What was the smallest search radius for Barack Obama that produced an average distance of 0.78 or better?


```python
7
```




    7



## Quality metrics for neighbors

The above analysis is limited by the fact that it was run with a single query, namely Barack Obama. We should repeat the analysis for the entirety of data. Iterating over all documents would take a long time, so let us randomly choose 10 documents for our analysis.

For each document, we first compute the true 25 nearest neighbors, and then run LSH multiple times. We look at two metrics:

* Precision@10: How many of the 10 neighbors given by LSH are among the true 25 nearest neighbors?
* Average cosine distance of the neighbors from the query

Then we run LSH multiple times with different search radii.


```python
def brute_force_query(vec, data, k):
    num_data_points = data.shape[0]
    
    # Compute distances for ALL data points in training set
    nearest_neighbors = graphlab.SFrame({'id':range(num_data_points)})
    nearest_neighbors['distance'] = pairwise_distances(data, vec, metric='cosine').flatten()
    
    return nearest_neighbors.topk('distance', k, reverse=True)
```

The following cell will run LSH with multiple search radii and compute the quality metrics for each run. Allow a few minutes to complete.


```python
max_radius = 17
precision = {i:[] for i in xrange(max_radius)}
average_distance  = {i:[] for i in xrange(max_radius)}
query_time  = {i:[] for i in xrange(max_radius)}

np.random.seed(0)
num_queries = 10
for i, ix in enumerate(np.random.choice(corpus.shape[0], num_queries, replace=False)):
    print('%s / %s' % (i, num_queries))
    ground_truth = set(brute_force_query(corpus[ix,:], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors
    
    for r in xrange(1,max_radius):
        start = time.time()
        result, num_candidates = query(corpus[ix,:], model, k=10, max_search_radius=r)
        end = time.time()

        query_time[r].append(end-start)
        # precision = (# of neighbors both in result and ground_truth)/10.0
        precision[r].append(len(set(result['id']) & ground_truth)/10.0)
        average_distance[r].append(result['distance'][1:].mean())
```

    0 / 10
    1 / 10
    2 / 10
    3 / 10
    4 / 10
    5 / 10
    6 / 10
    7 / 10
    8 / 10
    9 / 10
    


```python
plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(average_distance[i]) for i in xrange(1,17)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(precision[i]) for i in xrange(1,17)], linewidth=4, label='Precison@10')
plt.xlabel('Search radius')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(query_time[i]) for i in xrange(1,17)], linewidth=4, label='Query time')
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output_100_0.png)



![png](output_100_1.png)



![png](output_100_2.png)


The observations for Barack Obama generalize to the entire dataset.

## Effect of number of random vectors

Let us now turn our focus to the remaining parameter: the number of random vectors. We run LSH with different number of random vectors, ranging from 5 to 20. We fix the search radius to 3.

Allow a few minutes for the following cell to complete.


```python
precision = {i:[] for i in xrange(5,20)}
average_distance  = {i:[] for i in xrange(5,20)}
query_time = {i:[] for i in xrange(5,20)}
num_candidates_history = {i:[] for i in xrange(5,20)}
ground_truth = {}

np.random.seed(0)
num_queries = 10
docs = np.random.choice(corpus.shape[0], num_queries, replace=False)

for i, ix in enumerate(docs):
    ground_truth[ix] = set(brute_force_query(corpus[ix,:], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors

for num_vector in xrange(5,20):
    print('num_vector = %s' % (num_vector))
    model = train_lsh(corpus, num_vector, seed=143)
    
    for i, ix in enumerate(docs):
        start = time.time()
        result, num_candidates = query(corpus[ix,:], model, k=10, max_search_radius=3)
        end = time.time()
        
        query_time[num_vector].append(end-start)
        precision[num_vector].append(len(set(result['id']) & ground_truth[ix])/10.0)
        average_distance[num_vector].append(result['distance'][1:].mean())
        num_candidates_history[num_vector].append(num_candidates)
```

    num_vector = 5
    num_vector = 6
    num_vector = 7
    num_vector = 8
    num_vector = 9
    num_vector = 10
    num_vector = 11
    num_vector = 12
    num_vector = 13
    num_vector = 14
    num_vector = 15
    num_vector = 16
    num_vector = 17
    num_vector = 18
    num_vector = 19
    


```python
plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(average_distance[i]) for i in xrange(5,20)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('# of random vectors')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(precision[i]) for i in xrange(5,20)], linewidth=4, label='Precison@10')
plt.xlabel('# of random vectors')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(query_time[i]) for i in xrange(5,20)], linewidth=4, label='Query time (seconds)')
plt.xlabel('# of random vectors')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(num_candidates_history[i]) for i in xrange(5,20)], linewidth=4,
         label='# of documents searched')
plt.xlabel('# of random vectors')
plt.ylabel('# of documents searched')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output_105_0.png)



![png](output_105_1.png)



![png](output_105_2.png)



![png](output_105_3.png)


We see a similar trade-off between quality and performance: as the number of random vectors increases, the query time goes down as each bin contains fewer documents on average, but on average the neighbors are likewise placed farther from the query. On the other hand, when using a small enough number of random vectors, LSH becomes very similar brute-force search: Many documents appear in a single bin, so searching the query bin alone covers a lot of the corpus; then, including neighboring bins might result in searching all documents, just as in the brute-force approach.


```python

```
