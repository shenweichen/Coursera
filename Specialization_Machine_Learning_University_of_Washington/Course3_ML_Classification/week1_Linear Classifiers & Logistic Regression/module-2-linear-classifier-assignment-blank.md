
# Predicting sentiment from product reviews


The goal of this first notebook is to explore logistic regression and feature engineering with existing GraphLab functions.

In this notebook you will use product review data from Amazon.com to predict whether the sentiments about a product (from its reviews) are positive or negative.

* Use SFrames to do some feature engineering
* Train a logistic regression model to predict the sentiment of product reviews.
* Inspect the weights (coefficients) of a trained logistic regression model.
* Make a prediction (both class and probability) of sentiment for a new product review.
* Given the logistic regression weights, predictors and ground truth labels, write a function to compute the **accuracy** of the model.
* Inspect the coefficients of the logistic regression model and interpret their meanings.
* Compare multiple logistic regression models.

Let's get started!
    
## Fire up GraphLab Create

Make sure you have the latest version of GraphLab Create.


```python
from __future__ import division
import graphlab
import math
import string
```

    A newer version of GraphLab Create (v2.0.1) is available! Your current version is v1.10.1.
    
    You can use pip to upgrade the graphlab-create package. For more information see https://dato.com/products/create/upgrade.
    

# Data preperation

We will use a dataset consisting of baby product reviews on Amazon.com.


Download Link:https://s3.amazonaws.com/static.dato.com/files/coursera/course-3/amazon_baby.gl.zip


```python
products = graphlab.SFrame('amazon_baby.gl/')
```

    This trial license of GraphLab Create is assigned to 519589356@qq.com and will expire on July 24, 2016. Please contact trial@turi.com for licensing options or to request a free non-commercial license for academic use.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v1.10.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1468467786.log.0
    

Now, let us see a preview of what the dataset looks like.


```python
products
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Flannel Wipes</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">These flannel wipes are<br>OK, but in my opinion ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Wipe Pouch</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">it came early and was not<br>disappointed. i love ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Annas Dream Full Quilt<br>with 2 Shams ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Very soft and comfortable<br>and warmer than it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This is a product well<br>worth the purchase.  I ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">All of my kids have cried<br>non-stop when I tried to ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">When the Binky Fairy came<br>to our house, we didn't ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A Tale of Baby's Days<br>with Peter Rabbit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lovely book, it's bound<br>tightly so you may no ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Perfect for new parents.<br>We were able to keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A friend of mine pinned<br>this product on Pinte ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
    </tr>
</table>
[183531 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Build the word count vector for each review

Let us explore a specific example of a baby product.



```python
products[269]
```




    {'name': 'The First Years Massaging Action Teether',
     'rating': 5.0,
     'review': 'A favorite in our house!'}



Now, we will perform 2 simple data transformations:

1. Remove punctuation using [Python's built-in](https://docs.python.org/2/library/string.html) string functionality.
2. Transform the reviews into word-counts.

**Aside**. In this notebook, we remove all punctuations for the sake of simplicity. A smarter approach to punctuations would preserve phrases such as "I'd", "would've", "hadn't" and so forth. See [this page](https://www.cis.upenn.edu/~treebank/tokenization.html) for an example of smart handling of punctuations.


```python
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

review_without_puctuation = products['review'].apply(remove_punctuation)
products['word_count'] = graphlab.text_analytics.count_words(review_without_puctuation)
```

Now, let us explore what the sample example above looks like after these 2 transformations. Here, each entry in the **word_count** column is a dictionary where the key is the word and the value is a count of the number of times the word occurs.


```python
products[269]['word_count']
```




    {'a': 1L, 'favorite': 1L, 'house': 1L, 'in': 1L, 'our': 1L}



## Extract sentiments

We will **ignore** all reviews with *rating = 3*, since they tend to have a neutral sentiment.


```python
products = products[products['rating'] != 3]
len(products)
```




    166752



Now, we will assign reviews with a rating of 4 or higher to be *positive* reviews, while the ones with rating of 2 or lower are *negative*. For the sentiment column, we use +1 for the positive class label and -1 for the negative class label.


```python
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Wipe Pouch</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">it came early and was not<br>disappointed. i love ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'love': 1L,<br>'it': 3L, 'highly': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Annas Dream Full Quilt<br>with 2 Shams ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Very soft and comfortable<br>and warmer than it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'quilt': 1L,<br>'it': 1L, 'comfortable': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This is a product well<br>worth the purchase.  I ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'ingenious':<br>1L, 'love': 2L, 'is': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">All of my kids have cried<br>non-stop when I tried to ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'all': 2L,<br>'help': 1L, 'cried': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">When the Binky Fairy came<br>to our house, we didn't ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'cute': 1L,<br>'help': 2L, 'habit': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A Tale of Baby's Days<br>with Peter Rabbit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lovely book, it's bound<br>tightly so you may no ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'shop': 1L, 'be': 1L,<br>'is': 1L, 'bound': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Perfect for new parents.<br>We were able to keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'all': 1L,<br>'right': 1L, 'able': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A friend of mine pinned<br>this product on Pinte ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'fantastic':<br>1L, 'help': 1L, 'give': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'standarad':<br>1L, 'another': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this journal and<br>our nanny uses it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 2L, 'nannys': 1L,<br>'just': 1L, 'sleep': 2L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[166752 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Now, we can see that the dataset contains an extra column called **sentiment** which is either positive (+1) or negative (-1).

## Split data into training and test sets

Let's perform a train/test split with 80% of the data in the training set and 20% of the data in the test set. We use `seed=1` so that everyone gets the same result.


```python
train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)
```

    133416
    33336
    

# Train a sentiment classifier with logistic regression

We will now use logistic regression to create a sentiment classifier on the training data. This model will use the column **word_count** as a feature and the column **sentiment** as the target. We will use `validation_set=None` to obtain same results as everyone else.

**Note:** This line may take 1-2 minutes.


```python
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)
```


<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 133416</pre>



<pre>Number of classes           : 2</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 121712</pre>



<pre>Number of coefficients    : 121713</pre>



<pre>Starting L-BFGS</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training-accuracy |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>| 1         | 5        | 0.000002  | 2.206724     | 0.840754          |</pre>



<pre>| 2         | 9        | 3.000000  | 3.226436     | 0.931350          |</pre>



<pre>| 3         | 10       | 3.000000  | 3.626501     | 0.882046          |</pre>



<pre>| 4         | 11       | 3.000000  | 3.983450     | 0.954076          |</pre>



<pre>| 5         | 12       | 3.000000  | 4.320345     | 0.960964          |</pre>



<pre>| 6         | 13       | 3.000000  | 4.663678     | 0.975033          |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>TERMINATED: Terminated due to numerical difficulties.</pre>



<pre>This model may not be ideal. To improve it, consider doing one of the following:
(a) Increasing the regularization.
(b) Standardizing the input data.
(c) Removing highly correlated features.
(d) Removing `inf` and `NaN` values in the training data.</pre>



```python
sentiment_model
```




    Class                         : LogisticClassifier
    
    Schema
    ------
    Number of coefficients        : 121713
    Number of examples            : 133416
    Number of classes             : 2
    Number of feature columns     : 1
    Number of unpacked features   : 121712
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.01
    
    Training Summary
    ----------------
    Solver                        : lbfgs
    Solver iterations             : 6
    Solver status                 : TERMINATED: Terminated due to numerical difficulties.
    Training time (sec)           : 4.9956
    
    Settings
    --------
    Log-likelihood                : inf
    
    Highest Positive Coefficients
    -----------------------------
    word_count[mobileupdate]      : 41.9847
    word_count[placeid]           : 41.7354
    word_count[labelbox]          : 41.151
    word_count[httpwwwamazoncomreviewrhgg6qp7tdnhbrefcmcrprcmtieutf8asinb00318cla0nodeid]: 40.0454
    word_count[knobskeeping]      : 36.2091
    
    Lowest Negative Coefficients
    ----------------------------
    word_count[probelm]           : -44.9283
    word_count[impulsejeep]       : -43.081
    word_count[infantsyoung]      : -39.5945
    word_count[cutereditafter]    : -35.6875
    word_count[avacado]           : -35.0542



**Aside**. You may get an warning to the effect of "Terminated due to numerical difficulties --- this model may not be ideal". It means that the quality metric (to be covered in Module 3) failed to improve in the last iteration of the run. The difficulty arises as the sentiment model puts too much weight on extremely rare words. A way to rectify this is to apply regularization, to be covered in Module 4. Regularization lessens the effect of extremely rare words. For the purpose of this assignment, however, please proceed with the model above.

Now that we have fitted the model, we can extract the weights (coefficients) as an SFrame as follows:


```python
weights = sentiment_model.coefficients
weights.column_names()
```




    ['name', 'index', 'class', 'value', 'stderr']



There are a total of `121713` coefficients in the model. Recall from the lecture that positive weights $w_j$ correspond to weights that cause positive sentiment, while negative weights correspond to negative sentiment. 

Fill in the following block of code to calculate how many *weights* are positive ( >= 0). (**Hint**: The `'value'` column in SFrame *weights* must be positive ( >= 0)).


```python
num_positive_weights = len(weights[weights['value']>=0])
num_negative_weights = len(weights[weights['value']<0])

print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights
```

    Number of positive weights: 68419 
    Number of negative weights: 53294 
    

**Quiz question:** How many weights are >= 0?

## Making predictions with logistic regression

Now that a model is trained, we can make predictions on the **test data**. In this section, we will explore this in the context of 3 examples in the test dataset.  We refer to this set of 3 examples as the **sample_test_data**.


```python
sample_test_data = test_data[10:13]
print sample_test_data['rating']
sample_test_data
```

    [5.0, 2.0, 1.0]
    




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Our Baby Girl Memory Book</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Absolutely love it and<br>all of the Scripture in ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'all': 1L,<br>'love': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Wall Decor Removable<br>Decal Sticker - Colorful ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Would not purchase again<br>or recommend. The decals ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'wall': 1L,<br>'them': 1L, 'decals': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">New Style Trailing Cherry<br>Blossom Tree Decal ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Was so excited to get<br>this product for my baby ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'money': 1L,<br>'into': 1L, 'it': 3L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
</table>
[3 rows x 5 columns]<br/>
</div>



Let's dig deeper into the first row of the **sample_test_data**. Here's the full review:


```python
sample_test_data[0]['review']
```




    'Absolutely love it and all of the Scripture in it.  I purchased the Baby Boy version for my grandson when he was born and my daughter-in-law was thrilled to receive the same book again.'



That review seems pretty positive.

Now, let's see what the next row of the **sample_test_data** looks like. As we could guess from the sentiment (-1), the review is quite negative.


```python
sample_test_data[1]['review']
```




    'Would not purchase again or recommend. The decals were thick almost plastic like and were coming off the wall as I was applying them! The would NOT stick! Literally stayed stuck for about 5 minutes then started peeling off.'



We will now make a **class** prediction for the **sample_test_data**. The `sentiment_model` should predict **+1** if the sentiment is positive and **-1** if the sentiment is negative. Recall from the lecture that the **score** (sometimes called **margin**) for the logistic regression model  is defined as:

$$
\mbox{score}_i = \mathbf{w}^T h(\mathbf{x}_i)
$$ 

where $h(\mathbf{x}_i)$ represents the features for example $i$.  We will write some code to obtain the **scores** using GraphLab Create. For each row, the **score** (or margin) is a number in the range **[-inf, inf]**.


```python
scores = sentiment_model.predict(sample_test_data, output_type='margin')
print scores
```

    [6.734619727058244, -5.734130996759758, -14.668460404467641]
    

### Predicting sentiment

These scores can be used to make class predictions as follows:

$$
\hat{y} = 
\left\{
\begin{array}{ll}
      +1 & \mathbf{w}^T h(\mathbf{x}_i) > 0 \\
      -1 & \mathbf{w}^T h(\mathbf{x}_i) \leq 0 \\
\end{array} 
\right.
$$

Using scores, write code to calculate $\hat{y}$, the class predictions:


```python
yhat = [1 if score >0 else -1 for score in scores]
yhat
```




    [1, -1, -1]



Run the following code to verify that the class predictions obtained by your calculations are the same as that obtained from GraphLab Create.


```python
print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data)
```

    Class predictions according to GraphLab Create:
    [1L, -1L, -1L]
    

**Checkpoint**: Make sure your class predictions match with the one obtained from GraphLab Create.

### Probability predictions

Recall from the lectures that we can also calculate the probability predictions from the scores using:
$$
P(y_i = +1 | \mathbf{x}_i,\mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))}.
$$

Using the variable **scores** calculated previously, write code to calculate the probability that a sentiment is positive using the above formula. For each row, the probabilities should be a number in the range **[0, 1]**.


```python
P = [1/(1+math.exp(-score))  for score in scores]
P
```




    [0.9988123848377185, 0.0032232681818022873, 4.26155799665916e-07]



**Checkpoint**: Make sure your probability predictions match the ones obtained from GraphLab Create.


```python
print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data, output_type='probability')
```

    Class predictions according to GraphLab Create:
    [0.9988123848377185, 0.0032232681818022886, 4.261557996659157e-07]
    

** Quiz Question:** Of the three data points in **sample_test_data**, which one (first, second, or third) has the **lowest probability** of being classified as a positive review?

# Find the most positive (and negative) review

We now turn to examining the full test dataset, **test_data**, and use GraphLab Create to form predictions on all of the test data points for faster performance.

Using the `sentiment_model`, find the 20 reviews in the entire **test_data** with the **highest probability** of being classified as a **positive review**. We refer to these as the "most positive reviews."

To calculate these top-20 reviews, use the following steps:
1.  Make probability predictions on **test_data** using the `sentiment_model`. (**Hint:** When you call `.predict` to make predictions on the test data, use option `output_type='probability'` to output the probability rather than just the most likely class.)
2.  Sort the data according to those predictions and pick the top 20. (**Hint:** You can use the `.topk` method on an SFrame to find the top k rows sorted according to the value of a specified column.)


```python
test_data['probability'] = sentiment_model.predict(test_data,output_type='probability')
```


```python
test_data[test_data['probability']==1.0]['name']
```




    dtype: str
    Rows: ?
    ['Munchkin Mozart Magic Cube', 'BABYBJORN Potty Chair - Red', 'Safety 1st Tot-Lok Four Lock Assembly', 'Summer Infant Complete Nursery Care Kit', 'Leachco Snoogle Total Body Pillow', 'HALO SleepSack Micro-Fleece Wearable Blanket, Soft Pink, Small', 'Peg Perego Primo Viaggio Car Seat / Infant Carrier with LATCH Base - Black Sable', 'Capri Stroller - Red Tech', 'Wizard Convertible Car Seat with LATCH in Midnight Print', 'Britax Marathon Convertible Car Seat, Granite', 'Britax Decathlon Convertible Car Seat, Tiffany', 'North States Supergate Pressure Mount Clear Choice Wood Gate', 'Fisher-Price Deluxe Jumperoo', "Lilly Gold Sit 'n' Stroll 5 in 1 Car Seat and Stroller Combination, Tuxedo Black (sunshade is not included in the offering)", 'Fisher-Price Rainforest Melodies and Lights Deluxe Gym', 'JP Lizzy Chocolate Ice Classic Tote Set', 'Cloud b Sound Machine Soother, Sleep Sheep', 'Shermag Glider Rocker Combo, Pecan with Oatmeal', 'Traveling Toddler Car Seat Travel Accessory', 'Ameda Purely Yours Breast Pump - Carry All', 'Moby Wrap Original 100% Cotton Baby Carrier, Red', 'Moby Wrap Original 100% Cotton Baby Carrier, Red', 'bumGenius One-Size Cloth Diaper Twilight', 'Medela Supplemental Nursing System', 'Peg Perego Aria Light Weight One Hand Fold Stroller in Moka', 'KidCo Magnet Lock Starter Set', "P'Kolino Silly Soft Seating in Tias, Green", 'The First Years True Fit Convertible Car Seat, Monet', 'The First Years True Fit Convertible Car Seat, Monet', 'ESPRIT Sun Speed Stroller - Apple Green', 'Medela Contact Nipple Shield, Small', 'Evenflo X Sport Plus Convenience Stroller - Christina', 'We Sell Mats 36 Sq Ft Alphabet and Number Floor Mat', 'Gerber First Essential Clear View BPA Free Plastic Nurser With Latex Nipple, Assorted Colors, 9 Ounce, 3 Pack', 'Britax Frontier Booster Car Seat', "Dr. Brown's 3 Pack BPA Free Polypropylene Bottle, 8 oz", 'Carters Easy Fit Velour Plush Crib Fitted Sheet, Chocolate', 'Evenflo 6 Pack Classic Glass Bottle, 4-Ounce', 'Philips AVENT BPA Free Contemporary Freeflow Pacifier, 0-6 Months, 2-Pack, Colors and Designs May Vary', 'Skip Hop Studio Diaper Bag, Black Dot', 'Regalo Easy Step Walk Thru Gate, White', "Yookidoo Flow 'N Fill Spout Bath Toy (9m+)", 'Munchie Mug - Top Rated Spill Resistant Snack Cup for Toddlers. Ages 1 to 4 years. Made in AMERICA. - BPA and phthalate free. FDA compliant materials. - Blue Top', 'The Nesting Pillow - Organic Nursing Pillow with Washable Slip Cover', 'Simple Wishes Hands-Free Breastpump Bra, Pink, XS-L', 'Delta Universal 6 Drawer Dresser, Black Cherry', 'Baby BeeHinds Magic Alls Minkee All In One Pocket Diaper Snaps - Marine Green Small', 'green sprouts Stacking Cup Set, Colors may vary', 'green sprouts Stacking Cup Set, Colors may vary', 'Baby Einstein Around The World Discovery Center', 'Dream On Me / Mia Moda  Atmosferra Stroller, Nero', 'Freemie Hands-Free Concealable Breast Pump Collection System', 'Orbit Baby Stroller Travel System G2, Mocha', 'timi &amp; leslie Charlie 7-Piece Diaper Bag Set, Light Brown', 'Infantino Wrap and Tie Baby Carrier, Black Blueberries', 'OXO Tot Dishwasher Basket, Orange', 'North States Supergate Extra Tall Easy Close Gate, Bronze', 'Bla bla kids Mini Mcnuttie the Squirrel', 'Baby Planet Endangered Species Sport Lemur Frog Stroller', 'Lamaze High-Contrast Panda Rattle', 'RECARO ProRIDE Convertible Car Seat, Misty', 'RECARO ProRIDE Convertible Car Seat, Misty', 'RECARO ProRIDE Convertible Car Seat, Misty', 'Annabel Karmel Lunch Box', 'Eddie Bauer Trailmaker Travel System - Sinclair', 'Sorelle Tuscany Crib N More, Espresso', 'Badger Basket 3 Pack Polka Dot Nesting Trapezoid Shape Folding Baskets, Pink', 'Combi Kobuk Air-Thru, Licorice', 'bumGenius One-Size Snap Closure Cloth Diaper 4.0 - Bubble', "Fisher-Price Cradle 'N Swing,  My Little Snugabunny", 'Skip Hop Bento Diaper Tote Bag, Black', 'Roan Rocco Classic Pram Stroller 2-in-1 with Bassinet and Seat Unit - Coffee', 'Quinny 2012 Buzz Stroller, Rebel Red', 'Meeno Baby Cool Me Seat Liner Car Seat - Pink', 'HABA Heart Princess Pacifier Chain', 'Door Monkey, Childproof Door Lock &amp; Pinch Guard', 'JJ Cole Swag Diaper Bag, Bronze Drop', 'Safety 1st Magnetic Locking System', 'Ingenuity Cradle and Sway Swing, Bella Vista', 'Boon Lawn Countertop Drying Rack, Green', 'Cybex Aton Infant Car Seat - Coffee', 'Britax 2012 B-Agile Stroller, Red', 'Maxi-Cosi Pria 70 with Tiny Fit Convertible Car Seat', "Graco Pack 'n Play Element Playard - Flint", 'Display Box - 30 singles', 'Go-Go Babyz Sidekick Bliss Diaper Bag, Yellow', 'Jolly Jumper Arctic Sneak A Peek Infant Car Seat Cover Black', 'Diono RadianR100 Convertible Car Seat, Dune', 'Diono RadianRXT Convertible Car Seat, Plum', 'Diono RadianRXT Convertible Car Seat, Plum', 'Phil&amp;Teds Promenade Buggy Single Stroller Black', 'Prince Lionheart weePOD Basix, Ash Grey', 'Inglesina 2013 Fast Table Chair, Liquirizia', 'Inglesina 2012 Trip Stroller, Ibisco', 'Baby Jogger City Mini GT Single Stroller, Shadow/Orange', 'Baby Jogger City Mini GT Double Stroller, Shadow/Orange', "The Original CJ's BuTTer (All Natural Mango Sugar Mint, 12 oz. tub)", 'Rainy Day Indoor Playground toddler swing to be used with support system', 'Fisher-Price Grow with Me High Chair, Bunny', 'Ikea 36 Pcs Kalas Kids Plastic BPA Free Flatware, Bowl, Plate, Tumbler Set, Colorful', ... ]



**Quiz Question**: Which of the following products are represented in the 20 most positive reviews? [multiple choice]


Now, let us repeat this excercise to find the "most negative reviews." Use the prediction probabilities to find the  20 reviews in the **test_data** with the **lowest probability** of being classified as a **positive review**. Repeat the same steps above but make sure you **sort in the opposite order**.


```python
test_data.topk('probability',20,True)['name']
```




    dtype: str
    Rows: 20
    ['Jolly Jumper Arctic Sneak A Peek Infant Car Seat Cover Black', "Levana Safe N'See Digital Video Baby Monitor with Talk-to-Baby Intercom and Lullaby Control (LV-TW501)", 'Snuza Portable Baby Movement Monitor', 'Fisher-Price Ocean Wonders Aquarium Bouncer', 'VTech Communications Safe &amp; Sounds Full Color Video and Audio Monitor', 'Safety 1st High-Def Digital Monitor', 'Chicco Cortina KeyFit 30 Travel System in Adventure', 'Prince Lionheart Warmies Wipes Warmer', 'Valco Baby Tri-mode Twin Stroller EX- Hot Chocolate', 'Adiri BPA Free Natural Nurser Ultimate Bottle Stage 1 White, Slow Flow (0-3 months)', 'Munchkin Nursery Projector and Sound System, White', 'The First Years True Choice P400 Premium Digital Monitor, 2 Parent Unit', 'Nuby Natural Touch Silicone Travel Infa Feeder, Colors May Vary, 3 Ounce', 'Peg-Perego Tatamia High Chair, White Latte', 'Fisher-Price Royal Potty', 'Safety 1st Exchangeable Tip 3 in 1 Thermometer', 'Safety 1st Lift Lock and Swing Gate', 'Evenflo Take Me Too Premiere Tandem Stroller - Castlebay', 'Cloth Diaper Sprayer--styles may vary', 'The First Years 3 Pack Breastflow Bottle, 9 Ounce']



**Quiz Question**: Which of the following products are represented in the 20 most negative reviews?  [multiple choice]

## Compute accuracy of the classifier

We will now evaluate the accuracy of the trained classifer. Recall that the accuracy is given by


$$
\mbox{accuracy} = \frac{\mbox{# correctly classified examples}}{\mbox{# total examples}}
$$

This can be computed as follows:

* **Step 1:** Use the trained model to compute class predictions (**Hint:** Use the `predict` method)
* **Step 2:** Count the number of data points when the predicted class labels match the ground truth labels (called `true_labels` below).
* **Step 3:** Divide the total number of correct predictions by the total number of data points in the dataset.

Complete the function below to compute the classification accuracy:


```python
def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    ## YOUR CODE HERE
    predictions = model.predict(data)
    
    # Compute the number of correctly classified examples
    ## YOUR CODE HERE
    correct = sum(predictions==true_labels)

    # Then compute accuracy by dividing num_correct by total number of examples
    ## YOUR CODE HERE
    accuracy = correct/len(true_labels)
    
    return accuracy
```

Now, let's compute the classification accuracy of the **sentiment_model** on the **test_data**.


```python
get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
```




    0.9145368370530358



**Quiz Question**: What is the accuracy of the **sentiment_model** on the **test_data**? Round your answer to 2 decimal places (e.g. 0.76).

**Quiz Question**: Does a higher accuracy value on the **training_data** always imply that the classifier is better?

## Learn another classifier with fewer words

There were a lot of words in the model we trained above. We will now train a simpler logistic regression model using only a subet of words that occur in the reviews. For this assignment, we selected a 20 words to work with. These are:


```python
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
```


```python
len(significant_words)
```




    20



For each review, we will use the **word_count** column and trim out all words that are **not** in the **significant_words** list above. We will use the [SArray dictionary trim by keys functionality]( https://dato.com/products/create/docs/generated/graphlab.SArray.dict_trim_by_keys.html). Note that we are performing this on both the training and test set.


```python
train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
```

Let's see what the first example of the dataset looks like:


```python
train_data[0]['review']
```




    'it came early and was not disappointed. i love planet wise bags and now my wipe holder. it keps my osocozy wipes moist and does not leak. highly recommend it.'



The **word_count** column had been working with before looks like the following:


```python
print train_data[0]['word_count']
```

    {'and': 3L, 'love': 1L, 'it': 3L, 'highly': 1L, 'osocozy': 1L, 'bags': 1L, 'leak': 1L, 'moist': 1L, 'does': 1L, 'recommend': 1L, 'was': 1L, 'wipes': 1L, 'disappointed': 1L, 'early': 1L, 'not': 2L, 'now': 1L, 'holder': 1L, 'wipe': 1L, 'keps': 1L, 'wise': 1L, 'i': 1L, 'planet': 1L, 'my': 2L, 'came': 1L}
    

Since we are only working with a subet of these words, the column **word_count_subset** is a subset of the above dictionary. In this example, only 2 `significant words` are present in this review.


```python
print train_data[0]['word_count_subset']
```

    {'love': 1L, 'disappointed': 1L}
    

## Train a logistic regression model on a subset of data

We will now build a classifier with **word_count_subset** as the feature and **sentiment** as the target. 


```python
simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
simple_model
```


<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 133416</pre>



<pre>Number of classes           : 2</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 20</pre>



<pre>Number of coefficients    : 21</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-accuracy |</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>| 1         | 2        | 0.194058     | 0.862917          |</pre>



<pre>| 2         | 3        | 0.317886     | 0.865713          |</pre>



<pre>| 3         | 4        | 0.427106     | 0.866478          |</pre>



<pre>| 4         | 5        | 0.528759     | 0.866748          |</pre>



<pre>| 5         | 6        | 0.633577     | 0.866815          |</pre>



<pre>| 6         | 7        | 0.735380     | 0.866815          |</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    Class                         : LogisticClassifier
    
    Schema
    ------
    Number of coefficients        : 21
    Number of examples            : 133416
    Number of classes             : 2
    Number of feature columns     : 1
    Number of unpacked features   : 20
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.01
    
    Training Summary
    ----------------
    Solver                        : newton
    Solver iterations             : 6
    Solver status                 : SUCCESS: Optimal solution found.
    Training time (sec)           : 0.76
    
    Settings
    --------
    Log-likelihood                : 44323.7254
    
    Highest Positive Coefficients
    -----------------------------
    word_count_subset[loves]      : 1.6773
    word_count_subset[perfect]    : 1.5145
    word_count_subset[love]       : 1.3654
    (intercept)                   : 1.2995
    word_count_subset[easy]       : 1.1937
    
    Lowest Negative Coefficients
    ----------------------------
    word_count_subset[disappointed]: -2.3551
    word_count_subset[return]     : -2.1173
    word_count_subset[waste]      : -2.0428
    word_count_subset[broke]      : -1.658
    word_count_subset[money]      : -0.8979



We can compute the classification accuracy using the `get_classification_accuracy` function you implemented earlier.


```python
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
```




    0.8693004559635229



Now, we will inspect the weights (coefficients) of the **simple_model**:


```python
simple_model.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">class</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2995449552</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0120888541331</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">disappointed</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.35509250061</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0504149888557</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">love</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.36543549368</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0303546295109</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">well</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.504256746398</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.021381300631</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">product</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.320555492996</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0154311321362</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">loves</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.67727145556</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0482328275384</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">little</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.520628636025</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0214691475665</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">work</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.621700012425</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0230330597946</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">easy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.19366189833</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.029288869202</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">great</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.94469126948</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0209509926591</td>
    </tr>
</table>
[21 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Let's sort the coefficients (in descending order) by the **value** to obtain the coefficients with the most positive effect on the sentiment.


```python
simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
```

    +-------------------+--------------+-------+-----------------+-----------------+
    |        name       |    index     | class |      value      |      stderr     |
    +-------------------+--------------+-------+-----------------+-----------------+
    | word_count_subset |    loves     |   1   |  1.67727145556  | 0.0482328275384 |
    | word_count_subset |   perfect    |   1   |  1.51448626703  |  0.049861952294 |
    | word_count_subset |     love     |   1   |  1.36543549368  | 0.0303546295109 |
    |    (intercept)    |     None     |   1   |   1.2995449552  | 0.0120888541331 |
    | word_count_subset |     easy     |   1   |  1.19366189833  |  0.029288869202 |
    | word_count_subset |    great     |   1   |  0.94469126948  | 0.0209509926591 |
    | word_count_subset |    little    |   1   |  0.520628636025 | 0.0214691475665 |
    | word_count_subset |     well     |   1   |  0.504256746398 |  0.021381300631 |
    | word_count_subset |     able     |   1   |  0.191438302295 | 0.0337581955697 |
    | word_count_subset |     old      |   1   | 0.0853961886678 | 0.0200863423025 |
    | word_count_subset |     car      |   1   |  0.058834990068 | 0.0168291532091 |
    | word_count_subset |     less     |   1   | -0.209709815216 |  0.040505735954 |
    | word_count_subset |   product    |   1   | -0.320555492996 | 0.0154311321362 |
    | word_count_subset |    would     |   1   | -0.362308947711 | 0.0127544751985 |
    | word_count_subset |     even     |   1   |  -0.51173855127 | 0.0199612760261 |
    | word_count_subset |     work     |   1   | -0.621700012425 | 0.0230330597946 |
    | word_count_subset |    money     |   1   | -0.897884155776 | 0.0339936732836 |
    | word_count_subset |    broke     |   1   |  -1.65796447838 | 0.0580878907166 |
    | word_count_subset |    waste     |   1   |   -2.042773611  | 0.0644702932444 |
    | word_count_subset |    return    |   1   |  -2.11729659718 | 0.0578650807241 |
    | word_count_subset | disappointed |   1   |  -2.35509250061 | 0.0504149888557 |
    +-------------------+--------------+-------+-----------------+-----------------+
    [21 rows x 5 columns]
    
    

**Quiz Question**: Consider the coefficients of **simple_model**. There should be 21 of them, an intercept term + one for each word in **significant_words**. How many of the 20 coefficients (corresponding to the 20 **significant_words** and *excluding the intercept term*) are positive for the `simple_model`?


```python
sum(simple_model.coefficients['value']>0)-1
```




    10L



**Quiz Question**: Are the positive words in the **simple_model** (let us call them `positive_significant_words`) also positive words in the **sentiment_model**?


```python
positive_significant_words = simple_model.coefficients[simple_model.coefficients['value']>0]['index']
```


```python
for word in positive_significant_words:
    if word not in sentiment_model.coefficients[sentiment_model.coefficients['value']>0]['index']:
        print 'false'
```

# Comparing models

We will now compare the accuracy of the **sentiment_model** and the **simple_model** using the `get_classification_accuracy` method you implemented above.

First, compute the classification accuracy of the **sentiment_model** on the **train_data**:


```python
get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
```




    0.979440247046831



Now, compute the classification accuracy of the **simple_model** on the **train_data**:


```python
get_classification_accuracy(simple_model, train_data, train_data['sentiment'])
```




    0.8668150746537147



**Quiz Question**: Which model (**sentiment_model** or **simple_model**) has higher accuracy on the TRAINING set?

Now, we will repeat this excercise on the **test_data**. Start by computing the classification accuracy of the **sentiment_model** on the **test_data**:


```python
get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
```




    0.9145368370530358



Next, we will compute the classification accuracy of the **simple_model** on the **test_data**:


```python
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
```




    0.8693004559635229



**Quiz Question**: Which model (**sentiment_model** or **simple_model**) has higher accuracy on the TEST set?

## Baseline: Majority class prediction

It is quite common to use the **majority class classifier** as the a baseline (or reference) model for comparison with your classifier model. The majority classifier model predicts the majority class for all data points. At the very least, you should healthily beat the majority class classifier, otherwise, the model is (usually) pointless.

What is the majority class in the **train_data**?


```python
num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print num_positive
print num_negative
```

    112164
    21252
    

Now compute the accuracy of the majority class classifier on **test_data**.

**Quiz Question**: Enter the accuracy of the majority class classifier model on the **test_data**. Round your answer to two decimal places (e.g. 0.76).


```python
accuracy_baseline = (test_data['sentiment'] == +1).sum()/len(test_data)
print accuracy_baseline
```

    0.842782577394
    

**Quiz Question**: Is the **sentiment_model** definitely better than the majority class classifier (the baseline)?
