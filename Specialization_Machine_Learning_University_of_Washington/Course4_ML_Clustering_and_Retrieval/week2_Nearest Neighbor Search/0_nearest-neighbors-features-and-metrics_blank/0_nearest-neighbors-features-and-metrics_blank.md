
# Nearest Neighbors

When exploring a large set of documents -- such as Wikipedia, news articles, StackOverflow, etc. -- it can be useful to get a list of related material. To find relevant documents you typically
* Decide on a notion of similarity
* Find the documents that are most similar 

In the assignment you will
* Gain intuition for different notions of similarity and practice finding similar documents. 
* Explore the tradeoffs with representing documents using raw word counts and TF-IDF
* Explore the behavior of different distance metrics by looking at the Wikipedia pages most similar to President Obama’s page.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import necessary packages

As usual we need to first import the Python packages that we will need.

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
import graphlab
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'
```

## Load Wikipedia dataset
#### Download Link:https://static.dato.com/files/coursera/course-4/people_wiki.gl.zip

We will be using the same dataset of Wikipedia pages that we used in the Machine Learning Foundations course (Course 1). Each element of the dataset consists of a link to the wikipedia article, the name of the person, and the text of the article (in lowercase).  


```python
wiki = graphlab.SFrame('people_wiki.gl')
```

    This non-commercial license of GraphLab Create for academic use is assigned to last.fantasy@qq.com and will expire on July 25, 2017.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1469595190.log.0
    


```python
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
    </tr>
</table>
[59071 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Extract word count vectors

As we have seen in Course 1, we can extract word count vectors using a GraphLab utility function.  We add this as a column in `wiki`.


```python
wiki['word_count'] = graphlab.text_analytics.count_words(wiki['text'])
```


```python
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'since': 1L, 'carltons':<br>1L, 'being': 1L, '2005': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'precise': 1L, 'thomas':<br>1L, 'closely': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'just': 1L, 'issued':<br>1L, 'mainly': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L,<br>'bauforschung': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'legendary': 1L,<br>'gangstergenka': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'now': 1L, 'currently':<br>1L, 'less': 1L, 'being': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'exclusive': 2L,<br>'producer': 1L, 'tribe': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'taxi': 1L, 'salon': 1L,<br>'gangs': 1L, 'being': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'houston': 1L,<br>'frankie': 1L, 'labels': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'phenomenon': 1L,<br>'deborash': 1L, ...</td>
    </tr>
</table>
[59071 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Find nearest neighbors

Let's start by finding the nearest neighbors of the Barack Obama page using the word count vectors to represent the articles and Euclidean distance to measure distance.  For this, again will we use a GraphLab Create implementation of nearest neighbor search.


```python
model = graphlab.nearest_neighbors.create(wiki, label='name', features=['word_count'],
                                          method='brute_force', distance='euclidean')
```


<pre>Starting brute force nearest neighbors model training.</pre>


Let's look at the top 10 nearest neighbors by performing the following query:


```python
model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 5.014ms      |</pre>



<pre>| Done         |         | 100         | 284.256ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.0756708171</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">George W. Bush</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.3947670438</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lawrence Summers</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.1524549651</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mitt Romney</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.1662826401</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Francisco Barrio</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.3318042492</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Walter Mondale</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.4005494464</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Wynn Normington Hugh-<br>Jones ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.4965751818</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Don Bonker</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.633318168</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Andy Anstett</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.9594372252</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



All of the 10 people are politicians, but about half of them have rather tenuous connections with Obama, other than the fact that they are politicians.

* Francisco Barrio is a Mexican politician, and a former governor of Chihuahua.
* Walter Mondale and Don Bonker are Democrats who made their career in late 1970s.
* Wynn Normington Hugh-Jones is a former British diplomat and Liberal Party official.
* Andy Anstett is a former politician in Manitoba, Canada.

Nearest neighbors with raw word counts got some things right, showing all politicians in the query result, but missed finer and important details.

For instance, let's find out why Francisco Barrio was considered a close neighbor of Obama.  To do this, let's look at the most frequently used words in each of Barack Obama and Francisco Barrio's pages:


```python
def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)
```


```python
obama_words = top_words('Barack Obama')
obama_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">act</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
</table>
[273 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
barrio_words = top_words('Francisco Barrio')
barrio_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">chihuahua</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">governor</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[225 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Let's extract the list of most frequent words that appear in both Obama's and Barrio's documents. We've so far sorted all words from Obama and Barrio's articles by their word frequencies. We will now use a dataframe operation known as **join**. The **join** operation is very useful when it comes to playing around with data: it lets you combine the content of two tables using a shared column (in this case, the word column). See [the documentation](https://dato.com/products/create/docs/generated/graphlab.SFrame.join.html) for more details.

For instance, running
```
obama_words.join(barrio_words, on='word')
```
will extract the rows from both tables that correspond to the common words.


```python
combined_words = obama_words.join(barrio_words, on='word')
combined_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count.1</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">as</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">was</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
</table>
[56 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Since both tables contained the column named `count`, SFrame automatically renamed one of them to prevent confusion. Let's rename the columns to tell which one is for which. By inspection, we see that the first column (`count`) is for Obama and the second (`count.1`) for Barrio.


```python
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Obama</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Barrio</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">as</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">was</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
</table>
[56 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



**Note**. The **join** operation does not enforce any particular ordering on the shared column. So to obtain, say, the five common words that appear most often in Obama's article, sort the combined table by the Obama column. Don't forget `ascending=False` to display largest counts first.


```python
combined_words.sort('Obama', ascending=False)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Obama</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Barrio</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">as</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">was</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
</table>
[56 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



**Quiz Question**. Among the words that appear in both Barack Obama and Francisco Barrio, take the 5 that appear most frequently in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?

Hint:
* Refer to the previous paragraph for finding the words that appear in both articles. Sort the common words by their frequencies in Obama's article and take the largest five.
* Each word count vector is a Python dictionary. For each word count vector in SFrame, you'd have to check if the set of the 5 common words is a subset of the keys of the word count vector. Complete the function `has_top_words` to accomplish the task.
  - Convert the list of top 5 words into set using the syntax
```
set(common_words)
```
    where `common_words` is a Python list. See [this link](https://docs.python.org/2/library/stdtypes.html#set) if you're curious about Python sets.
  - Extract the list of keys of the word count dictionary by calling the [`keys()` method](https://docs.python.org/2/library/stdtypes.html#dict.keys).
  - Convert the list of keys into a set as well.
  - Use [`issubset()` method](https://docs.python.org/2/library/stdtypes.html#set) to check if all 5 words are among the keys.
* Now apply the `has_top_words` function on every row of the SFrame.
* Compute the sum of the result column to obtain the number of articles containing all the 5 top words.


```python
common_words = set(combined_words.sort('Obama', ascending=False)[0:5]['word']) # YOUR CODE HERE
def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())   # YOUR CODE HERE
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words.issubset(unique_words)  # YOUR CODE HERE

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
print sum(wiki['has_top_words']) # YOUR CODE HERE
```

    56066
    

**Checkpoint**. Check your `has_top_words` function on two random articles:


```python
print 'Output from your function:', has_top_words(wiki[32]['word_count'])
print 'Correct output: True'
print 'Also check the length of unique_words. It should be 167'
```

    Output from your function: True
    Correct output: True
    Also check the length of unique_words. It should be 167
    


```python
print 'Output from your function:', has_top_words(wiki[33]['word_count'])
print 'Correct output: False'
print 'Also check the length of unique_words. It should be 188'
```

    Output from your function: False
    Correct output: False
    Also check the length of unique_words. It should be 188
    

**Quiz Question**. Measure the pairwise distance between the Wikipedia pages of Barack Obama, George W. Bush, and Joe Biden. Which of the three pairs has the smallest distance?

Hint: To compute the Euclidean distance between two dictionaries, use `graphlab.toolkits.distances.euclidean`. Refer to [this link](https://dato.com/products/create/docs/generated/graphlab.toolkits.distances.euclidean.html) for usage.


```python
print graphlab.toolkits.distances.euclidean(wiki[wiki['name']=='Barack Obama']['word_count'][0],wiki[wiki['name']=='George W. Bush']['word_count'][0])
print graphlab.toolkits.distances.euclidean(wiki[wiki['name']=='Barack Obama']['word_count'][0],wiki[wiki['name']=='Joe Biden']['word_count'][0])
print graphlab.toolkits.distances.euclidean(wiki[wiki['name']=='Joe Biden']['word_count'][0],wiki[wiki['name']=='George W. Bush']['word_count'][0])
```

    34.3947670438
    33.0756708171
    32.7566787083
    

**Quiz Question**. Collect all words that appear both in Barack Obama and George W. Bush pages.  Out of those words, find the 10 words that show up most often in Obama's page. 


```python
bush_words = top_words('George W. Bush')
combined_words = obama_words.join(bush_words, on='word')
combined_words.sort('count',ascending = False)[0:10]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count.1</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">22</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">act</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">as</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
</table>
[10 rows x 3 columns]<br/>
</div>



**Note.** Even though common words are swamping out important subtle differences, commonalities in rarer political words still matter on the margin. This is why politicians are being listed in the query result instead of musicians, for example. In the next subsection, we will introduce a different metric that will place greater emphasis on those rarer words.

## TF-IDF to the rescue

Much of the perceived commonalities between Obama and Barrio were due to occurrences of extremely frequent words, such as "the", "and", and "his". So nearest neighbors is recommending plausible results sometimes for the wrong reasons. 

To retrieve articles that are more relevant, we should focus more on rare words that don't happen in every article. **TF-IDF** (term frequency–inverse document frequency) is a feature representation that penalizes words that are too common.  Let's use GraphLab Create's implementation of TF-IDF and repeat the search for the 10 nearest neighbors of Barack Obama:


```python
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count'])
```


```python
model_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                 method='brute_force', distance='euclidean')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 8.021ms      |</pre>



<pre>| Done         |         | 100         | 235.142ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Phil Schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">106.861013691</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jeff Sessions</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">108.871674216</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.045697909</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.108106165</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bob Menendez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.781867105</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.95778808</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">James A. Guest</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.413888718</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Roland Grossenbacher</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.4706087</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tulsi Gabbard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.696997999</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



Let's determine whether this list makes sense.
* With a notable exception of Roland Grossenbacher, the other 8 are all American politicians who are contemporaries of Barack Obama.
* Phil Schiliro, Jesse Lee, Samantha Power, and Eric Stern worked for Obama.

Clearly, the results are more plausible with the use of TF-IDF. Let's take a look at the word vector for Obama and Schilirio's pages. Notice that TF-IDF representation assigns a weight to each word. This weight captures relative importance of that word in the document. Let us sort the words in Obama's article by their TF-IDF weights; we do the same for Schiliro's article as well.


```python
def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)
```


```python
obama_tf_idf = top_words_tf_idf('Barack Obama')
obama_tf_idf
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">weight</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.2956530721</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">act</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">27.678222623</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">iraq</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17.747378588</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">control</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.8870608452</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">law</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.7229357618</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">ordered</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.5333739509</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">military</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13.1159327785</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">involvement</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.7843852412</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">response</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.7843852412</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">democratic</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.4106886973</td>
    </tr>
</table>
[273 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
schiliro_tf_idf
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">weight</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21.9729907785</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">staff</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">15.8564416352</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">congressional</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13.5470876563</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">daschleschiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.9864953892</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.62125623824</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">waxman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.04058524017</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">president</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.03358661416</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014from</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.68391029623</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">law</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.36146788088</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">consultant</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.91310403725</td>
    </tr>
</table>
[119 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Using the **join** operation we learned earlier, try your hands at computing the common words shared by Obama's and Schiliro's articles. Sort the common words by their TF-IDF weights in Obama's document.


```python
combined_words = obama_tf_idf.join(schiliro_tf_idf,on='word')
combined_words.sort('weight',ascending =False)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">weight</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">weight.1</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.2956530721</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.62125623824</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">law</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.7229357618</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.36146788088</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">democratic</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.4106886973</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.20534434867</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">senate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.1642881797</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.3880960599</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">presidential</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.3869554189</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.69347770945</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">president</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.22686929133</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.03358661416</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">policy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.09538628214</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.04769314107</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">states</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.47320098963</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.82440032988</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">office</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.24817282322</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.62408641161</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2011</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.10704127031</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.40469418021</td>
    </tr>
</table>
[47 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



The first 10 words should say: Obama, law, democratic, Senate, presidential, president, policy, states, office, 2011.

**Quiz Question**. Among the words that appear in both Barack Obama and Phil Schiliro, take the 5 that have largest weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 5 words?


```python
common_words = set(combined_words[0:5]['word'])  # YOUR CODE HERE

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())   # YOUR CODE HERE
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return common_words.issubset(unique_words)  # YOUR CODE HERE

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
print sum(wiki['has_top_words'])  # YOUR CODE HERE
```

    14
    

Notice the huge difference in this calculation using TF-IDF scores instead  of raw word counts. We've eliminated noise arising from extremely common words.

## Choosing metrics

You may wonder why Joe Biden, Obama's running mate in two presidential elections, is missing from the query results of `model_tf_idf`. Let's find out why. First, compute the distance between TF-IDF features of Obama and Biden.

**Quiz Question**. Compute the Euclidean distance between TF-IDF features of Obama and Biden. Hint: When using Boolean filter in SFrame/SArray, take the index 0 to access the first match.


```python
print graphlab.toolkits.distances.euclidean(wiki[wiki['name']=='Barack Obama']['tf_idf'][0],wiki[wiki['name']=='Joe Biden']['tf_idf'][0])
```

    123.29745601
    

The distance is larger than the distances we found for the 10 nearest neighbors, which we repeat here for readability:


```python
model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 0us          |</pre>



<pre>| Done         |         | 100         | 216.02ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Phil Schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">106.861013691</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jeff Sessions</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">108.871674216</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.045697909</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.108106165</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bob Menendez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.781867105</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.95778808</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">James A. Guest</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.413888718</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Roland Grossenbacher</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.4706087</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tulsi Gabbard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.696997999</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



But one may wonder, is Biden's article that different from Obama's, more so than, say, Schiliro's? It turns out that, when we compute nearest neighbors using the Euclidean distances, we unwittingly favor short articles over long ones. Let us compute the length of each Wikipedia document, and examine the document lengths for the 100 nearest neighbors to Obama's page.


```python
def compute_length(row):
    return len(row['text'].split(' '))

wiki['length'] = wiki.apply(compute_length) 
```


```python
nearest_neighbors_euclidean = model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
nearest_neighbors_euclidean = nearest_neighbors_euclidean.join(wiki[['name', 'length']], on={'reference_label':'name'})
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 15.607ms     |</pre>



<pre>| Done         |         | 100         | 308.345ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
nearest_neighbors_euclidean.sort('rank')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">length</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">540</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Phil Schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">106.861013691</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">208</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jeff Sessions</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">108.871674216</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">230</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.045697909</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">216</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.108106165</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">310</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bob Menendez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.781867105</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">220</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.95778808</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">255</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">James A. Guest</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.413888718</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">215</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Roland Grossenbacher</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.4706087</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">201</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tulsi Gabbard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.696997999</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">228</td>
    </tr>
</table>
[100 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



To see how these document lengths compare to the lengths of other documents in the corpus, let's make a histogram of the document lengths of Obama's 100 nearest neighbors and compare to a histogram of document lengths for all documents.


```python
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])

plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output_65_0.png)


Relative to the rest of Wikipedia, nearest neighbors of Obama are overwhemingly short, most of them being shorter than 300 words. The bias towards short articles is not appropriate in this application as there is really no reason to  favor short articles over long articles (they are all Wikipedia articles, after all). Many of the Wikipedia articles are 300 words or more, and both Obama and Biden are over 300 words long.

**Note**: For the interest of computation time, the dataset given here contains _excerpts_ of the articles rather than full text. For instance, the actual Wikipedia article about Obama is around 25000 words. Do not be surprised by the low numbers shown in the histogram.

**Note:** Both word-count features and TF-IDF are proportional to word frequencies. While TF-IDF penalizes very common words, longer articles tend to have longer TF-IDF vectors simply because they have more words in them.

To remove this bias, we turn to **cosine distances**:
$$
d(\mathbf{x},\mathbf{y}) = 1 - \frac{\mathbf{x}^T\mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$
Cosine distances let us compare word distributions of two articles of varying lengths.

Let us train a new nearest neighbor model, this time with cosine distances.  We then repeat the search for Obama's 100 nearest neighbors.


```python
model2_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                  method='brute_force', distance='cosine')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
nearest_neighbors_cosine = model2_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
nearest_neighbors_cosine = nearest_neighbors_cosine.join(wiki[['name', 'length']], on={'reference_label':'name'})
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 7.018ms      |</pre>



<pre>| Done         |         | 100         | 292.338ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
nearest_neighbors_cosine.sort('rank')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">length</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">540</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.703138676734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">414</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.742981902328</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">310</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Hillary Rodham Clinton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.758358397887</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">580</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.770561227601</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">255</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Robert Gibbs</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.784677504751</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">257</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Holder</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.788039072943</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">232</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.790926415366</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">216</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Henry Waxman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.798322602893</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">279</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe the Plumber</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.799466360042</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">217</td>
    </tr>
</table>
[100 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



From a glance at the above table, things look better.  For example, we now see Joe Biden as Barack Obama's nearest neighbor!  We also see Hillary Clinton on the list.  This list looks even more plausible as nearest neighbors of Barack Obama.

Let's make a plot to better visualize the effect of having used cosine distance in place of Euclidean on our TF-IDF vectors.


```python
plt.figure(figsize=(10.5,4.5))
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
```


    <matplotlib.figure.Figure at 0x15f380b8>



![png](output_73_1.png)


Indeed, the 100 nearest neighbors using cosine distance provide a sampling across the range of document lengths, rather than just short articles like Euclidean distance provided.

**Moral of the story**: In deciding the features and distance measures, check if they produce results that make sense for your particular application.

# Problem with cosine distances: tweets vs. long articles

Happily ever after? Not so fast. Cosine distances ignore all document lengths, which may be great in certain situations but not in others. For instance, consider the following (admittedly contrived) example.

```
+--------------------------------------------------------+
|                                             +--------+ |
|  One that shall not be named                | Follow | |
|  @username                                  +--------+ |
|                                                        |
|  Democratic governments control law in response to     |
|  popular act.                                          |
|                                                        |
|  8:05 AM - 16 May 2016                                 |
|                                                        |
|  Reply   Retweet (1,332)   Like (300)                  |
|                                                        |
+--------------------------------------------------------+
```

How similar is this tweet to Barack Obama's Wikipedia article? Let's transform the tweet into TF-IDF features, using an encoder fit to the Wikipedia dataset.  (That is, let's treat this tweet as an article in our Wikipedia dataset and see what happens.)


```python
sf = graphlab.SFrame({'text': ['democratic governments control law in response to popular act']})
sf['word_count'] = graphlab.text_analytics.count_words(sf['text'])

encoder = graphlab.feature_engineering.TFIDF(features=['word_count'], output_column_prefix='tf_idf')
encoder.fit(wiki)
sf = encoder.transform(sf)
sf
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf.word_count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">democratic governments<br>control law in response ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'control': 1L,<br>'democratic': 1L, 'act': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'control':<br>3.721765211295327, ...</td>
    </tr>
</table>
[1 rows x 3 columns]<br/>
</div>



Let's look at the TF-IDF vectors for this tweet and for Barack Obama's Wikipedia entry, just to visually see their differences.


```python
tweet_tf_idf = sf[0]['tf_idf.word_count']
tweet_tf_idf
```




    {'act': 3.4597778278724887,
     'control': 3.721765211295327,
     'democratic': 3.1026721743330414,
     'governments': 4.167571323949673,
     'in': 0.0009654063501214492,
     'law': 2.4538226269605703,
     'popular': 2.764478952022998,
     'response': 4.261461747058352,
     'to': 0.04694493768179923}




```python
obama = wiki[wiki['name'] == 'Barack Obama']
obama
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">has_top_words</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Barack_Obama&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">barack hussein obama ii<br>brk husen bm born august ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations': 1L,<br>'represent': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">length</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations':<br>3.811771079388818, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">540</td>
    </tr>
</table>
[? rows x 7 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>



Now, compute the cosine distance between the Barack Obama article and this tweet:


```python
obama_tf_idf = obama[0]['tf_idf']
graphlab.toolkits.distances.cosine(obama_tf_idf, tweet_tf_idf)
```




    0.7059183777794327



Let's compare this distance to the distance between the Barack Obama article and all of its Wikipedia 10 nearest neighbors:


```python
model2_tf_idf.query(obama, label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 9.023ms      |</pre>



<pre>| Done         |         | 100         | 278.261ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.703138676734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.742981902328</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Hillary Rodham Clinton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.758358397887</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.770561227601</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Robert Gibbs</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.784677504751</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Holder</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.788039072943</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.790926415366</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Henry Waxman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.798322602893</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe the Plumber</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.799466360042</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



With cosine distances, the tweet is "nearer" to Barack Obama than everyone else, except for Joe Biden!  This probably is not something we want. If someone is reading the Barack Obama Wikipedia page, would you want to recommend they read this tweet? Ignoring article lengths completely resulted in nonsensical results. In practice, it is common to enforce maximum or minimum document lengths. After all, when someone is reading a long article from _The Atlantic_, you wouldn't recommend him/her a tweet.
