
# Hierarchical Clustering

**Hierarchical clustering** refers to a class of clustering methods that seek to build a **hierarchy** of clusters, in which some clusters contain others. In this assignment, we will explore a top-down approach, recursively bipartitioning the data using k-means.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import packages

The following code block will check if you have the correct version of GraphLab Create. Any version later than 1.8.5 will do. To upgrade, read [this page](https://turi.com/download/upgrade-graphlab-create.html).


```python
import graphlab
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
%matplotlib inline

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'
```

## Load the Wikipedia dataset
#### Download Link:https://static.dato.com/files/coursera/course-4/people_wiki.gl.zip
#### Download Link:https://static.dato.com/files/coursera/course-4/em_utilities.py


```python
wiki = graphlab.SFrame('people_wiki.gl/')
```

    [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: C:\Users\51958\AppData\Local\Temp\graphlab_server_1470502623.log.0
    

    This non-commercial license of GraphLab Create for academic use is assigned to last.fantasy@qq.com and will expire on July 25, 2017.
    

As we did in previous assignments, let's extract the TF-IDF features:


```python
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])
```

To run k-means on this dataset, we should convert the data matrix into a sparse matrix.


```python
from em_utilities import sframe_to_scipy # converter

# This will take about a minute or two.
tf_idf, map_index_to_word = sframe_to_scipy(wiki, 'tf_idf')
```

To be consistent with the k-means assignment, let's normalize all vectors to have unit norm.


```python
from sklearn.preprocessing import normalize
tf_idf = normalize(tf_idf)
```

## Bipartition the Wikipedia dataset using k-means

Recall our workflow for clustering text data with k-means:

1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
2. Extract the data matrix from the dataframe.
3. Run k-means on the data matrix with some value of k.
4. Visualize the clustering results using the centroids, cluster assignments, and the original dataframe. We keep the original dataframe around because the data matrix does not keep auxiliary information (in the case of the text dataset, the title of each article).

Let us modify the workflow to perform bipartitioning:

1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
2. Extract the data matrix from the dataframe.
3. Run k-means on the data matrix with k=2.
4. Divide the data matrix into two parts using the cluster assignments.
5. Divide the dataframe into two parts, again using the cluster assignments. This step is necessary to allow for visualization.
6. Visualize the bipartition of data.

We'd like to be able to repeat Steps 3-6 multiple times to produce a **hierarchy** of clusters such as the following:
```
                      (root)
                         |
            +------------+-------------+
            |                          |
         Cluster                    Cluster
     +------+-----+             +------+-----+
     |            |             |            |
   Cluster     Cluster       Cluster      Cluster
```
Each **parent cluster** is bipartitioned to produce two **child clusters**. At the very top is the **root cluster**, which consists of the entire dataset.

Now we write a wrapper function to bipartition a given cluster using k-means. There are three variables that together comprise the cluster:

* `dataframe`: a subset of the original dataframe that correspond to member rows of the cluster
* `matrix`: same set of rows, stored in sparse matrix format
* `centroid`: the centroid of the cluster (not applicable for the root cluster)

Rather than passing around the three variables separately, we package them into a Python dictionary. The wrapper function takes a single dictionary (representing a parent cluster) and returns two dictionaries (representing the child clusters).


```python
def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    '''cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster'''
    
    data_matrix = cluster['matrix']
    dataframe   = cluster['dataframe']
    
    # Run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, n_jobs=-1)    
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    
    # Divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0], \
                                                      data_matrix[cluster_assignment==1]
    
    # Divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = graphlab.SArray(cluster_assignment) # minor format conversion
    dataframe_left_child, dataframe_right_child     = dataframe[cluster_assignment_sa==0], \
                                                      dataframe[cluster_assignment_sa==1]
        
    
    # Package relevant variables for the child clusters
    cluster_left_child  = {'matrix': data_matrix_left_child,
                           'dataframe': dataframe_left_child,
                           'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}
    
    return (cluster_left_child, cluster_right_child)
```

The following cell performs bipartitioning of the Wikipedia dataset. Allow 20-60 seconds to finish.

Note. For the purpose of the assignment, we set an explicit seed (`seed=1`) to produce identical outputs for every run. In pratical applications, you might want to use different random seeds for all runs.


```python
wiki_data = {'matrix': tf_idf, 'dataframe': wiki} # no 'centroid' for the root cluster
left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=8, seed=1)
```

Let's examine the contents of one of the two clusters, which we call the `left_child`, referring to the tree visualization above.


```python
left_child
```




    {'centroid': array([  0.00000000e+00,   8.57526623e-06,   0.00000000e+00, ...,
              1.38560691e-04,   6.46049863e-05,   2.26551103e-05]),
     'dataframe': Columns:
     	URI	str
     	name	str
     	text	str
     	tf_idf	dict
     
     Rows: Unknown
     
     Data:
     +-------------------------------+-------------------------------+
     |              URI              |              name             |
     +-------------------------------+-------------------------------+
     | <http://dbpedia.org/resour... |         Digby Morrell         |
     | <http://dbpedia.org/resour... | Paddy Dunne (Gaelic footba... |
     | <http://dbpedia.org/resour... |         Ceiron Thomas         |
     | <http://dbpedia.org/resour... |          Adel Sellimi         |
     | <http://dbpedia.org/resour... |          Vic Stasiuk          |
     | <http://dbpedia.org/resour... |          Leon Hapgood         |
     | <http://dbpedia.org/resour... |           Dom Flora           |
     | <http://dbpedia.org/resour... |           Bob Reece           |
     | <http://dbpedia.org/resour... | Bob Adams (American football) |
     | <http://dbpedia.org/resour... |           Marc Logan          |
     +-------------------------------+-------------------------------+
     +-------------------------------+-------------------------------+
     |              text             |             tf_idf            |
     +-------------------------------+-------------------------------+
     | digby morrell born 10 octo... | {'since': 1.45537671730804... |
     | paddy dunne was a gaelic f... | {'all': 3.2862224869824943... |
     | ceiron thomas born 23 octo... | {'thomas': 19.921640781374... |
     | adel sellimi arabic was bo... | {'coach': 5.44426411898705... |
     | victor john stasiuk born m... | {'leagues': 3.892260543300... |
     | leon duane hapgood born 7 ... | {'albion': 5.6732894101834... |
     | dominick a dom flora born ... | {'all': 1.6431112434912472... |
     | robert scott reece born ja... | {'leagues': 3.892260543300... |
     | robert bruce bob adams bor... | {'coach': 8.16639617848058... |
     | marc anthony logan born ma... | {'cincinnati': 4.392081929... |
     +-------------------------------+-------------------------------+
     [? rows x 4 columns]
     Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
     You can use sf.materialize() to force materialization.,
     'matrix': <11510x547979 sparse matrix of type '<type 'numpy.float64'>'
     	with 1885831 stored elements in Compressed Sparse Row format>}



And here is the content of the other cluster we named `right_child`.


```python
right_child
```




    {'centroid': array([  3.00882137e-06,   0.00000000e+00,   2.88868244e-06, ...,
              1.10291526e-04,   9.00609890e-05,   2.03703564e-05]),
     'dataframe': Columns:
     	URI	str
     	name	str
     	text	str
     	tf_idf	dict
     
     Rows: Unknown
     
     Data:
     +-------------------------------+---------------------+
     |              URI              |         name        |
     +-------------------------------+---------------------+
     | <http://dbpedia.org/resour... |    Alfred J. Lewy   |
     | <http://dbpedia.org/resour... |    Harpdog Brown    |
     | <http://dbpedia.org/resour... | Franz Rottensteiner |
     | <http://dbpedia.org/resour... |        G-Enka       |
     | <http://dbpedia.org/resour... |    Sam Henderson    |
     | <http://dbpedia.org/resour... |    Aaron LaCrate    |
     | <http://dbpedia.org/resour... |   Trevor Ferguson   |
     | <http://dbpedia.org/resour... |     Grant Nelson    |
     | <http://dbpedia.org/resour... |     Cathy Caruth    |
     | <http://dbpedia.org/resour... |     Sophie Crumb    |
     +-------------------------------+---------------------+
     +-------------------------------+-------------------------------+
     |              text             |             tf_idf            |
     +-------------------------------+-------------------------------+
     | alfred j lewy aka sandy le... | {'precise': 6.443200606955... |
     | harpdog brown is a singer ... | {'just': 2.700729968710864... |
     | franz rottensteiner born i... | {'all': 1.6431112434912472... |
     | henry krvits born 30 decem... | {'legendary': 4.2808562943... |
     | sam henderson born october... | {'now': 1.96695239252401, ... |
     | aaron lacrate is an americ... | {'exclusive': 10.455187230... |
     | trevor ferguson aka john f... | {'taxi': 6.052021456094502... |
     | grant nelson born 27 april... | {'houston': 3.935505942157... |
     | cathy caruth born 1955 is ... | {'phenomenon': 5.750053426... |
     | sophia violet sophie crumb... | {'zwigoff': 20.58669641733... |
     +-------------------------------+-------------------------------+
     [? rows x 4 columns]
     Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
     You can use sf.materialize() to force materialization.,
     'matrix': <47561x547979 sparse matrix of type '<type 'numpy.float64'>'
     	with 8493452 stored elements in Compressed Sparse Row format>}



## Visualize the bipartition

We provide you with a modified version of the visualization function from the k-means assignment. For each cluster, we print the top 5 words with highest TF-IDF weights in the centroid and display excerpts for the 8 nearest neighbors of the centroid.


```python
def display_single_tf_idf_cluster(cluster, map_index_to_word):
    '''map_index_to_word: SFrame specifying the mapping betweeen words and column indices'''
    
    wiki_subset   = cluster['dataframe']
    tf_idf_subset = cluster['matrix']
    centroid      = cluster['centroid']
    
    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroid.argsort()[::-1]
    for i in xrange(5):
        print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroid[idx[i]])),
    print('')
    
    # Compute distances from the centroid to all data points in the cluster.
    distances = pairwise_distances(tf_idf_subset, [centroid], metric='euclidean').flatten()
    # compute nearest neighbors of the centroid within the cluster.
    nearest_neighbors = distances.argsort()
    # For 8 nearest neighbors, print the title as well as first 180 characters of text.
    # Wrap the text at 80-character mark.
    for i in xrange(8):
        text = ' '.join(wiki_subset[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        print('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset[nearest_neighbors[i]]['name'],
              distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
    print('')
```

Let's visualize the two child clusters:


```python
display_single_tf_idf_cluster(left_child, map_index_to_word)
```

    league:0.040 season:0.036 team:0.029 football:0.029 played:0.028 
    * Todd Williams                                      0.95468
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    * Gord Sherven                                       0.95622
      gordon r sherven born august 21 1963 in gravelbourg saskatchewan and raised in mankota sas
      katchewan is a retired canadian professional ice hockey forward who played
    * Justin Knoedler                                    0.95639
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    * Chris Day                                          0.95648
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Tony Smith (footballer, born 1957)                 0.95653
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Ashley Prescott                                    0.95761
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * Leslie Lea                                         0.95802
      leslie lea born 5 october 1942 in manchester is an english former professional footballer 
      he played as a midfielderlea began his professional career with blackpool
    * Tommy Anderson (footballer)                        0.95818
      thomas cowan tommy anderson born 24 september 1934 in haddington is a scottish former prof
      essional footballer he played as a forward and was noted for
    
    


```python
display_single_tf_idf_cluster(right_child, map_index_to_word)
```

    she:0.025 her:0.017 music:0.012 he:0.011 university:0.011 
    * Anita Kunz                                         0.97401
      anita e kunz oc born 1956 is a canadianborn artist and illustratorkunz has lived in london
       new york and toronto contributing to magazines and working
    * Janet Jackson                                      0.97472
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Madonna (entertainer)                              0.97475
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * %C3%81ine Hyland                                   0.97536
      ine hyland ne donlon is emeritus professor of education and former vicepresident of univer
      sity college cork ireland she was born in 1942 in athboy co
    * Jane Fonda                                         0.97621
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    * Christine Robertson                                0.97643
      christine mary robertson born 5 october 1948 is an australian politician and former austra
      lian labor party member of the new south wales legislative council serving
    * Pat Studdy-Clift                                   0.97643
      pat studdyclift is an australian author specialising in historical fiction and nonfictionb
      orn in 1925 she lived in gunnedah until she was sent to a boarding
    * Alexandra Potter                                   0.97646
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    
    

The left cluster consists of athletes, whereas the right cluster consists of non-athletes. So far, we have a single-level hierarchy consisting of two clusters, as follows:

```
                                           Wikipedia
                                               +
                                               |
                    +--------------------------+--------------------+
                    |                                               |
                    +                                               +
                 Athletes                                      Non-athletes
```

Is this hierarchy good enough? **When building a hierarchy of clusters, we must keep our particular application in mind.** For instance, we might want to build a **directory** for Wikipedia articles. A good directory would let you quickly narrow down your search to a small set of related articles. The categories of athletes and non-athletes are too general to facilitate efficient search. For this reason, we decide to build another level into our hierarchy of clusters with the goal of getting more specific cluster structure at the lower level. To that end, we subdivide both the `athletes` and `non-athletes` clusters.

## Perform recursive bipartitioning

### Cluster of athletes

To help identify the clusters we've built so far, let's give them easy-to-read aliases:


```python
athletes = left_child
non_athletes = right_child
```

Using the bipartition function, we produce two child clusters of the athlete cluster:


```python
# Bipartition the cluster of athletes
left_child_athletes, right_child_athletes = bipartition(athletes, maxiter=100, num_runs=8, seed=1)
```

The left child cluster mainly consists of baseball players:


```python
display_single_tf_idf_cluster(left_child_athletes, map_index_to_word)
```

    baseball:0.111 league:0.103 major:0.051 games:0.046 season:0.045 
    * Steve Springer                                     0.89344
      steven michael springer born february 11 1961 is an american former professional baseball 
      player who appeared in major league baseball as a third baseman and
    * Dave Ford                                          0.89598
      david alan ford born december 29 1956 is a former major league baseball pitcher for the ba
      ltimore orioles born in cleveland ohio ford attended lincolnwest
    * Todd Williams                                      0.89823
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    * Justin Knoedler                                    0.90097
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    * Kevin Nicholson (baseball)                         0.90607
      kevin ronald nicholson born march 29 1976 is a canadian baseball shortstop he played part 
      of the 2000 season for the san diego padres of
    * Joe Strong                                         0.90638
      joseph benjamin strong born september 9 1962 in fairfield california is a former major lea
      gue baseball pitcher who played for the florida marlins from 2000
    * James Baldwin (baseball)                           0.90674
      james j baldwin jr born july 15 1971 is a former major league baseball pitcher he batted a
      nd threw righthanded in his 11season career he
    * James Garcia                                       0.90729
      james robert garcia born february 3 1980 is an american former professional baseball pitch
      er who played in the san francisco giants minor league system as
    
    

On the other hand, the right child cluster is a mix of football players and ice hockey players:


```python
display_single_tf_idf_cluster(right_child_athletes, map_index_to_word)
```

    season:0.034 football:0.033 team:0.031 league:0.029 played:0.027 
    * Gord Sherven                                       0.95562
      gordon r sherven born august 21 1963 in gravelbourg saskatchewan and raised in mankota sas
      katchewan is a retired canadian professional ice hockey forward who played
    * Ashley Prescott                                    0.95656
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * Chris Day                                          0.95656
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Jason Roberts (footballer)                         0.95658
      jason andre davis roberts mbe born 25 january 1978 is a former professional footballer and
       now a football punditborn in park royal london roberts was
    * Todd Curley                                        0.95743
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    * Tony Smith (footballer, born 1957)                 0.95801
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Sol Campbell                                       0.95802
      sulzeer jeremiah sol campbell born 18 september 1974 is a former england international foo
      tballer a central defender he had a 19year career playing in the
    * Richard Ambrose                                    0.95924
      richard ambrose born 10 june 1972 is a former australian rules footballer who played with 
      the sydney swans in the australian football league afl he
    
    

**Note**. Concerning use of "football"

The occurrences of the word "football" above refer to [association football](https://en.wikipedia.org/wiki/Association_football). This sports is also known as "soccer" in United States (to avoid confusion with [American football](https://en.wikipedia.org/wiki/American_football)). We will use "football" throughout when discussing topic representation.

Our hierarchy of clusters now looks like this:
```
                                           Wikipedia
                                               +
                                               |
                    +--------------------------+--------------------+
                    |                                               |
                    +                                               +
                 Athletes                                      Non-athletes
                    +
                    |
        +-----------+--------+
        |                    |
        |                    +
        +                 football/
     baseball            ice hockey
```

Should we keep subdividing the clusters? If so, which cluster should we subdivide? To answer this question, we again think about our application. Since we organize our directory by topics, it would be nice to have topics that are about as coarse as each other. For instance, if one cluster is about baseball, we expect some other clusters about football, basketball, volleyball, and so forth. That is, **we would like to achieve similar level of granularity for all clusters.**

Notice that the right child cluster is more coarse than the left child cluster. The right cluster posseses a greater variety of topics than the left (ice hockey/football vs. baseball). So the right child cluster should be subdivided further to produce finer child clusters.

Let's give the clusters aliases as well:


```python
baseball            = left_child_athletes
ice_hockey_football = right_child_athletes
```

### Cluster of ice hockey players and football players

In answering the following quiz question, take a look at the topics represented in the top documents (those closest to the centroid), as well as the list of words with highest TF-IDF weights.

**Quiz Question**. Bipartition the cluster of ice hockey and football players. Which of the two child clusters should be futher subdivided?

**Note**. To achieve consistent results, use the arguments `maxiter=100, num_runs=8, seed=1` when calling the `bipartition` function.

1. The left child cluster
2. The right child cluster


```python
left_child_cluster, right_child_cluster = bipartition(ice_hockey_football, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child_cluster, map_index_to_word)
```

    football:0.048 season:0.043 league:0.041 played:0.036 coach:0.034 
    * Todd Curley                                        0.94582
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    * Tony Smith (footballer, born 1957)                 0.94609
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Chris Day                                          0.94626
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Jason Roberts (footballer)                         0.94635
      jason andre davis roberts mbe born 25 january 1978 is a former professional footballer and
       now a football punditborn in park royal london roberts was
    * Ashley Prescott                                    0.94635
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * David Hamilton (footballer)                        0.94928
      david hamilton born 7 november 1960 is an english former professional association football
       player who played as a midfielder he won caps for the england
    * Richard Ambrose                                    0.94944
      richard ambrose born 10 june 1972 is a former australian rules footballer who played with 
      the sydney swans in the australian football league afl he
    * Neil Grayson                                       0.94960
      neil grayson born 1 november 1964 in york is an english footballer who last played as a st
      riker for sutton towngraysons first club was local
    
    


```python
display_single_tf_idf_cluster(right_child_cluster, map_index_to_word)
```

    championships:0.045 tour:0.044 championship:0.035 world:0.031 won:0.031 
    * Alessandra Aguilar                                 0.93847
      alessandra aguilar born 1 july 1978 in lugo is a spanish longdistance runner who specialis
      es in marathon running she represented her country in the event
    * Heather Samuel                                     0.93964
      heather barbara samuel born 6 july 1970 is a retired sprinter from antigua and barbuda who
       specialized in the 100 and 200 metres in 1990
    * Viola Kibiwot                                      0.94006
      viola jelagat kibiwot born december 22 1983 in keiyo district is a runner from kenya who s
      pecialises in the 1500 metres kibiwot won her first
    * Ayelech Worku                                      0.94022
      ayelech worku born june 12 1979 is an ethiopian longdistance runner most known for winning
       two world championships bronze medals on the 5000 metres she
    * Krisztina Papp                                     0.94068
      krisztina papp born 17 december 1982 in eger is a hungarian long distance runner she is th
      e national indoor record holder over 5000 mpapp began
    * Petra Lammert                                      0.94209
      petra lammert born 3 march 1984 in freudenstadt badenwrttemberg is a former german shot pu
      tter and current bobsledder she was the 2009 european indoor champion
    * Morhad Amdouni                                     0.94210
      morhad amdouni born 21 january 1988 in portovecchio is a french middle and longdistance ru
      nner he was european junior champion in track and cross country
    * Brian Davis (golfer)                               0.94360
      brian lester davis born 2 august 1974 is an english professional golferdavis was born in l
      ondon he turned professional in 1994 and became a member
    
    

**Caution**. The granularity criteria is an imperfect heuristic and must be taken with a grain of salt. It takes a lot of manual intervention to obtain a good hierarchy of clusters.

* **If a cluster is highly mixed, the top articles and words may not convey the full picture of the cluster.** Thus, we may be misled if we judge the purity of clusters solely by their top documents and words. 
* **Many interesting topics are hidden somewhere inside the clusters but do not appear in the visualization.** We may need to subdivide further to discover new topics. For instance, subdividing the `ice_hockey_football` cluster led to the appearance of golf.

**Quiz Question**. Which diagram best describes the hierarchy right after splitting the `ice_hockey_football` cluster? Refer to the quiz form for the diagrams.


```python
left_child_cluster, right_child_cluster = bipartition(ice_hockey_football, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child_cluster, map_index_to_word)
```

    football:0.048 season:0.043 league:0.041 played:0.036 coach:0.034 
    * Todd Curley                                        0.94582
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    * Tony Smith (footballer, born 1957)                 0.94609
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Chris Day                                          0.94626
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Jason Roberts (footballer)                         0.94635
      jason andre davis roberts mbe born 25 january 1978 is a former professional footballer and
       now a football punditborn in park royal london roberts was
    * Ashley Prescott                                    0.94635
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * David Hamilton (footballer)                        0.94928
      david hamilton born 7 november 1960 is an english former professional association football
       player who played as a midfielder he won caps for the england
    * Richard Ambrose                                    0.94944
      richard ambrose born 10 june 1972 is a former australian rules footballer who played with 
      the sydney swans in the australian football league afl he
    * Neil Grayson                                       0.94960
      neil grayson born 1 november 1964 in york is an english footballer who last played as a st
      riker for sutton towngraysons first club was local
    
    


```python
display_single_tf_idf_cluster(right_child_cluster, map_index_to_word)
```

    championships:0.045 tour:0.044 championship:0.035 world:0.031 won:0.031 
    * Alessandra Aguilar                                 0.93847
      alessandra aguilar born 1 july 1978 in lugo is a spanish longdistance runner who specialis
      es in marathon running she represented her country in the event
    * Heather Samuel                                     0.93964
      heather barbara samuel born 6 july 1970 is a retired sprinter from antigua and barbuda who
       specialized in the 100 and 200 metres in 1990
    * Viola Kibiwot                                      0.94006
      viola jelagat kibiwot born december 22 1983 in keiyo district is a runner from kenya who s
      pecialises in the 1500 metres kibiwot won her first
    * Ayelech Worku                                      0.94022
      ayelech worku born june 12 1979 is an ethiopian longdistance runner most known for winning
       two world championships bronze medals on the 5000 metres she
    * Krisztina Papp                                     0.94068
      krisztina papp born 17 december 1982 in eger is a hungarian long distance runner she is th
      e national indoor record holder over 5000 mpapp began
    * Petra Lammert                                      0.94209
      petra lammert born 3 march 1984 in freudenstadt badenwrttemberg is a former german shot pu
      tter and current bobsledder she was the 2009 european indoor champion
    * Morhad Amdouni                                     0.94210
      morhad amdouni born 21 january 1988 in portovecchio is a french middle and longdistance ru
      nner he was european junior champion in track and cross country
    * Brian Davis (golfer)                               0.94360
      brian lester davis born 2 august 1974 is an english professional golferdavis was born in l
      ondon he turned professional in 1994 and became a member
    
    


```python
left_child_cluster, right_child_cluster = bipartition(right_child_cluster, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child_cluster, map_index_to_word)
```

    championships:0.052 world:0.035 championship:0.033 she:0.031 team:0.030 
    * Heather Samuel                                     0.93046
      heather barbara samuel born 6 july 1970 is a retired sprinter from antigua and barbuda who
       specialized in the 100 and 200 metres in 1990
    * Alessandra Aguilar                                 0.93142
      alessandra aguilar born 1 july 1978 in lugo is a spanish longdistance runner who specialis
      es in marathon running she represented her country in the event
    * Viola Kibiwot                                      0.93148
      viola jelagat kibiwot born december 22 1983 in keiyo district is a runner from kenya who s
      pecialises in the 1500 metres kibiwot won her first
    * Ayelech Worku                                      0.93160
      ayelech worku born june 12 1979 is an ethiopian longdistance runner most known for winning
       two world championships bronze medals on the 5000 metres she
    * Krisztina Papp                                     0.93207
      krisztina papp born 17 december 1982 in eger is a hungarian long distance runner she is th
      e national indoor record holder over 5000 mpapp began
    * Antonina Yefremova                                 0.93520
      antonina yefremova born 19 july 1981 is a ukrainian sprinter who specializes in the 400 me
      tres yefremova received a twoyear ban in 2012 for using
    * Morhad Amdouni                                     0.93595
      morhad amdouni born 21 january 1988 in portovecchio is a french middle and longdistance ru
      nner he was european junior champion in track and cross country
    * Shitaye Eshete                                     0.93615
      shitaye eshete habtegebrel born 21 may 1990 is an ethiopianborn longdistance runner who co
      mpetes internationally for bahrainshe first began competing for the oilrich gulf state
    
    


```python
display_single_tf_idf_cluster(right_child_cluster, map_index_to_word)
```

    tour:0.254 pga:0.210 golf:0.137 open:0.073 golfer:0.062 
    * Bob Heintz                                         0.75967
      robert edward heintz born may 1 1970 is an american professional golfer who plays on the n
      ationwide tourheintz was born in syosset new york he
    * Sonny Skinner                                      0.76478
      sonny skinner born august 18 1960 is an american professional golfer who plays on the cham
      pions tourskinner was born in portsmouth virginia he turned professional
    * Todd Barranger                                     0.77142
      todd barranger born october 19 1968 is an american professional golfer who played on the p
      ga tour asian tour and the nationwide tourbarranger joined the
    * Tim Conley                                         0.77167
      tim conley born december 8 1958 is an american professional golfer who played on the pga t
      our nationwide tour and most recently the champions tourconley
    * Bruce Zabriski                                     0.77427
      bruce zabriski born august 3 1957 is an american professional golfer who played on the pga
       tour european tour and the nationwide tourzabriski joined the
    * Ted Purdy                                          0.77965
      theodore townsend purdy born august 15 1973 is an american professional golfer purdy was b
      orn in phoenix arizona he graduated from brophy college preparatory in
    * Russell Knox                                       0.78116
      russell knox born 21 june 1985 is a scottish professional golfer who plays on the pga tour
      knox played on the nga hooters tour from 2008
    * Dick Mast                                          0.78330
      richard mast born march 23 1951 is an american professional golfer who has played on the p
      ga tour nationwide tour and champions tourmast was born
    
    

### Cluster of non-athletes

Now let us subdivide the cluster of non-athletes.


```python
# Bipartition the cluster of non-athletes
left_child_non_athletes, right_child_non_athletes = bipartition(non_athletes, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child_non_athletes, map_index_to_word)
```

    university:0.016 he:0.013 she:0.013 law:0.012 served:0.012 
    * Barry Sullivan (lawyer)                            0.97227
      barry sullivan is a chicago lawyer and as of july 1 2009 the cooney conway chair in advoca
      cy at loyola university chicago school of law
    * Kayee Griffin                                      0.97444
      kayee frances griffin born 6 february 1950 is an australian politician and former australi
      an labor party member of the new south wales legislative council serving
    * Christine Robertson                                0.97450
      christine mary robertson born 5 october 1948 is an australian politician and former austra
      lian labor party member of the new south wales legislative council serving
    * James A. Joseph                                    0.97464
      james a joseph born 1935 is an american former diplomatjoseph is professor of the practice
       of public policy studies at duke university and founder of
    * David Anderson (British Columbia politician)       0.97492
      david a anderson pc oc born august 16 1937 in victoria british columbia is a former canadi
      an cabinet minister educated at victoria college in victoria
    * Mary Ellen Coster Williams                         0.97594
      mary ellen coster williams born april 3 1953 is a judge of the united states court of fede
      ral claims appointed to that court in 2003
    * Sven Erik Holmes                                   0.97600
      sven erik holmes is a former federal judge and currently the vice chairman legal risk and 
      regulatory and chief legal officer for kpmg llp a
    * Andrew Fois                                        0.97652
      andrew fois is an attorney living and working in washington dc as of april 9 2012 he will 
      be serving as the deputy attorney general
    
    


```python
display_single_tf_idf_cluster(right_child_non_athletes, map_index_to_word)
```

    she:0.039 her:0.030 music:0.023 film:0.021 album:0.015 
    * Madonna (entertainer)                              0.96092
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * Janet Jackson                                      0.96153
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Cher                                               0.96540
      cher r born cherilyn sarkisian may 20 1946 is an american singer actress and television ho
      st described as embodying female autonomy in a maledominated industry
    * Laura Smith                                        0.96600
      laura smith is a canadian folk singersongwriter she is best known for her 1995 single shad
      e of your love one of the years biggest hits
    * Natashia Williams                                  0.96677
      natashia williamsblach born august 2 1978 is an american actress and former wonderbra camp
      aign model who is perhaps best known for her role as shane
    * Anita Kunz                                         0.96716
      anita e kunz oc born 1956 is a canadianborn artist and illustratorkunz has lived in london
       new york and toronto contributing to magazines and working
    * Maggie Smith                                       0.96747
      dame margaret natalie maggie smith ch dbe born 28 december 1934 is an english actress she 
      made her stage debut in 1952 and has had
    * Lizzie West                                        0.96752
      lizzie west born in brooklyn ny on july 21 1973 is a singersongwriter her music can be des
      cribed as a blend of many genres including
    
    

The first cluster consists of scholars, politicians, and government officials whereas the second consists of musicians, artists, and actors. Run the following code cell to make convenient aliases for the clusters.


```python
scholars_politicians_etc = left_child_non_athletes
musicians_artists_etc = right_child_non_athletes
```

**Quiz Question**. Let us bipartition the clusters `scholars_politicians_etc` and `musicians_artists_etc`. Which diagram best describes the resulting hierarchy of clusters for the non-athletes? Refer to the quiz for the diagrams.

**Note**. Use `maxiter=100, num_runs=8, seed=1` for consistency of output.


```python
left_child_scholars_politicians, right_child_scholars_politicians = bipartition(scholars_politicians_etc, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child_scholars_politicians, map_index_to_word)
```

    party:0.039 election:0.036 minister:0.033 she:0.028 elected:0.026 
    * Kayee Griffin                                      0.95170
      kayee frances griffin born 6 february 1950 is an australian politician and former australi
      an labor party member of the new south wales legislative council serving
    * Marcelle Mersereau                                 0.95417
      marcelle mersereau born february 14 1942 in pointeverte new brunswick is a canadian politi
      cian a civil servant for most of her career she also served
    * Lucienne Robillard                                 0.95453
      lucienne robillard pc born june 16 1945 is a canadian politician and a member of the liber
      al party of canada she sat in the house
    * Maureen Lyster                                     0.95590
      maureen anne lyster born 10 september 1943 is an australian politician she was an australi
      an labor party member of the victorian legislative assembly from 1985
    * Liz Cunningham                                     0.95690
      elizabeth anne liz cunningham is an australian politician she was an independent member of
       the legislative assembly of queensland from 1995 to 2015 representing the
    * Carol Skelton                                      0.95780
      carol skelton pc born december 12 1945 in biggar saskatchewan is a canadian politician she
       is a member of the security intelligence review committee which
    * Stephen Harper                                     0.95816
      stephen joseph harper pc mp born april 30 1959 is a canadian politician who is the 22nd an
      d current prime minister of canada and the
    * Doug Lewis                                         0.95875
      douglas grinslade doug lewis pc qc born april 17 1938 is a former canadian politician a ch
      artered accountant and lawyer by training lewis entered the
    
    


```python
politicians, scholars = bipartition(left_child_scholars_politicians, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(politicians, map_index_to_word)
```

    district:0.048 law:0.043 republican:0.041 senate:0.039 court:0.038 
    * Jean Constance Hamilton                            0.93732
      jean constance hamilton born 1945 is a senior united states district judge of the united s
      tates district court for the eastern district of missouriborn in
    * Audrey B. Collins                                  0.93936
      audrey b collins born 1945 is a former united states district judge and an associate justi
      ce of the second district court of appeal for the
    * George B. Daniels                                  0.94002
      george benjamin daniels born 1953 is a united states federal judge for the united states d
      istrict court for the southern district of new yorkdaniels was
    * Levin H. Campbell                                  0.94154
      levin hicks campbell born january 2 1927 is an american federal appellate judge on senior 
      status with the united states court of appeals for the
    * Carol Bagley Amon                                  0.94190
      carol bagley amon born 1946 is the chief judge of the united states district court for the
       eastern district of new yorkamon was born in
    * James G. Carr                                      0.94226
      james g carr born july 7 1940 is a federal district judge for the united states district c
      ourt for the northern district of ohiocarr was
    * Carol E. Jackson                                   0.94300
      carol e jackson born 1952 is a united states federal judgeborn in st louis missouri jackso
      n received a ba from wellesley college in 1973 followed
    * Margaret B. Seymour                                0.94414
      margaret b seymour born 1947 is a senior united states district judgeborn in washington dc
       seymour received a ba from howard university in 1969 and
    
    


```python
left, right = bipartition(politicians, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left, map_index_to_word)
```

    republican:0.054 senate:0.045 district:0.040 state:0.037 democratic:0.034 
    * Bob Smith (American politician)                    0.93546
      robert c smith redirects here for the 19th century british astrologer see robert cross smi
      throbert clinton bob smith born march 30 1941 is an american
    * Bob Menendez                                       0.93731
      robert bob menendez born january 1 1954 is the senior united states senator from new jerse
      y he is a member of the democratic party first
    * Micheal R. Williams                                0.94093
      micheal r williams born february 6 1955 in knoxville tennessee is a tennessee politician w
      ho formerly served in the tennessee state senate and was elected
    * Cleo Fields                                        0.94267
      cleo c fields born november 22 1962 is an american attorney politician and member of the d
      emocratic party from the us state of louisiana he
    * Steve Rauschenberger                               0.94395
      steve rauschenberger born august 29 1956 elgin illinois served as a republican member of t
      he illinois state senate from 1993 to 2007 he was first
    * Winston Bryant                                     0.94475
      winston bryant born october 3 1938 is a former democratic secretary of state 19771978 the 
      tenth lieutenant governor 19811991 and attorney general 19911999 of the
    * Elizabeth Ames Jones                               0.94491
      elizabeth ames jones born october 29 1956 is a former member of the texas house of represe
      ntatives and the texas railroad commission the regulatory body
    * Clyde C. Holloway                                  0.94549
      clyde cecil holloway born november 28 1943 is an american politician small business owner 
      and member of the republican party who currently serves as one
    
    


```python
display_single_tf_idf_cluster(right, map_index_to_word)
```

    court:0.122 law:0.105 judge:0.099 district:0.069 justice:0.056 
    * George B. Daniels                                  0.87157
      george benjamin daniels born 1953 is a united states federal judge for the united states d
      istrict court for the southern district of new yorkdaniels was
    * Jean Constance Hamilton                            0.87804
      jean constance hamilton born 1945 is a senior united states district judge of the united s
      tates district court for the eastern district of missouriborn in
    * James G. Carr                                      0.88299
      james g carr born july 7 1940 is a federal district judge for the united states district c
      ourt for the northern district of ohiocarr was
    * William G. Young                                   0.88375
      william glover young born 1940 is a united states federal judge for the district of massac
      husetts young was born in huntington new york he attended
    * D. Brock Hornby                                    0.88448
      david brock hornby born april 21 1944 is a federal judge serving on the united states dist
      rict court for the district of maineborn in brandon
    * Levin H. Campbell                                  0.89139
      levin hicks campbell born january 2 1927 is an american federal appellate judge on senior 
      status with the united states court of appeals for the
    * Carol Bagley Amon                                  0.89310
      carol bagley amon born 1946 is the chief judge of the united states district court for the
       eastern district of new yorkamon was born in
    * Sandra J. Feuerstein                               0.89413
      sandra jeanne feuerstein born 1946 is a senior judge of the united states district court f
      or the eastern district of new yorkborn in new york
    
    


```python
display_single_tf_idf_cluster(scholars, map_index_to_word)
```

    party:0.057 minister:0.054 election:0.046 elected:0.029 she:0.028 
    * Lucienne Robillard                                 0.93913
      lucienne robillard pc born june 16 1945 is a canadian politician and a member of the liber
      al party of canada she sat in the house
    * Stephen Harper                                     0.94085
      stephen joseph harper pc mp born april 30 1959 is a canadian politician who is the 22nd an
      d current prime minister of canada and the
    * Marcelle Mersereau                                 0.94462
      marcelle mersereau born february 14 1942 in pointeverte new brunswick is a canadian politi
      cian a civil servant for most of her career she also served
    * Kayee Griffin                                      0.94499
      kayee frances griffin born 6 february 1950 is an australian politician and former australi
      an labor party member of the new south wales legislative council serving
    * Monique Landry                                     0.94550
      monique landry pc born december 25 1937 is a former canadian politician a physiotherapist 
      and administrator she was first elected to the canadian house of
    * Carol Skelton                                      0.94575
      carol skelton pc born december 12 1945 in biggar saskatchewan is a canadian politician she
       is a member of the security intelligence review committee which
    * Maureen Lyster                                     0.94604
      maureen anne lyster born 10 september 1943 is an australian politician she was an australi
      an labor party member of the victorian legislative assembly from 1985
    * Paul Martin                                        0.94659
      paul edgar philippe martin pc cc born august 28 1938 also known as paul martin jr is a can
      adian politician who was the 21st prime
    
    


```python
display_single_tf_idf_cluster(right_child_scholars_politicians, map_index_to_word)
```

    university:0.018 research:0.015 professor:0.013 he:0.013 president:0.010 
    * Lawrence W. Green                                  0.97495
      lawrence w green is best known by health education researchers as the originator of the pr
      ecede model and codeveloper of the precedeproceed model which has
    * James A. Joseph                                    0.97506
      james a joseph born 1935 is an american former diplomatjoseph is professor of the practice
       of public policy studies at duke university and founder of
    * Timothy Luke                                       0.97584
      timothy w luke is university distinguished professor of political science in the college o
      f liberal arts and human sciences as well as program chair of
    * Archie Brown                                       0.97628
      archibald haworth brown cmg fba commonly known as archie brown born 10 may 1938 is a briti
      sh political scientist and historian in 2005 he became
    * Jerry L. Martin                                    0.97687
      jerry l martin is chairman emeritus of the american council of trustees and alumni he serv
      ed as president of acta from its founding in 1995
    * Ren%C3%A9e Fox                                     0.97713
      rene c fox a summa cum laude graduate of smith college in 1949 earned her phd in sociology
       in 1954 from radcliffe college harvard university
    * Robert Bates (political scientist)                 0.97732
      robert hinrichs bates born 1942 is an american political scientist he is eaton professor o
      f the science of government in the departments of government and
    * Ferdinand K. Levy                                  0.97737
      ferdinand k levy was a famous management scientist with several important contributions to
       system analysis he was a professor at georgia tech from 1972 until
    
    


```python
left_child_musicians_artists, right_child_musicians_artists = bipartition(musicians_artists_etc, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child_musicians_artists, map_index_to_word)
```

    she:0.124 her:0.092 film:0.015 actress:0.015 music:0.014 
    * Janet Jackson                                      0.93373
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Barbara Hershey                                    0.93506
      barbara hershey born barbara lynn herzstein february 5 1948 once known as barbara seagull 
      is an american actress in a career spanning nearly 50 years
    * Lauren Royal                                       0.93716
      lauren royal born march 3 circa 1965 is a book writer from california royal has written bo
      th historic and novelistic booksa selfproclaimed angels baseball fan
    * Alexandra Potter                                   0.93802
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    * Cher                                               0.93804
      cher r born cherilyn sarkisian may 20 1946 is an american singer actress and television ho
      st described as embodying female autonomy in a maledominated industry
    * Madonna (entertainer)                              0.93806
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * Jane Fonda                                         0.93836
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    * Ellina Graypel                                     0.93926
      ellina graypel born july 19 1972 is an awardwinning russian singersongwriter she was born 
      near the volga river in the heart of russia she spent
    
    


```python
left_child, right_child = bipartition(left_child_musicians_artists, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child, map_index_to_word)
```

    miss:0.354 pageant:0.202 usa:0.120 she:0.114 her:0.064 
    * Tanya Wilson                                       0.69478
      tanya wilson born june 13 1950 is a beauty queen from honolulu hawaii who held the miss us
      a 1972 titlewilson placed second runnerup in the
    * Amy Crawford (pageant titleholder)                 0.69632
      amy crawford is a beauty queen from auburn washington who has competed in the miss usa pag
      eantcrawford won the miss washington usa 2005 title in
    * Katee Doland                                       0.69895
      katee dolandmink born 1980 in arvada colorado is a former beauty pageant titleholder who r
      epresented colorado in miss teen usa 1998 miss usa 2001 and
    * Ellen Chapman                                      0.70382
      ellen chapman keegan born 1982 is a beauty queen from san jose california who has competed
       in the miss usa pageantchapmans first pageant title was
    * Marin Poole                                        0.70792
      marin morgan poole is a beauty queen from logan utah who has competed in the miss teen usa
       and miss usa pageantspoole won the miss
    * Amber Copley                                       0.71379
      amber brooke copley born august 2 1985 is a beauty queen from abingdon virginia who has co
      mpeted in the miss teen usa and miss usa
    * Kasi Kelly                                         0.71669
      kasi laine kelly born october 1 1981 is a beauty queen from bridgeport texas who has compe
      ted in the miss usa pageant she was born
    * Jennifer Dupont                                    0.71968
      jennifer dupont born march 24 1981 in shreveport louisiana is an american beauty pageant c
      ontestant who represented louisiana in miss teen usa 1998 miss usa
    
    


```python
display_single_tf_idf_cluster(right_child, map_index_to_word)
```

    she:0.125 her:0.093 film:0.016 actress:0.015 music:0.014 
    * Janet Jackson                                      0.93311
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Barbara Hershey                                    0.93446
      barbara hershey born barbara lynn herzstein february 5 1948 once known as barbara seagull 
      is an american actress in a career spanning nearly 50 years
    * Lauren Royal                                       0.93680
      lauren royal born march 3 circa 1965 is a book writer from california royal has written bo
      th historic and novelistic booksa selfproclaimed angels baseball fan
    * Madonna (entertainer)                              0.93744
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * Cher                                               0.93745
      cher r born cherilyn sarkisian may 20 1946 is an american singer actress and television ho
      st described as embodying female autonomy in a maledominated industry
    * Alexandra Potter                                   0.93754
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    * Jane Fonda                                         0.93789
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    * Ellina Graypel                                     0.93940
      ellina graypel born july 19 1972 is an awardwinning russian singersongwriter she was born 
      near the volga river in the heart of russia she spent
    
    


```python
display_single_tf_idf_cluster(right_child_musicians_artists, map_index_to_word)
```

    music:0.027 film:0.023 album:0.017 band:0.016 art:0.015 
    * Julian Knowles                                     0.96904
      julian knowles is an australian composer and performer specialising in new and emerging te
      chnologies his creative work spans the fields of composition for theatre dance
    * Peter Combe                                        0.97080
      peter combe born 20 october 1948 is an australian childrens entertainer and musicianmusica
      l genre childrens musiche has had 22 releases including seven gold albums two
    * Craig Pruess                                       0.97121
      craig pruess born 1950 is an american composer musician arranger and gold platinum record 
      producer who has been living in britain since 1973 his career
    * Ceiri Torjussen                                    0.97170
      ceiri torjussen born 1976 is a composer who has contributed music to dozens of film and te
      levision productions in the ushis music was described by
    * Brenton Broadstock                                 0.97192
      brenton broadstock ao born 1952 is an australian composerbroadstock was born in melbourne 
      he studied history politics and music at monash university and later composition
    * Michael Peter Smith                                0.97318
      michael peter smith born september 7 1941 is a chicagobased singersongwriter rolling stone
       magazine once called him the greatest songwriter in the english language he
    * Marc Hoffman                                       0.97357
      marc hoffman born april 16 1961 is a composer of concert music and music for film pianist 
      vocalist recording artist and music educator hoffman grew
    * Tom Bancroft                                       0.97378
      tom bancroft born 1967 london is a british jazz drummer and composer he began drumming age
      d seven and started off playing jazz with his father
    
    


```python
left_child, right_child = bipartition(right_child_musicians_artists, maxiter=100, num_runs=8, seed=1)
```


```python
display_single_tf_idf_cluster(left_child, map_index_to_word)
```

    music:0.055 album:0.037 band:0.035 orchestra:0.022 released:0.020 
    * Brenton Broadstock                                 0.95802
      brenton broadstock ao born 1952 is an australian composerbroadstock was born in melbourne 
      he studied history politics and music at monash university and later composition
    * Prince (musician)                                  0.96148
      prince rogers nelson born june 7 1958 known by his mononym prince is an american singerson
      gwriter multiinstrumentalist and actor he has produced ten platinum albums
    * Tom Bancroft                                       0.96192
      tom bancroft born 1967 london is a british jazz drummer and composer he began drumming age
      d seven and started off playing jazz with his father
    * Julian Knowles                                     0.96210
      julian knowles is an australian composer and performer specialising in new and emerging te
      chnologies his creative work spans the fields of composition for theatre dance
    * Will.i.am                                          0.96256
      william adams born march 15 1975 known by his stage name william pronounced will i am is a
      n american rapper songwriter entrepreneur actor dj record
    * Dan Siegel (musician)                              0.96297
      dan siegel born in seattle washington is a pianist composer and record producer his earlie
      r music has been described as new age while his more
    * Tony Mills (musician)                              0.96311
      tony mills born 7 july 1962 in solihull england is an english rock singer best known for h
      is work with shy and tnthailing from birmingham
    * Don Robertson (composer)                           0.96346
      don robertson born 1942 is an american composerdon robertson was born in 1942 in denver co
      lorado and began studying music with conductor and pianist antonia
    
    


```python
display_single_tf_idf_cluster(right_child, map_index_to_word)
```

    film:0.036 art:0.023 television:0.016 theatre:0.016 series:0.016 
    * Justin Edgar                                       0.96759
      justin edgar is a british film directorborn in handsworth birmingham on 18 august 1971 edg
      ar graduated from portsmouth university in 1996 with a first class
    * Bill Bennett (director)                            0.96795
      bill bennett born 1953 is an australian film director producer and screenwriterhe dropped 
      out of medicine at queensland university in 1972 and joined the australian
    * Paul Swadel                                        0.96824
      paul swadel is a new zealand film director and producerhe has directed and produced many s
      uccessful short films which have screened in competition at cannes
    * Anton Hecht                                        0.96893
      anton hecht is an english artist born in london in 2007 he asked musicians from around the
       durham area to contribute to a soundtrack for
    * Robert Braiden                                     0.96948
      robert braiden is an australian film director and writer born in sydney he grew up in moor
      ebank liverpool new south wales and now currently lives
    * Joseph Laban                                       0.96966
      joseph israel laban is an awardwinning tagalog journalist and independent filmmaker he is 
      also a palancawinning playwright and a fulbright scholar he holds a masters
    * Tikoy Aguiluz                                      0.96993
      amable tikoy aguiluz is an award winning filipino film director film producer screenwriter
       and cinematographer he also founded the cinemanila international film festival in manila
    * Laura Neri                                         0.97003
      laura neri greek is a director of greek and italian origins born in brussels belgium she g
      raduated from the usc school of cinematic arts in
    
    
