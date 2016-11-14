
---

_You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._

---


```python
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
```

# Assignment 4 - Hypothesis Testing
This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

Definitions:
* A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
* A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
* A _recession bottom_ is the quarter within a recession which had the lowest GDP.
* A _university town_ is a city which has a high percentage of university students compared to the total population of the city.

**Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)

The following data files are available for this assignment:
* From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
* From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
* From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.

Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.


```python
# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
```


```python
def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan","Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State","RegionName"]  )'''
    f = open('university_towns.txt')
    State=[]
    RegionName=[]
    statename=None
    for line in f.readlines():
        line = line.strip()
        if line.endswith('[edit]'):
            statename=line[:line.find('[')].strip()
        else:
            State.append(statename)
            if line.find('(') != -1:
                line = line[:line.find('(')]
            RegionName.append(line.strip())
            #RegionName.append(line[:line.find('(')].strip(''))
    ans = pd.DataFrame({'State':State,'RegionName':RegionName},columns=['State','RegionName'])
    return ans
get_list_of_university_towns()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>RegionName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Auburn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alabama</td>
      <td>Florence</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alabama</td>
      <td>Jacksonville</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alabama</td>
      <td>Livingston</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alabama</td>
      <td>Montevallo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Alabama</td>
      <td>Troy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Alabama</td>
      <td>Tuscaloosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alabama</td>
      <td>Tuskegee</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alaska</td>
      <td>Fairbanks</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Arizona</td>
      <td>Flagstaff</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arizona</td>
      <td>Tempe</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Arizona</td>
      <td>Tucson</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Arkansas</td>
      <td>Arkadelphia</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Arkansas</td>
      <td>Conway</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Arkansas</td>
      <td>Fayetteville</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Arkansas</td>
      <td>Jonesboro</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Arkansas</td>
      <td>Magnolia</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Arkansas</td>
      <td>Monticello</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Arkansas</td>
      <td>Russellville</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Arkansas</td>
      <td>Searcy</td>
    </tr>
    <tr>
      <th>20</th>
      <td>California</td>
      <td>Angwin</td>
    </tr>
    <tr>
      <th>21</th>
      <td>California</td>
      <td>Arcata</td>
    </tr>
    <tr>
      <th>22</th>
      <td>California</td>
      <td>Berkeley</td>
    </tr>
    <tr>
      <th>23</th>
      <td>California</td>
      <td>Chico</td>
    </tr>
    <tr>
      <th>24</th>
      <td>California</td>
      <td>Claremont</td>
    </tr>
    <tr>
      <th>25</th>
      <td>California</td>
      <td>Cotati</td>
    </tr>
    <tr>
      <th>26</th>
      <td>California</td>
      <td>Davis</td>
    </tr>
    <tr>
      <th>27</th>
      <td>California</td>
      <td>Irvine</td>
    </tr>
    <tr>
      <th>28</th>
      <td>California</td>
      <td>Isla Vista</td>
    </tr>
    <tr>
      <th>29</th>
      <td>California</td>
      <td>University Park, Los Angeles</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>487</th>
      <td>Virginia</td>
      <td>Wise</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Virginia</td>
      <td>Chesapeake</td>
    </tr>
    <tr>
      <th>489</th>
      <td>Washington</td>
      <td>Bellingham</td>
    </tr>
    <tr>
      <th>490</th>
      <td>Washington</td>
      <td>Cheney</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Washington</td>
      <td>Ellensburg</td>
    </tr>
    <tr>
      <th>492</th>
      <td>Washington</td>
      <td>Pullman</td>
    </tr>
    <tr>
      <th>493</th>
      <td>Washington</td>
      <td>University District, Seattle</td>
    </tr>
    <tr>
      <th>494</th>
      <td>West Virginia</td>
      <td>Athens</td>
    </tr>
    <tr>
      <th>495</th>
      <td>West Virginia</td>
      <td>Buckhannon</td>
    </tr>
    <tr>
      <th>496</th>
      <td>West Virginia</td>
      <td>Fairmont</td>
    </tr>
    <tr>
      <th>497</th>
      <td>West Virginia</td>
      <td>Glenville</td>
    </tr>
    <tr>
      <th>498</th>
      <td>West Virginia</td>
      <td>Huntington</td>
    </tr>
    <tr>
      <th>499</th>
      <td>West Virginia</td>
      <td>Montgomery</td>
    </tr>
    <tr>
      <th>500</th>
      <td>West Virginia</td>
      <td>Morgantown</td>
    </tr>
    <tr>
      <th>501</th>
      <td>West Virginia</td>
      <td>Shepherdstown</td>
    </tr>
    <tr>
      <th>502</th>
      <td>West Virginia</td>
      <td>West Liberty</td>
    </tr>
    <tr>
      <th>503</th>
      <td>Wisconsin</td>
      <td>Appleton</td>
    </tr>
    <tr>
      <th>504</th>
      <td>Wisconsin</td>
      <td>Eau Claire</td>
    </tr>
    <tr>
      <th>505</th>
      <td>Wisconsin</td>
      <td>Green Bay</td>
    </tr>
    <tr>
      <th>506</th>
      <td>Wisconsin</td>
      <td>La Crosse</td>
    </tr>
    <tr>
      <th>507</th>
      <td>Wisconsin</td>
      <td>Madison</td>
    </tr>
    <tr>
      <th>508</th>
      <td>Wisconsin</td>
      <td>Menomonie</td>
    </tr>
    <tr>
      <th>509</th>
      <td>Wisconsin</td>
      <td>Milwaukee</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Wisconsin</td>
      <td>Oshkosh</td>
    </tr>
    <tr>
      <th>511</th>
      <td>Wisconsin</td>
      <td>Platteville</td>
    </tr>
    <tr>
      <th>512</th>
      <td>Wisconsin</td>
      <td>River Falls</td>
    </tr>
    <tr>
      <th>513</th>
      <td>Wisconsin</td>
      <td>Stevens Point</td>
    </tr>
    <tr>
      <th>514</th>
      <td>Wisconsin</td>
      <td>Waukesha</td>
    </tr>
    <tr>
      <th>515</th>
      <td>Wisconsin</td>
      <td>Whitewater</td>
    </tr>
    <tr>
      <th>516</th>
      <td>Wyoming</td>
      <td>Laramie</td>
    </tr>
  </tbody>
</table>
<p>517 rows × 2 columns</p>
</div>




```python
def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    data = pd.read_excel('gdplev.xls',skiprows=8,header=None,names=['ayear','aGDPc','aGDP2009','na','year','GDPc','GDP2009','na2'])
    data = pd.DataFrame(data,columns=['year','GDPc','GDP2009'])
    data.set_index('year',inplace=True)
    data = data.ix[-66::]
    change = [0]
    ans = None
    for i in range(1,data.shape[0]):
        if data['GDP2009'].values[i] - data['GDP2009'].values[i-1]>0:
            change.append(1)
        else:
            change.append(-1)
    for i in range(1,data.shape[0]-1):
        if change[i]==-1 and change[i+1]==-1:
            ans= data.index[i]
            break
    return ans
get_recession_start()
```




    '2008q3'




```python
def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    data = pd.read_excel('gdplev.xls',skiprows=8,header=None,names=['ayear','aGDPc','aGDP2009','na','year','GDPc','GDP2009','na2'])
    data = pd.DataFrame(data,columns=['year','GDPc','GDP2009'])
    data.set_index('year',inplace=True)
    data = data.ix[-66::]
    change = [0]
    ans = None
    for i in range(1,data.shape[0]):
        if data['GDP2009'].values[i] - data['GDP2009'].values[i-1]>0:
            change.append(1)
        else:
            change.append(-1)
    for i in range(1,data.shape[0]-1):
        if change[i]==-1 and change[i+1]==-1:
            for j in range(i+2,data.shape[0]-1):
                if change[j]==1 and change[j+1]==1:
                    ans = data.index[j+1]
                    return ans
    
get_recession_end()
```




    '2009q4'




```python
def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    data = pd.read_excel('gdplev.xls',skiprows=8,header=None,names=['ayear','aGDPc','aGDP2009','na','year','GDPc','GDP2009','na2'])
    data = pd.DataFrame(data,columns=['year','GDPc','GDP2009'])
    data.set_index('year',inplace=True)
    data = data.ix[-66::]
    start = list(data.index.values).index(get_recession_start())
    end = list(data.index.values).index(get_recession_end())
    ans = data.GDP2009[start:end].argmin()
    return ans
get_recession_bottom()
```




    '2009q2'




```python
def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    
    years=[];
    field=["State","RegionName"]
    for i in ['2000-','2001-','2002-','2003-','2004-','2005-','2006-','2007-','2008-','2009-','2010-','2011-','2012-','2013-','2014-','2015-']:
        for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            years.append(i+j)
    for i in ['01','02','03','04','05','06','07','08']:
        years.append('2016-'+i)
    df = pd.read_csv('City_Zhvi_AllHomes.csv')
    df1=pd.DataFrame(df,columns=field)
    df = pd.DataFrame(df,columns=years)
    
    dic = dict(zip(years,years))
    quarter = set()
    for key in dic.keys():
        if key.endswith('01') or key.endswith('02') or key.endswith('03'):
            dic[key]=key[0:4]+'q1'
        elif key.endswith('04') or key.endswith('05') or key.endswith('06'):
            dic[key]=key[0:4]+'q2'
        elif key.endswith('07') or key.endswith('08')or key.endswith('09'):
            dic[key]=key[0:4]+'q3'
        else:
             dic[key]=key[0:4]+'q4'
        quarter.add(dic[key])
    df.rename(columns=dic,inplace=True)
    grouped = df.groupby(df.columns, axis=1).mean()
    df= pd.merge(df1,grouped,how='inner',left_index=True,right_index=True)
    df['State']=df['State'].apply(lambda x :states[x])
    df.set_index(field,inplace=True)
    return df
convert_housing_data_to_quarters()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>2000q1</th>
      <th>2000q2</th>
      <th>2000q3</th>
      <th>2000q4</th>
      <th>2001q1</th>
      <th>2001q2</th>
      <th>2001q3</th>
      <th>2001q4</th>
      <th>2002q1</th>
      <th>2002q2</th>
      <th>...</th>
      <th>2014q2</th>
      <th>2014q3</th>
      <th>2014q4</th>
      <th>2015q1</th>
      <th>2015q2</th>
      <th>2015q3</th>
      <th>2015q4</th>
      <th>2016q1</th>
      <th>2016q2</th>
      <th>2016q3</th>
    </tr>
    <tr>
      <th>State</th>
      <th>RegionName</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New York</th>
      <th>New York</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.154667e+05</td>
      <td>5.228000e+05</td>
      <td>5.280667e+05</td>
      <td>5.322667e+05</td>
      <td>5.408000e+05</td>
      <td>5.572000e+05</td>
      <td>5.728333e+05</td>
      <td>5.828667e+05</td>
      <td>5.916333e+05</td>
      <td>587200.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>Los Angeles</th>
      <td>2.070667e+05</td>
      <td>2.144667e+05</td>
      <td>2.209667e+05</td>
      <td>2.261667e+05</td>
      <td>2.330000e+05</td>
      <td>2.391000e+05</td>
      <td>2.450667e+05</td>
      <td>2.530333e+05</td>
      <td>2.619667e+05</td>
      <td>2.727000e+05</td>
      <td>...</td>
      <td>4.980333e+05</td>
      <td>5.090667e+05</td>
      <td>5.188667e+05</td>
      <td>5.288000e+05</td>
      <td>5.381667e+05</td>
      <td>5.472667e+05</td>
      <td>5.577333e+05</td>
      <td>5.660333e+05</td>
      <td>5.774667e+05</td>
      <td>584050.0</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <th>Chicago</th>
      <td>1.384000e+05</td>
      <td>1.436333e+05</td>
      <td>1.478667e+05</td>
      <td>1.521333e+05</td>
      <td>1.569333e+05</td>
      <td>1.618000e+05</td>
      <td>1.664000e+05</td>
      <td>1.704333e+05</td>
      <td>1.755000e+05</td>
      <td>1.775667e+05</td>
      <td>...</td>
      <td>1.926333e+05</td>
      <td>1.957667e+05</td>
      <td>2.012667e+05</td>
      <td>2.010667e+05</td>
      <td>2.060333e+05</td>
      <td>2.083000e+05</td>
      <td>2.079000e+05</td>
      <td>2.060667e+05</td>
      <td>2.082000e+05</td>
      <td>212000.0</td>
    </tr>
    <tr>
      <th>Pennsylvania</th>
      <th>Philadelphia</th>
      <td>5.300000e+04</td>
      <td>5.363333e+04</td>
      <td>5.413333e+04</td>
      <td>5.470000e+04</td>
      <td>5.533333e+04</td>
      <td>5.553333e+04</td>
      <td>5.626667e+04</td>
      <td>5.753333e+04</td>
      <td>5.913333e+04</td>
      <td>6.073333e+04</td>
      <td>...</td>
      <td>1.137333e+05</td>
      <td>1.153000e+05</td>
      <td>1.156667e+05</td>
      <td>1.162000e+05</td>
      <td>1.179667e+05</td>
      <td>1.212333e+05</td>
      <td>1.222000e+05</td>
      <td>1.234333e+05</td>
      <td>1.269333e+05</td>
      <td>128700.0</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <th>Phoenix</th>
      <td>1.118333e+05</td>
      <td>1.143667e+05</td>
      <td>1.160000e+05</td>
      <td>1.174000e+05</td>
      <td>1.196000e+05</td>
      <td>1.215667e+05</td>
      <td>1.227000e+05</td>
      <td>1.243000e+05</td>
      <td>1.265333e+05</td>
      <td>1.283667e+05</td>
      <td>...</td>
      <td>1.642667e+05</td>
      <td>1.653667e+05</td>
      <td>1.685000e+05</td>
      <td>1.715333e+05</td>
      <td>1.741667e+05</td>
      <td>1.790667e+05</td>
      <td>1.838333e+05</td>
      <td>1.879000e+05</td>
      <td>1.914333e+05</td>
      <td>195200.0</td>
    </tr>
    <tr>
      <th>Nevada</th>
      <th>Las Vegas</th>
      <td>1.326000e+05</td>
      <td>1.343667e+05</td>
      <td>1.354000e+05</td>
      <td>1.370000e+05</td>
      <td>1.395333e+05</td>
      <td>1.417333e+05</td>
      <td>1.433667e+05</td>
      <td>1.461333e+05</td>
      <td>1.493333e+05</td>
      <td>1.509333e+05</td>
      <td>...</td>
      <td>1.700667e+05</td>
      <td>1.734000e+05</td>
      <td>1.754667e+05</td>
      <td>1.775000e+05</td>
      <td>1.816000e+05</td>
      <td>1.867667e+05</td>
      <td>1.906333e+05</td>
      <td>1.946000e+05</td>
      <td>1.972000e+05</td>
      <td>199950.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>San Diego</th>
      <td>2.229000e+05</td>
      <td>2.343667e+05</td>
      <td>2.454333e+05</td>
      <td>2.560333e+05</td>
      <td>2.672000e+05</td>
      <td>2.762667e+05</td>
      <td>2.845000e+05</td>
      <td>2.919333e+05</td>
      <td>3.012333e+05</td>
      <td>3.128667e+05</td>
      <td>...</td>
      <td>4.802000e+05</td>
      <td>4.890333e+05</td>
      <td>4.964333e+05</td>
      <td>5.033667e+05</td>
      <td>5.120667e+05</td>
      <td>5.197667e+05</td>
      <td>5.254667e+05</td>
      <td>5.293333e+05</td>
      <td>5.362333e+05</td>
      <td>539750.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <th>Dallas</th>
      <td>8.446667e+04</td>
      <td>8.386667e+04</td>
      <td>8.486667e+04</td>
      <td>8.783333e+04</td>
      <td>8.973333e+04</td>
      <td>8.930000e+04</td>
      <td>8.906667e+04</td>
      <td>9.090000e+04</td>
      <td>9.256667e+04</td>
      <td>9.380000e+04</td>
      <td>...</td>
      <td>1.066333e+05</td>
      <td>1.089000e+05</td>
      <td>1.115333e+05</td>
      <td>1.137000e+05</td>
      <td>1.211333e+05</td>
      <td>1.285667e+05</td>
      <td>1.346000e+05</td>
      <td>1.405000e+05</td>
      <td>1.446000e+05</td>
      <td>149300.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>San Jose</th>
      <td>3.742667e+05</td>
      <td>4.065667e+05</td>
      <td>4.318667e+05</td>
      <td>4.555000e+05</td>
      <td>4.706667e+05</td>
      <td>4.702000e+05</td>
      <td>4.568000e+05</td>
      <td>4.455667e+05</td>
      <td>4.414333e+05</td>
      <td>4.577667e+05</td>
      <td>...</td>
      <td>6.794000e+05</td>
      <td>6.970333e+05</td>
      <td>7.149333e+05</td>
      <td>7.314333e+05</td>
      <td>7.567333e+05</td>
      <td>7.764000e+05</td>
      <td>7.891333e+05</td>
      <td>8.036000e+05</td>
      <td>8.189333e+05</td>
      <td>822200.0</td>
    </tr>
    <tr>
      <th>Florida</th>
      <th>Jacksonville</th>
      <td>8.860000e+04</td>
      <td>8.970000e+04</td>
      <td>9.170000e+04</td>
      <td>9.310000e+04</td>
      <td>9.440000e+04</td>
      <td>9.560000e+04</td>
      <td>9.706667e+04</td>
      <td>9.906667e+04</td>
      <td>1.012333e+05</td>
      <td>1.034333e+05</td>
      <td>...</td>
      <td>1.207667e+05</td>
      <td>1.217333e+05</td>
      <td>1.231667e+05</td>
      <td>1.241667e+05</td>
      <td>1.269000e+05</td>
      <td>1.301333e+05</td>
      <td>1.320000e+05</td>
      <td>1.339667e+05</td>
      <td>1.372000e+05</td>
      <td>139900.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>San Francisco</th>
      <td>4.305000e+05</td>
      <td>4.644667e+05</td>
      <td>4.835333e+05</td>
      <td>4.930000e+05</td>
      <td>4.940667e+05</td>
      <td>4.961333e+05</td>
      <td>5.041000e+05</td>
      <td>5.134000e+05</td>
      <td>5.204333e+05</td>
      <td>5.381667e+05</td>
      <td>...</td>
      <td>9.269333e+05</td>
      <td>9.545333e+05</td>
      <td>9.687667e+05</td>
      <td>1.000733e+06</td>
      <td>1.060800e+06</td>
      <td>1.095100e+06</td>
      <td>1.105467e+06</td>
      <td>1.121767e+06</td>
      <td>1.119267e+06</td>
      <td>1106400.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <th>Austin</th>
      <td>1.429667e+05</td>
      <td>1.452667e+05</td>
      <td>1.494667e+05</td>
      <td>1.557333e+05</td>
      <td>1.612333e+05</td>
      <td>1.607333e+05</td>
      <td>1.595333e+05</td>
      <td>1.600333e+05</td>
      <td>1.589667e+05</td>
      <td>1.575000e+05</td>
      <td>...</td>
      <td>2.488667e+05</td>
      <td>2.528000e+05</td>
      <td>2.581333e+05</td>
      <td>2.665000e+05</td>
      <td>2.750333e+05</td>
      <td>2.816333e+05</td>
      <td>2.872333e+05</td>
      <td>2.935000e+05</td>
      <td>3.014333e+05</td>
      <td>304450.0</td>
    </tr>
    <tr>
      <th>Michigan</th>
      <th>Detroit</th>
      <td>6.616667e+04</td>
      <td>6.830000e+04</td>
      <td>6.676667e+04</td>
      <td>6.703333e+04</td>
      <td>6.750000e+04</td>
      <td>6.836667e+04</td>
      <td>6.926667e+04</td>
      <td>6.996667e+04</td>
      <td>7.100000e+04</td>
      <td>7.233333e+04</td>
      <td>...</td>
      <td>3.730000e+04</td>
      <td>3.710000e+04</td>
      <td>3.713333e+04</td>
      <td>3.620000e+04</td>
      <td>3.583333e+04</td>
      <td>3.706667e+04</td>
      <td>3.836667e+04</td>
      <td>3.796667e+04</td>
      <td>3.746667e+04</td>
      <td>37900.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <th>Columbus</th>
      <td>9.436667e+04</td>
      <td>9.583333e+04</td>
      <td>9.713333e+04</td>
      <td>9.826667e+04</td>
      <td>9.940000e+04</td>
      <td>1.002667e+05</td>
      <td>1.010667e+05</td>
      <td>1.022000e+05</td>
      <td>1.034000e+05</td>
      <td>1.048000e+05</td>
      <td>...</td>
      <td>1.031333e+05</td>
      <td>1.045000e+05</td>
      <td>1.064333e+05</td>
      <td>1.078667e+05</td>
      <td>1.094333e+05</td>
      <td>1.115667e+05</td>
      <td>1.150000e+05</td>
      <td>1.167000e+05</td>
      <td>1.182000e+05</td>
      <td>120100.0</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <th>Memphis</th>
      <td>7.250000e+04</td>
      <td>7.320000e+04</td>
      <td>7.386667e+04</td>
      <td>7.400000e+04</td>
      <td>7.416667e+04</td>
      <td>7.493333e+04</td>
      <td>7.550000e+04</td>
      <td>7.606667e+04</td>
      <td>7.633333e+04</td>
      <td>7.676667e+04</td>
      <td>...</td>
      <td>6.810000e+04</td>
      <td>6.910000e+04</td>
      <td>7.116667e+04</td>
      <td>7.053333e+04</td>
      <td>6.870000e+04</td>
      <td>6.866667e+04</td>
      <td>6.953333e+04</td>
      <td>7.090000e+04</td>
      <td>7.416667e+04</td>
      <td>75900.0</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <th>Charlotte</th>
      <td>1.269333e+05</td>
      <td>1.283667e+05</td>
      <td>1.302000e+05</td>
      <td>1.315667e+05</td>
      <td>1.329333e+05</td>
      <td>1.332000e+05</td>
      <td>1.328000e+05</td>
      <td>1.331000e+05</td>
      <td>1.343667e+05</td>
      <td>1.353667e+05</td>
      <td>...</td>
      <td>1.494667e+05</td>
      <td>1.506333e+05</td>
      <td>1.527333e+05</td>
      <td>1.551667e+05</td>
      <td>1.579000e+05</td>
      <td>1.601667e+05</td>
      <td>1.628667e+05</td>
      <td>1.664667e+05</td>
      <td>1.694333e+05</td>
      <td>172400.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <th>El Paso</th>
      <td>7.626667e+04</td>
      <td>7.686667e+04</td>
      <td>7.673333e+04</td>
      <td>7.730000e+04</td>
      <td>7.823333e+04</td>
      <td>7.830000e+04</td>
      <td>7.743333e+04</td>
      <td>7.680000e+04</td>
      <td>7.660000e+04</td>
      <td>7.640000e+04</td>
      <td>...</td>
      <td>1.118000e+05</td>
      <td>1.117333e+05</td>
      <td>1.117667e+05</td>
      <td>1.115000e+05</td>
      <td>1.113000e+05</td>
      <td>1.110667e+05</td>
      <td>1.102667e+05</td>
      <td>1.106667e+05</td>
      <td>1.114667e+05</td>
      <td>112200.0</td>
    </tr>
    <tr>
      <th>Massachusetts</th>
      <th>Boston</th>
      <td>2.069333e+05</td>
      <td>2.191667e+05</td>
      <td>2.331000e+05</td>
      <td>2.425000e+05</td>
      <td>2.496000e+05</td>
      <td>2.570667e+05</td>
      <td>2.669333e+05</td>
      <td>2.749667e+05</td>
      <td>2.825000e+05</td>
      <td>2.893000e+05</td>
      <td>...</td>
      <td>4.266667e+05</td>
      <td>4.314333e+05</td>
      <td>4.407333e+05</td>
      <td>4.485000e+05</td>
      <td>4.553667e+05</td>
      <td>4.639667e+05</td>
      <td>4.716333e+05</td>
      <td>4.826000e+05</td>
      <td>4.903667e+05</td>
      <td>501700.0</td>
    </tr>
    <tr>
      <th>Washington</th>
      <th>Seattle</th>
      <td>2.486000e+05</td>
      <td>2.556000e+05</td>
      <td>2.625333e+05</td>
      <td>2.674000e+05</td>
      <td>2.710000e+05</td>
      <td>2.724333e+05</td>
      <td>2.741667e+05</td>
      <td>2.781667e+05</td>
      <td>2.805000e+05</td>
      <td>2.846000e+05</td>
      <td>...</td>
      <td>4.418000e+05</td>
      <td>4.515000e+05</td>
      <td>4.591667e+05</td>
      <td>4.679333e+05</td>
      <td>4.933667e+05</td>
      <td>5.142667e+05</td>
      <td>5.334667e+05</td>
      <td>5.517333e+05</td>
      <td>5.755333e+05</td>
      <td>589700.0</td>
    </tr>
    <tr>
      <th>Maryland</th>
      <th>Baltimore</th>
      <td>5.966667e+04</td>
      <td>5.950000e+04</td>
      <td>5.883333e+04</td>
      <td>5.950000e+04</td>
      <td>5.956667e+04</td>
      <td>6.013333e+04</td>
      <td>6.210000e+04</td>
      <td>6.340000e+04</td>
      <td>6.366667e+04</td>
      <td>6.490000e+04</td>
      <td>...</td>
      <td>1.092333e+05</td>
      <td>1.095333e+05</td>
      <td>1.073667e+05</td>
      <td>1.080667e+05</td>
      <td>1.114333e+05</td>
      <td>1.139667e+05</td>
      <td>1.139000e+05</td>
      <td>1.146667e+05</td>
      <td>1.147333e+05</td>
      <td>115150.0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <th>Denver</th>
      <td>1.622333e+05</td>
      <td>1.678333e+05</td>
      <td>1.743333e+05</td>
      <td>1.803333e+05</td>
      <td>1.865000e+05</td>
      <td>1.925333e+05</td>
      <td>1.964000e+05</td>
      <td>1.991000e+05</td>
      <td>2.012333e+05</td>
      <td>2.024333e+05</td>
      <td>...</td>
      <td>2.708667e+05</td>
      <td>2.775000e+05</td>
      <td>2.872333e+05</td>
      <td>2.976333e+05</td>
      <td>3.103667e+05</td>
      <td>3.205000e+05</td>
      <td>3.301000e+05</td>
      <td>3.355667e+05</td>
      <td>3.427667e+05</td>
      <td>351550.0</td>
    </tr>
    <tr>
      <th>District of Columbia</th>
      <th>Washington</th>
      <td>1.377667e+05</td>
      <td>1.442000e+05</td>
      <td>1.487000e+05</td>
      <td>1.477000e+05</td>
      <td>1.497667e+05</td>
      <td>1.551333e+05</td>
      <td>1.646333e+05</td>
      <td>1.725333e+05</td>
      <td>1.805000e+05</td>
      <td>1.933000e+05</td>
      <td>...</td>
      <td>4.469333e+05</td>
      <td>4.530000e+05</td>
      <td>4.603000e+05</td>
      <td>4.661667e+05</td>
      <td>4.810667e+05</td>
      <td>4.934000e+05</td>
      <td>5.009000e+05</td>
      <td>5.041000e+05</td>
      <td>5.058000e+05</td>
      <td>516250.0</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <th>Nashville</th>
      <td>1.138333e+05</td>
      <td>1.152667e+05</td>
      <td>1.158667e+05</td>
      <td>1.169333e+05</td>
      <td>1.180333e+05</td>
      <td>1.191667e+05</td>
      <td>1.201000e+05</td>
      <td>1.208000e+05</td>
      <td>1.215667e+05</td>
      <td>1.226333e+05</td>
      <td>...</td>
      <td>1.607000e+05</td>
      <td>1.623000e+05</td>
      <td>1.669000e+05</td>
      <td>1.714667e+05</td>
      <td>1.762667e+05</td>
      <td>1.818000e+05</td>
      <td>1.892000e+05</td>
      <td>1.950667e+05</td>
      <td>2.003667e+05</td>
      <td>206100.0</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <th>Milwaukee</th>
      <td>7.803333e+04</td>
      <td>7.906667e+04</td>
      <td>8.103333e+04</td>
      <td>8.233333e+04</td>
      <td>8.403333e+04</td>
      <td>8.556667e+04</td>
      <td>8.706667e+04</td>
      <td>8.840000e+04</td>
      <td>8.953333e+04</td>
      <td>9.136667e+04</td>
      <td>...</td>
      <td>9.216667e+04</td>
      <td>9.216667e+04</td>
      <td>9.196667e+04</td>
      <td>9.333333e+04</td>
      <td>9.410000e+04</td>
      <td>9.413333e+04</td>
      <td>9.456667e+04</td>
      <td>9.466667e+04</td>
      <td>9.636667e+04</td>
      <td>98850.0</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <th>Tucson</th>
      <td>1.018333e+05</td>
      <td>1.029667e+05</td>
      <td>1.044667e+05</td>
      <td>1.056667e+05</td>
      <td>1.072000e+05</td>
      <td>1.087667e+05</td>
      <td>1.105667e+05</td>
      <td>1.128000e+05</td>
      <td>1.150000e+05</td>
      <td>1.172000e+05</td>
      <td>...</td>
      <td>1.424667e+05</td>
      <td>1.434333e+05</td>
      <td>1.442333e+05</td>
      <td>1.441667e+05</td>
      <td>1.451333e+05</td>
      <td>1.466000e+05</td>
      <td>1.481667e+05</td>
      <td>1.495333e+05</td>
      <td>1.511667e+05</td>
      <td>152700.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <th>Portland</th>
      <td>1.528000e+05</td>
      <td>1.547667e+05</td>
      <td>1.565667e+05</td>
      <td>1.574667e+05</td>
      <td>1.599000e+05</td>
      <td>1.618000e+05</td>
      <td>1.642667e+05</td>
      <td>1.677667e+05</td>
      <td>1.707667e+05</td>
      <td>1.741333e+05</td>
      <td>...</td>
      <td>2.822333e+05</td>
      <td>2.872667e+05</td>
      <td>2.955333e+05</td>
      <td>3.019333e+05</td>
      <td>3.119000e+05</td>
      <td>3.257333e+05</td>
      <td>3.430667e+05</td>
      <td>3.560000e+05</td>
      <td>3.698000e+05</td>
      <td>387050.0</td>
    </tr>
    <tr>
      <th>Oklahoma</th>
      <th>Oklahoma City</th>
      <td>7.643333e+04</td>
      <td>7.750000e+04</td>
      <td>7.856667e+04</td>
      <td>7.916667e+04</td>
      <td>7.983333e+04</td>
      <td>8.040000e+04</td>
      <td>8.113333e+04</td>
      <td>8.173333e+04</td>
      <td>8.260000e+04</td>
      <td>8.343333e+04</td>
      <td>...</td>
      <td>1.180333e+05</td>
      <td>1.189667e+05</td>
      <td>1.201000e+05</td>
      <td>1.208000e+05</td>
      <td>1.223667e+05</td>
      <td>1.247000e+05</td>
      <td>1.271000e+05</td>
      <td>1.279000e+05</td>
      <td>1.293000e+05</td>
      <td>130300.0</td>
    </tr>
    <tr>
      <th>Nebraska</th>
      <th>Omaha</th>
      <td>1.128000e+05</td>
      <td>1.141000e+05</td>
      <td>1.167333e+05</td>
      <td>1.189000e+05</td>
      <td>1.208667e+05</td>
      <td>1.197667e+05</td>
      <td>1.178667e+05</td>
      <td>1.174000e+05</td>
      <td>1.180667e+05</td>
      <td>1.176333e+05</td>
      <td>...</td>
      <td>1.301000e+05</td>
      <td>1.303000e+05</td>
      <td>1.325000e+05</td>
      <td>1.330667e+05</td>
      <td>1.344667e+05</td>
      <td>1.367333e+05</td>
      <td>1.400667e+05</td>
      <td>1.416333e+05</td>
      <td>1.426667e+05</td>
      <td>143450.0</td>
    </tr>
    <tr>
      <th>New Mexico</th>
      <th>Albuquerque</th>
      <td>1.258667e+05</td>
      <td>1.267000e+05</td>
      <td>1.264333e+05</td>
      <td>1.267333e+05</td>
      <td>1.271000e+05</td>
      <td>1.277333e+05</td>
      <td>1.285667e+05</td>
      <td>1.299000e+05</td>
      <td>1.310667e+05</td>
      <td>1.321000e+05</td>
      <td>...</td>
      <td>1.632667e+05</td>
      <td>1.640000e+05</td>
      <td>1.648000e+05</td>
      <td>1.651667e+05</td>
      <td>1.659000e+05</td>
      <td>1.665333e+05</td>
      <td>1.673333e+05</td>
      <td>1.691000e+05</td>
      <td>1.706333e+05</td>
      <td>171900.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>Fresno</th>
      <td>9.410000e+04</td>
      <td>9.526667e+04</td>
      <td>9.646667e+04</td>
      <td>9.823333e+04</td>
      <td>1.005667e+05</td>
      <td>1.035667e+05</td>
      <td>1.072333e+05</td>
      <td>1.103000e+05</td>
      <td>1.140333e+05</td>
      <td>1.185333e+05</td>
      <td>...</td>
      <td>1.696333e+05</td>
      <td>1.736000e+05</td>
      <td>1.781333e+05</td>
      <td>1.804667e+05</td>
      <td>1.820333e+05</td>
      <td>1.857000e+05</td>
      <td>1.874667e+05</td>
      <td>1.890333e+05</td>
      <td>1.927333e+05</td>
      <td>196450.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Texas</th>
      <th>Granite Shoals</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.169667e+05</td>
      <td>1.175333e+05</td>
      <td>1.175333e+05</td>
      <td>1.171667e+05</td>
      <td>1.191000e+05</td>
      <td>1.216000e+05</td>
      <td>1.280000e+05</td>
      <td>1.337667e+05</td>
      <td>1.400667e+05</td>
      <td>146450.0</td>
    </tr>
    <tr>
      <th>Maryland</th>
      <th>Piney Point</th>
      <td>1.556667e+05</td>
      <td>1.551667e+05</td>
      <td>1.584667e+05</td>
      <td>1.637000e+05</td>
      <td>1.634000e+05</td>
      <td>1.648333e+05</td>
      <td>1.647000e+05</td>
      <td>1.679000e+05</td>
      <td>1.782667e+05</td>
      <td>1.812000e+05</td>
      <td>...</td>
      <td>2.964000e+05</td>
      <td>3.090000e+05</td>
      <td>3.092333e+05</td>
      <td>3.095667e+05</td>
      <td>3.017000e+05</td>
      <td>3.052333e+05</td>
      <td>3.099667e+05</td>
      <td>3.195000e+05</td>
      <td>3.241667e+05</td>
      <td>324600.0</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <th>Maribel</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.306000e+05</td>
      <td>1.289667e+05</td>
      <td>1.296333e+05</td>
      <td>1.312667e+05</td>
      <td>1.301333e+05</td>
      <td>1.297333e+05</td>
      <td>1.293000e+05</td>
      <td>1.278333e+05</td>
      <td>1.292667e+05</td>
      <td>134200.0</td>
    </tr>
    <tr>
      <th>Idaho</th>
      <th>Middleton</th>
      <td>1.060667e+05</td>
      <td>1.043333e+05</td>
      <td>1.019000e+05</td>
      <td>1.041667e+05</td>
      <td>1.061667e+05</td>
      <td>1.083667e+05</td>
      <td>1.110333e+05</td>
      <td>1.112333e+05</td>
      <td>1.141000e+05</td>
      <td>1.141667e+05</td>
      <td>...</td>
      <td>1.443667e+05</td>
      <td>1.457000e+05</td>
      <td>1.462333e+05</td>
      <td>1.461667e+05</td>
      <td>1.477333e+05</td>
      <td>1.482000e+05</td>
      <td>1.511333e+05</td>
      <td>1.539000e+05</td>
      <td>1.571667e+05</td>
      <td>160750.0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <th>Bennett</th>
      <td>1.329000e+05</td>
      <td>1.358333e+05</td>
      <td>1.398000e+05</td>
      <td>1.446667e+05</td>
      <td>1.483000e+05</td>
      <td>1.521000e+05</td>
      <td>1.542333e+05</td>
      <td>1.562000e+05</td>
      <td>1.587333e+05</td>
      <td>1.606333e+05</td>
      <td>...</td>
      <td>1.514667e+05</td>
      <td>1.620667e+05</td>
      <td>1.714000e+05</td>
      <td>1.780333e+05</td>
      <td>1.844333e+05</td>
      <td>1.916667e+05</td>
      <td>1.958000e+05</td>
      <td>1.997667e+05</td>
      <td>2.074667e+05</td>
      <td>212600.0</td>
    </tr>
    <tr>
      <th>New Hampshire</th>
      <th>East Hampstead</th>
      <td>1.618333e+05</td>
      <td>1.691000e+05</td>
      <td>1.739667e+05</td>
      <td>1.805000e+05</td>
      <td>1.909000e+05</td>
      <td>1.950667e+05</td>
      <td>1.992667e+05</td>
      <td>2.074000e+05</td>
      <td>2.123000e+05</td>
      <td>2.122333e+05</td>
      <td>...</td>
      <td>2.495000e+05</td>
      <td>2.521000e+05</td>
      <td>2.557333e+05</td>
      <td>2.587333e+05</td>
      <td>2.613667e+05</td>
      <td>2.616000e+05</td>
      <td>2.688000e+05</td>
      <td>2.725333e+05</td>
      <td>2.778000e+05</td>
      <td>282450.0</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <th>Garden City</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.055000e+05</td>
      <td>1.043000e+05</td>
      <td>1.047667e+05</td>
      <td>1.060333e+05</td>
      <td>9.606667e+04</td>
      <td>9.930000e+04</td>
      <td>1.034333e+05</td>
      <td>1.062667e+05</td>
      <td>1.116667e+05</td>
      <td>113600.0</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <th>Mountainburg</th>
      <td>5.716667e+04</td>
      <td>6.433333e+04</td>
      <td>6.783333e+04</td>
      <td>6.900000e+04</td>
      <td>6.866667e+04</td>
      <td>6.386667e+04</td>
      <td>6.376667e+04</td>
      <td>6.546667e+04</td>
      <td>6.533333e+04</td>
      <td>6.600000e+04</td>
      <td>...</td>
      <td>8.160000e+04</td>
      <td>8.506667e+04</td>
      <td>8.846667e+04</td>
      <td>8.903333e+04</td>
      <td>8.556667e+04</td>
      <td>8.370000e+04</td>
      <td>9.043333e+04</td>
      <td>9.833333e+04</td>
      <td>1.019000e+05</td>
      <td>103400.0</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <th>Oostburg</th>
      <td>1.072667e+05</td>
      <td>1.081000e+05</td>
      <td>1.124333e+05</td>
      <td>1.155000e+05</td>
      <td>1.191000e+05</td>
      <td>1.204333e+05</td>
      <td>1.203667e+05</td>
      <td>1.196333e+05</td>
      <td>1.198667e+05</td>
      <td>1.185667e+05</td>
      <td>...</td>
      <td>1.295667e+05</td>
      <td>1.279333e+05</td>
      <td>1.274333e+05</td>
      <td>1.270667e+05</td>
      <td>1.274000e+05</td>
      <td>1.303333e+05</td>
      <td>1.320333e+05</td>
      <td>1.327667e+05</td>
      <td>1.341000e+05</td>
      <td>136350.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>Twin Peaks</th>
      <td>9.736667e+04</td>
      <td>1.001667e+05</td>
      <td>1.013333e+05</td>
      <td>1.017000e+05</td>
      <td>1.040000e+05</td>
      <td>1.076667e+05</td>
      <td>1.098333e+05</td>
      <td>1.111333e+05</td>
      <td>1.132000e+05</td>
      <td>1.166000e+05</td>
      <td>...</td>
      <td>1.501000e+05</td>
      <td>1.475333e+05</td>
      <td>1.460667e+05</td>
      <td>1.435000e+05</td>
      <td>1.523000e+05</td>
      <td>1.552667e+05</td>
      <td>1.591667e+05</td>
      <td>1.641667e+05</td>
      <td>1.679667e+05</td>
      <td>173500.0</td>
    </tr>
    <tr>
      <th>New York</th>
      <th>Upper Brookville</th>
      <td>1.230967e+06</td>
      <td>1.230967e+06</td>
      <td>1.237700e+06</td>
      <td>1.261567e+06</td>
      <td>1.295167e+06</td>
      <td>1.340033e+06</td>
      <td>1.403667e+06</td>
      <td>1.481933e+06</td>
      <td>1.536167e+06</td>
      <td>1.562033e+06</td>
      <td>...</td>
      <td>1.780633e+06</td>
      <td>1.749233e+06</td>
      <td>1.729467e+06</td>
      <td>1.749867e+06</td>
      <td>1.789600e+06</td>
      <td>1.777267e+06</td>
      <td>1.834367e+06</td>
      <td>1.904500e+06</td>
      <td>1.944067e+06</td>
      <td>1968800.0</td>
    </tr>
    <tr>
      <th>Hawaii</th>
      <th>Volcano</th>
      <td>9.870000e+04</td>
      <td>1.053667e+05</td>
      <td>1.146667e+05</td>
      <td>1.247667e+05</td>
      <td>1.181333e+05</td>
      <td>1.194000e+05</td>
      <td>1.232667e+05</td>
      <td>1.211667e+05</td>
      <td>1.233000e+05</td>
      <td>1.169000e+05</td>
      <td>...</td>
      <td>2.064667e+05</td>
      <td>2.276333e+05</td>
      <td>2.332000e+05</td>
      <td>2.346333e+05</td>
      <td>2.323667e+05</td>
      <td>2.249667e+05</td>
      <td>2.324333e+05</td>
      <td>2.420667e+05</td>
      <td>2.489667e+05</td>
      <td>247850.0</td>
    </tr>
    <tr>
      <th>South Carolina</th>
      <th>Wedgefield</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.436667e+04</td>
      <td>7.026667e+04</td>
      <td>7.206667e+04</td>
      <td>7.570000e+04</td>
      <td>7.206667e+04</td>
      <td>7.033333e+04</td>
      <td>6.903333e+04</td>
      <td>6.886667e+04</td>
      <td>7.426667e+04</td>
      <td>80700.0</td>
    </tr>
    <tr>
      <th>Michigan</th>
      <th>Williamston</th>
      <td>1.591667e+05</td>
      <td>1.613000e+05</td>
      <td>1.643000e+05</td>
      <td>1.662000e+05</td>
      <td>1.664333e+05</td>
      <td>1.686333e+05</td>
      <td>1.716667e+05</td>
      <td>1.750333e+05</td>
      <td>1.786667e+05</td>
      <td>1.793333e+05</td>
      <td>...</td>
      <td>1.657000e+05</td>
      <td>1.689333e+05</td>
      <td>1.708667e+05</td>
      <td>1.744333e+05</td>
      <td>1.758667e+05</td>
      <td>1.794667e+05</td>
      <td>1.823000e+05</td>
      <td>1.814667e+05</td>
      <td>1.824000e+05</td>
      <td>183000.0</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <th>Decatur</th>
      <td>6.360000e+04</td>
      <td>6.440000e+04</td>
      <td>6.566667e+04</td>
      <td>6.673333e+04</td>
      <td>6.720000e+04</td>
      <td>6.770000e+04</td>
      <td>6.650000e+04</td>
      <td>6.540000e+04</td>
      <td>6.460000e+04</td>
      <td>6.490000e+04</td>
      <td>...</td>
      <td>8.966667e+04</td>
      <td>9.256667e+04</td>
      <td>9.470000e+04</td>
      <td>9.350000e+04</td>
      <td>9.490000e+04</td>
      <td>9.543333e+04</td>
      <td>9.700000e+04</td>
      <td>9.650000e+04</td>
      <td>9.663333e+04</td>
      <td>96850.0</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <th>Briceville</th>
      <td>4.000000e+04</td>
      <td>4.173333e+04</td>
      <td>4.366667e+04</td>
      <td>4.490000e+04</td>
      <td>4.480000e+04</td>
      <td>4.530000e+04</td>
      <td>4.463333e+04</td>
      <td>4.370000e+04</td>
      <td>4.446667e+04</td>
      <td>4.340000e+04</td>
      <td>...</td>
      <td>5.623333e+04</td>
      <td>5.423333e+04</td>
      <td>5.260000e+04</td>
      <td>4.963333e+04</td>
      <td>4.590000e+04</td>
      <td>4.793333e+04</td>
      <td>4.360000e+04</td>
      <td>4.080000e+04</td>
      <td>4.180000e+04</td>
      <td>40850.0</td>
    </tr>
    <tr>
      <th>Indiana</th>
      <th>Edgewood</th>
      <td>9.170000e+04</td>
      <td>9.186667e+04</td>
      <td>9.293333e+04</td>
      <td>9.490000e+04</td>
      <td>9.893333e+04</td>
      <td>1.000667e+05</td>
      <td>1.008333e+05</td>
      <td>1.010000e+05</td>
      <td>1.021667e+05</td>
      <td>1.017667e+05</td>
      <td>...</td>
      <td>9.213333e+04</td>
      <td>9.406667e+04</td>
      <td>9.466667e+04</td>
      <td>9.586667e+04</td>
      <td>9.433333e+04</td>
      <td>9.663333e+04</td>
      <td>9.996667e+04</td>
      <td>9.943333e+04</td>
      <td>9.996667e+04</td>
      <td>100950.0</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <th>Palmyra</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.227667e+05</td>
      <td>1.269333e+05</td>
      <td>1.262333e+05</td>
      <td>1.223000e+05</td>
      <td>1.204667e+05</td>
      <td>1.198000e+05</td>
      <td>1.258000e+05</td>
      <td>1.276667e+05</td>
      <td>1.328667e+05</td>
      <td>137750.0</td>
    </tr>
    <tr>
      <th>Maryland</th>
      <th>Saint Inigoes</th>
      <td>1.480667e+05</td>
      <td>1.476000e+05</td>
      <td>1.572333e+05</td>
      <td>1.633667e+05</td>
      <td>1.642333e+05</td>
      <td>1.682000e+05</td>
      <td>1.665000e+05</td>
      <td>1.653333e+05</td>
      <td>1.673000e+05</td>
      <td>1.688000e+05</td>
      <td>...</td>
      <td>2.822333e+05</td>
      <td>2.884333e+05</td>
      <td>2.869667e+05</td>
      <td>2.847000e+05</td>
      <td>2.807667e+05</td>
      <td>2.778333e+05</td>
      <td>2.768333e+05</td>
      <td>2.793333e+05</td>
      <td>2.826333e+05</td>
      <td>281400.0</td>
    </tr>
    <tr>
      <th>Indiana</th>
      <th>Marysville</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.166000e+05</td>
      <td>1.151000e+05</td>
      <td>1.165000e+05</td>
      <td>1.118667e+05</td>
      <td>1.118000e+05</td>
      <td>1.156667e+05</td>
      <td>1.201667e+05</td>
      <td>1.282333e+05</td>
      <td>1.232333e+05</td>
      <td>124200.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>Forest Falls</th>
      <td>1.135333e+05</td>
      <td>1.144000e+05</td>
      <td>1.141667e+05</td>
      <td>1.111333e+05</td>
      <td>1.134333e+05</td>
      <td>1.130000e+05</td>
      <td>1.130333e+05</td>
      <td>1.151667e+05</td>
      <td>1.187000e+05</td>
      <td>1.250667e+05</td>
      <td>...</td>
      <td>1.653667e+05</td>
      <td>1.675000e+05</td>
      <td>1.771000e+05</td>
      <td>1.765333e+05</td>
      <td>1.818000e+05</td>
      <td>1.911667e+05</td>
      <td>1.987333e+05</td>
      <td>1.886333e+05</td>
      <td>1.898667e+05</td>
      <td>186650.0</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <th>Bois D Arc</th>
      <td>1.078000e+05</td>
      <td>1.069667e+05</td>
      <td>1.071000e+05</td>
      <td>1.081000e+05</td>
      <td>1.107000e+05</td>
      <td>1.136667e+05</td>
      <td>1.126333e+05</td>
      <td>1.127333e+05</td>
      <td>1.130667e+05</td>
      <td>1.154000e+05</td>
      <td>...</td>
      <td>1.375667e+05</td>
      <td>1.375667e+05</td>
      <td>1.404000e+05</td>
      <td>1.450333e+05</td>
      <td>1.475667e+05</td>
      <td>1.463000e+05</td>
      <td>1.494333e+05</td>
      <td>1.468667e+05</td>
      <td>1.437667e+05</td>
      <td>144000.0</td>
    </tr>
    <tr>
      <th>Virginia</th>
      <th>Henrico</th>
      <td>1.285667e+05</td>
      <td>1.307667e+05</td>
      <td>1.322667e+05</td>
      <td>1.332667e+05</td>
      <td>1.352333e+05</td>
      <td>1.367333e+05</td>
      <td>1.386000e+05</td>
      <td>1.413333e+05</td>
      <td>1.435333e+05</td>
      <td>1.461333e+05</td>
      <td>...</td>
      <td>2.016333e+05</td>
      <td>2.040000e+05</td>
      <td>2.059000e+05</td>
      <td>2.065667e+05</td>
      <td>2.104333e+05</td>
      <td>2.121000e+05</td>
      <td>2.139667e+05</td>
      <td>2.160333e+05</td>
      <td>2.162000e+05</td>
      <td>220150.0</td>
    </tr>
    <tr>
      <th>New Jersey</th>
      <th>Diamond Beach</th>
      <td>1.739667e+05</td>
      <td>1.831000e+05</td>
      <td>1.889667e+05</td>
      <td>1.931333e+05</td>
      <td>1.944000e+05</td>
      <td>2.102667e+05</td>
      <td>2.302667e+05</td>
      <td>2.486667e+05</td>
      <td>2.599333e+05</td>
      <td>2.656333e+05</td>
      <td>...</td>
      <td>3.818000e+05</td>
      <td>3.878667e+05</td>
      <td>3.876667e+05</td>
      <td>3.931667e+05</td>
      <td>3.980000e+05</td>
      <td>3.992333e+05</td>
      <td>4.004333e+05</td>
      <td>4.045333e+05</td>
      <td>4.039000e+05</td>
      <td>399000.0</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <th>Gruetli Laager</th>
      <td>3.540000e+04</td>
      <td>3.546667e+04</td>
      <td>3.666667e+04</td>
      <td>3.730000e+04</td>
      <td>3.773333e+04</td>
      <td>3.790000e+04</td>
      <td>3.936667e+04</td>
      <td>4.040000e+04</td>
      <td>4.156667e+04</td>
      <td>4.163333e+04</td>
      <td>...</td>
      <td>5.556667e+04</td>
      <td>5.636667e+04</td>
      <td>5.713333e+04</td>
      <td>5.890000e+04</td>
      <td>6.536667e+04</td>
      <td>6.950000e+04</td>
      <td>7.170000e+04</td>
      <td>7.533333e+04</td>
      <td>7.646667e+04</td>
      <td>77500.0</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <th>Town of Wrightstown</th>
      <td>1.017667e+05</td>
      <td>1.054000e+05</td>
      <td>1.113667e+05</td>
      <td>1.148667e+05</td>
      <td>1.259667e+05</td>
      <td>1.299000e+05</td>
      <td>1.299000e+05</td>
      <td>1.294333e+05</td>
      <td>1.319000e+05</td>
      <td>1.342000e+05</td>
      <td>...</td>
      <td>1.448667e+05</td>
      <td>1.468667e+05</td>
      <td>1.492333e+05</td>
      <td>1.486667e+05</td>
      <td>1.493333e+05</td>
      <td>1.498667e+05</td>
      <td>1.499333e+05</td>
      <td>1.498333e+05</td>
      <td>1.512667e+05</td>
      <td>155000.0</td>
    </tr>
    <tr>
      <th>New York</th>
      <th>Urbana</th>
      <td>7.920000e+04</td>
      <td>8.166667e+04</td>
      <td>9.170000e+04</td>
      <td>9.836667e+04</td>
      <td>9.486667e+04</td>
      <td>9.853333e+04</td>
      <td>1.029667e+05</td>
      <td>9.803333e+04</td>
      <td>9.396667e+04</td>
      <td>9.460000e+04</td>
      <td>...</td>
      <td>1.321333e+05</td>
      <td>1.370333e+05</td>
      <td>1.400667e+05</td>
      <td>1.417000e+05</td>
      <td>1.378667e+05</td>
      <td>1.364667e+05</td>
      <td>1.361667e+05</td>
      <td>1.389667e+05</td>
      <td>1.442000e+05</td>
      <td>143000.0</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <th>New Denmark</th>
      <td>1.145667e+05</td>
      <td>1.192667e+05</td>
      <td>1.260667e+05</td>
      <td>1.319667e+05</td>
      <td>1.438000e+05</td>
      <td>1.469667e+05</td>
      <td>1.483667e+05</td>
      <td>1.491667e+05</td>
      <td>1.531333e+05</td>
      <td>1.567333e+05</td>
      <td>...</td>
      <td>1.745667e+05</td>
      <td>1.811667e+05</td>
      <td>1.861667e+05</td>
      <td>1.876000e+05</td>
      <td>1.886667e+05</td>
      <td>1.884333e+05</td>
      <td>1.889333e+05</td>
      <td>1.910667e+05</td>
      <td>1.928333e+05</td>
      <td>197600.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>Angels</th>
      <td>1.510000e+05</td>
      <td>1.559000e+05</td>
      <td>1.581000e+05</td>
      <td>1.674667e+05</td>
      <td>1.768333e+05</td>
      <td>1.837667e+05</td>
      <td>1.902333e+05</td>
      <td>1.845667e+05</td>
      <td>1.840333e+05</td>
      <td>1.861333e+05</td>
      <td>...</td>
      <td>2.444667e+05</td>
      <td>2.540667e+05</td>
      <td>2.599333e+05</td>
      <td>2.601000e+05</td>
      <td>2.506333e+05</td>
      <td>2.635000e+05</td>
      <td>2.795000e+05</td>
      <td>2.765333e+05</td>
      <td>2.716000e+05</td>
      <td>269950.0</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <th>Holland</th>
      <td>1.510333e+05</td>
      <td>1.505000e+05</td>
      <td>1.532333e+05</td>
      <td>1.558333e+05</td>
      <td>1.618667e+05</td>
      <td>1.657333e+05</td>
      <td>1.680333e+05</td>
      <td>1.674000e+05</td>
      <td>1.657667e+05</td>
      <td>1.619667e+05</td>
      <td>...</td>
      <td>2.012667e+05</td>
      <td>2.015667e+05</td>
      <td>2.012667e+05</td>
      <td>2.060000e+05</td>
      <td>2.076000e+05</td>
      <td>2.128667e+05</td>
      <td>2.178333e+05</td>
      <td>2.219667e+05</td>
      <td>2.280333e+05</td>
      <td>234950.0</td>
    </tr>
  </tbody>
</table>
<p>10730 rows × 67 columns</p>
</div>




```python
def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    rec_start = '2008q2'
    rec_bottom = get_recession_bottom()
    df = convert_housing_data_to_quarters()
    ut = get_list_of_university_towns()
    ut.set_index(['State','RegionName'],inplace=True)
    
    dfut = pd.merge(df,ut,how='inner',left_index=True,right_index=True)
    dfnut = df.drop(ut.index)
    dfut = dfut.dropna()
    dfnut = dfnut.dropna()
    
    nut_start = dfnut[rec_start]
    nut_bottom = dfnut[rec_bottom]
    ut_start = dfut[rec_start]
    ut_bottom = dfut[rec_bottom]
    ut_price_ratio = ut_start.divide(ut_bottom)
    nut_price_ratio = nut_start.divide(nut_bottom)
    
    p = ttest_ind(ut_price_ratio, nut_price_ratio).pvalue
    if ut_price_ratio.mean() > nut_price_ratio.mean():
        better = "non-university town"
    else:
        better = "university town"

    different = p<0.01
    result  = (different, p, better)
    
    return result
run_ttest()
```




    (True, 0.0051648663279200407, 'university town')


