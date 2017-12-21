
# coding: utf-8

# In[5]:


yourFilePath="C:\\Users\\udayg\\Desktop\\final project\\kaggle-survey-2017\\multipleChoiceResponses.csv"
curencyConverion="C:\\Users\\udayg\\Desktop\\final project\\kaggle-survey-2017\\conversionRates.csv"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
plt.style.use('seaborn-paper')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from matplotlib_venn import venn2
from subprocess import check_output

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
import ipywidgets as widgets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score , accuracy_score, roc_curve, auc
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

import operator
import os,sys
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn_pandas import DataFrameMapper

def f(x):
    return x

def loadData(yourFilePath):
    df=pd.read_csv(yourFilePath, low_memory=False,encoding='ISO-8859-1')
    return df

def summaryColumn(columnName):
    column_summary=df_raw[[columnName]].groupby([columnName]).size().reset_index(name='counts')
    column_summary=column_summary.sort_values(['counts'],ascending=[0])
    column_category=list(column_summary[columnName])
    category_counts=list(column_summary['counts'])
#     plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos=np.arange(len(column_category))
    ax.barh(y_pos, category_counts, align='center',
        color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(column_category)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Count')
    ax.set_title(columnName)
    plt.show()
    return column_summary


# Loading the data from files using pandas reav_csv api call. We set the encoding to ISO-8859-1 and the low memory option to False.

# In[7]:


data = loadData(yourFilePath)
rates = loadData(curencyConverion)
df_raw=loadData(yourFilePath)
columnList=list(df_raw.columns)


# ## First step is to visualize all the data elements.
# We used the IPython interactive display panel to allow users to chose the attribute and visualize the distribution for that attribute.

# In[9]:


result=interactive(summaryColumn,columnName=columnList);
display(result)


# Based on the understanding the we gained by looking at data distribution in each attribute, we then proceed to do exploratory analysis across attributes.

# ## Second we tried to see if there is correlation between current job and the major of the respondent

# In[10]:


# Major and CurrentJobTitle Relation   
f,ax=plt.subplots(1,2,figsize=(20,10))                                                                                  
sns.countplot(y = data['MajorSelect'],ax=ax[0],order= data['MajorSelect'].value_counts().index)                         
ax[0].set_title('Major')                                                                                                
ax[0].set_ylabel('')                                                                                                    
sns.countplot(y= data['CurrentJobTitleSelect'],ax=ax[1],order=data['CurrentJobTitleSelect'].value_counts().index)       
ax[1].set_title('Current Job')                                                                                          
ax[1].set_ylabel('')                                                                                                    
plt.subplots_adjust(wspace=0.8)                                                                                         
plt.show()   


# We notice that most data scientists majored in computer science

# #### We analyse the most important work langauges. See if there are any intresting observations to be made there. Lets us see if we can get some insights of usage between python and R.
# 
# Show the work language  which are frequently used. We find python and R are top two.  

# In[11]:



work_tools = data['WorkToolsSelect'].dropna().str.split(',')              
tools = []                                                                
for wktools in work_tools:                                                
  for tool in wktools:                                                  
      tools.append(tool)                                                
result = pd.Series(tools).value_counts()[:10]                             
plt.subplots(figsize=(10,10))                                             
sns.barplot(result.values,result.index)                                   
plt.title('Work Tools')                                                   
plt.show()              


# We looked at the type of tools Data scientist use at work. Its mostly Python, R and SQL 

# In[13]:


resp = data.dropna(subset=['WorkToolsSelect'])                                                                
resp = pd.merge(resp,rates,left_on='CompensationCurrency',right_on='originCountry',how='left')                
python = resp[(resp['WorkToolsSelect'].str.contains('Python'))&(~resp['WorkToolsSelect'].str.contains('R'))]  
R = resp[(~resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]       
both = resp[(resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))] 
# python and R users recommendations:                                                
p_reconmd = python['LanguageRecommendationSelect'].value_counts()[:2]                
r_reconmd = R['LanguageRecommendationSelect'].value_counts()[:2]                     
labels1 = p_reconmd.index                                                            
values1 = p_reconmd.values                                                           
labels2 = r_reconmd.index                                                            
values2 = r_reconmd.values                                                           
f,ax = plt.subplots(1,2,figsize=(10,10))                                             
ax[0].pie(values1, labels = labels1,autopct='%1.1f%%', shadow=False, startangle=90)  
ax[0].axis('equal')                                                                  
ax[0].set_title('Python Users Recommendation')                                       
ax[1].pie(values2, labels = labels2,autopct='%1.1f%%', shadow=False, startangle=90)  
ax[1].axis('equal')                                                                  
ax[1].set_title('R Users Recommendation')                                            
plt.show()      


# On further analysis we find that majority of python users use only python and a few use C++. But quite a few R users also use python. This reinforced the idea that python is the predominant lanaguage of choice for data scientists.

# In[15]:


#  python and R salary compare:  
py_sal=(pd.to_numeric(python['CompensationAmount'].dropna(),errors='coerce')*python['exchangeRate']).dropna()
py_avr_sal = pd.Series(py_sal).median()

R_sal=(pd.to_numeric(R['CompensationAmount'].dropna(),errors='coerce')*R['exchangeRate']).dropna()
R_avr_sal = pd.Series(R_sal).median()

both_sal=(pd.to_numeric(both['CompensationAmount'].dropna(),errors='coerce')*both['exchangeRate']).dropna()
both_avr_sal = pd.Series(both_sal).median()
print ('Median Salary For Individual using Python:',py_avr_sal)
print ('Median Salary For Individual using R:',R_avr_sal)
print ('Median Salary For Individual knowing both languages:',both_avr_sal)


# Clearly the salary for Python is higher. Note that this is a median salary across the word. So a $1000 difference could be significant based on which part of the world you are in.

# #### Next we start looking at job titles and see how it effects there compensation and inclination to switch their jobs.

# In[18]:



salary=data[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()   
salary=pd.merge(salary,rates,left_on='CompensationCurrency',right_on='originCountry',how='left')                       
                                                 
salary['Salary'] = pd.to_numeric(salary['CompensationAmount'],errors='coerce') * salary['exchangeRate'].dropna()       
salary_null = pd.isnull(salary['Salary'])                                                                              
                                                 
salary_null_false= salary['Salary'][salary_null == False][salary['Salary'] >= 0]                                       

#Compensation By Job Title   
sal_job = salary.groupby('CurrentJobTitleSelect')['Salary'].median().to_frame().sort_values(by='Salary',ascending=False)
plt.subplots(figsize=(10,10))                                                                                           
sns.barplot(sal_job.Salary,sal_job.index)                                                                               
plt.title('Compensation By Job Title')                                                                                  
plt.show()                   


# Next we see who many of the respondents who are employed write code regularly
# #### Does Data Scientist always need to write code?

# In[20]:



employed=['Employed full-time','Independent contractor, freelancer, or self-employed','Employed part-time']
df_employed=df_raw[df_raw['EmploymentStatus'].isin(employed)]
df_employed[['CodeWriter']].groupby(['CodeWriter']).size().reset_index(name='counts')


# 80% of the respondents who are employed write code. 
# 
# We can now check how many of the respondents are looking to switch their jobs.

# In[21]:


switcher=df_raw[df_raw['CareerSwitcher']=='Yes']
switcher[['CurrentJobTitleSelect']].groupby(['CurrentJobTitleSelect']).size().reset_index(name='counts').sort_values(['counts'],ascending=[0])


# Software Developer, programmer and Business analyst are the major population who are seeking for new opportunities which reflects: 
# a. Great mobility of these roles. There are tons of opportunities for people with these skills.
# b. High expectation of future career development and better compensation
# 
# 

# In[23]:


df_raw['TitleFit_Score']=df_raw['TitleFit'].apply(lambda x: 5 if x=='Perfectly' else (3 if x=='Fine' else 1))
titleFit_idx=df_raw['TitleFit'].isnull()
df_withTitleFit=df_raw[~titleFit_idx]
print(df_withTitleFit.groupby(['CurrentJobTitleSelect'])['TitleFit_Score'].mean())


# Business Analyst and Engineer overall have the lowest 'title fit' rate which might because that their work has too diversed responsibility which are hard to be summazied with one title.
# 
# Given that we have analyzed respondents opinions on on the langauges, tools and their titles. Given how often people switch jobs and the opportunities they have, the next logical question to ask would be,
# 
# What are the indicators of Job Satisfaction and can we predict how job satisfaction of an employee based on the the environmnet they are in. 
# 
# Job Satisfaction is a attribute on the responses. We use that as the category label.
# 
# ### Predicting JobSatisfaction 

# In[27]:


df_raw_pre=loadData(yourFilePath)
obj_df1 = df_raw_pre.select_dtypes(include=['object']).copy()
obj_df1[obj_df1.isnull().any(axis=1)]

nonCatlistColumns = ['Age','LearningCategorySelftTaught','LearningCategoryOnlineCourses','LearningCategoryWork','LearningCategoryUniversity',
     'LearningCategoryKaggle','LearningCategoryOther','TimeGatheringData',
     'TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights','TimeOtherSelect']

def convertToInt(amt):
    try:
        if ',' in amt:
            return float(amt.replace(',',''))
        elif amt != '-':
            return float(amt)
    except:
        return float(amt)

        
def convertToString(str):
    if '' == str:
        return ''
    else:
        return str
    
def groupSatifaction(val):
    if 0 < val and val < 7:
        return 1
    #elif 5<= val and 7 > val:
     #   return 2
    elif 7<= val:
        return 2
    else:
        return 0
    
compensation = ['CompensationAmount','CompensationCurrency']
comp = obj_df1['CompensationAmount']
compCur = obj_df1['CompensationCurrency']

comp = comp.apply(convertToInt )
compCur = compCur.apply(convertToString)
obj_df1['CompensationAmount'] = comp
    
columns = list(df_raw_pre.columns)
for column in columns:
    if column in nonCatlistColumns:
        continue
    obj_df1[column] = obj_df1[column].astype('category').cat.codes


# In[30]:


def featurize():
    return DataFrameMapper(obj_df.columns)


pipeline = Pipeline([('featurize', featurize()), ('forest', RandomForestClassifier())])

y = obj_df1['JobSatisfaction'].apply(groupSatifaction)
X = obj_df1[obj_df.columns.drop('JobSatisfaction')]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(max_depth=2, random_state=0)
clf_rfc.fit(X_train, y_train)

i=0
columns = list(X.columns)
for val in np.nditer(clf_rfc.feature_importances_):
    if val > 0.01 :
        print (val, columns[i])
    i = i+1    



# In[29]:


print(clf_rfc.score(X_test, y_test))


# We can see the we can predict if an employee is satisfied based on other attributes. We belive that this is very useful for companies to determine employee satisfaction. The features that are needed for prediction would already be avaliable with companies. This model gives us the ability to predict satisfaction based on the know attributes about a company.

# Another intresting observation is that WorktoolSelect is the most important attribute for employee satisfaction. It means that people are passionate about their work. The tools they use matters to them. So can we help identify the learning paltforms that people find useful. This will help people pick better work tools and thus lead to greater employee job satisfaction.

# In[31]:


df_raw[['MLToolNextYearSelect']].groupby(['MLToolNextYearSelect']).size().reset_index(name='counts').sort_values(['counts'],ascending=[0])


# Tool for the future, TensorFlow, Python, R, Spark, Hadoop -- very clear trend of future

# In[32]:


df_raw[['MLMethodNextYearSelect']].groupby(['MLMethodNextYearSelect']).size().reset_index(name='counts').sort_values(['counts'],ascending=[0])


# Algos for the future, Deep learning, Neural Nets, Time Series Analysis, Bayesian Methods, Text mining -- very clear trend of future.
# 
# Lets look at learning platform usefulness for data scientists.

# In[33]:


learningPlatformSurvey=['LearningPlatformUsefulnessArxiv','LearningPlatformUsefulnessBlogs'
                        ,'LearningPlatformUsefulnessCollege','LearningPlatformUsefulnessCompany'
                        ,'LearningPlatformUsefulnessConferences','LearningPlatformUsefulnessFriends'
                        ,'LearningPlatformUsefulnessKaggle','LearningPlatformUsefulnessNewsletters'
                        ,'LearningPlatformUsefulnessCommunities'
                        ,'LearningPlatformUsefulnessDocumentation','LearningPlatformUsefulnessCourses'
                        ,'LearningPlatformUsefulnessProjects','LearningPlatformUsefulnessPodcasts'
                        ,'LearningPlatformUsefulnessSO','LearningPlatformUsefulnessTextbook'
                        ,'LearningPlatformUsefulnessTradeBook','LearningPlatformUsefulnessTutoring'
                        ,'LearningPlatformUsefulnessYouTube']
for surveyTarget in learningPlatformSurvey:
    df_raw['{}_Score'.format(surveyTarget)]=df_raw[surveyTarget].apply(lambda x: 5 if x=='Very useful' else (3 if x=='Somewhat useful' else 1))

    
# for DataScientist
df_raw_DataScientist=df_raw[df_raw['CurrentJobTitleSelect']=='Data Scientist']
for surveyTarget in learningPlatformSurvey:
    print ('{} average rate is {}'.format(surveyTarget,df_raw_DataScientist['{}_Score'.format(surveyTarget)].mean()))


# For Data Scientist group, we find almost all the ratings of the platforms are higher than the whole population which show the strong willingness to learn new skills for this community and also it shows that data science area is an actively moving industry. Stack Overflow are highly recommended in this community and very interestingly college is not highly rated as by the whole population

# In[34]:


# for Software engineer
df_raw_SoftwareEngineer=df_raw[df_raw['CurrentJobTitleSelect']=='Software Developer/Software Engineer']
for surveyTarget in learningPlatformSurvey:
    print ('{} average rate is {}'.format(surveyTarget,df_raw_SoftwareEngineer['{}_Score'.format(surveyTarget)].mean()))


# Comparatively, Software engineer is a more mature industry even thought the techniques and tools are still actively updated everyday. They are not as passionate as data scientist in learning. And also we see a drop in the college rating for this community which is very interesting. And surprisingly, SO also gets a lower rate.
# 
# There are a lot of tools and each has its own usefulness based on job title, the usefulness is very close. Can we do pair analysis and see which two learning platforms together would be most useful.

# In[35]:


for surveyTarget in learningPlatformSurvey:
    df_raw['{}_Flag'.format(surveyTarget)]=df_raw[surveyTarget].apply(lambda x: 1 if x=='Very useful' else (1 if x=='Somewhat useful' else 0))
df_platform=df_raw[['{}_Flag'.format(_) for _ in learningPlatformSurvey]]

platformPick=[]
for idx,item in df_platform.iterrows():
    tmp=[]
    for platform in ['{}_Flag'.format(_) for _ in learningPlatformSurvey]:
        if item[platform]==1:
            tmp.append(platform.replace('_Flag','').replace('LearningPlatformUsefulness',''))
    if len(tmp)>0:
        platformPick.append(tmp)
        


# In[38]:


from apyori import apriori
rules = apriori(platformPick, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
results=list(rules)[:3]
print(results)


# In[42]:


def load_dataset():
    "Load the sample dataset."
    return basketList


def createC1(dataset):
    "Create a list of candidate item sets of size one."
    c1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    #frozenset because it will be a ket of a dictionary.
    return map(frozenset, c1)


def scanD(dataset, candidates, min_support):
    "Returns all candidates that meets a minimum support level"
    sscnt = {}
    for tid in dataset:
        for can in candidates:
            if can.issubset(tid):
                sscnt.setdefault(can, 0)
                sscnt[can] += 1

    num_items = float(len(dataset))
    retlist = []
    support_data = {}
    for key in sscnt:
        support = sscnt[key] / num_items
        if support >= min_support:
            retlist.insert(0, key)
        support_data[key] = support
    return retlist, support_data


def aprioriGen(freq_sets, k):
    "Generate the joint transactions from candidate sets"
    retList = []
    lenLk = len(freq_sets)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(freq_sets[i] | freq_sets[j])
    return retList


def apriori(dataset, minsupport=0.5):
    "Generate a list of candidate item sets"
    C1 = createC1(dataset)
    D = map(set, dataset)
    L1, support_data = scanD(D, C1, minsupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minsupport)
        support_data.update(supK)
        L.append(Lk)
        k += 1

    return L, support_data


# In[43]:


apriori(platformPick,0.4)

