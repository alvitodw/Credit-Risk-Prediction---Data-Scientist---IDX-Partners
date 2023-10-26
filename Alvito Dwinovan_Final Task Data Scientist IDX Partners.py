#!/usr/bin/env python
# coding: utf-8

# **Alvito Dwinovan Wibowo**
# * Final Task Data Scientist IDX Partners x Rakamin  Academy
# 
# Sebagai tugas akhir dari masa kontrakmu sebagai intern Data Scientist di ID/X Partners, kali ini kamu akan dilibatkan dalam projek dari sebuah lending company. Kamu akan berkolaborasi dengan berbagai departemen lain dalam projek ini untuk menyediakan solusi teknologi bagi company tersebut. Kamu diminta untuk membangun model yang dapat memprediksi credit risk menggunakan dataset yang disediakan oleh company yang terdiri dari data pinjaman yang diterima dan yang ditolak. Selain itu kamu juga perlu mempersiapkan media visual untuk mempresentasikan solusi ke klien. Pastikan media visual yang kamu buat jelas, mudah dibaca, dan komunikatif. Pengerjaan end-to-end solution ini dapat dilakukan di Programming Language pilihanmu dengan tetap mengacu kepada framework/methodology Data Science.

# In[1]:


#Import Package
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


import pandas as pd

# Load dataset from Excel file
df = pd.read_csv("loan_data_2007_2014.csv", low_memory=False)
df.head()
df.describe()
df.info()
df.shape


# In[4]:


# Menghapus kolom yang tidak perlu atau kurang relevan (menurut saya) dalam menentukan prediksi credit risk
columns_to_delete =["Unnamed: 0","id","member_id","emp_title","emp_length","desc","mths_since_last_delinq","mths_since_last_record","revol_util",
                    "total_acc","last_pymnt_d","next_pymnt_d","last_credit_pull_d","collections_12_mths_ex_med","mths_since_last_major_derog",
                    "annual_inc_joint","dti_joint","verification_status_joint","acc_now_delinq","tot_coll_amt","tot_cur_bal","open_acc_6m",
                    "open_il_6m", "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util", "total_rev_hi_lim","inq_fi",
                    "total_cu_tl", "inq_last_12m",'term', 'grade', 'sub_grade', 'home_ownership', 'issue_d', 'pymnt_plan', 'url', 'purpose', 
                    'title', 'zip_code', 'addr_state', 'initial_list_status', 'application_type', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
                    'all_util', 'policy_code', 'earliest_cr_line', 'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'funded_amnt_inv', 'out_prncp_inv',
                      'total_pymnt_inv', 'funded_amnt', 'total_rec_prncp', 'installment']


df1 = df.drop(columns=columns_to_delete)
df1.head()


# In[5]:


# Hitung jumlah total baris dalam dataframe
total_rows = len(df1)

# Hitung threshold sebagai 80% dari jumlah total baris
threshold = 0.8 * total_rows

# Hitung jumlah nilai null dalam setiap kolom
null_counts = df1.isnull().sum()

# Tentukan ambang batas (misalnya, 2 nilai null)
threshold = 2  # Ganti dengan nilai ambang batas yang diinginkan

# Dapatkan indeks kolom yang akan dihapus (lebih dari ambang batas dan bukan 'annual_inc' dan 'open_acc)
columns_to_drop = null_counts[null_counts > threshold].index
columns_to_drop = [col for col in columns_to_drop if col != 'annual_inc' and col != 'open_acc']

# Hapus kolom-kolom yang memenuhi kriteria
df1 = df1.drop(columns_to_drop, axis=1)

# Menghapus data duplikat berdasarkan semua kolom
df1= df1.drop_duplicates()
df1.head()


# In[6]:


# Deskripsi jumlah data null dan duplikat

print("=========== Sum null of dataset================== ")
print(df1.isnull().sum())
print("=========== Sum Duplicate of dataset================== ")
print(df1.duplicated().sum())
df1.shape


# In[7]:


#Transformasi variabel kategori menjadi numerik dengan Encode
new_category=["verification_status","loan_status"]

df1_category = df1[new_category]
df1_category.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df1_cat_encode= df1_category.copy()
for col in df1_cat_encode.select_dtypes(include='O').columns:
    df1_cat_encode[col]=le.fit_transform(df1_cat_encode[col])
df1_cat_encode


# In[8]:


df1=df1.drop(["verification_status","loan_status"],axis=1)
df1=df1.join(df1_cat_encode)
df1.head()


# In[9]:


#Handling Missing Value
print('=======================================\n')
print('Handling Missing Values variabel Annual_Inc pada Data :')
df1['annual_inc'] = df1['annual_inc'].fillna(df1['annual_inc'].median())
print('Handling Missing Values variabel Open_acc pada Data :')
df1['open_acc'] = df1['open_acc'].fillna(df1['open_acc'].median())
print('Handling Missing Values variabel Open_acc pada Data :')
df1['open_acc'] = df1['open_acc'].fillna(df1['open_acc'].median())
print('Handling Missing Values variabel total_pymnt pada Data :')
df1['total_pymnt'] = df1['total_pymnt'].fillna(df1['total_pymnt'].mean())
print('Handling Missing Values variabel total_rec_int pada Data :')
df1['total_rec_int'] = df1['total_rec_int'].fillna(df1['total_rec_int'].mean())
print('Handling Missing Values variabel total_rec_late_fee pada Data :')
df1['total_rec_late_fee'] = df1['total_rec_late_fee'].fillna(df1['total_rec_late_fee'].mean())
print('Handling Missing Values variabel recoveries pada Data :')
df1['recoveries'] = df1['recoveries'].fillna(df1['recoveries'].mean())
print('Handling Missing Values variabel last_pymnt_amnt pada Data :')
df1['last_pymnt_amnt'] = df1['last_pymnt_amnt'].fillna(df1['last_pymnt_amnt'].mean())
print('Handling Missing Values variabel collection_recovery_fee pada Data :')
df1['collection_recovery_fee'] = df1['collection_recovery_fee'].fillna(df1['collection_recovery_fee'].mean())
print(df1.isnull().sum())


# In[10]:


df1.describe()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
fig, axes = plt.subplots(3, 5, figsize=(16, 10))

columns_to_analyze = ['loan_status', 'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'open_acc', 'revol_bal', 'out_prncp', 'total_pymnt', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'verification_status']

for i, column in enumerate(columns_to_analyze):
    row = i // 5
    col = i % 5
    sns.boxplot(data=df1, x=column, ax=axes[row, col], color='blue', notch=True)
    
    # Menentukan posisi label
    positions = [0]
    labels = [column]
    axes[row, col].set_xticks(positions)
    axes[row, col].set_xticklabels(labels, rotation=45)
    
    axes[row, col].set_title(f'Box Plot {column}', fontdict={'fontweight':'bold', 'fontsize':12})

plt.tight_layout()
plt.show()


# In[12]:


scaler = StandardScaler()
scaler.fit(df1)
scaled_features = scaler.transform(df1)
df1_scaled = pd.DataFrame(scaled_features,columns = df1.columns)
df1_scaled.head()


# In[13]:


columns = ['loan_amnt', 'int_rate','revol_bal','out_prncp','total_pymnt','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt']
data = df1[columns]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaled_features = scaler.transform(data)
df1_scaled = pd.DataFrame(scaled_features,columns =data.columns)
df1_scaled=df1_scaled.join(df1_cat_encode)
df1_scaled.head()
df1_scaled.info()


# In[14]:


from sklearn.linear_model import LogisticRegression
import warnings

#split dataset
y = df1_scaled["loan_status"]
x = df1_scaled.drop(['loan_status'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 88)


# In[15]:


x_train


# In[16]:


x_test


# In[17]:


y_train


# In[18]:


y_test


# In[19]:


#logistic regression

warnings.filterwarnings('ignore')

logreg = LogisticRegression(random_state = 88)
logreg.fit(x_train, y_train)


# In[20]:


y_predict = logreg.predict(x_test)
y_predict_train = logreg.predict(x_train)
y_predict1 = logreg.predict_proba(x_test)


# In[23]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#metode evaluasi

print('\nconfustion matrix') # generate the confusion matrix
print(confusion_matrix(y_test, y_predict))

print('\naccuracy')
print(accuracy_score(y_test, y_predict))

print('\nclassification report')
print(classification_report(y_test, y_predict))


# Exploratory Data Analysis

# In[24]:


# Heatmap
fig = plt.figure(figsize=(15,15))
corr = df1.corr()
sns.heatmap(corr, annot=True, square=True,)
plt.yticks(rotation=0)

plt.show()


# In[25]:


loan_status = df1['loan_status'].value_counts(sort=True)

fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)

#Grafik_1
loan_status.plot(ax=ax1, kind='bar', rot=0, color='maroon')
ax1.set_title('Status Loan')
ax1.set_ylabel('JUMLAH')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)


# In[26]:


loan_status = df1['verification_status'].value_counts(sort=True)

fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)

#Grafik_1
loan_status.plot(ax=ax1, kind='bar', rot=0, color='yellow')
ax1.set_title('Status Verification')
ax1.set_ylabel('JUMLAH')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)


# In[27]:


loan_status = df['grade'].value_counts(sort=True)

fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)

#Grafik_1
loan_status.plot(ax=ax1, kind='bar', rot=0, color='blue')
ax1.set_title('Grade')
ax1.set_ylabel('JUMLAH')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

