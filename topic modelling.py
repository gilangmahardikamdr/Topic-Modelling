#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pymysql as connection
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud, STOPWORDS


# In[79]:


df_nas_raw =  pd.read_sql(query,conn)


# In[3]:

df_lok =  pd.read_sql(query,conn)


# In[4]:

df_sek =  pd.read_sql(query,conn)


# In[5]:


produk_id_nas = df_nas['Produk_Id'].tolist()
produk_id_lok = df_lok['Produk_Id'].tolist()
produk_id_sek = df_sek['Produk_Id'].tolist()


# In[6]:


val_nas = [str(i) for i in produk_id_nas]
val_lok = [str(i) for i in produk_id_lok]
val_sek = [str(i) for i in produk_id_sek]


# In[7]:


from tqdm import tqdm


# In[11]:


temp_nas =[]

for i in tqdm(val_nas[:14584]):

# In[105]:


df_nas = pd.concat(temp_nas)


# In[106]:


df_nas = df_nas.merge(df_nas_raw[['Produk_Id', 'nama_produk']], how ='inner', left_on='id', right_on='Produk_Id')


# In[107]:


df_nas['des'] = df_nas['nama_produk'] + ', ' +  df_nas['label_atribut'] + ', ' + df_nas['atribut_value']


# In[108]:


df_nas['des'] = df_nas['des'].str.replace('Keterangan Lainnya,', '')


# In[109]:


df_nas['produk_id'] = df_nas['id'].copy()


# In[110]:


temp_df_nas = df_nas.drop_duplicates(subset=['id', 'des'])


# In[111]:


df_nas_fin = temp_df_nas[(temp_df_nas['atribut_value']!='-') & ((~temp_df_nas['atribut_value'].isnull()))]


# In[112]:


df_nas_pre = df_nas_fin.groupby(['produk_id'])['des'].apply(', '.join).reset_index()


# In[113]:


df_nas_pre['pre'] = df_nas_pre['des'].apply(lambda x: x.lower())


# In[114]:


import re

df_nas_pre['pre'] = df_nas_pre['des'].apply(lambda x: re.sub(r"\d+", "", x))


# In[115]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[116]:


# import StemmerFactory class
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
output = df_nas_pre['pre'].apply(lambda x: stopword.remove(x))
# stemming process
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
output_2   = output.apply(lambda x: stemmer.stem(x))
# sentence = 'Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan'


# In[117]:


vectorizer_wtf = TfidfVectorizer(analyzer='word')
X_wtf = vectorizer_wtf.fit_transform(output_2)


# In[118]:


from sklearn.decomposition import LatentDirichletAllocation


# In[122]:


#LDA
lda = LatentDirichletAllocation(n_components=60, learning_decay=0.9)
X_lda = lda.fit(X_wtf)

#Plot topics function. Code from: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(10, 6, figsize=(30, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
#Show topics
n_top_words = 5
feature_names = vectorizer_wtf.get_feature_names()
plot_top_words(X_lda, feature_names, n_top_words, '')


# In[124]:


#LDA
lda = LatentDirichletAllocation(n_components=20, learning_decay=0.9)
X_lda = lda.fit(X_wtf)

#Plot topics function. Code from: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(5, 4, figsize=(30, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
#Show topics
n_top_words = 5
feature_names = vectorizer_wtf.get_feature_names()
plot_top_words(X_lda, feature_names, n_top_words, '')


# In[123]:


from sklearn.cluster import KMeans


# In[90]:


sse={}
for k in np.arange(100,900,100):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_wtf)
    sse[k] = kmeans.inertia_
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel('Values for K')
plt.ylabel('SSE')
plt.show();


# In[91]:


kmeans = KMeans(n_clusters=200)
kmeans.fit(X_wtf)
result = pd.concat([df_nas_pre['pre'],pd.DataFrame(X_wtf.toarray(),columns=vectorizer_wtf.get_feature_names())],axis=1)
result['cluster'] = kmeans.predict(X_wtf)


# In[107]:


res = result[['pre','cluster']].copy()


# In[123]:


#res['id'] = df_nas_pre['produk_id'].copy()
res['nama_produk'] = df_nas['nama_produk'].copy()
res['nama_kategori'] = df_nas['nama_kategori'].copy()


# In[110]:


temp_res_nas = res.merge(df_nas[['Produk_Id','nama_kategori']], left_on='id', right_on='Produk_Id', how='inner')


# In[127]:


res[res['cluster']==138]


# In[131]:


df_nas[df_nas['Produk_Id']==1471034]


# In[120]:


temp_res_nas.groupby(['cluster']).agg({'nama_kategori':'nunique'}).reset_index().sort_values(by=['nama_kategori'], ascending = False).head(50)


# In[77]:


df_nas_pre['stem'] = df_nas_pre['label_atribut'].apply(lambda x: stemmer.stem(x))


# In[ ]:





# In[ ]:


temp_lok =[]

for i in tqdm(val_lok):

    conn = connection.connect(
         host="10.254.0.142",    # your host, usually localhost
         user="telkom-dsc",         # your username 
         passwd="9859c4b85623",  # your password
         db="lkpp_datamart")

    query = """select pk.id
	 , pka.label_atribut label_atribut
	 , pav.atribut_value atribut_value
from produk_katalog pk 
inner join produk_kategori_atribut pka 
on pk.produk_kategori_id = pka.produk_kategori_id
inner join produk_atribut_value pav
on pav.produk_kategori_atribut_id = pka.id and pk.id = pav.produk_id
inner join komoditas k
on k.id = pk.komoditas_id 
where pka.active = 1 and pav.active = 1 and pk.id = """ + i + """;"""
    temp_df =  pd.read_sql(query,conn)
    temp_lok.append(temp_df)


# In[ ]:


temp_sek =[]

for i in tqdm(val_sek):

    conn = connection.connect(
         host="10.254.0.142",    # your host, usually localhost
         user="telkom-dsc",         # your username 
         passwd="9859c4b85623",  # your password
         db="lkpp_datamart")

    query = """select pk.id
	 , pka.label_atribut label_atribut
	 , pav.atribut_value atribut_value
from produk_katalog pk 
inner join produk_kategori_atribut pka 
on pk.produk_kategori_id = pka.produk_kategori_id
inner join produk_atribut_value pav
on pav.produk_kategori_atribut_id = pka.id and pk.id = pav.produk_id
inner join komoditas k
on k.id = pk.komoditas_id 
where pka.active = 1 and pav.active = 1 and pk.id = """ + i + """;"""
    temp_df =  pd.read_sql(query,conn)
    temp_sek.append(temp_df)


# In[34]:


temp_df


# In[29]:


', '.join(temp_df['label_atribut'] + ', ' + temp_df['atribut_value'])


# In[24]:


pd.DataFrame(temp_nas[0])


# In[ ]:


conn = connection.connect(
         host="10.254.0.142",    # your host, usually localhost
         user="telkom-dsc",         # your username 
         passwd="9859c4b85623",  # your password
         db="lkpp_datamart")

query = """select pk.id
	 , GROUP_CONCAT(pka.label_atribut) label_atribut
	 , GROUP_CONCAT(pav.atribut_value) atribut_value
from produk_katalog pk 
inner join produk_kategori_atribut pka 
on pk.produk_kategori_id = pka.produk_kategori_id
inner join produk_atribut_value pav
on pav.produk_kategori_atribut_id = pka.id
inner join komoditas k
on k.id = pk.komoditas_id 
where k.komoditas_kategori_id = 5 and pk.id in ( """ + ', '.join(val_lok) + """ )
group by pk.id
LIMIT 20000;"""
df_ket_lok =  pd.read_sql(query,conn)


# In[15]:


conn = connection.connect(
         host="10.254.0.142",    # your host, usually localhost
         user="telkom-dsc",         # your username 
         passwd="9859c4b85623",  # your password
         db="lkpp_datamart")

query = """select pk.id
	 , GROUP_CONCAT(pka.label_atribut) label_atribut
	 , GROUP_CONCAT(pav.atribut_value) atribut_value
from produk_katalog pk 
inner join produk_kategori_atribut pka 
on pk.produk_kategori_id = pka.produk_kategori_id
inner join produk_atribut_value pav
on pav.produk_kategori_atribut_id = pka.id
inner join komoditas k
on k.id = pk.komoditas_id 
where k.komoditas_kategori_id = 6 and pk.id in ( """ + ', '.join(val_sek) + """ )
group by pk.id
LIMIT 20000;"""
df_ket_sek =  pd.read_sql(query,conn)


# In[12]:


val_nas + val_lok + val_sek

