#!/usr/bin/env python
# coding: utf-8

# In[115]:


import numpy as np
import pandas as pd


# 

# In[116]:


books = pd.read_csv("books.csv", sep=";", error_bad_lines=False, encoding='latin-1')


# In[117]:


books.head()


# In[118]:


books.columns


# In[119]:


books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]


# In[120]:


books.head()


# In[121]:


books.rename(columns={'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)


# In[122]:


books.head()


# In[123]:


users = pd.read_csv("users.csv", sep=";", error_bad_lines=False, encoding='latin-1')


# In[124]:


users.head()


# In[125]:


users.rename(columns={"User-ID":'user_id', "Location":"location", "Age":"age"}, inplace=True)


# In[126]:


users.head(2)


# In[127]:


ratings = pd.read_csv('ratings.csv', sep=";", error_bad_lines=False, encoding='latin-1')


# In[128]:


ratings.head(2)


# In[129]:


ratings.rename(columns={"User-ID":"user_id","Book-Rating":"rating"}, inplace=True)


# In[130]:


users.head(1)


# In[131]:


books.shape, users.shape, ratings.shape


# In[132]:


ratings.head(2)


# In[133]:


x = ratings['user_id'].value_counts()>200


# In[134]:


y = x[x].index


# In[135]:


y


# In[136]:


ratings = ratings[ratings['user_id'].isin(y)]


# In[137]:


ratings.shape


# In[138]:


ratings.head()


# In[139]:


ratings_with_books = ratings.merge(books, on="ISBN")


# In[140]:


ratings_with_books.head()


# In[141]:


ratings_with_books.shape


# In[142]:


number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()


# In[143]:


number_rating.head(3)


# In[144]:


number_rating.rename(columns={'rating':'number_of_ratings'}, inplace=True)


# In[145]:


number_rating.head(3)


# In[146]:


final_rating = ratings_with_books.merge(number_rating, on='title')


# In[147]:


final_rating.head()


# In[148]:


final_rating.shape


# In[149]:


final_rating =final_rating[final_rating['number_of_ratings']>=50]


# In[150]:


final_rating.shape


# In[151]:


final_rating.drop_duplicates(['user_id', 'title'], inplace=True)


# In[152]:


final_rating.head()


# In[153]:


book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')


# In[154]:


book_pivot.shape


# In[155]:


book_pivot.fillna(0, inplace=True)


# In[156]:


book_pivot


# In[157]:


from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)


# In[158]:


book_sparse


# In[159]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')


# In[160]:


model.fit(book_sparse)


# In[161]:


distances, suggestions = model.kneighbors(book_pivot.iloc[54, :].values.reshape(1,-1), n_neighbors=6)


# In[162]:


distances


# In[163]:


suggestions


# In[164]:


for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])


# In[165]:


np.where(book_pivot.index=='Animal Farm')[0][0]


# In[166]:


def recommend_book(book_name):
    book_suggestions = []
    book_id = np.where(book_pivot.index==book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1,-1), n_neighbors=6)
    for i in range(len(suggestions)):
            if i==0:
                print(f"The suggestions for {book_name} are : ")
            if not i:
                print(book_pivot.index[suggestions[i]])
                book_suggestions.append(book_pivot.index[suggestions[i]])
    return book_suggestions


# In[167]:


l = recommend_book('Animal Farm')


# In[168]:


l


# In[ ]:





# In[ ]:




