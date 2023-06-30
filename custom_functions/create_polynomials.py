#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def create_polynomials(mydf):
    for column in list(mydf.select_dtypes('number')):
        mydf[column+'_2'] = mydf[column]**2
        mydf[column+'_3'] = mydf[column]**3
        mydf[column+'_0.5'] = mydf[column]**0.5

