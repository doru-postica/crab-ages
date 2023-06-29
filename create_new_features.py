#!/usr/bin/env python
# coding: utf-8

# In[1]:


def create_new_features(mydf):
    mydf['Volume'] = mydf['Length'] * mydf['Diameter'] * mydf['Height']
    mydf['Shucked_Weight_Ratio'] = mydf['Shucked Weight'] / mydf['Weight']
    mydf['Volume_Weight_Ratio'] = mydf['Volume'] / mydf['Weight']
    mydf['Surface_Area'] = 2 * (mydf['Length'] * mydf['Diameter'] + mydf['Length'] * mydf['Height'] + mydf['Diameter'] * mydf['Height'])
    mydf['Shell_Weight_Ratio'] = mydf['Shell Weight'] / mydf['Weight']
    mydf['Viscera_Weight_Ratio'] = mydf['Viscera Weight'] / mydf['Weight']
    mydf['shell_to_body_ratio'] = mydf['Shell Weight'] / (mydf['Shucked Weight'] + mydf['Viscera Weight'])
    mydf['weight_difference'] = mydf['Weight'] - (mydf['Shucked Weight'] + mydf['Viscera Weight'])
    mydf['length_to_diameter_ratio'] = mydf['Length'] / mydf['Diameter']
    mydf['weight_to_size_ratio'] = mydf['Weight'] / mydf['Length']
    mydf['meat_density'] = (mydf['Shucked Weight'] + mydf['Viscera Weight']) / (mydf['Length'] * mydf['Diameter'] * mydf['Height'])
    mydf['shell_density'] = mydf['Shell Weight'] / (mydf['Length'] * mydf['Diameter'] * mydf['Height'])


# In[ ]:




