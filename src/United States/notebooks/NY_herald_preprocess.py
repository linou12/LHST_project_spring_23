#!/usr/bin/env python
# coding: utf-8

# # New York Herald Filtered

# In[3]:


import os
import tarfile
import pandas as pd
import subprocess
import  tarfile


# ## Process the data
# 

# In[ ]:





# In[7]:


import os
import tarfile
import pandas as pd
import subprocess
import  tarfile

dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_bindweed_ver02.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_bindweed_ver02.csv', index=False)


# In[ ]:


import os
import tarfile
import pandas as pd
import subprocess
import  tarfile

dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_crowfoot_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_crowfoot_ver01.csv', index=False)


# In[ ]:


dates_us = set()    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_deadnettle_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_deadnettle_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_eucalyptus_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_eucalyptus_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_fairymoss_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_fairymoss_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_goldenglow_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_goldenglow_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_houseleek_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_houseleek_ver01.csv', index=False)


# In[ ]:





dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_itchweed_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_itchweed_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_juneberry_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_juneberry_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_kudzu_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_kudzu_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_laceflower_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_laceflower_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_marcus_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_marcus_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_nosebleed_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_nosebleed_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_poppy_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_poppy_ver01.csv', index=False)


# In[ ]:




dates_us = set()
    
with open('/scratch/students/bousbina/corpus/USA/dates_us.txt','w') as f:
        f.write(str(dates_us)) 

# path to the tar.bz2 file
file_path = '/scratch/students/bousbina/corpus/USA/dlc_quercitron_ver01.tar.bz2'

# extract the tar.bz2 file
# with tarfile.open(file_path, "r:bz2") as tar:
#     tar.extractall()


# create an empty DataFrame to store the data
data = pd.DataFrame(columns=["date", "text"])

subprocess.run(["tar", "-xvf",  file_path], stdout=subprocess.PIPE, text=True)
tar = tarfile.open(file_path)

# extract the publication date from the file name
for m in tar:
        if ".txt" in m.name:
            pub_date = m.name.split("/")[1] + "-" + m.name.split("/")[2] + "-" + m.name.split("/")[3] 
            dates_us.add(pub_date)
            f=tar.extractfile(m)
            text=f.read()
#     add the data to the DataFrame
data = data.append({'date': pub_date, 'text': text}, ignore_index=True)

# save the DataFrame to a CSV file
data.to_csv('/scratch/students/bousbina/corpus/USA/dlc_quercitron_ver01.csv', index=False)

