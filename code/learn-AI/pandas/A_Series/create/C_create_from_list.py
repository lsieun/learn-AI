import pandas as pd

#使用list作为data，且不指明index
s1 = pd.Series(data=[10,20,30,40,50])
print(s1)

#使用list作为data，指明index
s2 = pd.Series(data=[10,20,30,40,50],index=['A','B','C','D','E'])
print(s2)