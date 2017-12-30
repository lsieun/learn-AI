import pandas as pd

data = {'a':0,'b':1,'c':2}
s1 = pd.Series(data)
print(s1)

# s2 = pd.Series(data,index=['b','c','d','a'])
# print(s2)