import pandas as pd
import numpy as np

data = np.array(['a','b','c','d'])
s1 = pd.Series(data)
print(s1)
# s2 = pd.Series(data,index={10,11,12,13})
# print(s2)