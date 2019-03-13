import csv
import pandas as pd
read2=pd.read_csv('/home/yousef/Documents/tensor_p_p1_1/tensor_P__11/1/u1.base.csv')
print(read2.info())
print(read2['first'].unique())
print(read2['second'].unique())
print(read2['third'].unique())
print(read2['fourth'].unique())
#------------------------------
#------------------------------
print(read2['first'].unique())
print(read2['second'].unique())
print(read2['third'].unique())
print(read2['fourth'].unique())
#------------------------------
print("")
print("")
print(len(read2['first'].unique().tolist()))
print(len(read2['second'].unique().tolist()))
print(len(read2['third'].unique().tolist()))
print(len(read2['fourth'].unique().tolist()))
#-------------------------------------------------
#-------------------------------------------------








