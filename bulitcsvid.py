# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 20:23:06 2016

@author: Jiao
"""

import numpy as np
import pandas as pd
from sklearn import metrics
def submission_tranfer():
    submission_eight=pd.read_csv('result_test.csv',sep=' ')
    a=submission_eight
    submission = pd.read_csv('result_end1.csv',sep=' ')
    submission['education'] = a['education']
    submission.to_csv('result_end1.csv',sep=' ',index=False)
    print "finish"
def calculate_result(actual,pred):
    c=0.000
    for i in range(len(actual)):
        if actual[i]==pred[i]:
            c+=1
    #d=float(c)/len(a)
    print 'right:{0:.3f}'.format(float(c)/len(actual)) 

#a=[1,0,2,1,0,2,2,0,1,0]
#b=[1,2,1,2,1,1,1,2,2,2]
#calculate_result(a,b)
#c=1
#submission_eight=pd.read_csv('result_devised.csv',sep=' ')
#submission = pd.read_csv('result_end1.csv',sep=' ')
submission_tranfer()
#filename='test.csv'
#f=open(filename,'r')#,encoding="gbk")   
#lines=f.readlines()
#f.close()        
#result=[]
#for i in lines:
#    result.append(i.split('\t',4))
#result=np.array(result)    
#df_data=pd.DataFrame(data=result[:,:5],index=range(len(result)),\
#                         columns=['id','age','gender','education','dataValue'])
#
#submission = pd.read_csv('result_3_devise.csv',sep=' ')
#submission['value'] = df_data['dataValue']
#submission.to_csv('result_3_devise.csv',sep=' ',index=False)


