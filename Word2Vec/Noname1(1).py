import jieba
import numpy as np
import pandas as pd
def preprocess(path):
    #¶ÁÈëÍ£´Ê±í¡¢¹¹½¨Í£´Ê×Öµä
    new_table = []
    data = []
    f = open(path,'r')
    label =[]
    for line in f:
        #±àÂë×ª»»and°´¿Õ¸ñÇÐ¸î¡¢ÌáÈ¡²éÑ¯¹Ø¼ü×Ö¡¢Ã¿¸öÓÃ»§ºÏ²¢³ÉÎªÒ»¾ä»°
        single_line = line.decode('gb18030','ignore').encode('utf-8').strip('\n').split('\t')
        key_words = single_line[4:]
        user_word = ''
        for item in key_words:
            seg_list = jieba.cut(item)
            for check in seg_list:
                check = check.encode('utf-8')
                if  single_line[3]!='0':
                    user_word = user_word + ' ' + check
        if single_line[3]!='0':
            label.append(single_line[:4] )
            data.append(user_word + '\n')
    trainLabel = pd.DataFrame(np.array(label), columns=['id', 'age', 'gender', 'education'])
    trainLabel.to_csv('trainLabel_1.csv', sep=' ', index=False)
    trainData = pd.DataFrame({'query': data})
    trainData.to_csv('trainData-1.csv', sep=' ', index=False)

import pandas as pd

def preprocess1(path):
    f = open(path, 'r')
    label = []
    data = []
    for line in f:
        # ±àÂë×ª»»and°´¿Õ¸ñÇÐ¸î¡¢ÌáÈ¡²éÑ¯¹Ø¼ü×Ö¡¢Ã¿¸öÓÃ»§ºÏ²¢³ÉÎªÒ»¾ä»°
        single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
        key_words = single_line[1:]
        user_word = ''
        for item in key_words:
            seg_list = jieba.cut(item)
            for check in seg_list:
                check = check.encode('utf-8')
                user_word = user_word + ' ' + check
        label.append(single_line[0:1])
        data.append(user_word + '\n')
f = open('../../sougodownload/sougo_first/user_tag_query.2W.TEST', 'r')
label = []
data = []
for line in f:
        # ±àÂë×ª»»and°´¿Õ¸ñÇÐ¸î¡¢ÌáÈ¡²éÑ¯¹Ø¼ü×Ö¡¢Ã¿¸öÓÃ»§ºÏ²¢³ÉÎªÒ»¾ä»°
    single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
#    key_words = single_line[1:]
#    user_word = ''
#    for item in key_words:
#        seg_list = jieba.cut(item)
#        for check in seg_list:
#            check = check.encode('utf-8')
#            user_word = user_word + ' ' + check
#    data.append(user_word + '\n')        
    label.append(single_line[0:1])
testLabel = pd.DataFrame(np.array(label),columns=['id'])
testLabel['education']='1'                         
'''------------------------------------------'''
#trainLabel = pd.DataFrame(np.array(label),columns=['id','age','gender','education'])
#
#    testLabel = pd.DataFrame(np.array(label), columns=['id','age','gender','education'])
#    testLabel.to_csv('testLabel_1.csv', sep=' ', index=False)
#    testData = pd.DataFrame({'query': data})
#    testData.to_csv('testData_1.csv', sep=' ', index=False)
#
#
#
#preprocess('train_word_bag/user_tag_query.10W.TRAIN')
#trainData = pd.read_csv('trainData-1.csv',sep=' ')
#trainLabel= pd.read_csv('trainLabel_1.csv',sep=' ')
#train_Data = trainData['query'].values.tolist()
#train_Label = trainLabel['education'].values.tolist()
#print len(train_Label)
#print len(train_Data)
#
#print 'data'
#
#preprocess1('../../sougodownload/rematch/user_tag_query.10W.TEST')
#testData = pd.read_csv('testData_1.csv', sep=' ')
#testLabel= pd.read_csv('testLabel_1.csv',sep=' ')
#print len(testLabel)
#print type(testLabel)
#print type(testData)
#test_Data = testData['query'].values.tolist()
#print len(test_Data)
#test_Label = testLabel['education'].values.tolist()
#y_train, y_test = train_Label,test_Label
#
