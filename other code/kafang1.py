# -*- coding: utf-8 -*-
# @Time    : 2016/10/21 0:14
# @Author  : Aries
# @Site    : 
# @File    : kafang-1.py
# @Software: PyCharm Community Edition

def Chi_Square(Word_Repetition,Word_Non_Repetition):
# Word_Repetition {'userid': {'gender':'cutWordList'},'userid':{'gender', 'cutWordList'}]
# Word_Non_Repetition {'1':list(man_include_repeat),'2':list(woman_include_repeat)]

    Chi_List = {}
    n =len(Word_Repetition)
    for label,list_non_Repetition in Word_Non_Repetition.iteritems():
        _tem_label = {}
        print label,'label'
        for tem_word in list_non_Repetition:
            print '2'
            for word in tem_word:
                count1, count2, count3, count4 = 0, 0, 0, 0
                for userid ,  tem_dict in Word_Repetition.iteritems():
                    for label_1,query_list in tem_dict.iteritems():
                        if label == label_1 and word in query_list:
                            count1 += 1
                        elif label == label_1 and word not in query_list:
                            count3 += 1
                        elif label != label_1 and word in query_list:
                            count2 += 1
                        elif label != label_1 and word not in query_list:
                            count4 += 1
                tem_1 = count1*count4-count2*count3
                try:
                    chi_1 = (pow(tem_1, 2)*n) / ( (count1+count3)*(count1+count2)*(count2+count4)*(count3+count4))
                except Exception :
                    chi_1 = 0
                finally:
                    _tem_label[word] = chi_1
        l = sorted(_tem_label.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        Chi_List.setdefault(label,l[:2000]) #不同类别可能出现相同的词，暂时没有处理
    feature = {}
    n = 1
    for label ,chi in Chi_List.iteritems():
        for tuple1 in chi:
            feature.setdefault(tuple1[0],n)
            n += 1
    f = open(r'D:\CCFBDC\testResoult\feature.txt','w')
    for k ,v in feature.iteritems():
        f.write(str(k)+':')
        f.write(str(v))
        f.write('\n')

    f.close()
    return feature


