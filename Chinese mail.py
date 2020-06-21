import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def createWord(index,trainNum):
    stopWordPath = "chemail/trec06c/中文停用词表.txt"
    with open(stopWordPath, 'rb') as fp:
        stopword = fp.read().decode('utf-8')
    stopWordList = stopword.splitlines()
    f = open(index, 'r')
    wordList = []

    for i in range(trainNum):
        line = f.readline()
        if (line[0] == 'h'):
            address = 'chemail/trec06c/data/' + line[12:15] + '/' + line[16:19]
            voclist = open(address, "r", encoding="gb2312").read()
        else:
            address = 'chemail/trec06c/data/' + line[13:16] + '/' + line[17:20]
            voclist = open(address, "r", encoding="gb2312").read()
        voclist = ''.join(re.findall('[\u4e00-\u9fa5]', voclist))
        voclist = jieba.cut(voclist)
        newtext = ''
        for word in voclist:
            if word not in stopWordList and word != ' ':
                newtext += word
                newtext += " "
        wordList.append(newtext)
    return wordList,stopWordList

def get_label(index,trainNum):
    f=open(index,'r')
    labelList = []
    for i in range(trainNum):
        line = f.readline()
        if (line[0] == 'h'):
            labelList.append(1)
        else:
            labelList.append(0)
    return labelList

def naiveBayes(index,trainNum):
    testbegin = int(0.7 * trainNum)
    wordList, stopWordList = createWord(index,trainNum)
    labelList=get_label(index,trainNum)
    train_ham = []
    train_spam = []
    test = wordList[testbegin:trainNum]
    test_label = labelList[testbegin:trainNum]

    for i in range(0, testbegin):
        if (labelList[i] == 1):
            train_ham.append(wordList[i])
        else:
            train_spam.append(wordList[i])
    train_matrix = CountVectorizer()
    freTrain =train_matrix.fit_transform(wordList[0:testbegin]).toarray()
    trainVoc =train_matrix.vocabulary_

    test_matrix = CountVectorizer(stop_words=stopWordList, vocabulary=trainVoc)
    freHam = (test_matrix.fit_transform(train_ham)).toarray()
    freSpam = (test_matrix.fit_transform(train_spam)).toarray()
    freTest = (test_matrix.fit_transform(test)).toarray()

    hamProb = np.sum(freHam, axis=0)
    totalHam = np.sum(hamProb)
    hamProb[hamProb == 0] = 0.01
    hamProb = hamProb / totalHam

    spamProb = np.sum(freSpam, axis=0)
    totalSpam = np.sum(spamProb)
    spamProb[spamProb == 0] = 0.01
    spamProb = spamProb / totalSpam

    accuracy = []
    ph = (np.shape(freHam)[0]) / (np.shape(freTrain)[0])
    ps = 1 - ph
    for i in range(trainNum - testbegin):
        ph1 = (hamProb ** freTest[i]).prod().astype(float)
        ps1 = (spamProb ** freTest[i]).prod().astype(float)
        if((ps*ps1+ph*ph1)!=0):
            finalProb = ps * ps1 / (ps * ps1 + ph * ph1)
            if finalProb > 0.5:
               predict = 0
            else:
               predict = 1
            if predict == test_label[i]:
               accuracy.append(1)
            else:
               accuracy.append(0)
    print("accurate is %5.3f for %d emails" % (np.mean(accuracy), trainNum))

index='chemail/trec06c/delay/newindex'
naiveBayes(index,4000)