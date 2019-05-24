"""
A conversation analyzer that takes a processed facebook conversation json export
and allows data to be graphed accoring to gensim word vector associations, as
well as training a naive bayes predictor to give a prediction of a greater probability
of association of a provided phrase to a particular member of the conversation

@author: Tyler Matthews
"""
import json
import string
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from textblob import Word

# preprocess data
def processPhrases(phrasesForUsers):
    print('processing file...')
    preProcessPhrasesForUsers = {}
    for user in phrasesForUsers:
        preProcessPhrasesForUsers[user] = []
        phrases = phrasesForUsers[user]
        for phrase in phrases:
            updatedPhrase = massagePhrase(phrase)
            if len(updatedPhrase) > 0:
                preProcessPhrasesForUsers[user].append(updatedPhrase)
    print('processing complete')
    return preProcessPhrasesForUsers

# perform typical NLP transformations on input
def massagePhrase(phrase):
    updatedPhrase = [x.lower() for x in phrase] # lower case
    updatedPhrase = [''.join(c for c in s if c not in string.punctuation) for s in updatedPhrase] # remove punctuation
    updatedPhrase = [s for s in updatedPhrase if s] # remove empty
    updatedPhrase = [Word(word).lemmatize() for word in updatedPhrase] # lemmatize
    return updatedPhrase

# get data
def parseSeedFile():
    print('opening file...')
    try :
        with open('seed.json') as json_file:
            return processPhrases(json.load(json_file))
    except:
        print('error opening and parsing seed file (ensure you have a seed.json file ready for import)')
        exit()
    return phrasesForUsers

# charts data with optional annotation and minimum word count
def chartData(phrasesForUsers, annotatePoints, minCount):
    models = {}
    pcaModels = {}
    pcaTransformationResult = {}
    chartColor = 'red'
    for user in phrasesForUsers:
        models[user] = Word2Vec(phrasesForUsers[user], min_count=minCount)
        pcaModels[user] = models[user][models[user].wv.vocab]
        pcaTransformationResult[user] = PCA(n_components=2).fit_transform(pcaModels[user])
        pyplot.scatter(pcaTransformationResult[user][:, 0], pcaTransformationResult[user][:, 1], color=chartColor)
        words = list(models[user].wv.vocab)
        if annotatePoints:
            for i, word in enumerate(words):
                pyplot.annotate(word, xy=(pcaTransformationResult[user][i, 0], pcaTransformationResult[user][i, 1]))
        chartColor = 'green'
        print("Loaded chart data for %s" % user)
    pyplot.show()

# ask to annotate points
def annotatePoints():
    while True:
        decision = input('Annotate Points with words? (with huge datasets, this could take a long time to render) Y/N\n')
        if (decision == 'Y' or decision == 'N'):
            if (decision == 'Y'):
                return True
            else:
                return False

# ask for min word count
def minWordCount():
    while True:
        decision = input('Minimum count for words to be plotted (numeric):')
        try:
            return int(decision)
        except:
            print('invalid number')

# ask to remove stop words
def askRemoveStopWords(phrasesForUsers):
    while True:
        decision = input('Remove stop words (stop words are common words with little meaning and removing will reduce noise in data while graphing)? Y/N\n')
        if (decision == 'Y' or decision == 'N'):
            if (decision == 'Y'):
                return removeStopWords(phrasesForUsers)
            else:
                return phrasesForUsers

# remove stop words
def removeStopWords(phrasesForUsers):
    stop_words = set(stopwords.words('english'))
    updatededPhrases = json.loads(json.dumps(phrasesForUsers)) # deep copy
    for user in updatededPhrases:
        newPhrase = []
        for phrase in phrasesForUsers[user]:
            updatedPhrase = [word for word in phrase if word not in stop_words] # remove stop words
            updatedPhrase = [s for s in updatedPhrase if s] # remove empty
            if len(updatedPhrase) > 0:
                newPhrase.append(updatedPhrase)
    updatededPhrases[user] = newPhrase
    return updatededPhrases

# create a vocab list from all words
def createVocabList(phrasesForUsers):
    print('building vocabulary...')
    vocabSet = set([])  #create empty set
    for user in phrasesForUsers:
        for phrase in phrasesForUsers[user]:
            vocabSet = vocabSet | set(phrase) #union of the two sets
    print('complete')
    return list(vocabSet)

# count number of times words within input is in the given vocabulary
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# loop through all phrases to build word count vectors
def phrasesToWordCountVectors(allwords, phrasesForUsers):
    print('associating phrases to vocabulary...')
    wordCountVectors = []
    for user in phrasesForUsers:
        for phrase in phrasesForUsers[user]:
            wordCountVectors.append(bagOfWords2VecMN(allwords, phrase))
    print('complete')
    return wordCountVectors

# determine probabilities based on phrase to vocab association
# adapted from https://github.com/pbharrin/machinelearninginaction
def trainNB0(trainMatrix, trainCategory):
    print('determining probabilities...')
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to np.ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          #change to np.log()
    p0Vect = np.log(p0Num/p0Denom)          #change to np.log()
    print('complete')
    return p0Vect, p1Vect

# perform classification based on probability, adapted from https://github.com/pbharrin/machinelearninginaction
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec)
    p0 = sum(vec2Classify * p0Vec)
    if p1 > p0:
        return 1
    else:
        return 0

# build a test set from a subsection of provided data
def buildTestSet(phrasesForUsers, userKey, percent):
    print('building test set ' + userKey)
    testSet = []
    # grab n percent for testing
    for i in range(round(len(phrasesForUsers[userKey]) * (percent / 100.0))):
        index = np.random.randint(0, len(phrasesForUsers[userKey]))
        testSet.append(phrasesForUsers[userKey][index])
        del phrasesForUsers[userKey][index]
    print('complete')
    return testSet

# tests the predictor with known classified data
def testSetWithCorrectClass(key, testSet, correctClass, prob0, prob1, allwords):
    error = 0
    for phrase in testSet:
        prediction = classifyNB(bagOfWords2VecMN(allwords, phrase), prob0, prob1, .5)
        if prediction == correctClass:
            error += 1
    print('the error rate for ' + key + ' is ' + str(float(error / len(testSet))))

# perform naive bayes preduiction
def predict(phrasesForUsers):
    testSetPercent = int(input('enter percentage of data for test set\n'))

    allwords = createVocabList(askRemoveStopWords(phrasesForUsers)) # build vocabulary

    # build test sets from a subsection of the data
    user1TestSet = buildTestSet(phrasesForUsers, list(phrasesForUsers)[0], testSetPercent)
    user2TestSet = buildTestSet(phrasesForUsers, list(phrasesForUsers)[1], testSetPercent)

    # get phrases tied to users,  for trained classification
    user1Phrases = phrasesForUsers[list(phrasesForUsers)[0]]
    user2Phrases = phrasesForUsers[list(phrasesForUsers)[1]]
    # vector to associate class with phrases
    trainingCategories = ([0] * len(user1Phrases)) +  ([1] * len(user2Phrases))
    wordCountVectors = phrasesToWordCountVectors(allwords, phrasesForUsers) # associate phrases to vocab
    prob0, prob1 = trainNB0(wordCountVectors, trainingCategories) # determine probability
    print('predictor ready, running testset...\n')

    # check how accurate the predictions are, based on known data
    testSetWithCorrectClass(list(phrasesForUsers)[0], user1TestSet, 0, prob0, prob1, allwords)
    testSetWithCorrectClass(list(phrasesForUsers)[1], user2TestSet, 1, prob0, prob1, allwords)

    while True:
        phraseToPredict = input('Enter a phrase to predict association (\'exit\' to quit)\n')
        if (phraseToPredict == 'exit'):
            break
        print("Probably " + list(phrasesForUsers)[classifyNB(bagOfWords2VecMN(allwords, massagePhrase(phraseToPredict)), prob0, prob1, .5)])

# get name to import from conversation
def getMessageSenderNameToImport():
    return input('Message Sender Name to Import:')

# ask to grab another message sender
def shouldAddAnotherMessageSender():
    while True:
        decision = input('Add another (The conversation analyzer currently only bases predictions on two users)? Y/N?')
        if (decision == 'Y' or decision == 'N'):
            if (decision == 'Y'):
                return True
            else:
                return False

# A script to import the messages of a given users, from a facebook message json export
def importConversation():
    # start import process
    print('starting importer')
    try:
        file = open(input('enter json file name\n'))
        jsondata = json.load(file)

        names = [];
        while True:
            names.append(getMessageSenderNameToImport()) # get names to import from json
            if (shouldAddAnotherMessageSender() == False):
                break

        messagesPerUser = {}
        for message in jsondata['messages']:
            if message.get('content'):
                if (message.get('sender_name') in names and message.get('content') != 'You sent an attachment.'):
                    if (message.get('sender_name') not in messagesPerUser):
                        messagesPerUser[message.get('sender_name')] = []
                    messagesPerUser[message.get('sender_name')].append(message['content'].split())
        print(messagesPerUser)

        # save as json
        with open('seed.json', 'w') as seedfile:
            json.dump(messagesPerUser, seedfile)

        print('importer finished')
    except:
       print('failure to parse messages')

# program start
while True:
    decision = input("Decide:\n1. Run message importer\n2. Graph Data \n3. Get Prediction (Naive Bayes Classifier)\n4. Exit\n")
    if (decision == '1'):
        importConversation()
    elif (decision == '2'):
        chartData(askRemoveStopWords(parseSeedFile()), annotatePoints(), minWordCount())
    elif (decision =='3'):
        predict(parseSeedFile())
    elif (decision == '4'):
        exit()
