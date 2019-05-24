# conversation-analyzer

To use, you must first export and include relative to the current working directory of the program, your facebook message conversation.

Note: Requires Python 3.x, numpy, nltk stopwords, gemsim word2vec, sklearn, textblob and matplotlib

Sample run:
```
1. Run message importer
2. Graph Data 
3. Get Prediction (Naive Bayes Classifier)
4. Exit
```
Sample Prediction Flow:
```
processing file...
processing complete
enter percentage of data for test set
10
Remove stop words (stop words are common words with little meaning and removing will reduce noise in data while graphing)? Y/N
N
building vocabulary...
complete
building test set Kathleen Matthews
complete
building test set Tyler Matthews
complete
associating phrases to vocabulary...
complete
determining probabilities...
complete
predictor ready, running testset...

the error rate for Kathleen is 0.669496855345912
the error rate for Tyler is 0.7182343234323433

Enter a phrase to predict association ('exit' to quit)
Hello, how are you
Probably Kathleen


```

Sample Graph Data Flow:
```
Remove stop words (stop words are common words with little meaning and removing will reduce noise in data while graphing)? Y/N
Y
Annotate Points with words? (with huge datasets, this could take a long time to render) Y/N
N
Minimum count for words to be plotted (numeric):1
```
![Non-Annotated Graph](https://raw.githubusercontent.com/matthewstyler/conversation-analyzer/screenshots/screen1.png)
```
Remove stop words (stop words are common words with little meaning and removing will reduce noise in data while graphing)? Y/N
Y
Annotate Points with words? (with huge datasets, this could take a long time to render) Y/N
Y
Minimum count for words to be plotted (numeric):50
```
![Non-Annotated Graph](https://raw.githubusercontent.com/matthewstyler/conversation-analyzer/screenshots/screen2.png)
