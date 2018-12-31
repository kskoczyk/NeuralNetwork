import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from joblib import load


class Game:

    def __init__(self, title, platform, year, genre, publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales,
                 Global_Sales):
        self.title = title
        self.platform = platform
        self.year = year
        self.genre = genre
        self.publisher = publisher
        self.NA_Sales = NA_Sales
        self.EU_Sales = EU_Sales
        self.JP_Sales = JP_Sales
        self.Other_Sales = Other_Sales
        self.Global_Sales = Global_Sales

    def getTitle(self):
        return self.title

    def getPlatform(self):
        return self.platform

    def getYear(self):
        return self.year

    def getGenre(self):
        return self.genre

    def getPublisher(self):
        return self.publisher

    def getNASales(self):
        return self.NA_Sales

    def getEUSales(self):
        return self.EU_Sales

    def getJPSales(self):
        return self.JP_Sales

    def getOtherSales(self):
        return self.Other_Sales

    def getGlobalSales(self):
        return self.Global_Sales


class DataSet:

    def __init__(self):
        self.games = []
        self.csvfile = ''
        self.csvreader = ''
        self.gamesNumber = 0

    def setFile(self, file):
        self.csvfile = open(file, 'r', encoding="utf8", newline='')

    def loadFromFile(self):
        self.csvreader = csv.reader(self.csvfile, delimiter=';')
        i = 0
        for row in self.csvreader:
            if i != 0:
                self.games.append(Game(row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]))
            i += 1
        self.gamesNumber = i

    def getGame(self, number):
        return self.games[number]

    def getGamesNumber(self):
        return self.gamesNumber

#############################


def getWordID(word, dictionary):
    # do not save unknown/unspecified values to dictionary
    unknown = ["N/A", "Unknown"]
    if word in unknown:
        return -1

    if word in dictionary:
        return 0

    wordIndex = len(dictionary)
    dictionary.append(word)
    return wordIndex


def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def switchYearPeriod(year, first, last, period):
    yearList = []

    firstYear = first
    lastYear = last
    periodRange = period  # how many years in one period
    noPeriods = math.ceil((lastYear - firstYear + 1) / periodRange)

    if (not isNumber(year)) or (not firstYear <= int(year) <= lastYear):  # N/A or outside the range cases
        residue = -1
    else:
        residue = int(year) % firstYear

    appended = False
    for i in range(0, noPeriods):
        if (not appended) and (i * periodRange <= residue <= (i + 1) * periodRange - 1):
            yearList.append(1)
            appended = True
        else:
            yearList.append(0)
    return yearList


def convertDictionaryToBinary(word, dictionary):
    binaryList = []

    try:
        active = dictionary.index(word)
    except ValueError:
        print("NETWORK WARNING: The word \"%s\" was not found in the network's dictionary. The program will continue as"
              " normal but this value will have no effect on output." % word)
        binaryList.extend([0] * len(dictionary))
        return binaryList
    binaryList.extend([0] * active)  # filling with zeroes
    binaryList.append(1)
    binaryList.extend([0] * (len(dictionary) - active - 1))
    return binaryList


def main():
    # GET INPUT DATA
    data = DataSet()
    # data.setFile('./Input/input.csv')
    data.setFile('./Source/vgsalesYearSorted.csv')  # current one for testing
    data.loadFromFile()

    # DATA LOADING
    # load dynamic dictionaries for unique words based on year
    firstYear = 1980
    lastYear = 2017
    periodRange = 38  # how many years for each period

    path = "./Saved/Dictionaries/" + str(periodRange) + "-year-periods/"
    genreDictionaries = load(path + "genreDict")
    platformDictionaries = load(path + "platformDict")
    publisherDictionaries = load(path + "publisherDict")

    # PROCESS THE INPUT DATA
    trueResults = []
    predictedResults = []
    previousPeriod = -1
    for j in range(0, data.getGamesNumber() - 1):
        # FIND PERIOD
        year = int(data.getGame(j).getYear())
        try:
            period = switchYearPeriod(year, firstYear, lastYear, periodRange).index(1)
        except Exception:
            print("Error for %d element!" % j)

        # SELECT PERIOD'S DICTIONARY
        genrePeriodDictionary = genreDictionaries[period]
        platformPeriodDictionary = platformDictionaries[period]
        publisherPeriodDictionary = publisherDictionaries[period]

        # CREATE AN INPUT VECTOR FOR THE PERIOD
        inputVector = []
        inputVector.extend(convertDictionaryToBinary(data.getGame(j).getGenre(), genrePeriodDictionary))
        inputVector.extend(convertDictionaryToBinary(data.getGame(j).getPlatform(), platformPeriodDictionary))
        inputVector.extend(convertDictionaryToBinary(data.getGame(j).getPublisher(), publisherPeriodDictionary))
        trueResults.append(float(data.getGame(j).getGlobalSales()))

        # FIND THE MODEL
        if previousPeriod != period:  # save the time reducing unnecessary loads
            path = "./Saved/Models/" + str(periodRange) + "-year-periods/"
            startYear = firstYear + period * periodRange
            endYear = startYear + periodRange - 1
            modelList = load(path + str(startYear) + "-" + str(endYear) + "model")
            model = modelList[0]
            print("Using " + modelList[3])

        # PREDICT
        predictedVal = model.predict(np.array(inputVector).reshape(1, -1))
        if predictedVal < 0.01:  # neutralize impossible predictions to the lowest possible value
            predictedVal = 0.01
        predictedResults.append(predictedVal)
        previousPeriod = period

    print("Done. All inputs processed. Mean squared error for the whole data is: ")
    print(mean_squared_error(trueResults, predictedResults))

    resultNumberArray = []
    labels = []
    diffArray = []
    with open('./Output/Result.csv', mode='w', encoding="utf-8") as resultFile:
        resultWriter = csv.writer(resultFile, delimiter=';', quotechar='"')
        resultWriter.writerow(["Game title", "Platform", "Year", "Genre", "Publisher", "Global Sales", "Predicted Sales", "Prediction error"])
        for x in range(0, data.getGamesNumber()-1):
            diff = float(data.getGame(x).getGlobalSales())-predictedResults[x]
            resultNumberArray.append(x)
            labels.append(data.getGame(x).getTitle())
            diffArray.append(diff)
            resultWriter.writerow([data.getGame(x).getTitle(), data.getGame(x).getPlatform(), data.getGame(x).getYear(), data.getGame(x).getGenre(), data.getGame(x).getPublisher(), float(data.getGame(x).getGlobalSales()), predictedResults[x], diff])

    plt.bar(resultNumberArray, diffArray, align='center')
    plt.xticks(resultNumberArray, labels)
    plt.show()


main()
