import csv
import os
import random
import itertools

import numpy as np
import math

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load


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
    # GET DATA
    data = DataSet()
    data.setFile('./Source/vgsalesYearSorted.csv')
    data.loadFromFile()

    # CREATE LAYERS BASED ON TIME PERIOD
    firstYear = 1980
    lastYear = 2017
    periodRange = 3  # how many years for each period
    noPeriods = math.ceil((lastYear - firstYear + 1) / periodRange)

    # DATA NORMALIZATION
    # Expectations:
    # dataset is year sorted (it'll be randomly shuffled later)
    # every training field needs to have a year, no empty or N/A fields
    # "N/A" or "Unknown" values for other fields are handled but they will have no effect on the output

    trainVectors = []
    resultVectors = []

    genreDictionaries = []
    platformDictionaries = []
    publisherDictionaries = []

    periodStart = 0  # start of the given period in sorted dataset
    dataMax = data.getGamesNumber() - 2  # WHY -2? Probably something to do with how class DataSet processes data
    for period in range(0, noPeriods):
        closingYear = firstYear + (period + 1) * periodRange  # till which year given period lasts

        # create dynamic dictionaries of unique words for the period
        genrePeriodDictionary = []
        platformPeriodDictionary = []
        publisherPeriodDictionary = []

        periodEnd = -1
        for i in itertools.count(periodStart):
            # check if not beyond the specified period already
            if i > dataMax or int(data.getGame(i).getYear()) >= closingYear:
                periodEnd = i
                break

            getWordID(data.getGame(i).getGenre(), genrePeriodDictionary)
            getWordID(data.getGame(i).getPlatform(), platformPeriodDictionary)
            getWordID(data.getGame(i).getPublisher(), publisherPeriodDictionary)

        genreDictionaries.append(genrePeriodDictionary)
        platformDictionaries.append(platformPeriodDictionary)
        publisherDictionaries.append(publisherPeriodDictionary)

        # create a train and result vector for given period
        trainPeriodVector = []
        resultPeriodVector = []
        for j in range(periodStart, periodEnd):
            inputVector = []
            inputVector.extend(convertDictionaryToBinary(data.getGame(j).getPlatform(), platformPeriodDictionary))
            inputVector.extend(convertDictionaryToBinary(data.getGame(j).getGenre(), genrePeriodDictionary))
            inputVector.extend(convertDictionaryToBinary(data.getGame(j).getPublisher(), publisherPeriodDictionary))
            trainPeriodVector.append(inputVector)
            resultPeriodVector.append(float(data.getGame(j).getGlobalSales()))

        trainVectors.append(trainPeriodVector)
        resultVectors.append(resultPeriodVector)
        periodStart = periodEnd

    # SAVE DICTIONARIES
    path = "./Saved/Dictionaries/" + str(periodRange) + "-year-periods/"
    if not os.path.exists(path):
        os.makedirs(path)
    dump(genreDictionaries, path + "genreDict")
    dump(platformDictionaries, path + "platformDict")
    dump(publisherDictionaries, path + "publisherDict")

    # NETWORK - DATA TRAIN
    for period in range(0, noPeriods):
        print("%d period" % period)
        periodBest = None
        try:
            path = "./Saved/Models/" + str(periodRange) + "-year-periods/"
            startYear = firstYear + period * periodRange
            endYear = startYear + periodRange - 1
            bestSaved = load(path + str(startYear) + "-" + str(endYear) + "model")
            periodBest = bestSaved[1]
        except Exception:  # no model was saved on disk for this period
            pass
        sparseIterations = 3000  # how many iterations allowed without improvement
        lastImprovement = 0
        while lastImprovement < sparseIterations:
            # PREPARE DATA
            split = int(len(trainVectors[period]) / 3)

            # randomize X and Y values together
            c = list(zip(trainVectors[period], resultVectors[period]))
            random.shuffle(c)
            trainVectors[period], resultVectors[period] = zip(*c)

            # Split the data into training/testing sets
            salesXtrain = trainVectors[period][:-split]
            salesXtest = trainVectors[period][-split:]

            # Split the targets into training/testing sets
            salesYtrain = resultVectors[period][:-split]
            salesYtest = resultVectors[period][-split:]

            # LINEAR REGRESSION MODEL
            # Create linear regression object
            regr = linear_model.LinearRegression()

            # Train the model using the training sets
            regr.fit(salesXtrain, salesYtrain)

            # Make predictions using the testing set
            salesYpred = regr.predict(salesXtest)

            # test the model
            MSE = mean_squared_error(salesYtest, salesYpred)
            if periodBest is None or MSE < periodBest:
                periodBest = MSE
                lastImprovement = 0
                print("Found a better model! (linear) MSE: %f" % MSE)

                # save the model
                modelList = []
                modelList.extend([regr, MSE, r2_score(salesYtest, salesYpred), "Linear regression model"])
                path = "./Saved/Models/" + str(periodRange) + "-year-periods/"
                if not os.path.exists(path):
                    os.makedirs(path)
                startYear = firstYear + period * periodRange
                endYear = startYear + periodRange - 1
                dump(modelList, path + str(startYear) + "-" + str(endYear) + "model")

                startYear = firstYear + period * periodRange
                endYear = startYear + periodRange - 1
                dump(modelList, path + str(startYear) + "-" + str(endYear) + "model")

            # POLYNOMIAL MODEL
            for degrees in range(1, 5):
                model = make_pipeline(PolynomialFeatures(degrees), Ridge())  # , memory="./Cache")

                try:
                    model.fit(salesXtrain, salesYtrain)
                except MemoryError:
                    print("Fitting for %d degrees failed due to memory error" % degrees)
                    continue
                y_plot = model.predict(salesXtest)

                # test the model
                MSE = mean_squared_error(salesYtest, y_plot)
                if periodBest is None or MSE < periodBest:
                    periodBest = MSE
                    lastImprovement = 0
                    print("Found a better model! (polynomial, %d degree) MSE: %f" % (degrees, MSE))

                    # save the model
                    modelList = []
                    modelList.extend([model, MSE, r2_score(salesYtest, salesYpred), "Polynomial model, " + str(degrees) + " degrees"])
                    path = "./Saved/Models/" + str(periodRange) + "-year-periods/"
                    if not os.path.exists(path):
                        os.makedirs(path)

                    startYear = firstYear + period * periodRange
                    endYear = startYear + periodRange - 1
                    dump(modelList, path + str(startYear) + "-" + str(endYear) + "model")

            lastImprovement += 1
            print("%d iterations without improvement, %d period" % ((lastImprovement - 1), period))

    print("Done")


main()
