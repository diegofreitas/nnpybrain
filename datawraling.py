from pandas import DataFrame
from ggplot import *
from stocknn import *
from pybrain.datasets import SupervisedDataSet
import numpy as np
import pandas

__author__ = 'diego.freitas'

def toFloat(value):
    return float(value.lstrip())/100

company_code = 'AAPL34'

cotahist = open('COTAHIST_A2014.TXT','r')

bovespadf = DataFrame(columns=["TIPREG","DATA","CODBDI","CODNEG", "TPMERC", "NOMRES", "ESPECI", "PRAZOT", "MODREF", "PREABE", "PREMAX", "PREMIN", "PREMED", "PREULT", "PREOFC", "PREOFV", "TOTNEG", "QUATOT", "VOLTOT", "PREEXE", "INDOPC", "DATVEN", "FATCOT", "PTOEXE", "CODISI", "DISMES"])

for line in cotahist:
    if (line.startswith('00') or line.startswith('99') or len(line) == 0 or  not "PETR4" in line):
        continue
    bovespadf = bovespadf.append([{
                                      #"TIPREG": line[0:2],
                                      "DATA": line[2:10],
                                      "DIA": line[8:10],
                                      "CODBDI": line[10:12],
                                      "CODNEG": line[12:24].strip(),
                                      #"TPMERC": line[24:27],
                                      "NOMRES": line[27:39],
                                      #"ESPECI": line[39:49],
                                      #"PRAZOT": line[49:52],
                                      #"MODREF": line[52:56],
                                      "PREABE": toFloat(line[56:69]),
                                      #"PREMAX": line[69:82],
                                      #"PREMIN": line[82:95],
                                      #"PREMED": line[95:108],
                                      "PREULT": toFloat(line[108:121])
                                      #"PREOFC": line[121:134],
                                      #"PREOFV": line[134:147],
                                      #"TOTNEG": line[147:152],
                                      #"QUATOT": line[152:170],
                                      #"VOLTOT": line[170:188],
                                      #"PREEXE": line[188:201],
                                      #"INDOPC": line[201:202],
                                      #"DATVEN": line[202:210],
                                      #"FATCOT": line[210:217],
                                      #"PTOEXE": line[217:230],
                                      #"CODISI": line[230:242],
                                      #"DISMES": line[242:245]
                                  }])



bovespadf = bovespadf[["DIA","DATA","CODNEG","NOMRES","PREABE", "PREULT"]][(bovespadf.CODNEG == 'PETR4')]
bovespadf = bovespadf[pandas.notnull(bovespadf['PREULT'])]

bovespadf['TOMORROW_PREULT'] = bovespadf["PREULT"]
bovespadf.TOMORROW_PREULT = bovespadf.TOMORROW_PREULT.shift(-1)
bovespadf = bovespadf[pandas.notnull(bovespadf['TOMORROW_PREULT'])]


trainingdf = bovespadf[(bovespadf.DATA < '20140830')]
comparedf = bovespadf[(bovespadf.DATA > '20140830')]
comparedf = comparedf[(comparedf.DATA < '20141001')]


print bovespadf.head()

print trainingdf.head()
print comparedf.head()

dataset = SupervisedDataSet(1, 1)
for index, row in trainingdf.iterrows():
        dataset.addSample(row['PREULT'], row['TOMORROW_PREULT'])


nn = NeuralNetwork()
nn.train(dataset)


results = []
for index, row in comparedf.iterrows():
    results.append(round(nn.predict([row["PREULT"]]),2))

comparedf['NN_PREULT'] = results

print comparedf

#print ggplot(comparedf, aes('DIA', 'TOMORROW_PREULT')) + \
#    geom_line() + stat_smooth(colour='blue', span=0.3)

print ggplot(comparedf, aes(x='DIA', y='NN_PREULT')) + \
    geom_line() + stat_smooth(colour='red', span=0.3)