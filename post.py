import pandas
import os

dfs = []
for filename in os.listdir("ensemble"):
    dfs.append(pandas.read_csv("ensemble/" + filename)['time'])

agg = pandas.DataFrame()
for df in dfs:
    agg = pandas.concat((agg, df), axis=1)

agg.mean(axis=1).to_csv("best_result.csv")