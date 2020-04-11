import pandas as pd
import os
import time
import patoolib
import gzip
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import openpyxl
import datetime as dt

import Calculate_Factors
import Calculate_Profit_Loss
import Results_P_and_L

def resample(analyzed_df, resample_interval):
    min_1_low = analyzed_df.loc[:,"Low"].resample(resample_interval).apply(np.min)
    min_1_high = analyzed_df.loc[:,"High"].resample(resample_interval).apply(np.max)
    min_1_open = analyzed_df.loc[:,"Open"].resample(resample_interval).first()
    min_1_close = analyzed_df.loc[:,"Close"].resample(resample_interval).last()
    
    min_1_analyzed_df = pd.DataFrame.from_dict({"Open":min_1_open,"High":min_1_high,"Low":min_1_low,"Close":min_1_close})

    return min_1_analyzed_df


def assign_factors(min_1_analyzed_df, lookback, num_of_std_dev, index_frequency):
    min_1_analyzed_df_temp = min_1_analyzed_df

    min_1_analyzed_df_temp["DateTime_UTC_column"] = min_1_analyzed_df_temp.index
    min_1_analyzed_df["Middle_Band"] = min_1_analyzed_df["Close"].rolling(lookback).mean()
    min_1_analyzed_df["Upper_Band"] = min_1_analyzed_df["Middle_Band"] + \
    (min_1_analyzed_df["Close"].rolling(lookback).std()*num_of_std_dev)
    min_1_analyzed_df["Lower_Band"] = min_1_analyzed_df["Middle_Band"] - \
    (min_1_analyzed_df["Close"].rolling(lookback).std()*num_of_std_dev)
    min_1_analyzed_df["Prev_Lower_Band"] = min_1_analyzed_df["Lower_Band"].shift(1)
    min_1_analyzed_df["Prev_Upper_Band"] = min_1_analyzed_df["Upper_Band"].shift(1)
    min_1_analyzed_df["Upper_Lower_Band_Span"] = min_1_analyzed_df["Upper_Band"] - min_1_analyzed_df["Lower_Band"]
    min_1_analyzed_df["Candle_High_and_Close_above_U_Band"] = min_1_analyzed_df["Close"] > min_1_analyzed_df["Upper_Band"]
    min_1_analyzed_df["Prev_Candle_High_and_Close_above_U_Band"] = min_1_analyzed_df["Candle_High_and_Close_above_U_Band"].shift(1)
    min_1_analyzed_df.drop("Candle_High_and_Close_above_U_Band",axis=1,inplace=True)
    min_1_analyzed_df["Candle_Low_and_Close_below_L_Band"] = min_1_analyzed_df["Close"] < min_1_analyzed_df["Lower_Band"]
    min_1_analyzed_df["Prev_Candle_Low_and_Close_below_L_Band"] = min_1_analyzed_df["Candle_Low_and_Close_below_L_Band"].shift(1)
    min_1_analyzed_df.drop("Candle_Low_and_Close_below_L_Band",axis=1,inplace=True)
    #min_1_analyzed_df["Is_Prev_Cndl_Up"] = ((min_1_analyzed_df["Close"].shift(1) - min_1_analyzed_df["Open"].shift(1)) >= 0)
    #min_1_analyzed_df["Is_Prev_Cndl_Down"] = ((min_1_analyzed_df["Close"].shift(1) - min_1_analyzed_df["Open"].shift(1)) < 0)
    min_1_analyzed_df["Five_per_std"] = min_1_analyzed_df["Close"].rolling(5).std()
    min_1_analyzed_df["Prev_High"] = min_1_analyzed_df["High"].shift(1)
    min_1_analyzed_df["Prev_Low"] = min_1_analyzed_df["Low"].shift(1)

    def Dict_of_Closing_Prices_Creation(row):
        if row.DateTime_UTC_column > min_1_analyzed_df.iloc[lookback-1,:].name:
            Dict_of_Closing_Prices = {}
            for n in range(1, lookback):
                new_DateTime_UTC = row.DateTime_UTC_column-(index_frequency*(n))
                Dict_of_Closing_Prices[new_DateTime_UTC] = min_1_analyzed_df.loc[new_DateTime_UTC,"Close"]
            return Dict_of_Closing_Prices
        else:
            return np.nan

    min_1_analyzed_df["Dict_of_Closing_Prices"] = min_1_analyzed_df_temp.apply(Dict_of_Closing_Prices_Creation, axis=1)    

    return min_1_analyzed_df


def drop_na(min_1_analyzed_df):
    min_1_analyzed_df.dropna(inplace=True)
    return min_1_analyzed_df

def filter_certain_hours(min_1_analyzed_df, filter_hours):
    min_1_analyzed_df = min_1_analyzed_df[min_1_analyzed_df.index.hour.isin(filter_hours)]
    return min_1_analyzed_df

def include_period_num(min_1_analyzed_df):
    min_1_analyzed_df["Period_Number"] = range(0,min_1_analyzed_df.shape[0])
    return min_1_analyzed_df

## P&L time distribution
def P_and_L_time_distribution(min_1_analyzed_df_dist_analysis):
    min_1_analyzed_df_dist_analysis["Trade_Entry_Time"] = min_1_analyzed_df_dist_analysis.index
    min_1_analyzed_df_dist_analysis["Trade_Entry_Time_Shifted"] = min_1_analyzed_df_dist_analysis["Trade_Entry_Time"].shift(1)
    min_1_analyzed_df_dist_analysis.drop("Trade_Entry_Time",axis=1, inplace=True)
    min_1_analyzed_df_dist_analysis_exists_only = min_1_analyzed_df_dist_analysis.loc[((min_1_analyzed_df_dist_analysis \
    ["Action"] == "Took Profit") | (min_1_analyzed_df_dist_analysis["Action"] == "Stopped Out")),:]
    min_1_analyzed_df_dist_analysis_exists_only["Hour_of_Entry"] = min_1_analyzed_df_dist_analysis_exists_only \
    ["Trade_Entry_Time_Shifted"].apply(lambda x: x.hour)

    ## Time distribution of Profits
    min_1_analyzed_df_dist_analysis_exists_only[min_1_analyzed_df_dist_analysis_exists_only["Trade_Prft_Lss"]>0] \
    ["Hour_of_Entry"].hist(bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,23]);
    plt.title("Time distribution of profits")
    plt.show()

    ## Time distribution of losses
    min_1_analyzed_df_dist_analysis_exists_only[min_1_analyzed_df_dist_analysis_exists_only["Trade_Prft_Lss"]<=0] \
    ["Hour_of_Entry"].hist(bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]);
    plt.title("Time distribution of losses")
    plt.show()    
    




## Instrument
inst = "EURGBP"

## Parameters
trade_size = 20000
trading_date = "2018-03"

start_date = "2018-02-19"
end_date = "2020-04-03"

filter_days = []
filter_hours = range(8,20) # UTC timezone
filter_mins = []
filter_secs = []

# For resampling
resample_interval = "15T"

# For assigning factors
num_of_std_dev = 3
lookback = 20 # Days
index_frequency = pd.Timedelta(minutes=15)

# For calculating factors
stop_loss_buffer = 0.0010
take_profit_buffer = 0.0010

asdf

## Load data
df = pd.read_pickle(r"L:\Raw_1_sec_Bar_Data\FX\{}\Pickle\{}.pkl".format(inst,inst))
# analyzed_df = df[trading_date]
analyzed_df = df.loc[start_date:end_date,:]


## Resampling into 1 Minute bars
min_1_analyzed_df = resample(analyzed_df, resample_interval)


## Calculate factors
min_1_analyzed_df = assign_factors(min_1_analyzed_df, lookback, num_of_std_dev, index_frequency)


## Dropping N.As
min_1_analyzed_df = drop_na(min_1_analyzed_df)


## Include only certain times of the day
min_1_analyzed_df = filter_certain_hours(min_1_analyzed_df, filter_hours)


## Including "Period Number"
min_1_analyzed_df = include_period_num(min_1_analyzed_df)


## Calculating Factor's Values
min_1_analyzed_df = Calculate_Factors.calc_factors(min_1_analyzed_df, 
                                                    num_of_std_dev, 
                                                    stop_loss_buffer, 
                                                    take_profit_buffer)


## Calculating Factor's Profit and Loss
min_1_analyzed_df, min_1_analyzed_df_dist_analysis = Calculate_Profit_Loss.calc_profit_loss(min_1_analyzed_df,
                                                                                            trade_size)


## P&L time distribution profit and loss charts
P_and_L_time_distribution(min_1_analyzed_df_dist_analysis)


## Result (P & L)
Results_P_and_L.results_P_and_L(min_1_analyzed_df, trade_size)