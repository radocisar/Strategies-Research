import pandas as pd
import os
import time
import patoolib
import gzip
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import openpyxl
import datetime as dt
from sklearn.model_selection import ParameterGrid
import Calculate_Factors
import Calculate_Profit_Loss
import Results_P_and_L
import logging

logging.basicConfig(filename='logging.log',level=logging.INFO, format='%(asctime)s %(message)s', \
    datefmt='%m/%D/%Y %H:%M:%S')

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
def P_and_L_time_distribution(min_1_analyzed_df_dist_analysis, train_start_date_str, train_end_date_str):
    min_1_analyzed_df_dist_analysis.loc[:,"Trade_Entry_Time"] = min_1_analyzed_df_dist_analysis.index
    min_1_analyzed_df_dist_analysis.loc[:,"Trade_Entry_Time_Shifted"] = min_1_analyzed_df_dist_analysis["Trade_Entry_Time"].shift(1)
    min_1_analyzed_df_dist_analysis.drop("Trade_Entry_Time",axis=1, inplace=True)
    min_1_analyzed_df_dist_analysis_exists_only = min_1_analyzed_df_dist_analysis.loc[((min_1_analyzed_df_dist_analysis \
    ["Action"] == "Took Profit") | (min_1_analyzed_df_dist_analysis["Action"] == "Stopped Out")),:]
    min_1_analyzed_df_dist_analysis_exists_only.loc[:,"Hour_of_Entry"] = min_1_analyzed_df_dist_analysis_exists_only \
    ["Trade_Entry_Time_Shifted"].apply(lambda x: x.hour)

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ## Time distribution of Profits
    min_1_analyzed_df_dist_analysis_exists_only[min_1_analyzed_df_dist_analysis_exists_only["Trade_Prft_Lss"]>0] \
    ["Hour_of_Entry"].hist(bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,23], ax=ax1);
    ax1.set_title("Time distribution of profits")
    # plt.draw()
    # plt.show(block=False)
    # plt.xaxis_date()
    ax1.autoscale_view()
    # plt.savefig(f"./Charts/Profit_{train_start_date_str}_{train_end_date_str}.png")

    ## Time distribution of losses
    min_1_analyzed_df_dist_analysis_exists_only[min_1_analyzed_df_dist_analysis_exists_only["Trade_Prft_Lss"]<=0] \
    ["Hour_of_Entry"].hist(bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], ax=ax2);
    ax2.set_title("Time distribution of losses")
    # plt.draw()
    # plt.show(block=False)
    # plt.xaxis_date()
    ax2.autoscale_view()
    
    plt.savefig(f"./Charts/Loss_{train_start_date_str}_{train_end_date_str}.png")
    plt.close(fig=fig)
    




## Instrument
inst = "EURGBP"

## Load data
df = pd.read_pickle(r"L:\Raw_1_sec_Bar_Data\FX\{}\Pickle\{}.pkl".format(inst,inst))

## Adjusting start and end of time series
# remove any dates before first Sunday (FX Specific)
df["weekday"] = df.index.weekday
first_sunday_str = df.drop_duplicates("weekday",keep="first").loc[df["weekday"] == 6,:].index[0].strftime(format="%Y-%m-%d") # FX specific
df = df[first_sunday_str::]
# remove any dates after last Friday
last_friday_str = df.drop_duplicates("weekday",keep="last").loc[df["weekday"] == 4,:].index[0].strftime(format="%Y-%m-%d") # FX specific
df = df[:last_friday_str]



## Selecting, training and testing periods
three_weeks_dt = dt.timedelta(days=19) #for FX specifically
one_week_dt = dt.timedelta(days=5) #for FX specifically
last_friday_dt = df.drop_duplicates("weekday",keep="last").loc[df["weekday"] == 4,:].index[0].date()
mondays = np.unique(df.loc[df["weekday"]==0,"weekday"].index.date)
#n = 1 # increment by week
for i in mondays:
    if (i + three_weeks_dt + one_week_dt + dt.timedelta(days=2)) <= last_friday_dt:
        ## Assign train_start_date_dt
        # 19 days for training
        train_start_date_dt = i #df.iloc[n,:].name.date()
        ## Assign train_end_date_dt
        train_end_date_dt = train_start_date_dt + three_weeks_dt #one day after actual last day as last day doesn't count
        # continue with next i if there is less than 10 training data days
        if (train_end_date_dt - train_start_date_dt) < (three_weeks_dt - dt.timedelta(days=4)) or \
            (train_end_date_dt - train_start_date_dt) > (three_weeks_dt + dt.timedelta(days=1)):
            logging.info(f"ERROR...HENCE SKIPPING CYCLE: (train_start_date_dt - train_end_date_dt) is less than dt.timedelta(days=10): \
                train_start_date_dt: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
            logging.info(f"-----------------")
            logging.info(f"-----------------")
            continue
        ## Assign test_start_date_dt 
        # 6 days for testing
        unique_date_index = np.unique(df.index.date)
        # try:
        #     train_end_index_num = int(np.where(unique_date_index == train_end_date_dt - dt.timedelta(days=1))[0])
        #     try:
        #         # if next Monday is a true date in my d
        #         test_start_date_dt = unique_date_index[train_end_index_num + 2] #next business day
        #     except TypeError:
        #         # if next Tuesday is a true date in my data
        #         train_end_index_num = int(np.where(unique_date_index == train_end_date_dt + dt.timedelta(days=2))[0])
        #         test_start_date_dt = unique_date_index[train_end_index_num] #next business day
        #     else:
        #         # if neither Monday nor Tuesday is a true date in my data
        #         raise TypeError("when selecting \"test_start_date_dt\" neither next Monday nor Tuesday is a true date in my data")
        # except TypeError:
        #     problematic_date = train_end_date_dt # - dt.timedelta(days=1)
        if train_end_date_dt.weekday() == 5:
            try:
                # if next Monday is a true date in my data
                train_end_index_num = int(np.where(unique_date_index == train_end_date_dt + dt.timedelta(days=2))[0])
                test_start_date_dt = unique_date_index[train_end_index_num] #next business day
            except TypeError:
                # if next Tuesday is a true date in my data
                try:
                    train_end_index_num = int(np.where(unique_date_index == train_end_date_dt + dt.timedelta(days=3))[0])
                    test_start_date_dt = unique_date_index[train_end_index_num] #next business day
                except  TypeError:
                    # if neither Monday nor Tuesday is a true date in my data
                    # raise TypeError("when selecting \"test_start_date_dt\" neither next Monday nor Tuesday is a true date in my data")
                    logging.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_start_date_dt\" neither next Monday nor Tuesday is a \
                        true date in my data: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
                    logging.info(f"-----------------")
                    logging.info(f"-----------------")
                    continue
        elif train_end_date_dt.weekday() == 6:
            try:
                # if next Monday is a true date in my d
                train_end_index_num = int(np.where(unique_date_index == train_end_date_dt + dt.timedelta(days=1))[0])
                test_start_date_dt = unique_date_index[train_end_index_num] #next business day
            except TypeError:
                # if next Tuesday is a true date in my data
                try:
                    train_end_index_num = int(np.where(unique_date_index == train_end_date_dt + dt.timedelta(days=2))[0])
                    test_start_date_dt = unique_date_index[train_end_index_num] #next business day
                except TypeError:
                    # if neither Monday nor Tuesday is a true date in my data
                    # raise TypeError("when selecting \"test_start_date_dt\" neither next Monday nor Tuesday is a true date in my data")
                    logging.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_start_date_dt\" neither next Monday nor Tuesday is a \
                        true date in my data: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
                    logging.info(f"-----------------")
                    logging.info(f"-----------------")
                    continue
        else:
            raise TypeError("when selecting \"test_start_date_dt\" the \"train_end_date_ dt\" is neither Saturday nor Sunday")
            logging.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_start_date_dt\" the \"train_end_date_dt\" is neither \
                Saturday nor Sunday: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
            logging.info(f"-----------------")
            logging.info(f"-----------------")
            continue
        ## Assign test_end_date_dt
        try:
            test_end_index_num = int(np.where(unique_date_index == test_start_date_dt + one_week_dt)[0])
            test_end_date_dt = unique_date_index[test_end_index_num] #Friday
        # test_end_date_dt = test_start_date_dt + one_week_dt #one day after actual last day as last day doesn't count
        except TypeError:
            try:
                test_end_index_num = int(np.where(unique_date_index == test_start_date_dt + one_week_dt - dt.timedelta(days=1))[0])
                test_end_date_dt = unique_date_index[test_end_index_num] #next business day
            except TypeError:
                try:
                test_end_index_num = int(np.where(unique_date_index == test_start_date_dt + one_week_dt - dt.timedelta(days=2))[0])
                test_end_date_dt = unique_date_index[test_end_index_num] #next business day
                except TypeError:
                    logging.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_end_date_dt\" neither Friday nor Thursday exis: \
                        test_start_date_dt: {test_start_date_dt}")
                    logging.info(f"-----------------")
                    logging.info(f"-----------------")
                    continue
        if (test_end_date_dt - test_start_date_dt) < (one_week_dt - dt.timedelta(days=4)) or \
            (test_end_date_dt - test_start_date_dt) > (one_week_dt + dt.timedelta(days=1)):
            logging.info(f"ERROR...HENCE SKIPPING CYCLE: (train_start_date_dt - test_start_date_dt) is less than dt.timedelta(days=10): \
                train_start_date_dt: train_start_date_dt: {train_start_date_dt} - test_start_date_dt: {test_start_date_dt}")
            logging.info(f"-----------------")
            logging.info(f"-----------------")
            continue

        # TODO: test_start_date_dt should be a Saturday
        # TODO: Why is test_start_date_dt and test_start_date_dt not being used?

        ## Parameters
        trade_size = 20000
        # trading_date = "2018-03"

        # For resampling
        frequency = 15
        
        filter_days = []
        filter_hours = range(8,20) # UTC timezone
        filter_mins = []
        filter_secs = []


        # For assigning factors
        num_of_std_dev = 3
        lookback = 20 # Days
        

        # For calculating factors
        stop_loss_buffer = 0.0010
        take_profit_buffer = 0.0010


        ## Parameter Grid
        params = {"frequency":15, # For resampling
                "filter_hours":range(8,20), # For selecting most suitable tme of day to trade
                "num_of_std_dev":3, # For assigning factors
                "lookback":20, # For assigning factors
                "stop_loss_buffer":0.0010, # For calculating factors
                "take_profit_buffer":0.0010} # For calculating factors
        
        # ParameterGrid({})


        ## Filter for relevant trading period
        # analyzed_df = df[trading_date]
        analyzed_df = df.loc[train_start_date_dt:train_end_date_dt,:].copy()


        ## Resampling into 1 Minute bars
        resample_interval = f"{frequency}T"
        min_1_analyzed_df = resample(analyzed_df, resample_interval).copy()


        ## Assign factors
        index_frequency = pd.Timedelta(minutes=frequency)
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
        P_and_L_time_distribution(min_1_analyzed_df_dist_analysis, train_start_date_dt.strftime(format="%Y%m%d") \
            , train_end_date_dt.strftime(format="%Y%m%d"))


        ## Result (P & L)
        num_of_trades, P_N_L_Stats, gross_absolute_profit_loss, gross_percent_profit_loss, \
            commission, slippage, net_absolute_profit_loss, \
            net_percent_profit_loss = Results_P_and_L.results_P_and_L(min_1_analyzed_df, trade_size, \
                train_start_date_dt.strftime(format="%Y%m%d"), train_end_date_dt.strftime(format="%Y%m%d"))
        
        logging.info(f"Train Dates: {train_start_date_dt, train_end_date_dt}")
        logging.info(f"Test Dates: {test_start_date_dt, test_end_date_dt}")
        logging.info(f"Parameters: {frequency, filter_hours, num_of_std_dev, lookback, stop_loss_buffer, take_profit_buffer}")
        logging.info(f"-----RESULTS-----")
        logging.info(f"Profit and loss: {P_N_L_Stats}")
        logging.info(f"Number of trades: {num_of_trades}")
        logging.info(f"Gross absolute profit/loss: {gross_absolute_profit_loss}")
        logging.info(f"Gross % profit/loss: {gross_percent_profit_loss}")
        logging.info(f"Commission: {commission}")
        logging.info(f"Slippage: {slippage}")
        logging.info(f"Net absolute profit/loss: {net_absolute_profit_loss}")
        logging.info(f"Net % profit/loss: {net_percent_profit_loss}")
        logging.info(f"-----------------")
        logging.info(f"-----------------")

logging.shutdown()

