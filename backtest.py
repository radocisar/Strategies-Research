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
from tqdm import tqdm

param_search_logger = logging.getLogger("param_search_logger")
param_search_logger.setLevel(logging.INFO)
param_search_handler = logging.FileHandler(filename='param_search_log.log')
param_search_handler.setLevel(logging.INFO)

final_results_logger = logging.getLogger("final_results_logger")
final_results_logger.setLevel(logging.INFO)
final_results_handler = logging.FileHandler(filename='final_results_log.log')
final_results_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(message)s')

param_search_handler.setFormatter(formatter)
final_results_handler.setFormatter(formatter)

param_search_logger.addHandler(param_search_handler)
final_results_logger.addHandler(final_results_handler)



try:
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
    for i in tqdm(mondays, leave=False):
        if (i + three_weeks_dt + one_week_dt + dt.timedelta(days=2)) <= last_friday_dt:
            ## Assign train_start_date_dt
            # 19 days for training
            train_start_date_dt = i #df.iloc[n,:].name.date()
            ## Assign train_end_date_dt
            train_end_date_dt = train_start_date_dt + three_weeks_dt #one day after actual last day as last day doesn't count
            # continue with next i if there is less than 10 training data days
            if (train_end_date_dt - train_start_date_dt) < (three_weeks_dt - dt.timedelta(days=4)) or \
                (train_end_date_dt - train_start_date_dt) > (three_weeks_dt + dt.timedelta(days=1)):
                final_results_logger.info(f"ERROR...HENCE SKIPPING CYCLE: (train_start_date_dt - train_end_date_dt) is less than dt.timedelta(days=10): \
                    train_start_date_dt: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
                final_results_logger.info(f"-----------------")
                final_results_logger.info(f"-----------------")
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
                        final_results_logger.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_start_date_dt\" neither next Monday nor Tuesday is a \
                            true date in my data: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
                        final_results_logger.info(f"-----------------")
                        final_results_logger.info(f"-----------------")
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
                        final_results_logger.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_start_date_dt\" neither next Monday nor Tuesday is a \
                            true date in my data: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
                        final_results_logger.info(f"-----------------")
                        final_results_logger.info(f"-----------------")
                        continue
            else:
                raise TypeError("when selecting \"test_start_date_dt\" the \"train_end_date_ dt\" is neither Saturday nor Sunday")
                final_results_logger.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_start_date_dt\" the \"train_end_date_dt\" is neither \
                    Saturday nor Sunday: train_start_date_dt: {train_start_date_dt} - train_end_date_dt: {train_end_date_dt}")
                final_results_logger.info(f"-----------------")
                final_results_logger.info(f"-----------------")
                continue
            ## Assign test_end_date_dt
            try:
                # if starting from Saturday and Friday is available
                test_end_index_num = int(np.where(unique_date_index == test_start_date_dt + one_week_dt - dt.timedelta(days=1))[0])
                test_end_date_dt = unique_date_index[test_end_index_num] + dt.timedelta(days=1) # assign Friday (show Saturday)
            except TypeError:
                try:
                    # if starting from Saturday and Thursday is available or starting from Sunday and Friday is available
                    test_end_index_num = int(np.where(unique_date_index == test_start_date_dt + one_week_dt - dt.timedelta(days=2))[0])
                    test_end_date_dt = unique_date_index[test_end_index_num] + dt.timedelta(days=1) # assign Friday (show Saturday) or Thursday (show Friday)
                except TypeError:
                    # try:
                    #     test_end_index_num = int(np.where(unique_date_index == test_start_date_dt + one_week_dt - dt.timedelta(days=2))[0])
                    #     test_end_date_dt = unique_date_index[test_end_index_num] #next business day
                    # except TypeError:
                    final_results_logger.info(f"ERROR...HENCE SKIPPING CYCLE: when selecting \"test_end_date_dt\" neither Friday nor \
                        Thursday exist or \"test_end_date_dt\" is a Tuesday and Friday doesn't exist: \
                        test_start_date_dt: {test_start_date_dt}")
                    final_results_logger.info(f"-----------------")
                    final_results_logger.info(f"-----------------")
                    continue
            # if (test_end_date_dt - test_start_date_dt) < (one_week_dt - dt.timedelta(days=4)) or \
            #     (test_end_date_dt - test_start_date_dt) > (one_week_dt + dt.timedelta(days=1)):
            #     final_results_logger.info(f"ERROR...HENCE SKIPPING CYCLE: (train_start_date_dt - test_start_date_dt) is less than dt.timedelta(days=10): \
            #         train_start_date_dt: train_start_date_dt: {train_start_date_dt} - test_start_date_dt: {test_start_date_dt}")
            #     final_results_logger.info(f"-----------------")
            #     final_results_logger.info(f"-----------------")
            #     continue

            
            ## Filter for relevant trading period
            # analyzed_df = df[trading_date]
            analyzed_df = df.loc[train_start_date_dt:train_end_date_dt,:].copy()
            
            
            ## Parameters
            # trade_size = 20000
            # trading_date = "2018-03"

            # For resampling
            # frequency = 15
            
            # filter_days = []
            # filter_hours = range(8,20) # UTC timezone
            # filter_mins = []
            # filter_secs = []


            # For assigning factors
            # num_of_std_dev = 3
            # lookback = 20 # Days
            

            # For calculating factors
            # stop_loss_buffer = 0.0010
            # take_profit_buffer = 0.0010

            ## Testing run
            ## Parameter Grid
            """
            frequency" - For resampling
            "filter_hours" - For selecting most suitable tme of day to trade
            "num_of_std_dev" - For assigning factors
            "lookback" -  For assigning factors
            "stop_loss_buffer" - For calculating factors
            "take_profit_buffer" - For calculating factors
            """
            params = {
                "frequency":[5, 10, 15, 20],
                "filter_hours":[range(7,20), range(8,20), range(8,22)],
                "num_of_std_dev":[2, 2.5, 3, 3.5],
                "lookback":[10, 20, 30],
                "stop_loss_buffer":[0.0005, 0.0010, 0.0015],
                "take_profit_buffer":[0.0005, 0.0010, 0.0015]}
            # "trade_size":20000,
            tested_params = {}

            for param in ParameterGrid(params):
                trade_size = 20000 # param["trade_size"]
                frequency = param["frequency"]
                filter_hours = param["filter_hours"]
                num_of_std_dev = param["num_of_std_dev"]
                lookback = param["lookback"]
                stop_loss_buffer = param["stop_loss_buffer"]
                take_profit_buffer = param["take_profit_buffer"]

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
            
                param_search_logger.info(f"{{\"Train Start Date\":{train_start_date_dt}, \"Train End Date\":{train_end_date_dt}, \"Test Start Date\":{test_start_date_dt},\
                \"Test End Date\": {test_end_date_dt}}} | {{\"trade_size\":{trade_size}, \"frequency\":{frequency}, \"filter_hours\":{filter_hours}, \"filter_hours\":{filter_hours},\
                \"num_of_std_dev\":{num_of_std_dev}, \"lookback\":{lookback}, \"stop_loss_buffer\":{stop_loss_buffer}, \"take_profit_buffer\":{take_profit_buffer}}} | \
                {{\"Number of trades\":{num_of_trades}, \"Gross absolute profit/loss\":{gross_absolute_profit_loss}, \"Gross per cent profit/loss\":{gross_percent_profit_loss},\
                \"Commission\":{commission}, \"Slippage\":{slippage}, \"Net absolute profit/loss\":{net_absolute_profit_loss}, \"Net % profit/loss\":{net_percent_profit_loss}}}")

                ## Log parameteres used in training + net profit
                tested_params[str(param)] = net_percent_profit_loss


            ## Best parameters combination per training period
            best_params = max(tested_params, key=tested_params.get)

            ## Testing run
            ### run best model against test period
            trade_size = best_params["trade_size"]
            frequency = best_params["frequency"]
            filter_hours = best_params["filter_hours"]
            num_of_std_dev = param["num_of_std_dev"]
            lookback = best_params["lookback"]
            stop_loss_buffer = best_params["stop_loss_buffer"]
            take_profit_buffer = best_params["take_profit_buffer"]

            ## Resampling into 1 Minute brs
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

            ## Result (P & L)
            num_of_trades, P_N_L_Stats, gross_absolute_profit_loss, gross_percent_profit_loss, \
                commission, slippage, net_absolute_profit_loss, \
                net_percent_profit_loss = Results_P_and_L.results_P_and_L(min_1_analyzed_df, trade_size, \
                    train_start_date_dt.strftime(format="%Y%m%d"), train_end_date_dt.strftime(format="%Y%m%d"))

            final_results_logger.info(f"FINAL RESULT | {{\"Train Start Date\":{train_start_date_dt}, \"Train End Date\":{train_end_date_dt}, \"Test Start Date\":{test_start_date_dt}, \"Test End Date\": \
                    {test_end_date_dt}}} | {{\"trade_size\":{trade_size}, \"frequency\":{frequency}, \"filter_hours\":{filter_hours}, \
                    \"num_of_std_dev\":{num_of_std_dev}, \"lookback\":{lookback}, \"stop_loss_buffer\":{stop_loss_buffer}, \"take_profit_buffer\":{take_profit_buffer}}} \
                    | {{\"Number of trades\":{num_of_trades}, \"Gross absolute profit/loss\":{gross_absolute_profit_loss}, \"Gross per cent profit/loss\":{gross_percent_profit_loss}, \
                        \"Commission\":{commission}, \"Slippage\":{slippage}, \"Net absolute profit/loss\":{net_absolute_profit_loss}, \"Net % profit/loss\":{net_percent_profit_loss}}}")
            final_results_logger.info(f"-----------------")
            final_results_logger.info(f"-----------------")
            # final_results_logger.info(f"Train Dates: {train_start_date_dt, train_end_date_dt}")
            # final_results_logger.info(f"Test Dates: {test_start_date_dt, test_end_date_dt}")
            # final_results_logger.info(f"Parameters: {frequency, filter_hours, num_of_std_dev, lookback, stop_loss_buffer, take_profit_buffer}")
            # final_results_logger.info(f"-----RESULTS-----")
            # final_results_logger.info(f"Profit and loss: {P_N_L_Stats}")
            # final_results_logger.info(f"Number of trades: {num_of_trades}")
            # final_results_logger.info(f"Gross absolute profit/loss: {gross_absolute_profit_loss}")
            # final_results_logger.info(f"Gross % profit/loss: {gross_percent_profit_loss}")
            # final_results_logger.info(f"Commission: {commission}")
            # final_results_logger.info(f"Slippage: {slippage}")
            # final_results_logger.info(f"Net absolute profit/loss: {net_absolute_profit_loss}")
            # final_results_logger.info(f"Net % profit/loss: {net_percent_profit_loss}")
            # final_results_logger.info(f"-----------------")
            # final_results_logger.info(f"-----------------")

    param_search_logger.handlers.clear()
    final_results_logger.handlers.clear()
    logging.shutdown()
finally:
    param_search_logger.handlers.clear()
    final_results_logger.handlers.clear()
    logging.shutdown()