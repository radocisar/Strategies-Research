import pandas as pd
import numpy as np

def calc_factors(min_1_analyzed_df, num_of_std_dev, stop_loss_buffer, take_profit_buffer):
    """
    #### Entry:
    Test each of these:
    1. Upper or lower BB (3 std for 20 day middle band) breached
    2. AND the price needs to get back inside the BB
    3. AND the BB outter band span needs to be => 0.0015
    ---
    1. Upper or lower BB (2 std for 20 day middle band) breached
    2. AND the price needs to get back inside the BB
    3. AND the BB outter band span needs to be => 0.0015
    ---
    1. Upper or lower BB (3 std for 20 day middle band) touched
    2. AND the BB outter band span needs to be => 0.0015
    ---
    1. Upper or lower BB (2 std for 20 day middle band) touched
    2. AND the BB outter band span needs to be => 0.0015
    ---
    1. Upper or lower BB (3 std for 20 day middle band) touched
    2. AND the next candle needs to be red (if looking for short) or green (if looking for long)
    3. AND the openning price of the next candle needs to be no more than 40% between the lower/upper band and the middle band
    4. AND the BB outter band span needs to be => 0.0015
    ---
    1. Upper or lower BB (2 std for 20 day middle band) touched
    2. AND the next candle needs to be red (if looking for short) or green (if looking for long)
    3. AND the openning price of the next candle needs to be no more than 40% between the lower/upper band and the middle band
    4. AND the BB outter band span needs to be => 0.0015

    #### Stop/Exit:
    Test each of these:
    1. above/below the high/low of the candle that touched/breached BB upper/lower band
    2. above/below BB upper/lower band

    #### Take profit:
    1. 1 pip above(for short)/below (for long) the midle band (SMA)
    2. 1 pip above(for short)/below (for long) the lower/upper band
    3. 1 pip above(for short)/below (for long) the 50% between middle band (SMA) and lower/upper band
    4. 5 pips below entry price
    
    #### To make the strategy more profitable:
    """
    # class take_profit_stop_distance:
    #     tpd = 0.0005
    #     tsd = 0.0001

    class pos_open:    
        
        is_position_open = False
        entry_prc = 0
        long_short = ""
        exit_price_on_entry_value = 0
        take_prft_price = 0
        
        @classmethod
        def position_opened(cls, opened):
            if opened == True:
                cls.is_position_open = True
            else:
                cls.is_position_open = False
        @classmethod
        def entry_price(cls, entry_prcs):
            cls.entry_prc = round(entry_prcs,6)
        @classmethod
        def take_profit_price(cls, take_prft_prc):
            cls.take_prft_price = round(take_prft_prc,6)
        @classmethod
        def long_or_short_entry(cls, long_or_short):
            cls.long_short = long_or_short
        @classmethod
        def exit_price_on_entry(cls, prc):
            cls.exit_price_on_entry_value = round(prc,6)

    def calc_new_stats(Closing_prices, theo_price):
        Closing_prices.append(theo_price)
        new_mean_and_upper_lower_bands = {}
        new_mean_and_upper_lower_bands["new_middle_band"] = pd.Series(Closing_prices).mean()
        new_mean_and_upper_lower_bands["new_upper_band"] = new_mean_and_upper_lower_bands["new_middle_band"] + \
        (pd.Series(Closing_prices).std()*num_of_std_dev)
        new_mean_and_upper_lower_bands["new_lower_band"] = new_mean_and_upper_lower_bands["new_middle_band"] - \
        (pd.Series(Closing_prices).std()*num_of_std_dev)
        
        return new_mean_and_upper_lower_bands

    def enter_factor(row):
        if (row.Upper_Lower_Band_Span >= 0.0015) & (pos_open.is_position_open == False):
        ## For short:
            # The below IF statement takes care of standard scenario where candle gets above upper BB and then closes below it
            if (row.High > row.Upper_Band) & (row.Close < row.Upper_Band):
                # When last price equaled high price, the BB upper band shouldn't have been breached
                if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), row.High)["new_upper_band"] \
                > row.High:
                    # If it wasn't, then check when did the upper BB fall below high price
                    for test_price in np.linspace(row.High,row.Close,round((row.Close-row.High)/-0.00005)+1):
                        if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), test_price)["new_upper_band"] \
                        <= row.High:
                            # Once it did, the last price needs to:
                            #        a) tick twice while upper BB is below high for entry to occur
                            if (test_price - row.Close) > 0.0001:
                                pos_open.long_or_short_entry("short")
                                pos_open.entry_price(round(test_price,6) - 0.0001)
                                if row.Prev_Candle_High_and_Close_above_U_Band == True:
                                    if row.Prev_High > row.High:
                                        pos_open.exit_price_on_entry (row.Prev_High + stop_loss_buffer)
                                    else:
                                        pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
                                else:
                                    pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
                                #print("Hit 1")
                                entry = True
                            else:
                                #print("Hit 2")
                                entry = False
                            break
                        else:
                            entry = False
                # Following takes care of a scenario where the upper BB may have been breached on candle's high after previous
                # candle's high closed above upper BB
                elif (row.Prev_Candle_High_and_Close_above_U_Band == True) & (row.Low < row.Upper_Band):
                    for tst_price in np.linspace(row.High,row.Close,round((row.Close-row.High)/-0.00005)+1):
                            if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), tst_price)["new_upper_band"] \
                            > tst_price:
                                if (tst_price - row.Low) > 0.0001:
                                    pos_open.long_or_short_entry("short")
                                    pos_open.entry_price(round(tst_price,6) - 0.0001)
                                    if row.Prev_Candle_High_and_Close_above_U_Band == True:
                                        if row.Prev_High > row.High:
                                            pos_open.exit_price_on_entry (row.Prev_High + stop_loss_buffer)
                                        else:
                                            pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
                                    else:
                                        pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
                                    #print("Hit 3")
                                    entry = True
                                else:
                                    #print("Hit 4")
                                    entry = False
                            else:
                                entry = False
    #             # When last price equaled high price, the BB upper band was breached
    #             elif (calc_new_stats(list(row.Dict_of_Closing_Prices.values()), row.High)["new_upper_band"] \
    #             <= row.High) & (row.Low < row.Upper_Band):
    #                 for test_1_price in np.linspace(row.High,row.Close,round((row.Close-row.High)/-0.00005)+1):
    #                         if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), test_1_price)["new_upper_band"] \
    #                         > test_1_price:
    #                             if (test_1_price - row.Low) > 0.0001:
    #                                 pos_open.long_or_short_entry("short")
    #                                 pos_open.entry_price(round(test_1_price,6) - 0.0001)
    #                                 if row.Prev_Candle_High_and_Close_above_U_Band == True:
    #                                     if row.Prev_High > row.High:
    #                                         pos_open.exit_price_on_entry (row.Prev_High + stop_loss_buffer)
    #                                     else:
    #                                         pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
    #                                 else:
    #                                     pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
    #                                 #print("Hit 3")
    #                                 entry = True
    #                             else:
    #                                 #print("Hit 4")
    #                                 entry = False
    #                         else:
    #                             entry = False
                else:
                    #print("Short - something else happened - {}".format(row.DateTime_UTC_column))
                    entry = False
                            #        b) tick twice away from the BB while upper BB is below high, for entry to occur
                            #        c) close below BB while upper BB is below high, for entry to occur on the close (or next candle's open)
            # The below takes care of:
            #   1) a scenario where previous candles closed above BB and next candle opens at/below upper BB and then closes above it
            #   2) a very rare occurence when a candle closes above BB and the next candle gaps and stays below it (not 
            #      even its high gets above BB)
            elif (row.Prev_Candle_High_and_Close_above_U_Band == True) & (row.Low < row.Upper_Band):
                # 2)
                #        a) tick twice while under BB for entry to occur
                if (row.Close <= row.Upper_Band) & ((row.High - row.Close) > 0.0001):
                    pos_open.long_or_short_entry("short")
                    pos_open.entry_price(row.High - 0.0001)            
                    if row.Prev_Candle_High_and_Close_above_U_Band == True:
                        if row.Prev_High > row.High:
                            pos_open.exit_price_on_entry (row.Prev_High + stop_loss_buffer)
                        else:
                            pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
                    else:
                        pos_open.exit_price_on_entry (row.High + stop_loss_buffer)
                    #print("Hit 6")
                    entry = True
                # 1)
                else:
                    #print("Hit 7")
                    entry = False

        ## For long:
            # The below IF statement takes care of standard scenario where candle gets below lower BB and then closes above it
            elif (row.Low < row.Lower_Band) & (row.Close > row.Lower_Band):
                # When last price equaled low price, the BB lower band shouldn't have been breached
                if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), row.Low)["new_lower_band"] \
                < row.Low:
                    # If it wasn't, then check when did the lower BB rise above low price
                    for test_price in np.linspace(row.Low,row.Close,round((row.Close-row.Low)/0.00005)+1):
                        if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), test_price)["new_lower_band"] \
                        >= row.Low:
                            # Once it did, the last price needs to:
                            #        a) tick twice while lower BB is above low for entry to occur
                            if (row.Close - test_price) > 0.0001:
                                pos_open.long_or_short_entry("long")
                                pos_open.entry_price(round(test_price,6) + 0.0001)
                                if row.Prev_Candle_Low_and_Close_below_L_Band == True:
                                    if row.Prev_Low < row.Low:
                                        pos_open.exit_price_on_entry (row.Prev_Low - stop_loss_buffer)
                                    else:
                                        pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
                                else:
                                    pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
                                #print("Hit 8")
                                entry = True
                            else:
                                #print("Hit 9")
                                entry = False
                            break
                        else:
                            entry = False
                # Following takes care of a scenario where the lower BB may have been breached on candle's low after previous
                # candle's low closed below lower BB
                elif (row.Prev_Candle_Low_and_Close_below_L_Band == True) & (row.High > row.Lower_Band):
                    for tst_price in np.linspace(row.Low,row.Close,round((row.Close-row.Low)/0.00005)+1):
                            if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), tst_price)["new_lower_band"] \
                            < tst_price:
                                if (row.High - tst_price) > 0.0001:
                                    pos_open.long_or_short_entry("long")
                                    pos_open.entry_price(round(tst_price,6) + 0.0001)
                                    if row.Prev_Candle_Low_and_Close_below_L_Band == True:
                                        if row.Prev_Low < row.Low:
                                            pos_open.exit_price_on_entry (row.Prev_Low - stop_loss_buffer)
                                        else:
                                            pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
                                    else:
                                        pos_open.exit_price_on_entry (row.Low -stop_loss_buffer)
                                    #print("Hit 10")
                                    entry = True
                                else:
                                    #print("Hit 11")
                                    entry = False
                            else:
                                entry = False
                # When last price equaled low price, the BB lower band was breached
    #             elif (calc_new_stats(list(row.Dict_of_Closing_Prices.values()), row.Low)["new_lower_band"] \
    #             >= row.Low) & (row.High > row.Lower_Band):
    #                 for tst_price in np.linspace(row.Low,row.Close,round((row.Close-row.Low)/0.00005)+1):
    #                         if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), tst_price)["new_lower_band"] \
    #                         < tst_price:
    #                             if (row.High - tst_price) > 0.0001:
    #                                 pos_open.long_or_short_entry("long")
    #                                 pos_open.entry_price(round(tst_price,6) + 0.0001)
    #                                 if row.Prev_Candle_Low_and_Close_below_L_Band == True:
    #                                     if row.Prev_Low < row.Low:
    #                                         pos_open.exit_price_on_entry (row.Prev_Low - stop_loss_buffer)
    #                                     else:
    #                                         pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
    #                                 else:
    #                                     pos_open.exit_price_on_entry (row.Low -stop_loss_buffer)
    #                                 #print("Hit 10")
    #                                 entry = True
    #                             else:
    #                                 #print("Hit 11")
    #                                 entry = False
    #                         else:
    #                             entry = False
                else:
                    # print("Long - something else happened - {}".format(row.DateTime_UTC_column))
                    entry = False
                            #        b) tick twice away from the BB while upper BB is below high, for entry to occur
                            #        c) close below BB while upper BB is below high, for entry to occur on the close (or next candle's open)
            # The below takes care of:
            #   1) a scenario where next candle opens at/above lower BB and then closes below it
            #   2) a very rare occurence when a candle closes below BB and the next candle gaps and stays above it (not 
            #      even its low gets below BB)
            elif (row.Prev_Candle_Low_and_Close_below_L_Band == True) & (row.High > row.Lower_Band):
                #        a) tick twice while above BB for entry to occur
                if (row.Close >= row.Lower_Band) & ((row.Close - row.Low) > 0.0001):
                    pos_open.long_or_short_entry("long")
                    pos_open.entry_price(row.Low + 0.0001)            
                    if row.Prev_Candle_Low_and_Close_below_L_Band == True:
                        if row.Prev_Low < row.Low:
                            pos_open.exit_price_on_entry (row.Prev_Low - stop_loss_buffer)
                        else:
                            pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
                    else:
                        pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
                    #print("Hit 13")
                    entry = True
                else:
                    #print("Hit 14")
                    entry = False
            else:
                #print("Hit 15")
                entry = False
                #print("Hit 15 entry value: {}".format(entry))
    #     elif (row.Upper_Lower_Band_Span >= 0.0015) \
    #     & (((row.Low < row.Lower_Band) \
    #     & (row.Close > row.Lower_Band)) \
    #     | ((row.Prev_Candle_Low_and_Close_below_L_Band == True) \
    #     & (row.High > row.Lower_Band))) \
    #     & (pos_open.is_position_open == False):
    #         pos_open.long_or_short_entry("long")
    #         pos_open.entry_price(row.Lower_Band + 0.0001)
    #         if row.Prev_Candle_Low_and_Close_below_L_Band == True:
    #             if row.Prev_Low < row.Low:
    #                 pos_open.exit_price_on_entry (row.Prev_Low - stop_loss_buffer)
    #             else:
    #                 pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
    #         else:
    #             pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
    #         entry = True
        else:
            #print("Hit 16")
            entry = False
        #print("Pre-return entry value: {}".format(entry))
        return entry

    # (row.Middle_Band-((row.Middle_Band-row.Lower_Band)/2))
    # (row.Low < row.Prev_Low)
    def take_profit(row):
        # For long:
        # For testing absolute take profit number: if ((row.High > (pos_open.entry_prc + take_profit_stop_distance.tpd)) \
        if (row.High > (row.Middle_Band + take_profit_buffer)) \
        & (pos_open.long_short == "long") \
        & (pos_open.is_position_open == True):
            # Only trigger if:
                # a) candle's high truly breached middle band
            if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), row.High)["new_middle_band"] + take_profit_buffer \
                <= row.High:
                # then at what point did it breach it:
                for test_price in np.linspace(row.Low,row.High,round((row.High-row.Low)/0.00005)+1):
                    if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), test_price)["new_middle_band"] + take_profit_buffer \
                    <= test_price:
                        pos_open.take_profit_price(round(test_price,6) - 0.0001)
                        break
                take_prft = True
            else:
                take_prft = False
                # b) candle's high, itself, may not have breached middle band, but candle's last price may have at some 
                # point breached it
    #         for test_price in np.linspace(row.Low,row.Close,round((row.Close-row.Low)/0.00005)+1):
    #                     if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), test_price)["new_lower_band"] \
    #                     >= row.Low:
    #                         # Once it did, the last price needs to:
    #                         #        a) tick twice while lower BB is above low for entry to occur
    #                         if (row.Close - test_price) > 0.0001:
    #                             pos_open.long_or_short_entry("long")
    #                             pos_open.entry_price(round(test_price,6) + 0.0001)
    #                             if row.Prev_Candle_Low_and_Close_below_L_Band == True:
    #                                 if row.Prev_Low < row.Low:
    #                                     pos_open.exit_price_on_entry (row.Prev_Low - stop_loss_buffer)
    #                                 else:
    #                                     pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
    #                             else:
    #                                 pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
    #                             entry = True
    #                         else:
    #                             entry = False
    #                         break     
        # For short:
        # For testing absolute take profit number: elif ((row.Low < (pos_open.entry_prc - take_profit_stop_distance.tpd)) \
        elif (row.Low < (row.Middle_Band  - take_profit_buffer)) \
        & (pos_open.long_short == "short") \
        & (pos_open.is_position_open == True):
            # Only trigger if:
                # a) candle's low truly breached middle band - take profit buffer
            if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), row.Low)["new_middle_band"] - take_profit_buffer \
                >= row.Low:
                # then at what point did it breach it:
                for tst_price in np.linspace(row.High,row.Low,round((row.Low-row.High)/-0.00005)+1):
                    if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), tst_price)["new_middle_band"] - take_profit_buffer \
                    >= tst_price:
                        pos_open.take_profit_price(round(tst_price,6) + 0.0001)
                        break
                take_prft = True
            else:
                take_prft = False
                # b) candle's low, itself, may not have breached middle band, but candle's last price may have at some 
                # point breached it
    #         for test_price in np.linspace(row.Low,row.Close,round((row.Close-row.Low)/0.00005)+1):
    #                     if calc_new_stats(list(row.Dict_of_Closing_Prices.values()), test_price)["new_lower_band"] \
    #                     >= row.Low:
    #                         # Once it did, the last price needs to:
    #                         #        a) tick twice while lower BB is above low for entry to occur
    #                         if (row.Close - test_price) > 0.0001:
    #                             pos_open.long_or_short_entry("long")
    #                             pos_open.entry_price(round(test_price,6) + 0.0001)
    #                             if row.Prev_Candle_Low_and_Close_below_L_Band == True:
    #                                 if row.Prev_Low < row.Low:
    #                                     pos_open.exit_price_on_entry (row.Prev_Low - stop_loss_buffer)
    #                                 else:
    #                                     pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
    #                             else:
    #                                 pos_open.exit_price_on_entry (row.Low - stop_loss_buffer)
    #                             entry = True
    #                         else:
    #                             entry = False
    #                         break     
        else:
            take_prft = False
        return take_prft

    def stop_loss(row):
        # For long:
        if ((row.Low < pos_open.exit_price_on_entry_value) \
        & (pos_open.long_short == "long") \
        & (pos_open.is_position_open == True)):
            stop_lss = True
        # For short:
        elif ((row.High > pos_open.exit_price_on_entry_value) \
        & (pos_open.long_short == "short") \
        & (pos_open.is_position_open == True)):
            stop_lss = True
        else:
            stop_lss = False
        return stop_lss

    def action_func(row):

        #rtn = str(pos_open.is_position_open)
        
        # Enter position
        if enter_factor(row):
            pos_open.position_opened(opened=True)
            #pos_open.entry_price(row.High)
            #rtn = rtn + "_" + str(pos_open.is_position_open)
            return "Entered" + "_" + pos_open.long_short + "@" + str(pos_open.entry_prc)
            #print("3")
        
        # Check if stop loss criteria has been met
        elif stop_loss(row):
            pos_open.position_opened(opened=False)
            return "Stopped Out" + "_" + pos_open.long_short + "@" + str(pos_open.exit_price_on_entry_value)
        
        # Check if take profit criteria has been met
        elif take_profit(row):
            pos_open.position_opened(opened=False)
            #rtn = rtn + "_" + str(pos_open.is_position_open)
            return "Took Profit" + "_" + pos_open.long_short + "@" + str(pos_open.take_prft_price)
        # Else return "Waiting"
        else:
            #rtn = "else" 
            return "Waiting" #+ "_" + rtn

    def extract_trade_side(Split_Trade_Side_Action):
        if len(Split_Trade_Side_Action) > 1:
            return Split_Trade_Side_Action[-1]
        else:
            return "N.A"

    def extract_trade_price(Split_Trade_Prc_Action):
        if len(Split_Trade_Prc_Action) > 1:
            return Split_Trade_Prc_Action[-1]
        else:
            return np.nan

    def clean_trade_side(Clean_Trade_Side_Action):
        if len(Clean_Trade_Side_Action) > 1:
            return Clean_Trade_Side_Action[0]
        else:
            return np.nan

    def trim_trade_side(Split_Action):
        return Split_Action[0]
    #     if len(Split_Action) > 1:
    #         return Split_Action[1]
    #     else:
    #         return np.nan

    
    min_1_analyzed_df["Action"] = min_1_analyzed_df.apply(action_func, axis=1)
    
    min_1_analyzed_df["Long_Short"] = min_1_analyzed_df["Action"].str.split(pat="_").apply(extract_trade_side)
    min_1_analyzed_df["Trade_Price"] = min_1_analyzed_df["Long_Short"].str.split(pat="@").apply(extract_trade_price)
    min_1_analyzed_df["Long_Short"] = min_1_analyzed_df["Long_Short"].str.split(pat="@").apply(clean_trade_side)

    min_1_analyzed_df["Action"] = min_1_analyzed_df["Action"].str.split(pat="_").apply(trim_trade_side)

    return min_1_analyzed_df