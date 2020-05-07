import numpy as np

def calc_profit_loss(min_1_analyzed_df, trade_size):
    # class long_or_short_cls:
    #     # Long = 0
    #     # Short = 1
    #     long_or_short = 0 

    class Pos_Open:
        is_position_open = False
        @classmethod
        def position_opened(cls, opened):
            if opened == True:
                cls.is_position_open = True
            else:
                cls.is_position_open = False
            
    class Prft_lss_calc:
        entry_value = 0
        exit_value = 0
        prft_lss = 0
        # stop_price_at_entry = 0
        
        def round_down(self, n):
            return (np.floor(n*100))/100
            
        @classmethod
        def add_to_pos(cls, trade_price, long_or_short): # (cls, high_price, low_price, upper_band, lower_band, long_or_short)
            #For long:
            if long_or_short == "long":
                cls.entry_value = trade_price*trade_size # (lower_band + 0.0001)
                # cls.stop_price_at_entry = low_price
            #For short:
            else:
                cls.entry_value = trade_price*trade_size # (upper_band + 0.0001)
                # cls.stop_price_at_entry = high_price

        def prft_lss_cal_stopped_out(self, price, long_or_short):
            self.exit_value = price*trade_size
            # for long
            if long_or_short == "long":
                self.prft_lss = self.round_down(self.exit_value - Prft_lss_calc.entry_value)
            # for short
            else:
                self.prft_lss = self.round_down(Prft_lss_calc.entry_value - self.exit_value)
            return self.prft_lss
        
        def prft_lss_cal_took_prft(self, price, long_or_short):
            self.exit_value = price*trade_size
            # for long
            if long_or_short == "long":
                self.prft_lss = self.round_down(self.exit_value - Prft_lss_calc.entry_value)
            # for short
            else:
                self.prft_lss = self.round_down(Prft_lss_calc.entry_value - self.exit_value)
            return self.prft_lss
            
    def trd_price_and_prft(row):
        
        p_l_calc = Prft_lss_calc()
        
        if (row.Action == "Entered") & (Pos_Open.is_position_open == False):
            p_l_calc.add_to_pos(round(float(row.Trade_Price),6), row.Long_Short) # (row.High, row.Low, row.Upper_Band, row.Lower_Band, row.Long_Short)
            Pos_Open.position_opened(True)
            return [p_l_calc.entry_value,0]
        
        elif ((row.Action == "Took Profit") | (row.Action == "Stopped Out")) & (Pos_Open.is_position_open == True):
    #         Prft_lss_calc.close_pos(row.High)
            Pos_Open.position_opened(False)
            # if Took Profit
            if row.Action == "Took Profit":
                # for long:
                if row.Long_Short == "long":
                    pric = round(float(row.Trade_Price),6) # row.Middle_Band + 0.0009
                # for short:
                else:
                    pric = round(float(row.Trade_Price),6) # row.Middle_Band - 0.0009
                #list_to_return = 
                return [pric, p_l_calc.prft_lss_cal_took_prft(pric, row.Long_Short)]
            # if Stopped Out
            else:
                # for long:
                if row.Long_Short == "long":
                    stopped_pric = round(float(row.Trade_Price),6) # p_l_calc.stop_price_at_entry - 0.0006
                # for short:
                else:
                    stopped_pric = round(float(row.Trade_Price),6) # p_l_calc.stop_price_at_entry + 0.0006
                #list_to_return_1 = 
                return [stopped_pric, p_l_calc.prft_lss_cal_stopped_out(stopped_pric, row.Long_Short)]
        
        else:
            return[np.NaN,0]

    min_1_analyzed_df["Trade_Prc_and_Prft"] = min_1_analyzed_df.apply(trd_price_and_prft, axis=1)

    # def extract_prft_lss(Trade_Prc_and_Prft):
    #     if len(Split_Action) > 1:
    #         return Split_Action[-1]
    #     else:
    #         return np.nan

    min_1_analyzed_df["Trade_Prft_Lss"] = min_1_analyzed_df["Trade_Prc_and_Prft"].apply(lambda x: x[1])
    min_1_analyzed_df_dist_analysis = min_1_analyzed_df[min_1_analyzed_df["Action"] != "Waiting"].copy()

    return min_1_analyzed_df, min_1_analyzed_df_dist_analysis