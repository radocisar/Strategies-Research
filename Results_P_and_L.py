import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def results_P_and_L(min_1_analyzed_df, trade_size, train_start_date_str, train_end_date_str):
    # Gross P&L:
    gross_absolute_profit_loss = min_1_analyzed_df.loc[min_1_analyzed_df["Trade_Prft_Lss"] != 0.0,"Trade_Prft_Lss"].sum()
    gross_percent_profit_loss = (gross_absolute_profit_loss/trade_size)*100

    # Commission:
    commission = min_1_analyzed_df.loc[min_1_analyzed_df["Trade_Prft_Lss"] != 0.0,"Trade_Prft_Lss"].count()*2

    #Slippage:
    slippage = min_1_analyzed_df.loc[min_1_analyzed_df["Trade_Prft_Lss"] != 0.0,"Trade_Prft_Lss"].count()*(trade_size/10000)

    # Net P&L:
    net_absolute_profit_loss = gross_absolute_profit_loss - commission - slippage
    net_percent_profit_loss = (net_absolute_profit_loss/trade_size)*100

    # Number of trades:
    num_of_trades = min_1_analyzed_df.loc[min_1_analyzed_df["Trade_Prft_Lss"] != 0.0,"Trade_Prft_Lss"].count()

    # Print table
    profit_vs_loss = min_1_analyzed_df.loc[min_1_analyzed_df["Trade_Prft_Lss"] != 0.0,"Trade_Prft_Lss"] > 0
    profit_vs_loss.replace({True:"Profit",False:"Loss"}, inplace=True)
    s = min_1_analyzed_df.loc[min_1_analyzed_df["Trade_Prft_Lss"] != 0.0,"Trade_Prft_Lss"]
    pft_df = pd.concat([profit_vs_loss,s], axis=1)
    pft_df.columns = ["Trade_Prft_Lss_Name","Trade_Prft_Lss_Value"]
    P_N_L_Stats = pft_df.groupby(["Trade_Prft_Lss_Name"],sort=False).describe()
    P_N_L_Stats.rename(columns={"Trade_Prft_Lss_Value":""},level=0,inplace=True)
    P_N_L_Stats.index.name = None
    # P_N_L_Stats = pd.DataFrame({"Count":profit_vs_loss.value_counts(ascending=True).values,"Average":[s[s>0].mean(),
    # s[s<0].mean()]},index=profit_vs_loss.value_counts().index)

    print("# of Trades: {}".format(num_of_trades))
    print(P_N_L_Stats)
    print("Gross P&L: {} ({}%)".format(gross_absolute_profit_loss,gross_percent_profit_loss))
    print("Commission: {}".format(commission))
    print("Slippage: {}".format(slippage))
    print("Net P&L: {} ({}%)".format(net_absolute_profit_loss,net_percent_profit_loss))

    # Charting P&L:
    pnl_chart_df = min_1_analyzed_df["Trade_Prft_Lss"].replace(0,np.NaN)
    pnl_chart_df_cumsum = pnl_chart_df[pnl_chart_df < 1000000000].cumsum()
    pnl_chart_df_less_comm_and_slip = pnl_chart_df - 2 - trade_size/10000
    pnl_chart_df_less_comm_and_slip_cumsum = pnl_chart_df_less_comm_and_slip[pnl_chart_df_less_comm_and_slip < 1000000000].cumsum()

    fig1 = plt.figure(figsize=(15,15))
    ax4 = fig1.subplot(2,1,1)
    ax4.set_title("Net P&L", fontsize=20)
    pnl_chart_df_less_comm_and_slip_cumsum.plot(ax=ax4, linestyle="-", marker="o")
    ax5 = fig1.subplot(2,1,2)
    ax5.set_title("Gross P&L", fontsize=20)
    pnl_chart_df_cumsum.plot(ax=ax5, linestyle="-", marker="o")
    # plt.show(block=False)
    plt.savefig(f"./Charts/P_and_L_{train_start_date_str}_{train_end_date_str}.png"")
    # ax2 = plt.subplot(3,1,1)
    # min_1_analyzed_df["SMA10"].head(100).plot(ax=ax1, color="r")
    # ax3 = plt.subplot(3,1,1)
    # min_1_analyzed_df["SMA20"].head(100).plot(ax=ax1, color="y")

    return num_of_trades, P_N_L_Stats, gross_absolute_profit_loss, gross_percent_profit_loss, \
        commission, slippage, net_absolute_profit_loss, net_percent_profit_loss