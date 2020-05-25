#User defined library for visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objects as go
import seaborn as sns


#Create charts
class create_chart:

    def __init__(self,df, sample = False):

        if sample:
            self.df = df.sample(nrow = 10000)
        else:
            self.df = df


    def fill_rate(self, cols = None):
        """
        Calculates fill rate for each column

        Keyword Arguments:
            cols {list} -- Columns for which fill rate is to be calculated (default: {None})
        """
        if cols:
            col_list = cols
        else:
            col_list = self.df.columns
        
        fill_rate = [self.df[col].count()/len(self.df)*100 for col in col_list]
        iplot({'data': [
                        go.Bar(x= col_list,
                            y=fill_rate)
                        ],
                'layout': go.Layout(title = 'Fill Rate plot', xaxis=go.layout.XAxis(title='Columns'), yaxis=go.layout.YAxis(title='Fill Rate'))
                }, show_link=False) 


    def correlation(self,cols = None):
        """[summary]

        Keyword Arguments:
            cols {list} -- Columns for which metric is to be calculated (default: {None})
        """
        if cols:
            col_list = cols
        else:
            col_list = df.columns
        

        plt.figure()
        ax = sns.heatmap(self.df[col_list].corr(), annot = False, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    def missing_val_loc(self,cols = None):
        """Shows map of missing values (NaN)

        Arguments:
            cols {list} -- Columns for which metric is to be calculated (default: {None})
        """
        if cols:
            col_list = cols
        else:
            col_list = self.df.columns
        
        #Understasnd where the missing values occur
        plt.figure()
        ax = sns.heatmap(self.df.iloc[np.unique(np.where(self.df.isnull())[0]),:].isnull(),cbar=False)
        plt.title("Missing value location distribution")
        plt.show()


    def time_series(self,date_col,value_col):
        """Plots a customizable time series plot

        Arguments:
            date_col {list of string} 
            value_col {list of strings} 
        """

        fig = go.Figure([go.Scatter(x=self.df[date_col], y=self.df[value_col])])
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()


    def create_dist_plot(self,val_cols, group_col = None):
        """Creates univariate distribution plots with an interactive grouping col

        Arguments:
            val_cols {List of strings} -- Distributions will be plotted for these

        Keyword Arguments:
            group_col {List of string} -- Grouping will be done on this column (default: {None})
        """
        if group_col:
            for col in val_cols:
                iplot({'data': [go.Histogram(x=self.df[self.df[col] == group]) for group in self.df[group_col].unique()],
                    'layout': go.Layout(title = 'Dist plot for ' + col)
                    }, show_link=False) 

        else:
            for col in val_cols:
                iplot({'data': [go.Histogram(x=self.df[col])],
                    'layout': go.Layout(title = 'Dist plot for ' + col)
                    }, show_link=False) 




    def show_basic_stats(self):
        """
        Create an interactive plot which allows us to select the variable 
        and shows the following in a visual format.
        1. % of missing values - Heatmap based on the columns selected
        2. # of unique values - Based on columns selected
        3. Range of values - Based on columns selected
        """

        
        

if __name__ == '__main__':

    df = pd.read_csv(r'/Users/abhisheknamballa/Downloads/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
    print(df.info())

    a = create_chart(df)
    a.fill_rate()
    a.correlation()
    a.missing_val_loc()





        
        













