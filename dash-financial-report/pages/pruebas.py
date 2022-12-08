"""import matplotlib.pyplot as plt
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot([0,1,2], [10,20,3])
fig.savefig('Images.png')   # save the figure to file
plt.close(fig)  
"""
"""import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table # EDIT: see deprecation warnings below

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

table(ax, df)  # where df is your data frame

plt.savefig('mytable.png')

"""
import pandas as pd
import numpy as np
import dataframe_image as dfi
import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


def dataframe_toimage(df, name):
    df_styled = df.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled, name+".png")


df_prueba = pd.read_csv(DATA_PATH.joinpath("melb_data.csv"))
df_types = pd.DataFrame(df_prueba.dtypes)
dataframe_toimage(df_types, "imagen1")