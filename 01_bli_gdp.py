#%%
# 01_bli_gdp.py

import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import sklearn.linear_model

def prepare_country_stats(oecd_bli, gdp_per_capita):
		oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
		oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
		gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
		gdp_per_capita.set_index("Country", inplace=True)
		full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
		full_country_stats.sort_values(by="GDP per capita", inplace=True)
		remove_indices = [0, 1, 6, 8, 33, 34, 35]
		keep_indices = list(set(range(36)) - set(remove_indices))
		return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


bli = pd.read_csv("/home/yclian/.windows/Temp/bli.csv", thousands = ",")
gdp = pd.read_csv("/home/yclian/.windows/Temp/gdp.csv", thousands = ",", delimiter = '\t', encoding = "latin1", na_values = "n/a")

stats = prepare_country_stats(bli, gdp)
X = np.c_[stats["GDP per capita"]]
y = np.c_[stats["Life satisfaction"]]
stats.plot(kind = "scatter", x = "GDP per capita", y = "Life satisfaction")
plt.show()

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
X_new = [[9556]] # Malaysia GDP per capita
print(model.predict(X_new))

#%%
