import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

vgsales = pd.read_csv("./data/vgsales.csv", index_col=0)

#print(vgsales.info())

vgsales = vgsales.dropna(how='any')
print(vgsales.info())

"""
#各电子游戏平台销售额情况

plt.figure(figsize = (20,10))
sale_for_Platform = vgsales.groupby('Platform', as_index = False).sum().sort_values(by = 'Global_Sales', ascending = False)
sns.barplot(x = 'Global_Sales',
            y = 'Platform', data = sale_for_Platform)
plt.title('Global Sales in Millions $ by Platform')
plt.show()
"""
"""
#各电子游戏销售额情况

plt.figure(figsize = (20,10))
sale_for_game = vgsales.groupby('Name', as_index = False).sum().sort_values(by = 'Global_Sales', ascending = False).head(20)
sns.barplot(x = 'Global_Sales',
            y = 'Name', data = sale_for_game)
plt.title('Global Sales in Millions $ by Game')
plt.show()
"""
"""
#各电子游戏类型销售额情况

plt.figure(figsize = (20,10))
sale_for_Genre = vgsales.groupby('Genre', as_index = False).sum().sort_values(by = 'Global_Sales', ascending = False).head(20)
sns.barplot(x = 'Global_Sales', y = 'Genre', data = sale_for_Genre)
plt.title('Global Sales in Millions $ by Genre')
plt.show()
"""
"""
#各电子游戏发行商销售额情况

plt.figure(figsize = (20,10))
sale_for_Publisher = vgsales.groupby('Publisher', as_index = False).sum().sort_values(by = 'Global_Sales', ascending = False).head(20)
sns.barplot(x = 'Global_Sales', y = 'Publisher', data = sale_for_Publisher)
plt.title('Global Sales in Millions $ by Publisher')
plt.show()
"""


"""
#每年电子游戏销售额情况

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8

fig = plt.figure(figsize = (25,10))
ax = fig.add_axes([left, bottom, width, height])
sale_for_years = vgsales.groupby('Year', as_index = False).sum()

sns.lineplot(x = 'Year', y = 'NA_Sales', data = sale_for_years, color = 'blue', label = 'North_America_Sales', ax = ax)
sns.lineplot(x = 'Year', y = 'JP_Sales', data = sale_for_years, color = 'yellow', label = 'Japan_Sales', ax = ax)
sns.lineplot(x = 'Year', y = 'EU_Sales', data = sale_for_years, color = 'green', label = 'European_Union_Sales', ax = ax)
sns.lineplot(x = 'Year', y = 'Other_Sales', data = sale_for_years, color = 'pink', label = 'Other_Sales', ax = ax)
sns.lineplot(x = 'Year', y = 'Global_Sales', data = sale_for_years, color = 'red', label = 'Global_Sales', ax = ax)


ax.set_xlabel('Year', fontsize = 12)
ax.set_ylabel('Sales', fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_title('Sales per Year in Millions $', fontsize = 25)
plt.legend(fontsize = 20)
plt.show()

"""
"""
#通过北美洲的销售额来预测全球的销售额

vgsales = vgsales[vgsales['Year'] <= 2016.0]
sale_for_years = vgsales.groupby('Year', as_index = False).sum()
print(sale_for_years.info())
x = sale_for_years['NA_Sales'].values.reshape(-1, 1)
y = sale_for_years['Global_Sales']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1)

LR = LinearRegression()
LR.fit(X_train, y_train)
LR_score_train = LR.score(X_train, y_train)
print('LR Training score: ',LR_score_train)
LR_score_test = LR.score(X_test, y_test)
print('LR Testing score: ',LR_score_test)

svr = SVR(kernel="poly")
svr.fit(X_train, y_train)
svr_score_train = svr.score(X_train, y_train)
print('svr Training score: ', svr_score_train)
svr_score_test = svr.score(X_test, y_test)
print('svr Testing score: ', svr_score_test)
"""

#通过年份来预测某几年的销售额

vgsales = vgsales[vgsales['Year'] <= 2016.0]
sale_for_years = vgsales.groupby('Year', as_index = False).sum()
print(sale_for_years.info())
x = sale_for_years['Year'].values.reshape(-1, 1)
y = sale_for_years['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state = 1)

LR = LinearRegression()
LR.fit(X_train, y_train)
LR_score_train = LR.score(X_train, y_train)
print('LR Training score: ',LR_score_train)
LR_score_test = LR.score(X_test, y_test)
print('LR Testing score: ',LR_score_test)

