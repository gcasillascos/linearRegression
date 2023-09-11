import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


world_gdp = pd.read_csv('./data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5728786.csv')

print(world_gdp )
world_gdp.drop('Indicator Name',axis=1,inplace=True)
world_gdp.drop('Indicator Code',axis=1,inplace=True)

mex_gdp = world_gdp[world_gdp['Country Code']=='MEX']

mex_gdp.drop('Country Name',axis=1,inplace=True)
mex_gdp.drop('Country Code',axis=1,inplace=True)
mex_gdp.drop('Unnamed: 67',axis=1,inplace=True)
mex_gdp_t = mex_gdp.T


print(mex_gdp_t)

print(mex_gdp_t.info())
print(mex_gdp_t.shape)

mex_gdp_t = mex_gdp_t.assign(row_number=range(len(mex_gdp_t)))

mex_gdp_t['year'] = pd.to_numeric(mex_gdp_t.index, errors='ignore')


mex_gdp_t.rename(columns={154:'income','years':'year'},inplace=True)
mex_gdp_t.set_index(['row_number'],inplace=True)


print(mex_gdp_t)
print(mex_gdp_t.info())


# mex_gdp_t.plot()

 
sns.regplot(x='year', y='income', data=mex_gdp_t)
plt.savefig('asdfa.png')

lri = LinearRegression()
lri.fit(mex_gdp_t[['year']].values, mex_gdp_t.income)

predicted_gdp =lri.predict([[2023]])
print(predicted_gdp)
