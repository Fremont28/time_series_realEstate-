#6/23/18
#forecasting real estate prices in Spain and Portugal 

#import real estate datasets 
price_change=pd.read_csv('Price changes.csv',delimiter=";")
area=pd.read_csv('Built area, used area.csv',delimiter=";")
details=pd.read_csv('Details.csv',encoding='latin-1')

#merge real estate data on listing id
merge1=pd.merge(price_change,area,on=['listing_id'])
merge2=pd.merge(merge1,details,on=['listing_id'])

#counting frequent words in the real estate descriptions 
for word in df:
    print(df.count(word),word)

def word_count(str):
    counts=dict()
    words=str.split()

    for word in words:
        if word in counts:
            counts[word] +=1
        else:
            counts[word]=1
    return counts 

word_count(df1)

#search for real estate keywords 
garage=merge2.apply(lambda row: row.astype(str).str.contains('Garage','garage').any(), axis=1)
air_cond=merge2.apply(lambda row: row.astype(str).str.contains('Conditioning','conditioning').any(),axis=1)
tennis=merge2.apply(lambda row: row.astype(str).str.contains('Tennis').any(),axis=1)
ocean=merge2.apply(lambda row: row.astype(str).str.contains('Ocean','ocean').any(), axis=1)
apt=merge2.apply(lambda row: row.astype(str).str.contains('apartment','Apartment').any(), axis=1)
swim=merge2.apply(lambda row: row.astype(str).str.contains('Swimming','swimming').any(), axis=1)
jazz=merge2.apply(lambda row: row.astype(str).str.contains('Jacuzzi','Sauna').any(), axis=1)

#combine dataframes
garage1=pd.DataFrame(data=garage,columns=["garage"])
air_cond1=pd.DataFrame(data=garage,columns=["air_cond"])
tennis1=pd.DataFrame(data=tennis,columns=["tennis"])
ocean1=pd.DataFrame(data=ocean,columns=["ocean"])
swim1=pd.DataFrame(data=swim,columns=["swim"])
jazz1=pd.DataFrame(data=jazz,columns=["jazz"])
apt1=pd.DataFrame(data=apt,columns=["apt"])

ocean1.to_csv("oceans.csv")
tennis1.to_csv("tennis1.csv") 

#cbind dataframes and merge 
desc=pd.DataFrame(np.array(df))
desc.columns=['description']

#join dataframes 
df_test=pd.concat([desc,merge2])
#subset
df_test1=df_test[["listing_id","change_date","built_area","description","new_price",
"old_price","used_area","built_area"]]

#drop nan from the dataset 
df_test2=df_test1.dropna(how='all')
df_test2.shape 

###6/25/18********************************
#time series analysis 
mallorca1=pd.read_csv("mallorca.csv")
mallorca1.info() 

#convert to datetime 
mallorca1['new_date']=pd.to_datetime(mallorca1['change_date']) 
#create a time series object 
mallorca1.index=mallorca1['new_date']
del mallorca1['new_date']
mallorca1['2016']
mallorca1['2017']

#observations entre 2017-03-31 to 2017-06-30
mallorca1['2017-01-01':'2017-04-30'].describe() #3.306449e+06 avg. price (374)
mallorca1['2017-05-30':'2017-07-31'].describe() #3.394143e+06 avg. price (189)

#count obseravtions per timestamp 
mallorca1.groupby(level=0).count()
#listings per day
mallorca1.resample(mallorca1['new_price']).mean() #mean price per day
mallorca1['new_price'].mean() 
test_func1=mallorca1['built_area'].apply(lambda x:x*0.10)

#scatterplot (old price vs. new price)
sns.regplot(x=mallorca1['old_price'],y=mallorca1['new_price'])
plt.xlabel("Old Price (Euros)")
plt.ylabel("New Price (Euros)")
plt.show() 

#histogram (new price for real estate) 
sns.distplot(mallorca1['new_price'],bins=20)
plt.xlabel('New Price (Euros)')
plt.show()

#trends and time series seasonality 
#i. 12-month rolling avg (for each point take the average of the adjacent points)
new_price=mallorca2[['mean_price']]

#seasonality (spike summer, low-winter) rolling moving average 
new_price.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Date',fontsize=12)
plt.ylabel('New Price (Euros)')
plt.show() 

#first order differencing 
new_price.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Date',fontsize=10)
plt.show() 

#autocorrelation plot 
pd.plotting.autocorrelation_plot(new_price)
plt.show() 

#ARIMA time series forecasting ****************************

#train and test dataset split 
import pyramid
from pyramid.arima import auto_arima

stepwise_model = auto_arima(new_price, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

stepwise_model.aic() #lowest AIC is 12054 and seasonal_order=(1, 1, 1, 12) 

#train and test sets (for time series real estate forecasting)
train=new_price.loc['2016-01-18':'2017-03-31']
test=new_price.loc['2017-04-01':]

#train the model 
stepwise_model.fit(train)

#evaluate the model
future_forecast=stepwise_model.predict(n_periods=86)
future_forecast

future_forecast = pd.DataFrame(future_forecast,index = test.index)
future_forecast.columns=['Prediction']

#combine dataframes
test_future=pd.concat([test,future_forecast],axis=1)
test_future.head(4)
test_future.info()

#root-mean squared error 
rms=np.sqrt(mean_squared_error(test_future.mean_price,test_future.Prediction))
rms #3871425.9237282816

#compare predicted real estate price vs. actual real estate prices 
test_future.plot()
plt.xlabel("Date")
pyplot.show() 

#forecast into the future 
f=future_forecast[34:]

future_forecast1=stepwise_model.predict(n_periods=451) #8/10/2017-8/10/2018 
future_forecast1
future_forecast1_sub=future_forecast1[108:440] #starting from 9/1/17 to 8/1/18
future_forecast1_sub

sep=future_forecast1_sub[290:321].mean() 



