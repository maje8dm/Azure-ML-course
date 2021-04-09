#!/usr/bin/env python
# coding: utf-8

# # Flights Data Exploration Challenge
# 
# In this challge, you'll explore a real-world dataset containing flights data from the US Department of Transportation.
# 
# Let's start by loading and viewing the data.

# In[1]:


import pandas as pd

df_flights = pd.read_csv('data/flights.csv')
df_flights.head()


# The dataset contains observations of US domestic flights in 2013, and consists of the following fields:
# 
# - **Year**: The year of the flight (all records are from 2013)
# - **Month**: The month of the flight
# - **DayofMonth**: The day of the month on which the flight departed
# - **DayOfWeek**: The day of the week on which the flight departed - from 1 (Monday) to 7 (Sunday)
# - **Carrier**: The two-letter abbreviation for the airline.
# - **OriginAirportID**: A unique numeric identifier for the departure aiport
# - **OriginAirportName**: The full name of the departure airport
# - **OriginCity**: The departure airport city
# - **OriginState**: The departure airport state
# - **DestAirportID**: A unique numeric identifier for the destination aiport
# - **DestAirportName**: The full name of the destination airport
# - **DestCity**: The destination airport city
# - **DestState**: The destination airport state
# - **CRSDepTime**: The scheduled departure time
# - **DepDelay**: The number of minutes departure was delayed (flight that left ahead of schedule have a negative value)
# - **DelDelay15**: A binary indicator that departure was delayed by more than 15 minutes (and therefore considered "late")
# - **CRSArrTime**: The scheduled arrival time
# - **ArrDelay**: The number of minutes arrival was delayed (flight that arrived ahead of schedule have a negative value)
# - **ArrDelay15**: A binary indicator that arrival was delayed by more than 15 minutes (and therefore considered "late")
# - **Cancelled**: A binary indicator that the flight was cancelled
# 
# Your challenge is to explore the flight data to analyze possible factors that affect delays in departure or arrival of a flight.
# 
# 1. Start by cleaning the data.
#     - Identify any null or missing data, and impute appropriate replacement values.
#     - Identify and eliminate any outliers in the **DepDelay** and **ArrDelay** columns.
# 2. Explore the cleaned data.
#     - View summary statistics for the numeric fields in the dataset.
#     - Determine the distribution of the **DepDelay** and **ArrDelay** columns.
#     - Use statistics, aggregate functions, and visualizations to answer the following questions:
#         - *What are the average (mean) departure and arrival delays?*
#         - *How do the carriers compare in terms of arrival delay performance?*
#         - *Are some days of the week more prone to arrival days than others?*
#         - *Which departure airport has the highest average departure delay?*
#         - *Do **late** departures tend to result in longer arrival delays than on-time departures?*
#         - *Which route (from origin airport to destination airport) has the most **late** arrivals?*
#         - *Which route has the highest average arrival delay?*
#         
# Add markdown and code cells as requried to create your solution.
# 
# > **Note**: There is no single "correct" solution. A sample solution is provided in [01 - Flights Challenge.ipynb](01%20-%20Flights%20Solution.ipynb).

# ### Clean missing values
# 
# Find how many null values there are for each column.

# In[2]:


df_flights.isnull().sum()


# Hmm, looks like there are some null "late departure" indicators. Departures are considered late if the delay is 15 minutes or more, so let's see the delays for the ones with a null late indicator:

# In[3]:


df_flights[df_flights.isnull().any(axis=1)][['DepDelay','DepDel15']]


# We can't see them all in this display, but it looks like they may all have delay of 0. Let's check by looking at the summary statistics for these records:

# In[4]:


df_flights[df_flights.isnull().any(axis=1)].DepDelay.describe()


# The min, max, and mean are all 0; so it seems that none of these were actually *late* departures. Let's replace the missing **DepDel15** indicator with a 0 and confirm there are no more missing values.

# In[5]:


df_flights.DepDel15 = df_flights.DepDel15.fillna(0)
df_flights.isnull().sum()


# ### Clean outliers
# 
# View the distribution and summary statistics for the **DepDelay** and **ArrDelay** columns.

# In[6]:


# Function to show summary stats and distribution for a column
def show_distribution(var_data):
    from matplotlib import pyplot as plt

    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print(var_data.name,'\nMinimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle(var_data.name)

    # Show the figure
    fig.show()

# Call the function for each delay field
delayFields = ['DepDelay','ArrDelay']
for col in delayFields:
    show_distribution(df_flights[col])


# There are a outliers at the lower and upper ends of both variables - particularly at the upper end.
# 
# Let's trim the data so that we include only rows where the values for these fields are within the 1st and 90th percentile.

# In[7]:


# Trim outliers for ArrDelay based on 1% and 90% percentiles
ArrDelay_01pcntile = df_flights.ArrDelay.quantile(0.01)
ArrDelay_90pcntile = df_flights.ArrDelay.quantile(0.90)
df_flights = df_flights[df_flights.ArrDelay < ArrDelay_90pcntile]
df_flights = df_flights[df_flights.ArrDelay > ArrDelay_01pcntile]

# Trim outliers for DepDelay based on 1% and 90% percentiles
DepDelay_01pcntile = df_flights.DepDelay.quantile(0.01)
DepDelay_90pcntile = df_flights.DepDelay.quantile(0.90)
df_flights = df_flights[df_flights.DepDelay < DepDelay_90pcntile]
df_flights = df_flights[df_flights.DepDelay > DepDelay_01pcntile]

# View the revised distributions
for col in delayFields:
    show_distribution(df_flights[col])


# That looks a bit better.
# 
# ### Explore the data
# 
# Let's start with an overall view of the summary statistics for the numeric columns.

# In[8]:


df_flights.describe()


# #### What are the mean departure and arrival delays?

# In[9]:


df_flights[delayFields].mean()


# #### How do the carriers compare in terms of arrival delay performance?

# In[10]:


for col in delayFields:
    df_flights.boxplot(column=col, by='Carrier', figsize=(8,8))


# #### Are some days of the week more prone to arrival days than others?

# In[11]:


for col in delayFields:
    df_flights.boxplot(column=col, by='DayOfWeek', figsize=(8,8))


# #### Which departure airport has the highest average departure delay?

# In[12]:


departure_airport_group = df_flights.groupby(df_flights.OriginAirportName)

mean_departure_delays = pd.DataFrame(departure_airport_group['DepDelay'].mean()).sort_values('DepDelay', ascending=False)
mean_departure_delays.plot(kind = "bar", figsize=(12,12))
mean_departure_delays


# #### Do *late* departures tend to result in longer arrival delays than on-time departures?

# In[13]:


df_flights.boxplot(column='ArrDelay', by='DepDel15', figsize=(12,12))


# ### Which route (from origin airport to destination airport) has the most **late** arrivals?

# In[14]:


# Add a routes column
routes  = pd.Series(df_flights['OriginAirportName'] + ' > ' + df_flights['DestAirportName'])
df_flights = pd.concat([df_flights, routes.rename("Route")], axis=1)

# Group by routes
route_group = df_flights.groupby(df_flights.Route)
pd.DataFrame(route_group['ArrDel15'].sum()).sort_values('ArrDel15', ascending=False)


# #### Which route has the highest average arrival delay?

# In[15]:


pd.DataFrame(route_group['ArrDelay'].mean()).sort_values('ArrDelay', ascending=False)

