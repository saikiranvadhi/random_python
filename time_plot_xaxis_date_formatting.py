import matplotlib.pyplot as plt
from matplotlib import dates, ticker
import seaborn as sns
sns.set_style("darkgrid")

# % matplotlib inline

### Start a figure
fig = plt.figure(figsize=(12, 6))
axes = fig.add_axes([0, 0, 1, 1])
# Make sure the 'date_variable' is of format datetime
# client_trading['date_variable'] = pd.to_datetime(client_trading['date_variable'])
sns.lineplot(x='date_variable', y='sales', data=df, estimator='sum', ax=axes)
axes.set_title("Sales over Time")

# axes.xaxis.set_major_locator(MonthLocator())
# axes.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
# axes.xaxis.set_major_formatter(NullFormatter())
# axes.xaxis.set_minor_formatter(DateFormatter('%b'))
# plt.show()

# Define the date format
date_form = dates.DateFormatter("%y-%b")
axes.xaxis.set_major_formatter(date_form)
axes.xaxis.set_minor_formatter(ticker.NullFormatter())

# Ensure a major tick for each month
axes.xaxis.set_major_locator(dates.MonthLocator())
axes.xaxis.set_minor_locator(dates.MonthLocator(bymonthday=15))

plt.show()
