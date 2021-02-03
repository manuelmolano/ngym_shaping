# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

path = '/Users/leyre/Dropbox/mice_data'
os.getcwd()
os.chdir(path)
df_trials = pd.read_csv(path + '/global_trials.csv', sep=';')
df_params = pd.read_csv(path + '/global_params.csv', sep=';')
df_raw = pd.merge(df_params, df_trials, on=['session', 'subject_name'])

# GRAPHIC 1
# PERFORMANCE/ACCURACY VS ALL SESSIONS
# function to plot bar chart of performance/accuracy over all the sessions


def bar_plot(dataframe, x_var, y_var, col):
    f, ax = plt.subplots()
    ax.bar(x=dataframe.x_var, height=dataframe.y_var, color=col)
    ax.set(title='Plot of accuracy Vs session')
    plt.xlabel('x_var')
    plt.ylabel('y_var')
    plt.show()


# plot accuracy VS session
bar_plot(df_params, session, accuracy, 'purple')


# plot Performance VS session without definition
s, ay = plt.subplots()
ay.bar(x=df_params.session, height=df_params.performance, color='green')
ay.set(title='Plot of performance Vs session')
plt.xlabel('Sessions')
plt.ylabel('Performance')
plt.show()

# RANDOM PLOTS
new = df_params[['session', 'accuracy']]
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(new, hue='session', palette='Spectral')
# blue more sessions, red less sessions
# even for a lot or a few of sessions, the accuracy seems to be around 0.5

# RELATIONSHIP BETWEEN ACCURACY/PERFORMANCE AND SESSIONS
g = sns.lmplot(x='session', y='accuracy', data=df_params, palette='Set1')
# lineal relationship: as sessions increase, accuracy decreases
t = sns.lmplot(x='session', y='performance', data=df_params, palette='Set1')
# lineal relationship: as sessions increase, performance decreases


# GRAPHIC 2 questions
# Plot the accuracy for each single session
# Obtain one plot for session
graphic = []
counter = 0
for column in df_params.columns:
    if column == session:
        for i in column:
            if i == counter+1:
                graphic.append(df_params.accuracy[i])
        counter += 1
print(graphic)

# i don't know where is the problem
# here we can see that one column is called session
for column in df_params.columns:
    print(column)

# GRAPHIC 2 done
# Accuracy in each session


def acc_session(num_session):
    # find all the trials done in session1
    ind = df_params[df_params.session == num_session]
    # Obtain the accuracy of the trials dond in session 1
    acc = ind.iloc[:, [0]]  # 0 because accuracy is the first column
    plt.xlabel('Trials')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per session' + ' ' + str(num_session))
    plt.plot(acc)
    return acc


# GRAPHIC 3
# Duration for each phase
# Duration is the column time
dur = df_params.loc[:, 'time']
print(dur)

# it cannot be executed due to times are objects
# and not integers, so they cannot sum


# function to get seconds from time
def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


print(get_sec('1:23:45'))


def time_session(num_session):
    # find all the trials done in session1
    ind = df_params[df_params.session == num_session]
    # Obtain the time of the trials done in session 1
    time = ind.iloc[:, [39]]
    time = time.to_numpy()
    sumtimes = 0
    for i in time:
        t = get_sec(i)
        sumtimes += t
    return sumtimes
    plt.xlabel('Session' + ' ' + str(num_session))
    plt.ylabel('Duration (s)')
    plt.title('Duration per session' + ' ' + str(num_session))
    plt.plot(sumtimes)
    return time


time_session(1)

# trying to fix it
ind = df_params[df_params.session == 1]
time = ind.iloc[:, [39]]
print(time)
time = time.to_numpy()
print(time)
time = np.array_split(time, 10)

    



















