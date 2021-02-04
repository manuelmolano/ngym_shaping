# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

path = '/Users/leyre/Dropbox/mice_data'
path = '/home/manuel/mice_data'

# GRAPHIC 1
# PERFORMANCE/ACCURACY VS ALL SESSIONS
# function to plot bar chart of performance/accuracy over all the sessions


def plot_xvar_VS_yvar(df, x_var, y_var, col, xlabel='x_var', ylabel='y_var'):
    """
    plot x_var VS y_var.

    Parameters
    ----------
    df : dataframe
        dataframe containing data.
    x_var : str
        first variable.
    y_var : str
        second variable.
    col : str
        color.

    Returns
    -------
    None.

    """
    f, ax = plt.subplots()
    ax.plot(df[x_var], df[y_var], color=col)
    ax.set(title='Plot of accuracy VS session')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


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


def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def num_sessions_per_stage(df, subj):
    # find all the trials done in session1
    stg = df.loc[df['subject_name'] == subj, 'stage_number'].values
    perf = df.loc[df['subject_name'] == subj, 'performance'].values
    both = df.loc[df['subject_name'] == subj, ['performance', 'stage_number',
                                               'substage']].values

    print(both)
    return stg, perf

    # ind = df[df_params.session == num_session]
    # # Obtain the time of the trials done in session 1
    # time = ind.iloc[:, [39]]
    # time = time.to_numpy()
    # sumtimes = 0
    # for i in time:
    #     t = get_sec(i)
    #     sumtimes += t
    # plt.xlabel('Session' + ' ' + str(num_session))
    # plt.ylabel('Duration (s)')
    # plt.title('Duration per session' + ' ' + str(num_session))
    # plt.plot(sumtimes)
    # return time, sumtimes


if __name__ == '__main__':
    df_params = pd.read_csv(path + '/global_params.csv', sep=';')
    # num_sessions_per_stage(df=df_params, subj='C28')

    # plot accuracy VS session
    #  TODO: filter for each animal 'subject_name'
    X_df = df_params.loc[df_params['subject_name'] == 'C28', 'stage_number']
    print(X_df)

    # worse way to list the subjects
    # subjects = ['C30', 'C23', 'C28', 'C32', 'C31', 'C29', 'C25',
    #             'C26', 'C24', 'C27']

    # best way to do it
    subject_mat = df_params.subject_name
    subj_unq = np.unique(subject_mat)
    # I get the subjects and all the indexes in which they appear
    # However, the presentation is not clear
    num_ses_per_stg = defaultdict(list)
    for sbj in subj_unq:
        print('------')
        print(sbj)
        num_sessions_per_stage(df_params, sbj)

    # # for loops way, but I don't know how to use i+1
    # stages_subjects = []
    # for i in subjects:
    #     if i == i+1:
    #         continue
    #     else:
    #         x = num_sessions_per_stage(df_params, i)
    #         stages_subjects.append(x)
    # print(stages_subjects)

    # plot_xvar_VS_yvar(df=df_params, x_var='session', y_var='accuracy',
    #                   xlabel='Session', ylabel='Accuracy', col='purple')
# df_trials = pd.read_csv(path + '/global_trials.csv', sep=';')
# df_raw = pd.merge(df_params, df_trials, on=['session', 'subject_name'])
# # RANDOM PLOTS
# new = df_params[['session', 'accuracy']]
# sns.set(style='ticks', color_codes=True)
# g = sns.pairplot(new, hue='session', palette='Spectral')
# # blue more sessions, red less sessions
# # even for a lot or a few of sessions, the accuracy seems to be around 0.5

# # RELATIONSHIP BETWEEN ACCURACY/PERFORMANCE AND SESSIONS
# g = sns.lmplot(x='session', y='accuracy', data=df_params, palette='Set1')
# # lineal relationship: as sessions increase, accuracy decreases
# t = sns.lmplot(x='session', y='performance', data=df_params, palette='Set1')
# # lineal relationship: as sessions increase, performance decreases


# # GRAPHIC 2 questions
# # Plot the accuracy for each single session
# # Obtain one plot for session
# graphic = []
# counter = 0
# for column in df_params.columns:
#     if column == session:
#         for i in column:
#             if i == counter+1:
#                 graphic.append(df_params.accuracy[i])
#         counter += 1
# print(graphic)

# # i don't know where is the problem
# # here we can see that one column is called session
# for column in df_params.columns:
#     print(column)

# # GRAPHIC 2 done
# # Accuracy in each session


# # GRAPHIC 3
# # Duration for each phase
# # Duration is the column time
# dur = df_params.loc[:, 'time']
# print(dur)

# # it cannot be executed due to times are objects
# # and not integers, so they cannot sum


# # function to get seconds from time


# print(get_sec('1:23:45'))


# time_session(1)

# # trying to fix it
# ind = df_params[df_params.session == 1]
# time = ind.iloc[:, [39]]
# print(time)
# time = time.to_numpy()
# print(time)
# time = np.array_split(time, 10)   



















