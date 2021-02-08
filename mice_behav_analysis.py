# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    both = df.loc[df['subject_name'] == subj, ['performance', 'stage_number',
                                               'substage']].values

    print(both)
    return stg, both


def gen_repeating(s):
    """Generator of groups with repeated elements in an iterable,
    for example 'abbccc' would be [('a', 0, 0), ('b', 1, 2), ('c', 3, 5)]
    """
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        yield (s[i], i, j-1)
        i = j


def accuracy_sessions_subj(df, subj):
    col_list = ['#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f',
                '#990000']
    acc = df.loc[df['subject_name'] == subj, 'accuracy'].values
    stg = df.loc[df['subject_name'] == subj, 'stage_number'].values
    print(acc)
    print(stg)

    # create the extremes (a 0 at the beggining and a 1 at the ending)
    stg_exp = np.insert(stg, 0, 0)
    stg_exp = np.append(stg_exp, stg_exp[-1]+1)
    stg_diff = np.diff(stg_exp)
    stg_chng = np.where(stg_diff != 0)[0]
    print(stg_chng)
    # TODO: plot chunks: do diff, find changes,
    # plot chunks from change_t-1 to change_t
    for i_stg in range(1, len(stg_chng)):
        color = stg_exp[stg_chng[i_stg-1]+1]
        plt.plot(range(stg_chng[i_stg-1], min(stg_chng[i_stg]+1, len(acc))),
                 acc[stg_chng[i_stg-1]:min(stg_chng[i_stg]+1, len(acc))],
                 color=col_list[color-1])


if __name__ == '__main__':
    plt.close('all')
    df_params = pd.read_csv(path + '/global_params.csv', sep=';')
    accuracy_sessions_subj(df_params, 'C28')

    # filter for each animal 'subject_name'
    # X_df = df_params.loc[df_params['subject_name'] == 'C28', 'stage_number']
    # print(X_df)

    # obtain the list of subjects
    # subject_mat = df_params.subject_name
    # subj_unq = np.unique(subject_mat)
    # print(subj_unq)

    # # list of all accuracy values
    # accuracy_mat = df_params.accuracy
    # accu_unq = np.unique(accuracy_mat)
    # print(accu_unq)

    # # list of all the sessions
#     session_mat = df_params.session
#     session_unq = np.unique(session_mat)
#     print(session_unq)

    # # list of all the stages
    # stages_mat = df_params.stage_number
    # stages_unq = np.unique(stages_mat)
    # print(stages_unq)

    # # performance, stage and substage for each subject
    # for sbj in subj_unq:
    #     print('------')
    #     print(sbj)
    #     num_sessions_per_stage(df_params, sbj)

    # # find the stages for only one session
    # stages_28 = []
    # for i in num_sessions_per_stage(df_params, 'C28')[0]:
    #     stages_28.append(i)

    # plot accuracy VS session for each subject


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













