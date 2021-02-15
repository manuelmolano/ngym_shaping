# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
COLORS = sns.color_palette("mako", n_colors=3)

path = '/Users/leyre/Dropbox/mice_data/standard_training_2020'
path = '/home/manuel/mice_data/standard_training_2020'


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


def num_sessions_per_stage(df, subj):
    # find all the trials done in session1
    stg = df.loc[df['subject_name'] == subj, 'stage_number'].values
    both = df.loc[df['subject_name'] == subj, ['performance', 'stage_number',
                                               'substage']].values

    print(both)
    return stg, both


#def accuracy_sessions_subj(df, subj, ax):
#    acc = df.loc[df['subject_name'] == subj, 'accuracy'].values
#    stg = df.loc[df['subject_name'] == subj, 'stage_number'].values
#
#    # create the extremes (a 0 at the beggining and a 1 at the ending)
#    stg_exp = np.insert(stg, 0, 0)  # extended stages
#    stg_exp = np.append(stg_exp, stg_exp[-1]+1)
#    stg_diff = np.diff(stg_exp)  # change of stage
#    stg_chng = np.where(stg_diff != 0)[0]  # index where stages change
#    # _, ax_temp = plt.subplots(nrows=1, ncols=1)
#    # ax_temp.plot(stg_exp, label='stage')
#    # ax_temp.plot(stg_diff, label='stage diff')
#    # We go over indexes where stage changes and plot chunks from ind_t-1
#    # to ind_t
#    # TODO: try to separate results production from results plotting
#    for i_stg in range(1, len(stg_chng)):
#        # ax_temp.plot([stg_chng[i_stg-1], stg_chng[i_stg-1]], [0, 5], '--k')
#        color = stg_exp[stg_chng[i_stg-1]+1]-1
#        xs = range(stg_chng[i_stg-1], min(stg_chng[i_stg]+1, len(acc)))
#        accs = acc[stg_chng[i_stg-1]:min(stg_chng[i_stg]+1, len(acc))]
#        ax.plot(xs, accs, color=COLORS[color])
#        ax.axhline(0.5)
#        ax.set_title(subj)
#        ax.set_ylim(0.4, 1)
#    if subj == 'N01' or subj == 'N07' or subj == 'N13':
#        ax.set_ylabel('Accuracy')
#    if subj == 'N13' or subj == 'N14' or subj == 'N15' or\
#       subj == 'N16' or subj == 'N17' or subj == 'N18':
#        ax.set_xlabel('Session')
    
def accuracy_sessions_subj(df, subj):
    acc = df.loc[df['subject_name'] == subj, 'accuracy'].values
    stg = df.loc[df['subject_name'] == subj, 'stage_number'].values

    # create the extremes (a 0 at the beggining and a 1 at the ending)
    stg_exp = np.insert(stg, 0, 0)  # extended stages
    stg_exp = np.append(stg_exp, stg_exp[-1]+1)
    stg_diff = np.diff(stg_exp)  # change of stage
    stg_chng = np.where(stg_diff != 0)[0]  # index where stages change
    # We go over indexes where stage changes and plot chunks from ind_t-1
    # to ind_t
    acc_list = []
    xs_list = []
    color_list = []
    for i_stg in range(1, len(stg_chng)):
        color_list.append(stg_exp[stg_chng[i_stg-1]+1]-1)
        xs = range(stg_chng[i_stg-1], min(stg_chng[i_stg]+1, len(acc)))
        accs = acc[stg_chng[i_stg-1]:min(stg_chng[i_stg]+1, len(acc))]
        acc_list.append(accs)
        xs_list.append(xs)
    return acc_list, xs_list, color_list


def plot_accuracy_sessions_subj(acc, xs, color, ax):
    """
    

    Parameters
    ----------
    acc : TYPE
        DESCRIPTION.
    xs : TYPE
        DESCRIPTION.
    color : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # TODO: finish and add docstring
    for i_chnk, chnk in enumerate(acc):
        # ax_temp.plot([stg_chng[i_stg-1], stg_chng[i_stg-1]], [0, 5], '--k')
        ax.plot(xs[i_stg], acc[i_stg], color=COLORS[color])
        ax.axhline(0.5)
        ax.set_title(subj)
        ax.set_ylim(0.4, 1)
    if subj == 'N01' or subj == 'N07' or subj == 'N13':
        ax.set_ylabel('Accuracy')
    if subj == 'N13' or subj == 'N14' or subj == 'N15' or\
       subj == 'N16' or subj == 'N17' or subj == 'N18':
        ax.set_xlabel('Session')


def accuracy_at_stg_change(df, subj_unq):
    """The function returns the mean and standard deviation of the changes
    from a stage to another"""
    mat_perfs = {}
    for i_s, sbj in enumerate(subj_unq):
        acc = df.loc[df['subject_name'] == sbj, 'accuracy'].values
        stg = df.loc[df['subject_name'] == sbj, 'stage_number'].values
        # create the extremes (a 0 at the beggining and a 1 at the ending)
        stg_diff = np.diff(stg)  # change of stage
        stg_chng = np.where(stg_diff != 0)[0]  # index where stages change
        # We go over indexes where stage changes and plot chunks from ind_t-1
        # to ind_t
        prev_w = 10
        nxt_w = 10
        for i_stg in range(len(stg_chng)):
            # color = stg_exp[stg_chng[i_stg-1]+1]-1
            stg_prev = stg[stg_chng[i_stg]-1]  # get stage before the change
            stg_nxt = stg[stg_chng[i_stg]+1]  # get stage after the change
            assert stg_prev != stg_nxt, 'stages are supposed to be different'
            key = str(stg_prev)+'-'+str(stg_nxt)  # e.g. 1-2
            if key not in mat_perfs.keys():
                mat_perfs[key] = []
            # build chunk
            i_previo = max(0, stg_chng[i_stg]-prev_w)
            i_next = stg_chng[i_stg]+nxt_w
            chunk = -min(0, stg_chng[i_stg]-prev_w)*[np.nan] +\
                acc[i_previo:i_next].tolist() +\
                max(0, i_next-len(acc))*[np.nan]
            # add chunk to the dictionary
            mat_perfs[key].append(chunk)
    mat_mean_perfs = {}
    mat_std_perfs = {}
    for key in mat_perfs.keys():
        assert np.std([len(p) for p in mat_perfs[key]]) == 0
        mat_mean_perfs[key] = np.nanmean(np.array(mat_perfs[key]), axis=0)
        mat_std_perfs[key] = np.nanstd(np.array(mat_perfs[key]), axis=0)
    return mat_mean_perfs, mat_std_perfs


def plot_means_sem(means, std):
    plt.close('all')
    # TODO: for loop going through keys
    m = sorted(means.items())  # sorted by key, return a list of tuples
    x1, y1 = zip(*m)  # unpack a list of pairs into two tuples
    s = sorted(std.items())
    x2, y2 = zip(*s)
    plt.errorbar(x1, y1, y2, fmt='--o')  # TODO: use axis
    plt.title('Mean and std over stage change')
    plt.xlabel('Stage change')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    df_params = pd.read_csv(path + '/global_params.csv', sep=';')
    fig, ax = plt.subplots(nrows=3, ncols=6)
    ax = ax.flatten()
    subj_unq = np.unique(df_params.subject_name)
#    for i_s, sbj in enumerate(subj_unq):
#        accuracy_sessions_subj(df=df_params, subj=sbj, ax=ax[i_s])
#    fig.suptitle("Accuracy VS sessions", fontsize="x-large")
#    fig.legend([COLORS[0], COLORS[1], COLORS[2]],
#               labels=['Stage 1', 'Stage 2', 'Stage 3'],
#               loc="center right",   # Position of legend
#               borderaxespad=0.1,  # Small spacing around legend box
#               title='Color legend')
    # for i_s, sbj in enumerate(subj_unq):
    #     acc_sbj, xs_sbj = accuracy_sessions_subj(df=df_params, subj=sbj)
    #     plot_accuracy_sessions_subj(df=df_params, subj=sbj, acc=acc_sbj, 
    #                                 xs=xs_sbj, ax=ax[i_s])
    mat_mean_perfs, mat_std_perfs = accuracy_at_stg_change(df_params, subj_unq)
    plot_means_sem(mat_mean_perfs, mat_std_perfs)

    # obtain the list of subjects
#     subj_mat = df_params.subject_name
#     subj_unq = np.unique(subj_mat)
#     print(subj_unq)

    # # list of all accuracy values
    # accuracy_mat = df_params.accuracy
    # accu_unq = np.unique(accuracy_mat)
    # print(accu_unq)

    # # list of all the sessions
#     session_mat = df_params.session
#     session_unq = np.unique(session_mat)
#     print(session_unq)
