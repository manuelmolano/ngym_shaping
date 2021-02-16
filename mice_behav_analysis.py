# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
COLORS = sns.color_palette("mako", n_colors=3)

path = '/Users/leyre/Dropbox/mice_data/standard_training_2020'
#path = '/home/manuel/mice_data/standard_training_2020'


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


# def accuracy_sessions_subj(df, subj, ax):
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
    """
    Find accuracy values, number of sessions in each stage and color for each
    stage.

    Parameters
    ----------
    df : dataframe
        dataframe containing data.
    subj : str
        subject (each mouse).

    Returns
    -------
    For each mouse, it returns a list of the accuracies, a list of the
    sessions in each stage and a list with the colors of each stage.

    """
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


def plot_accuracy_sessions_subj(acc, xs, color, ax, subj):
    """
    The function plots accuracy over session for every subject, showing
    the stages the mice are in different colors.

    Parameters
    ----------
    acc : list
        list of the accuracy values for each subject.
    xs : list
        list of the segments where the subject is in the same stage (e.g.
        range(0, 8), range(7,11), range(10,23)).
    color : list
        list of colors corresponding to the stage.
    ax : numpy.ndarray
        Axes object where x and y-axis are rendered inside.
    subj : str
        Subject (each mouse)

    Returns
    -------
    The plot of accuracy over session for every subject.

    """
    # TODO: finish
    for i_chnk, chnk in enumerate(acc):
        ax.plot(xs[i_chnk], acc[i_chnk], color=color[i_chnk])
        ax.axhline(0.5)
        ax.set_title(subj)
        ax.set_ylim(0.4, 1)
    if subj == 'N01' or subj == 'N07' or subj == 'N13':
        ax.set_ylabel('Accuracy')
    if subj == 'N13' or subj == 'N14' or subj == 'N15' or\
       subj == 'N16' or subj == 'N17' or subj == 'N18':
        ax.set_xlabel('Session')


def accuracy_at_stg_change(df, subj_unq):
    """
    The function returns the mean and standard deviation of the changes
    from a stage to another.

    Parameters
    ----------
    df : dataframe
        dataframe containing data.
    subj_unq : numpy.ndarray
        array of strings with the name of all the subjects

    Returns
    -------
    Mean and standard deviation of each subject

    """
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


def plot_means_std(means, std):
    """
    Plot mean and standard deviation from the accuracies of each state of each
    mouse.

    Parameters
    ----------
    means : dict
        dictionary containing all the stage changes (e.g. '1-2', '2-3'...) and
        the accuracies associated to each change.
    std : dict
        dictionary containing all the stage changes (e.g. '1-2', '2-3'...) and
        the standard deviations associated to each change.

    Returns
    -------
    Plot of the mean and standard deviation of each stage for all the subjects

    """
    change1_2 = []
    change2_1 = []
    change2_3 = []
    change3_1 = []
    change3_2 = []
    for subject in subj_unq:
        for key, val in means.items():
            if key == ('1-2'):
                change1_2.append(val)
            elif key == ('2-1'):
                change2_1.append(val)
            elif key == ('2-3'):
                change2_3.append(val)
            elif key == ('3-1'):
                change3_1.append(val)
            elif key == ('3-2'):
                change3_2.append(val)
            mean1_2 = np.mean(change1_2, 0)
            mean2_1 = np.mean(change2_1, 0)
            mean2_3 = np.mean(change2_3, 0)
            mean3_1 = np.mean(change3_1, 0)
            mean3_2 = np.mean(change3_2, 0)
    mean_list = [mean1_2, mean2_1, mean2_3, mean3_1, mean3_2]
    fig, ax = plt.subplots()
    fig.suptitle('Mean Accuracy of changes')
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(mean_list[0])
    ax1.set_title('Change 1-2')
    ax1.set_ylim(0.5, 1)
    ax1.set_ylabel('Accuracy')
    ax1.axvline(10, color='black')
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Change 2-1')
    ax2.plot(mean_list[1])
    ax2.set_ylim(0.5, 1)
    ax2.axvline(10, color='black')
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Change 2-3')
    ax3.plot(mean_list[1])
    ax3.set_ylim(0.5, 1)
    ax3.axvline(10, color='black')
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Change 3-1')
    ax4.plot(mean_list[3])
    ax4.set_ylim(0.5, 1)
    ax4.set_xlabel('Session')
    ax4.set_ylabel('Accuracy')
    ax4.axvline(10, color='black')
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Change 3-2')
    ax5.plot(mean_list[4])
    ax5.set_ylim(0.5, 1)
    ax5.set_xlabel('Session')
    ax5.axvline(10, color='black')


if __name__ == '__main__':
    plt.close('all')
    df_params = pd.read_csv(path + '/global_params.csv', sep=';')
    subj_unq = np.unique(df_params.subject_name)
    # plot acc_sess todo junto que funciona
    fig, ax = plt.subplots(nrows=3, ncols=6)
    ax = ax.flatten()
#    for i_s, sbj in enumerate(subj_unq):
#        accuracy_sessions_subj(df=df_params, subj=sbj, ax=ax[i_s])
#    fig.suptitle("Accuracy VS sessions", fontsize="x-large")
#    fig.legend([COLORS[0], COLORS[1], COLORS[2]],
#               labels=['Stage 1', 'Stage 2', 'Stage 3'],
#               loc="center right",   # Position of legend
#               borderaxespad=0.1,  # Small spacing around legend box
#               title='Color legend')
    for i_s, sbj in enumerate(subj_unq):
        acc_sbj, xs_sbj, color_sbj = accuracy_sessions_subj(df=df_params,
                                                            subj=sbj)
        plot_accuracy_sessions_subj(acc=acc_sbj, xs=xs_sbj, color=color_sbj,
                                    ax=ax[i_s], subj=sbj)
    # plot mean
    mat_mean_perfs, mat_std_perfs = accuracy_at_stg_change(df_params, subj_unq)
    plot_means_std(mat_mean_perfs, mat_std_perfs)

    # obtain the list of subjects
#     subj_mat = df_params.subject_name
#     subj_unq = np.unique(subj_mat)
#     print(subj_unq)
