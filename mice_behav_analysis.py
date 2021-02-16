# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
COLORS = sns.color_palette("mako", n_colors=3)
COLORS_qlt = sns.color_palette("tab10")

# TODO: capital letters
path = '/Users/leyre/Dropbox/mice_data/standard_training_2020'
path = '/home/manuel/mice_data/standard_training_2020'
SV_FOLDER

def sv_fig(f, name):
    """
    Save figure.

    Parameters
    ----------
    f : fig
        figure to save.
    name : str
        name to use to save the figure.

    Returns
    -------
    None.

    """
    f.savefig(SV_FOLDER+'/'+name+'.svg', dpi=400, bbox_inches='tight')
    f.savefig(SV_FOLDER+'/'+name+'.png', dpi=400, bbox_inches='tight')


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


def accuracy_sessions_subj_old(df, subj, ax):
    acc = df.loc[df['subject_name'] == subj, 'accuracy'].values
    stg = df.loc[df['subject_name'] == subj, 'stage_number'].values

    # create the extremes (a 0 at the beggining and a 1 at the ending)
    stg_exp = np.insert(stg, 0, 0)  # extended stages
    stg_exp = np.append(stg_exp, stg_exp[-1]+1)
    stg_diff = np.diff(stg_exp)  # change of stage
    stg_chng = np.where(stg_diff != 0)[0]  # index where stages change
    # _, ax_temp = plt.subplots(nrows=1, ncols=1)
    # ax_temp.plot(stg_exp, label='stage')
    # ax_temp.plot(stg_diff, label='stage diff')
    # We go over indexes where stage changes and plot chunks from ind_t-1
    # to ind_t
    for i_stg in range(1, len(stg_chng)):
        # ax_temp.plot([stg_chng[i_stg-1], stg_chng[i_stg-1]], [0, 5], '--k')
        color = stg_exp[stg_chng[i_stg-1]+1]-1
        xs = range(stg_chng[i_stg-1], min(stg_chng[i_stg]+1, len(acc)))
        accs = acc[stg_chng[i_stg-1]:min(stg_chng[i_stg]+1, len(acc))]
        ax.plot(xs, accs, color=COLORS[color])
        ax.axhline(0.5)
        ax.set_title(subj)
        ax.set_ylim(0.4, 1)
    if subj == 'N01' or subj == 'N07' or subj == 'N13':
        ax.set_ylabel('Accuracy')
    if subj == 'N13' or subj == 'N14' or subj == 'N15' or\
       subj == 'N16' or subj == 'N17' or subj == 'N18':
        ax.set_xlabel('Session')


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
        ax.plot(xs[i_chnk], acc[i_chnk], color=COLORS[color[i_chnk]])
        ax.axhline(0.5)
        ax.set_title(subj)
        ax.set_ylim(0.4, 1)
    if subj in ['N01', 'N07', 'N13']:
        ax.set_ylabel('Accuracy')
    # TODO: do as above
    if subj == 'N13' or subj == 'N14' or subj == 'N15' or\
       subj == 'N16' or subj == 'N17' or subj == 'N18':
        ax.set_xlabel('Session')


def accuracy_at_stg_change(df, subj_unq, prev_w=10, nxt_w=10):
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
        for i_stg in range(len(stg_chng)):
            # color = stg_exp[stg_chng[i_stg-1]+1]-1
            stg_prev = stg[stg_chng[i_stg]]  # get stage before the change
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
    # TODO: save and show number of samples
    for key in mat_perfs.keys():
        assert np.std([len(p) for p in mat_perfs[key]]) == 0
        mat_mean_perfs[key] = np.nanmean(np.array(mat_perfs[key]), axis=0)
        sqrt_n_smpls = np.sqrt(np.array(mat_perfs[key]).shape[0])
        mat_std_perfs[key] =\
            np.nanstd(np.array(mat_perfs[key]), axis=0)/sqrt_n_smpls
    return mat_mean_perfs, mat_std_perfs


def plot_means_std(means, std, prev_w=10, nxt_w=10):
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
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(4, 3))
    ax = ax.flatten()
    fig.suptitle('Mean Accuracy of changes')
    xs = np.arange(-prev_w, nxt_w)
    for i_k, (key, val) in enumerate(means.items()):
        ax[i_k].errorbar(xs, val, std[key], label=key)
        ax[i_k].set_ylim(0.5, 1)
        ax[i_k].set_title(key)
        ax[i_k].set_xlabel('Session')
        ax[i_k].axvline(0, color='black', linestyle='--')


if __name__ == '__main__':
    plt.close('all')
    df_params = pd.read_csv(path + '/global_params.csv', sep=';')
    subj_unq = np.unique(df_params.subject_name)
    # plot acc_sess todo junto que funciona
    fig, ax = plt.subplots(nrows=3, ncols=6)
    ax = ax.flatten()
    for i_s, sbj in enumerate(subj_unq):
        acc_sbj, xs_sbj, color_sbj = accuracy_sessions_subj(df=df_params,
                                                            subj=sbj)
        plot_accuracy_sessions_subj(acc=acc_sbj, xs=xs_sbj, color=color_sbj,
                                    ax=ax[i_s], subj=sbj)
    # TODO: legend, horizontal line dashed and black
    # performance at stage change
    prev_w = 10
    nxt_w = 10
    mat_mean_perfs, mat_std_perfs = accuracy_at_stg_change(df_params, subj_unq,
                                                           prev_w=prev_w,
                                                           nxt_w=nxt_w)
    plot_means_std(mat_mean_perfs, mat_std_perfs, prev_w=prev_w, nxt_w=nxt_w)

    # obtain the list of subjects
#     subj_mat = df_params.subject_name
#     subj_unq = np.unique(subj_mat)
#     print(subj_unq)
