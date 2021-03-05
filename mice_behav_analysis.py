# import libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
COLORS = sns.color_palette("mako", n_colors=3)
COLORS_qlt = sns.color_palette("tab10", n_colors=80)
matplotlib.rcParams['font.size'] = 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

PATH = '/Users/leyre/Dropbox/mice_data/standard_training_2020'
SV_FOLDER = '/Users/leyre/Dropbox/mice_data/standard_training_2020'

#PATH = '/home/manuel/mice_data/standard_training_2020'
#SV_FOLDER = '/home/manuel/mice_data/standard_training_2020'


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


def plot_xvar_VS_yvar(df, x_var, y_var, col, xlabel='x_var', ylabel='y_var',
                      name='X variable VS Y variable'):
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
    Plot x VS y

    """
    f, ax = plt.subplots()
    ax.plot(df[x_var], df[y_var], color=col)
    ax.set(title='Plot of accuracy VS session')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    sv_fig(f, name)


def accuracy_sessions_subj(df, subj, stg=None):
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
    if stg is None:
        stg = df.loc[df['subject_name'] == subj, 'stage_number'].values

    # create the extremes (a 0 at the beggining and a 1 at the ending)
    stg_exp = np.insert(stg, 0, stg[0]-1)  # extended stages
    stg_exp = np.append(stg_exp, stg_exp[-1]+1)
    stg_diff = np.diff(stg_exp)  # change of stage
    stg_chng = np.where(stg_diff != 0)[0]  # index where stages change
    # We go over indexes where stage changes and plot chunks from ind_t-1
    # to ind_t
    acc_list = []  # list for accuracy
    xs_list = []  # list for the x axis
    stage_list = []  # list for the stages
    # iterate every stage to fill the lists
    for i_stg in range(1, len(stg_chng)):
        stage_list.append(stg_exp[stg_chng[i_stg-1]+1]-1)
        xs = range(stg_chng[i_stg-1], min(stg_chng[i_stg]+1, len(acc)+1))
        accs = acc[stg_chng[i_stg-1]:min(stg_chng[i_stg]+1, len(acc)+1)]
        acc_list.append(accs)
        xs_list.append(xs)
    return acc_list, xs_list, stage_list


def plot_accuracy_sessions_subj(acc, xs, col, ax, subj):
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
    for i_chnk, chnk in enumerate(acc):
        # iterate for every chunk, to paint the stages with different colors
        ax.plot(xs[i_chnk], acc[i_chnk], color=COLORS[col[i_chnk]])
    ax.set_title(subj)
    ax.set_ylim(0.4, 1)
    ax.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if subj in ['N01', 'N07', 'N13']:
        ax.set_ylabel('Accuracy')
    if subj in ['N13', 'N14', 'N15', 'N16', 'N17', 'N18']:
        ax.set_xlabel('Session')


def plot_final_acc_session_subj(subj_unq):
    """
    The function plots accuracy over session for all the subjects.

    Parameters
    ----------
    subj_unq : numpy.ndarray
        array of strings with the name of all the subjects

    Returns
    -------
    Plot of accuracy by session for every subject.

    """
    fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(8, 4),
                           gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    # leave some space between two figures, wspace is the horizontal gap and
    # hspace is the vertical gap
    ax = ax.flatten()
    # plot a subplot for each subject
    for i_s, sbj in enumerate(subj_unq):
        acc_sbj, xs_sbj, color_sbj = accuracy_sessions_subj(df=df_params,
                                                            subj=sbj)
        plot_accuracy_sessions_subj(acc=acc_sbj, xs=xs_sbj, col=color_sbj,
                                    ax=ax[i_s], subj=sbj)
    fig.suptitle("Accuracy VS sessions", fontsize="x-large")
    lines = [obj for obj in ax[0].properties()['children']  # all objs in ax[0]
             if isinstance(obj, matplotlib.lines.Line2D)  # that are lines
             and obj.get_linestyle() != '--']  # that are not dashed
    fig.legend(lines, ['Stage 1', 'Stage 2', 'Stage 3'],
               loc="center right",   # Position of legend
               borderaxespad=0.1,  # Small spacing around legend box
               title='Color legend')
    sv_fig(fig, 'Accuracy VS sessions')


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
    prev_w: int
        previous window size (default value:10)
    nxt_w: int
        previous window size (default value:10)

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
    # dictionary of the mean of the performance
    mat_mean_perfs = {}
    # dictionary of the standard deviation of the performace
    mat_std_perfs = {}
    # list of the number of samples of each change of stage
    number_samples = []
    for key in mat_perfs.keys():
        number_samples.append(len(mat_perfs[key]))  # save number of samples
        assert np.std([len(p) for p in mat_perfs[key]]) == 0
        mat_mean_perfs[key] = np.nanmean(np.array(mat_perfs[key]), axis=0)
        sqrt_n_smpls = np.sqrt(np.array(mat_perfs[key]).shape[0])
        mat_std_perfs[key] =\
            np.nanstd(np.array(mat_perfs[key]), axis=0)/sqrt_n_smpls
    return mat_mean_perfs, mat_std_perfs, number_samples


def plot_means_std(means, std, list_samples, prev_w=10, nxt_w=10):
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
    prev_w: int
        previous window size (default value:10)
    nxt_w: int
        previous window size (default value:10)

    Returns
    -------
    Plot of the mean and standard deviation of each stage for all the subjects

    """
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 4),
                           gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    ax = ax.flatten()
    fig.suptitle('Mean Accuracy of changes', fontsize='x-large')
    xs = np.arange(-prev_w, nxt_w)
    for i_k, (key, val) in enumerate(means.items()):
        ax[i_k].errorbar(xs, val, std[key], label=key)
        ax[i_k].set_ylim(0.5, 1)
        ax[i_k].set_title(key + ' (N='+str(list_samples[i_k])+')')
        ax[i_k].axvline(0, color='black', linestyle='--')
        # Hide the right and top spines
        ax[i_k].spines['right'].set_visible(False)
        ax[i_k].spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax[i_k].yaxis.set_ticks_position('left')
        ax[i_k].xaxis.set_ticks_position('bottom')
        if i_k in [0, 3]:
            ax[i_k].set_ylabel('Mean accuracy')
        if i_k in [3, 4]:
            ax[i_k].set_xlabel('Sessions after stage change')
    sv_fig(fig, 'Mean Accuracy of changes')


def concatenate_trials(df, subject):
    """
    Concatenates for each subject all hithistory variables (true/false),
    which describe the success of the trial.

    Parameters
    ----------
    df : dataframe
        data
    subject: str
        subject chosen

    Returns
    -------
    Performance of each subject by trials

    """
    df_hh = df[['hithistory', 'subject_name']]
    # make a group for each subjects
    df_grps = df_hh.groupby('subject_name')
    # obtain the performance for each subject
    df_sbj_perf = df_grps.get_group(subject)['hithistory'].values
    return df_sbj_perf


def plot_trials_subj(df, subject, df_sbj_perf, ax=None, conv_w=200):
    """
    Plots for each subject all hithistory variables (true/false),
    which describe the success of the trial.

    Parameters
    ----------
    df : dataframe
        data
    subject: str
        subject chosen
    df_sbj_perf: dataframe
        performance of each subject
    ax: None
        axes
    conv_w: int
        window used to smooth (convolving) the accuracy (default 200)

    Returns
    -------
    Plots a figure of the performance of the subject along trials

    """
    if ax is None:
        f, ax = plt.subplots()
    # plot the convolution of the performance of each subject
    ax.plot(np.convolve(df_sbj_perf, np.ones((conv_w,))/conv_w, mode='valid'))
    ax.set_title("Accuracy by trials of subject" + subject)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Accuracy (Hit: True or False)')
    session = df.loc[df['subject_name'] == sbj, 'session'].values
    # create the extremes (a 0 at the beggining and a 1 at the ending)
    ses_diff = np.diff(session)  # find change of stage
    ses_chng = np.where(ses_diff != 0)[0]  # find where is the previous change
    # plot a vertical line for every change os session
    for i in ses_chng:
        ax.axvline(i, color='black')


def plot_trials_subj_misses(df, subject, df_sbj_perf, conv_w=200):
    """
    Plots for each subject all hithistory variables (true/false),
    which describe the success of the trial.

    Parameters
    ----------
    df : dataframe
        data
    subject: str
        subject chosen
    df_sbj_perf: dataframe
        performance of each subject
    conv_w: int
        window used to smooth (convolving) the accuracy (default 200)

    Returns
    -------
    Plots a figure of the performance of the subject along trials

    """
    plt.figure()
    plt.plot(np.convolve(df_sbj_perf, np.ones((conv_w,))/conv_w, mode='valid'))
    plt.title("Accuracy by trials of subject taking into" +
              " account misses (" + subject+")")
    plt.xlabel('Trials')
    plt.ylabel('Accuracy (Hit: True or False)')
    session = df.loc[df['subject_name'] == sbj, 'session'].values
    # create the extremes (a 0 at the beggining and a 1 at the ending)
    ses_diff = np.diff(session)  # find change of stage
    ses_chng = np.where(ses_diff != 0)[0]  # find where is the previous change
    # plot a vertical line for every change os session
    for i in ses_chng:
        plt.axvline(i, color='black')


def plot_trials_subjects(df, conv_w=20):
    """
    Plots the performance of all the subjects along trials in the same figure.

    Parameters
    ----------
    df : dataframe
        data
    conv_w: int
        convolution window size (default value:20)

    Returns
    -------
    Plots a figure with subplots containing the performance of the subjects
    along all trials

    """
    # dataframe with only hithistory and subject_name columns
    df_hh = df[['hithistory', 'subject_name']]
    # make a group for each subject
    df_grps = df_hh.groupby('subject_name')
    subj_unq = np.unique(df_hh.subject_name)
    fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(6, 4),
                           gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    ax = ax.flatten()
    for i_s, sbj in enumerate(subj_unq):
        df_sbj_perf = df_grps.get_group(sbj)['hithistory'].values
        ax[i_s].plot(np.convolve(df_sbj_perf, np.ones((conv_w,))/conv_w))
        ax[i_s].set_title(sbj)
        ax[i_s].set_xlim(0, 30000)
        ax[i_s].spines['right'].set_visible(False)
        ax[i_s].spines['top'].set_visible(False)
        ax[i_s].yaxis.set_ticks_position('left')
        ax[i_s].xaxis.set_ticks_position('bottom')
        if sbj in ['N01', 'N07', 'N13']:
            ax[i_s].set_ylabel('Hit (T/F)')
        if sbj in ['N13', 'N14', 'N15', 'N16', 'N17', 'N18']:
            ax[i_s].set_xlabel('Trials')
    fig.suptitle("Accuracy over trials")


def concatenate_misses(df, subject):
    """
    Concatenates for each subject all hithistory variables (true/false),
    which describe the success of the trial.

    Parameters
    ----------
    df : dataframe
        data
    subject: str
        subject chosen

    Returns
    -------
    Performance of each subject by trials

    """
    df_hh = df[['misshistory', 'subject_name']]
    df_grps = df_hh.groupby('subject_name')
    df_sbj_perf = df_grps.get_group(subject)['misshistory'].values
    return df_sbj_perf


def plot_misses_subj(df, subject, df_sbj_perf, conv_w=200):
    """
    Plots for each subject all hithistory variables (true/false),
    which describe the success of the trial.

    Parameters
    ----------
    df : dataframe
        data
    subject: str
        subject chosen
    df_sbj_perf: dataframe
        performance of each subject
    conv_w: int
        window used to smooth (convolving) the accuracy (default 200)

    Returns
    -------
    Plots a figure of the performance of the subject along trials

    """
    plt.figure()
    plt.plot(np.convolve(df_sbj_perf, np.ones((conv_w,))/conv_w, mode='valid'))
    plt.title("Misses by trials of subject" + subject)
    plt.xlabel('Trials')
    plt.ylabel('Misses (True:responds or False:do anything)')
    session = df.loc[df['subject_name'] == sbj, 'session'].values
    # create the extremes (a 0 at the beggining and a 1 at the ending)
    ses_diff = np.diff(session)  # change of stage
    ses_chng = np.where(ses_diff != 0)[0]
    for i in ses_chng:
        plt.axvline(i, color='black')


def create_toy_dataset(df_trials):
    """
    Creates a set of data with subjects 01 and 02 and sessions 1,2 and 3.

    Parameters
    ----------
    df_trials : dataframe
        data.

    Returns
    -------
    Set of the original dataset

    """
    # creation of a miniset of data with subjects N01&N02 and sessions 1,2,3
    minitrials = df_trials.loc[((df_trials['subject_name'] == 'N01') |
                               (df_trials['subject_name'] == 'N02')) &
                               (df_trials['session'] < 4)]
    minitrials.to_csv(PATH + '/minitrials.csv', sep=';')
    return minitrials


def remove_misses(df):
    """
    Revomes misses of the accuracy.

    Parameters
    ----------
    df_trials : dataframe
        data.

    Returns
    -------
    Dataset without misses

    """
    # change the values of the performance to 0.5 when hithistory is False 
    # and misshistory is true in order to revome misses
    d1 = np.where((df['hithistory'] is False) & (df['misshistory'] is True))
    for index in d1:
        df['hithistory'][index] = 0.5
    return df


def create_stage_column(df, df_prms, subject):
    """
    Creation of an extra column for the stages in df_trials

    Parameters
    ----------
    df : dataframe
        data of trials
    df : dataframe
        data of sessions
    subject: str
        subject chosen

    Returns
    -------
    Dataframe with an extra column

    """
    # obtain values of the session in df_trials
    df_ss = df.loc[df.subject_name == subject, ['session']]
    # obtain values of the indexes and stages of the df_params
    _, indx_stg, stages = accuracy_sessions_subj(df=df_prms, subj=subject)
    sess_unq = np.unique(df_ss)
    trial_sess = np.zeros_like(df_ss).flatten()
    for ss in sess_unq:
        aux = [i_x for i_x, x in enumerate(indx_stg) if ss in x][0]
        indx = np.where(df_ss == ss)[0]  # find where the sessions of both
        # datasets are the same
        trial_sess[indx] = stages[aux]+1  # sessions are index+1
    df_trials_subject = df.loc[df.subject_name == subject]
    df_trials_subject['stage'] = trial_sess
    return df_trials_subject


def create_motor_column(df, df_prms, subject):
    """
    Creation of an extra column for the motor variable in df_trials

    Parameters
    ----------
    df : dataframe
        data of trials
    df : dataframe
        data of sessions
    subject: str
        subject chosen

    Returns
    -------
    Dataframe with an extra column

    """
    # obtain values of the motor values in df_trials
    motor = df_prms.loc[df_prms['subject_name'] == subject, 'motor'].values
    # obtain values of the indexes and motor stages of the df_params
    _, indx_mstg, mot_stg = accuracy_sessions_subj(df=df_prms, subj=subject,
                                                   stg=motor)
    df_ss = df.loc[df.subject_name == subject, ['session']]
    sess_unq = np.unique(df_ss)
    trial_sess = np.zeros_like(df_ss).flatten()
    for ss in sess_unq:
        aux = [i_x for i_x, x in enumerate(indx_mstg) if ss in x][0]
        indx = np.where(df_ss == ss)[0]  # find where the sessions of both
        # datasets are the same
        trial_sess[indx] = mot_stg[aux]+1  # sessions are index+1
    df_trials_subject = df.loc[df.subject_name == subject]
    df_trials_subject['motor_stage'] = trial_sess
    return df_trials_subject


def plot_stage_motor_delay_subject(subj, new_df, ax):
    """
    Plot a subplot for each subject

    Parameters
    ----------
    subj : str
        subject chosen
    new_df : dataframe
        data of sessions
    ax: axes

    Returns
    -------
    Subplot for a subject

    """
    ax.plot(new_df['stage'].values)
    ax.plot(new_df['motor_stage'].values)
    # convert all values to float64 to avoid errors
    new_df['Motor_out_start'] = pd.to_numeric(new_df['Motor_out_start'],
                                              errors='coerce')
    new_df['Motor_out_end'] = pd.to_numeric(new_df['Motor_out_end'],
                                            errors='coerce')
    new_df['Motor_in_end'] = pd.to_numeric(new_df['Motor_in_end'],
                                           errors='coerce')
    # sum a constant k=5 to see better plots and avoid overlapping
    ax.plot(new_df['Motor_out_end'].values+5)
    ax.plot(new_df['Motor_in_end'].values+5)
    ax.plot(new_df['delay_times_m'].values)
    ax.plot(new_df['delay_times_h'].values)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(subj)
    if subj in ['N13', 'N14', 'N15', 'N16', 'N17', 'N18']:
        ax.set_xlabel('Trials')


def plot_final_stage_motor_delay(subj_unq, df, df_prms):
    """
    Plot with a subplot for each subject showing the following variables:
        'Stage', 'Motor_stage', 'Motor_out_end', 'Motor_in_end',
        'delay_times_m', 'delay_times_h'

    Parameters
    ----------
    subj_unq : array
        list of subjects
    df : dataframe
        trials dataframe
    df_prms: dataframe
        params dataframe

    Returns
    -------
    Plot with a subplot for each subject

    """
    fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(12, 6),
                           gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    # leave some space between two figures, wspace is the horizontal gap and
    # hspace is the vertical gap
    ax = ax.flatten()
    for i_s, sbj in enumerate(subj_unq):
        # add the stage column to df_trials
        new_df_set = create_stage_column(df, df_prms, subject=sbj)
        # add the motor stage column to df_trials
        new_df_set = create_motor_column(new_df_set, df_prms, subject=sbj)
        plot_stage_motor_delay_subject(subj=sbj, new_df=new_df_set, ax=ax[i_s])
    fig.suptitle("Motor and Delay variables", fontsize="x-large")
    lines = [obj for obj in ax[0].properties()['children']  # all objs in ax[0]
             if isinstance(obj, matplotlib.lines.Line2D)  # that are lines
             and obj.get_linestyle() != '--']
    fig.legend(lines, ['Stage', 'Motor_stage', 'Motor_out_end',
                       'Motor_in_end', 'delay_times_m', 'delay_times_h'],
               loc="center right",   # Position of legend
               borderaxespad=0.1,  # Small spacing around legend box
               title='Color legend')
    sv_fig(fig, 'Motor and Delay variables')


if __name__ == '__main__':
    plt.close('all')
    do = False
    dataset = 'global_trials'  # global_trials-minitrials
    df_params = pd.read_csv(PATH + '/global_params.csv', sep=';')
    df_trials = pd.read_csv(PATH + '/'+dataset+'.csv', sep=';',
                            low_memory=False)
    subj_unq = np.unique(df_params.subject_name)
    # PLOT MOTOR AND DELAY VARIABLES ACROSS TRIALS FOR ALL THE SUBJECTS
    plot_final_stage_motor_delay(subj_unq, df=df_trials, df_prms=df_params)
    if do:
        # PLOT ACCURACY VS SESSION
        plot_final_acc_session_subj(subj_unq)
        # PLOT PERFORMANCE AT EACH STAGE
        prev_w = 10
        nxt_w = 10
        mat_mean_perfs, mat_std_perfs, num_samples =\
            accuracy_at_stg_change(df_params, subj_unq,
                                   prev_w=prev_w, nxt_w=nxt_w)
        plot_means_std(mat_mean_perfs, mat_std_perfs, num_samples,
                       prev_w=prev_w, nxt_w=nxt_w)
        # PLOT TRIALS ACCURACY OF ALL THE SUBJECTS
        for i_s, sbj in enumerate(subj_unq):
            df_sbj_perf = concatenate_trials(df_trials, sbj)
            plot_trials_subj(df_trials, sbj, df_sbj_perf, conv_w=200)
        # PLOT TRIALS ACCURACY OF ALL THE SUBJECTS CONSIDERING MISSES
        # remove misses
        df_trials_without_misses = remove_misses(df_trials)
        for i_s, sbj in enumerate(subj_unq):
            df_sbj_perf = concatenate_trials(df_trials_without_misses, sbj)
            plot_trials_subj_misses(df_trials_without_misses, sbj, df_sbj_perf,
                                    conv_w=200)
        # PLOT MISSES ACROSS TRIALS OF ALL THE SUBJECTS
        for i_s, sbj in enumerate(subj_unq):
            df_sbj_perf = concatenate_misses(df_trials, sbj)
            plot_misses_subj(df_trials, sbj, df_sbj_perf, conv_w=200)
