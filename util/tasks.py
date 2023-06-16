from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import re

SoPA_loadings = np.array([.66, -.22, .01, .44, -.26, .17, .01, .8, .53, -.09, -.11, .63, .51])
SoNA_loadings = np.array([-.07, .44, .71, -.39, .38, .69, .56, .12, -.25, .57, .52, .02, -.04])

def cohens_d(m1, sd1, m2, sd2):
    sd = np.sqrt(np.mean([sd1**2, sd2**2]))
    return (m2 - m1)/sd

def process_SoAS(subs, layout):
    SoA_scores = []
    for sub in subs:
        f = layout.get(return_type = 'file', subject = sub, task = 'SoAScale')[0]
        df = pd.read_csv(f, sep = '\t')
        SoPA_sub = np.sum(df.response * SoPA_loadings)
        SoNA_sub = np.sum(df.response * SoNA_loadings)
        soa = pd.Series({
            'SoPA': SoPA_sub, 'SoNA': SoNA_sub, 'subject': sub
        })
        SoA_scores.append(soa)
    SoA_scores = pd.DataFrame(SoA_scores)
    return SoA_scores

def process_libet(subs, layout):
    '''
    Returns
    --------
    meas_df : pd.DataFrame
        Contains a row for each subject/condition with mean estimation error.
        Outliers have already been removed.
    indiv_effect_sizes : pd.DataFrame
        Contains a row for each subject with Cohen's effect size estimates for
        for tone and for key conditions. Outliers are marked in the 'exclude'
        column but not removed yet.
    '''
    libet_data = []
    indiv_effect_sizes = []
    for sub in subs:
        f = layout.get(return_type = 'file', subject = sub, task = 'libet')[0]
        df = pd.read_csv(f, sep = '\t')
        sub_meas = []
        sub_effects = dict()
        assert(df.trial.unique().size == 4)
        for cond in df.trial.unique():
            tp, target = re.findall(r'(\w+)_(\w+)', cond)[0]
            delta = df.overest_ms[df.trial == cond][5:] # exclude 5 practice trials
            s = pd.Series({
                'condition': cond,
                'type': tp,
                'target': target,
                'estimation error (ms)': delta.mean(),
                'subject': sub
            })
            sub_meas.append(s)
            sub_effects[cond] = (delta.mean(), delta.std(ddof=1))
        m1, sd1 = sub_effects['baseline_key']
        m2, sd2 = sub_effects['operant_key']
        key_effect = cohens_d(m1, sd1, m2, sd2)
        m1, sd1 = sub_effects['baseline_tone']
        m2, sd2 = sub_effects['operant_tone']
        tone_effect = cohens_d(m1, sd1, m2, sd2)
        sub_effects = pd.Series({
            'binding: key': key_effect, 'binding: tone': -1*tone_effect,
            'subject': sub
        })
        indiv_effect_sizes.append(sub_effects)
        sub_meas = pd.DataFrame(sub_meas)
        libet_data.append(sub_meas)
    meas_df = pd.concat(libet_data)
    indiv_effect_sizes = pd.DataFrame(indiv_effect_sizes)
    # remove extreme outliers
    m = meas_df['estimation error (ms)'].mean()
    sd = meas_df['estimation error (ms)'].std()
    z = (meas_df['estimation error (ms)'] - m) / sd
    outliers = (meas_df.subject[np.abs(z) > 5]).unique()
    meas_df = meas_df[~meas_df.subject.isin(outliers)]
    is_outlier = indiv_effect_sizes.subject.isin(outliers)
    exclude = (~np.isfinite(indiv_effect_sizes['binding: key'])) | is_outlier
    indiv_effect_sizes['exclude'] = exclude
    return meas_df, indiv_effect_sizes

def process_dot_motion(subs, layout):
    meas = []
    for sub in subs:

        f = layout.get(return_type = 'file', subject = sub, task = 'dotMotion')[0]
        df = pd.read_csv(f, sep = '\t')

        exclude = False

        # exclude if only responded with extreme confidence
        resp_max = df.confidenceLevel == df.confidenceLevel.max()
        resp_min = df.confidenceLevel == df.confidenceLevel.min()
        resp_extremes = resp_max | resp_min
        if resp_extremes.mean() > .95:
            exclude = True

        # exclude if less than 40% accuracy on catch trials
        catch_corr_prop = df[df.trial == 'catch'].correct.mean()
        if catch_corr_prop < .4:
            exclude = True

        # and the original study's way of computing d prime was weird
        # so we don't use that exclusion criteria.

        # control threshold is average control level across last 5 reversals
        thres = df[df.reverse == True]['controlLevel'][-5:].mean()

        # metacognitive (Type II) AUROC
        auroc = roc_auc_score(df.correct, df.confidenceLevel)

        if auroc < .5:
            exclude = True

        soa = pd.Series({
            'control threshold': thres, 'AUROC': auroc,
            'subject': sub, 'exclude': exclude
        })
        meas.append(soa)

    meas_df = pd.DataFrame(meas)
    meas_df['log control threshold'] = np.log(meas_df['control threshold'])
    return meas_df
