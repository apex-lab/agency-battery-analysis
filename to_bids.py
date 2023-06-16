from datetime import datetime
import numpy as np
import pandas as pd
import os
import re
import json
import shutil
from mne_bids import BIDSPath
from collections import defaultdict

# hard-coded stuff
DATA_DIR = 'data'
OUTPUT_DIR = 'data_bids'

def extract_datetime(f):
    '''
    pulls timestamp out of pavlovia filename
    '''
    time_info = re.findall('(\d+)-(\d+)-(\d+)_(\d+)h(\d+)\.(\d+)\.(\d+)', f)[0]
    time_info = [int(t) for t in time_info]
    timestamp = datetime(*time_info[:-1], 1000*time_info[-1])
    return timestamp

def subtract_angles(a, b):
    '''
    returns signed angle between two angles

    Parameters
    -----------
    a : float
        in radians
    b : float
        in radians

    Returns
    ----------
    delta : float
        the difference a - b, in radians
    '''
    return (a - b + np.pi) % (2*np.pi) - np.pi

def unpack_survey_responses(row, questions):
    resps = json.loads(row.response)
    resps = pd.Series(resps) + 1 # so choices are 1 indexed
    resps = pd.DataFrame({'response': resps, 'question': questions})
    item = 'Q' + (resps.index.str.extract('(\d+)').astype(int) + 1).astype(str)
    resps.insert(0, 'trial', item.iloc[:, 0].to_numpy())
    return resps

# gather data files & metadata
fnames = os.listdir(DATA_DIR)
fnames = [f for f in fnames if '.csv' in f]
timestamps = [extract_datetime(f) for f in fnames]
fnames.sort(key = extract_datetime)
sub_ids = ['%03d'%(i + 1) for i in range(len(fnames))] # order of timestamps
fpaths = [os.path.join(DATA_DIR, f) for f in fnames]

# get survey questions so we may record them for posterity
with open('questions.json', 'r') as qf:
    survey_questions = json.load(qf)

worker_ids = defaultdict(lambda: None)
has_all_data = defaultdict(lambda: False)

for f, sub, t in zip(fpaths, sub_ids, timestamps):

    try:

        df = pd.read_csv(f)

        # save worker ID
        worker_ids[sub] = df.subject_id[0]

        ## parse libet data and save
        libet = df[df.trial_type == 'libet']
        libet = libet[['cond_bo', 'cond_kt', 'early', 'theta_initial',
                       'clock_start_ms', 'keypress_ms', 'theta_keypress',
                       'theta_tone', 'tone_ms', 'theta_target', 'spin_continue_ms',
                       'theta_est_0', 'theta_est', 'tone_delay_ms', 'timeout']]
        libet = libet[~(libet.early | libet.timeout)]
        cond = libet.cond_bo + '_' + libet.cond_kt
        libet.insert(0, 'trial', cond)
        libet = libet.drop(['cond_bo', 'cond_kt', 'timeout', 'early'], axis = 1)

        # compute difference between actual and estimate times
        delta = subtract_angles(libet.theta_est, libet.theta_target)
        delta = -delta # flip sign, since clock was moving counterclockwise

        # record difference
        clock_period_ms = 2560
        libet['overest_rad'] = delta
        libet['overest_ms'] = delta / (2*np.pi) * clock_period_ms

        # and save to tsv
        bids_path = BIDSPath(
            subject = sub,
            task = 'libet',
            datatype = 'beh',
            root = OUTPUT_DIR,
            suffix = 'beh',
            extension = 'tsv'
        )
        bids_path.mkdir()
        libet.to_csv(str(bids_path), sep = '\t', na_rep = 'n/a', index = False)

        # pull out dot-motion trials
        dot = df[df.trial_type == 'dot-motion']
        dot = dot[[
            'controlLevel', 'correct', 'confidenceLevel',
               'key_press', 'test_part', 'staircase', 'reverse'
        ]]
        # remove practice and timed-out trials
        dot = dot[(dot.test_part == 'dot_catch_trial') | (dot.test_part == 'dot_stimulus')]
        # add trial description
        trial_type = dot.test_part.str.extract('dot_([^\W_]+)').replace('stimulus', 'staircase')
        dot.insert(0, 'trial', trial_type.iloc[:, 0].to_numpy())
        dot = dot.drop(['test_part'], axis = 1)
        # and save
        bids_path = BIDSPath(
            subject = sub,
            task = 'dotMotion',
            datatype = 'beh',
            root = OUTPUT_DIR,
            suffix = 'beh',
            extension = 'tsv'
        )
        dot.to_csv(str(bids_path), sep = '\t', na_rep = 'n/a', index = False)

        ## SoA scale
        row = df[df.trial_type == 'survey-likert'].iloc[0]
        questions = survey_questions['Sense of Agency Scale']
        SoAScale = unpack_survey_responses(row, questions)
        bids_path = BIDSPath(
            subject = sub,
            task = 'SoAScale',
            datatype = 'beh',
            root = OUTPUT_DIR,
            suffix = 'beh',
            extension = 'tsv'
        )
        SoAScale.to_csv(str(bids_path), sep = '\t', na_rep = 'n/a', index = False)

        ## ESoS scale
        row = df[df.trial_type == 'survey-likert'].iloc[1]
        questions = survey_questions['Embodied Sense of Self Scale']
        ESoSScale = unpack_survey_responses(row, questions)
        bids_path = BIDSPath(
            subject = sub,
            task = 'ESoSScale',
            datatype = 'beh',
            root = OUTPUT_DIR,
            suffix = 'beh',
            extension = 'tsv'
        )
        ESoSScale.to_csv(str(bids_path), sep = '\t', na_rep = 'n/a', index = False)

        ## tellegan scale
        row = df[df.trial_type == 'survey-likert'].iloc[2]
        questions = survey_questions['Tellegan Absorption Scale']
        tellegan = unpack_survey_responses(row, questions)
        bids_path = BIDSPath(
            subject = sub,
            task = 'tellegan',
            datatype = 'beh',
            root = OUTPUT_DIR,
            suffix = 'beh',
            extension = 'tsv'
        )
        tellegan.response = tellegan.response == 1 # convert to boolean
        tellegan.to_csv(str(bids_path), sep = '\t', na_rep = 'n/a', index = False)

        # if you've gotten this far, all data seems to be there
        has_all_data[sub] = True

    except:
        continue

# save subject metadata
wids = [worker_ids[sub] for sub in sub_ids]
has_data = [has_all_data[sub] for sub in sub_ids]
participants = pd.DataFrame({
    'participant_id': sub_ids,
    'complete': has_data,
    'timestamp': [t.strftime("%Y-%m-%dT%H:%M:%S.%f") for t in timestamps],
    'prolific_id': wids
})
out_f = os.path.join(OUTPUT_DIR, 'participants.tsv')
participants.to_csv(out_f, sep = '\t', na_rep = 'n/a', index = False)

# and copy original data to BIDS source folder
src_dir = os.path.join(OUTPUT_DIR, 'source')
shutil.copytree(DATA_DIR, src_dir)
