from bids import BIDSLayout
import pandas as pd
import shutil
import os

BIDS_DIR = 'data_bids'
ANON_DIR = 'data_bids_anon'

# parse dataset
layout = BIDSLayout(BIDS_DIR, validate = False)

# get subjects who completed everything through the SoA scale
sub_ids = layout.get_subjects(task = 'SoAScale')
sub_ids.sort(key = int)

os.mkdir(ANON_DIR)

for sub in sub_ids:
    sub_dir_old = os.path.join(BIDS_DIR, 'sub-%s'%sub)
    sub_dir_new = os.path.join(ANON_DIR, 'sub-%s'%sub)
    shutil.copytree(sub_dir_old, sub_dir_new)

df = pd.read_csv(os.path.join(BIDS_DIR, 'participants.tsv'), sep = '\t', dtype = str)
data_complete = df.participant_id.isin(sub_ids)
# remove columns with identifying info (e.g. prolific worker ID)
df = df[data_complete][['participant_id', 'age', 'sex']]
tsv_f = os.path.join(ANON_DIR, 'participants.tsv')
df.to_csv(tsv_f, sep = '\t', na_rep = 'n/a', index = False)
