# Analysis of agency battery

The experiment code for generating the dataset analyzed here can be found on the Pavlovia GitLab [repo](https://gitlab.pavlovia.org/letitiayhho/agency-tasks).

A whirlwind tour of this repository:
* `to_bids.py` was used to convert the raw data downloaded from Pavlovia into a nice, clean [BIDS-like](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/07-behavioral-experiments.html) dataset. `questions.json` is just a helper file for this script, which adds the scale questions to the BIDS-like dataset. The output of `to_bids.py` still contains identifying data like Prolific worker IDs, so `anonymize.py` cleans up and anonymizes the dataset. If you've downloaded our data from OSF, these steps are already done for you.
* `analysis.ipynb` contains the main analysis. There are some helper functions in the `util` directory that are called in `analysis.ipynb`; these do the bulk of the work of cleaning and processing the task data.
* `demongraphics.ipynb` calculates basic demographic stats.
* Files called `Figure*.jpeg` are figures output by `analysis.ipynb`.
