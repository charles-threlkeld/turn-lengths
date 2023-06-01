"""
filter_data.py looks at all the .cha files and
returns a list of csv file names that meet the
transcription quality standards defined here.

The heuristics are:
ExactMatch -- words are an exact match
TimingOmission -- the Timing corpus was missing a word present in the DA corpus
DialogueActOmission -- DA corpus missing word present in timing corpus
TimingAbandon -- timing corpus transcribed word as abandoned that was complete
                 in DA corpus
AlternativeTranscription -- e.g. 'uh huh' vs, 'um hum'
DialogueActRepeat -- DA corpus repeats word not repeated in Timing corpus
TimingRepeat -- Timing corpus repeats word not repeated in DA corpus
WordError -- None of the above

To be returned, a transcription must have the following attributes:
ExactMatch > 0.9
WordError < 0.02
"""

import os

CHA_DIR = "/home/chas/Projects/switchboard-data/merge-corpora/"
CSV_DIR = "/home/chas/Projects/switchboard-data/basic-csvs/"


def get_data():
    """Use quality heuristics to create dataset."""

    cha_list = [x for x in os.listdir(CHA_DIR) if ".cha" in x]
    quality_fnames = []
    for cha_fname in cha_list:
        quality = {"@ExactMatch": None,
                   "@TimingOmission": None,
                   "@DialogueActOmission": None,
                   "@TimingAbandon": None,
                   "@AlternativeTranscription": None,
                   "@DialogueActRepeat": None,
                   "@TimingRepeat": None,
                   "@WordError": None}

        with open(CHA_DIR + cha_fname, 'r', encoding="UTF-8") as cha_file:
            lines = cha_file.readlines()

        for line in lines:
            if line.split(":")[0] in list(quality):
                quality[line.split(":")[0]] = float(line.split(":")[1])

        # Quality heuristics:
        if quality["@ExactMatch"] > 0.9 and \
           quality["@WordError"] < 0.02:
            convo_num = cha_fname.split(".")[0]
            quality_fnames.append(f"{CSV_DIR}{convo_num}.csv")
    return quality_fnames

# fs = get_data()
