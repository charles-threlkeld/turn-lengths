"""
turn length graph and models.

turn_lengths.py creates a csv file for all data used in this project.
The data consist of tcu lengths (ms), tcus per turn, and turn lengths (ms).

There are functions to build best-fit models of these data using the following:
tcu length ~ Gamma
tcus per turn ~ Geometric
turn length ~ Gamma

There are also functions to visualize the data and the fit models.
"""
import csv
import sys
from os.path import exists
import re
import warnings

import cmudict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
#import seaborn as sns

import filter_data

warnings.simplefilter(action='ignore', category=FutureWarning)

OUTPUT_DIR = "/home/chas/Projects/turn-length/"

VOWEL_RUNS = re.compile(
    "[aeiouy]+", flags=re.I)
EXCEPTIONS = re.compile(
    # Remove trailing e
    # e.g. smite, scared
    "[^aeiou]e[sd]?$|"
    # or adverbs ending in -ely
    # e.g. `nicely`
    + "[^e]ely$", flags=re.I)
ADDITIONAL = re.compile(
    # fixes incorrect subtractions from exceptions:
    # smile, scarred, raises, fated
    "[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$"
    # fixes miscellaneous issues:
    # flying, piano, video, prism, fire, fire, evaluate
    + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",
    flags=re.I)
SYLLABLE_DICT = cmudict.dict()

def count_syllables(words):
    """For a string of words, return the number of syllables."""
    def fallback_count_syllables(word):
        """If the cmudict fails, we'll use some vowel-counting regexes."""
        # Count consecutive strings of vowels
        vowel_runs = len(VOWEL_RUNS.findall(word))
        exceptions = len(EXCEPTIONS.findall(word))
        additional = len(ADDITIONAL.findall(word))
        return max(1, vowel_runs - exceptions + additional)

    turn_syllables = 0
    for word in words.split():
        word = word.split('_')[0].split('-')[-1]
        word_syllables = [len(list(phoneme for phoneme in phoneme_lists
                                   if phoneme[-1].isdigit()))
                          for phoneme_lists in SYLLABLE_DICT[word]]
        if word_syllables == []:
            turn_syllables += fallback_count_syllables(word)
        else:
            turn_syllables += word_syllables[0]
    return turn_syllables


def get_existing_data(turn_lengths_fname, tcu_lengths_fname):
    """Return dict of exisiting data on disk."""
    turn_lengths = []
    turn_syllables = []
    turn_words = []
    tcu_lengths = []
    tcu_syllables = []
    tcu_words = []
    tcus_per_turn = []
    with open(turn_lengths_fname, 'r', encoding="UTF-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            turn_lengths.append(float(row["turn_lengths"]))
            turn_words.append(float(row["turn_words"]))
            turn_syllables.append(float(row["turn_syllables"]))
            tcus_per_turn.append(int(row["tcus_per_turn"]))

    with open(tcu_lengths_fname, 'r', encoding="UTF-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            tcu_lengths.append(float(row["tcu_lengths"]))
            tcu_syllables.append(float(row["tcu_syllables"]))
            tcu_words.append(float(row["tcu_words"]))
        return {"tcus_per_turn": tcus_per_turn,
                "turn_lengths": turn_lengths,
                "turn_syllables": turn_syllables,
                "turn_words": turn_words,
                "tcu_lengths": tcu_lengths,
                "tcu_syllables": tcu_syllables,
                "tcu_words": tcu_words}


def make_fresh_data(file_list):
    """Create fresh data dict from file_list."""
    turn_lengths = []
    turn_syllables = []
    turn_words = []
    tcu_lengths = []
    tcu_syllables = []
    tcu_words = []
    tcus_per_turn = []
    for _, file_name in enumerate(file_list):
        with open(file_name, 'r',
                  encoding="UTF-8") as csv_file:
            reader = csv.DictReader(csv_file)
            prev_speaker = None
            tcu_count = 0
            turn_start = None
            current_turn_words = 0
            current_turn_syllables = 0
            for row in reader:
                # Turn lengths
                tcu_start = float(row["start"])
                tcu_end = float(row["end"])
                raw_words = row["words"]
                current_tcu_words = len(raw_words.split())
                current_tcu_syllables = count_syllables(raw_words)
                tcu_words.append(current_tcu_words)
                tcu_syllables.append(current_tcu_syllables)
                tcu_lengths.append(tcu_end - tcu_start)
                if not turn_start:
                    turn_start = tcu_start
                curr_speaker = row["speaker"]
                if prev_speaker == curr_speaker:
                    tcu_count += 1
                    current_turn_words += current_tcu_words
                    current_turn_syllables += current_tcu_syllables
                elif prev_speaker:
                    tcus_per_turn.append(tcu_count)
                    turn_words.append(current_turn_words)
                    turn_syllables.append(current_turn_syllables)
                    current_turn_words = current_tcu_words
                    current_turn_syllables = current_tcu_syllables
                    tcu_count = 1
                    turn_lengths.append(tcu_end - turn_start)
                    turn_start = None
                else:
                    tcu_count = 1
                    current_turn_words = current_tcu_words
                    current_turn_syllables = current_tcu_syllables
                prev_speaker = curr_speaker

    return {"turn_lengths": turn_lengths,
            "turn_words": turn_words,
            "turn_syllables": turn_syllables,
            "tcu_lengths": tcu_lengths,
            "tcu_words": tcu_words,
            "tcu_syllables": tcu_syllables,
            "tcus_per_turn": tcus_per_turn}


def output_new_csvs(turn_lengths_fname, tcu_lengths_fname, data_dict):
    """Write csvs to disk with the current data."""
    turn_lengths = data_dict["turn_lengths"]
    turn_words = data_dict["turn_words"]
    turn_syllables = data_dict["turn_syllables"]
    tcus_per_turn = data_dict["tcus_per_turn"]
    output = "turn_lengths,turn_words,turn_syllables,tcus_per_turn\n"
    for i, _ in enumerate(data_dict["turn_lengths"]):
        output += f"{turn_lengths[i]},{turn_words[i]},"
        output += f"{turn_syllables[i]},{tcus_per_turn[i]}\n"
    with open(turn_lengths_fname, 'w', encoding="UTF-8") as csv_file:
        csv_file.write(output)

    tcu_lengths = data_dict["tcu_lengths"]
    tcu_words = data_dict["tcu_words"]
    tcu_syllables = data_dict["tcu_syllables"]
    output = "tcu_lengths,tcu_words,tcu_syllables\n"
    for i, _ in enumerate(tcu_lengths):
        output += f"{tcu_lengths[i]},{tcu_words[i]},{tcu_syllables[i]}\n"
    with open(tcu_lengths_fname, 'w', encoding="UTF-8") as csv_file:
        csv_file.write(output)


def get_turn_length_data(file_list,
                         turn_lengths_fname="turn_lengths.csv",
                         tcu_lengths_fname="tcu_lengths.csv",
                         refresh=False):
    """
    Return dict of TRP info from `file_list`.

    Given a list of csv files, return a csv file name that contains
    the trps of all the tcu data and the inter/intra speaker status.

    Also output csvs to replicate the pipeline.
    """
    if exists(turn_lengths_fname) and exists(tcu_lengths_fname) and \
       not refresh:
        data_dict = get_existing_data(turn_lengths_fname, tcu_lengths_fname)
    else:
        data_dict = make_fresh_data(file_list)

    if refresh:
        output_new_csvs(turn_lengths_fname, tcu_lengths_fname, data_dict)

    return data_dict


def make_turn_length_model(data, exponential=False):
    """Create the pymc3 model for the turn lengths."""
    observed_turn_length = data["turn_lengths"]
    if exponential:
        with pm.Model() as turn_lengths_model:
            rate = pm.Gamma("rate",
                            mu=0.0002,
                            sd=0.1)
            turn_lengths = pm.Exponential("turn_lengths",
                                          lam=rate,
                                          observed=observed_turn_length)
            turn_length_trace = pm.sample(10000,
                                          tune=4000,
                                          target_accept=0.9,
                                          return_inferencedata=False)
            turn_length_posterior = \
                pm.sampling.sample_posterior_predictive(turn_length_trace,
                                                        keep_size=True)
            model = turn_lengths_model
    else:
        with pm.Model() as turn_lengths_model:
            rate = pm.Gamma("rate",
                            mu=0.0002,
                            sd=0.1)
            shape = pm.Gamma("shape",
                             mu=1,
                             sd=0.7)
            turn_lengths = pm.Gamma("turn_lengths",
                                    alpha=shape,
                                    beta=rate,
                                    observed=observed_turn_length)
            turn_length_trace = pm.sample(10000,
                                          tune=4000,
                                          target_accept=0.9,
                                          return_inferencedata=False)
            turn_length_posterior = \
                pm.sampling.sample_posterior_predictive(turn_length_trace,
                                                        keep_size=True)
            model = turn_lengths_model

    return {"trace": turn_length_trace,
            "posterior": turn_length_posterior,
            "model": model}


def make_turn_length_graph(data, gamma_model, exp_model,
                           figname="turn_lengths.png", show=False):
    """Create visuals for the tcu length data."""
    az.style.use("arviz-darkgrid")

    turn_lengths = list(filter(lambda x: 0 < x < 10000, data["turn_lengths"]))
    az.plot_dist(turn_lengths, label="Data", kind="hist",
                 hist_kwargs={"bins": 20, "align": "right"},
                 color=sns.color_palette()[1])
    az.plot_dist(gamma_model["posterior"]["turn_lengths"],
                 label="Gamma Model",
                 color=sns.color_palette()[0])
    az.plot_dist(exp_model["posterior"]["turn_lengths"],
                 label="Exponential Model",
                 color=sns.color_palette()[2])
    plt.xlim(1, 10500)
    plt.tick_params(axis="y", labelleft=False, left=False)
    plt.xticks(ticks=list(range(1000, 10001, 1000)),
               labels=list(range(1, 11, 1)))
    plt.title("Turn Duration")
    plt.xlabel("Seconds (s)")
    plt.ylabel("Density")
    plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close()

    return figname


def make_turn_words_model(data, exponential=False):
    """Create the pymc3 model for the turn word counts."""
    observed_turn_words = data["turn_words"]
    if exponential:
        with pm.Model() as turn_words_model:
            rate = pm.Gamma("rate",
                            mu=0.07,
                            sd=0.2)
            turn_words = pm.Exponential("turn_words",
                                        lam=rate,
                                        observed=observed_turn_words)
            turn_words_trace = pm.sample(10000,
                                          tune=4000,
                                          target_accept=0.9,
                                          return_inferencedata=False)
            turn_words_posterior = \
                pm.sampling.sample_posterior_predictive(turn_words_trace,
                                                        keep_size=True)
            model = turn_words_model
    else:
        with pm.Model() as turn_words_model:
            shape = pm.Gamma("shape",
                             mu=0.07,
                             sd=0.2)
            rate = pm.Gamma("rate",
                            mu=1,
                            sd=0.4)
            turn_words = pm.Gamma("turn_words",
                                  alpha=shape,
                                  beta=rate,
                                  observed=observed_turn_words)
            turn_words_trace = pm.sample(10000,
                                         tune=4000,
                                         target_accept=0.9,
                                         return_inferencedata=False)
            turn_words_posterior = \
                pm.sampling.sample_posterior_predictive(turn_words_trace,
                                                        keep_size=True)
            model = turn_words_model

    return {"trace": turn_words_trace,
            "posterior": turn_words_posterior,
            "model": model}


def make_turn_words_graph(data, gamma_model, exp_model,
                          figname="turn_words.png", show=False):
    """Create visuals for the tcu length data."""
    az.style.use("arviz-darkgrid")

    turn_words = list(filter(lambda x: 0 < x < 50, data["turn_words"]))
    az.plot_dist(turn_words, label="Data", kind="hist",
                 hist_kwargs={"bins": 25, "align": "mid"},
                 color=sns.color_palette()[1])
    az.plot_dist(gamma_model["posterior"]["turn_words"],
                 label="Gamma Model",
                 color=sns.color_palette()[0])
    az.plot_dist(exp_model["posterior"]["turn_words"],
                 label="Exponential Model",
                 color=sns.color_palette()[2])
    plt.xlim(1, 50)
    plt.tick_params(axis="y", labelleft=False, left=False)
    plt.title("Turn Word Count")
    plt.xticks(ticks=list(range(1, 50, 10)),
               labels=list(range(1, 50, 10)))
    plt.xlabel("Number of Words")
    plt.ylabel("Density")
    plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close()

    return figname


def make_turn_syllables_model(data, exponential=False):
    """Create the pymc3 model for the turn syllable counts."""
    observed_turn_syllables = data["turn_syllables"]
    if exponential:
        with pm.Model() as turn_syllables_model:
            rate = pm.Gamma("rate",
                            mu=0.07,
                            sd=0.2)
            turn_syllables = pm.Exponential("turn_syllables",
                                        lam=rate,
                                        observed=observed_turn_syllables)
            turn_syllables_trace = pm.sample(10000,
                                             tune=4000,
                                             target_accept=0.9,
                                             return_inferencedata=False)
            turn_syllables_posterior = \
                pm.sampling.sample_posterior_predictive(turn_syllables_trace,
                                                        keep_size=True)
            model = turn_syllables_model
    else:
        with pm.Model() as turn_syllables_model:
            shape = pm.Gamma("shape",
                             mu=0.07,
                             sd=0.2)
            rate = pm.Gamma("rate",
                            mu=1,
                            sd=0.4)
            turn_syllables = pm.Gamma("turn_syllables",
                                      alpha=shape,
                                      beta=rate,
                                      observed=observed_turn_syllables)
            turn_syllables_trace = pm.sample(10000,
                                             tune=4000,
                                             target_accept=0.9,
                                             return_inferencedata=False)
            turn_syllables_posterior = \
                pm.sampling.sample_posterior_predictive(turn_syllables_trace,
                                                        keep_size=True)
            model = turn_syllables_model

    return {"trace": turn_syllables_trace,
            "posterior": turn_syllables_posterior,
            "model": model}


def make_turn_syllables_graph(data, gamma_model, exp_model,
                              figname="turn_syllables.png",
                              show=False):
    """Create visuals for the tcu length data."""
    az.style.use("arviz-darkgrid")

    turn_syllables = list(filter(lambda x: 0 < x < 50, data["turn_syllables"]))
    az.plot_dist(turn_syllables, label="Data", kind="hist",
                 hist_kwargs={"bins": 25, "align": "mid"},
                 color=sns.color_palette()[1])
    az.plot_dist(gamma_model["posterior"]["turn_syllables"],
                 label="Gamma Model",
                 color=sns.color_palette()[0])
    az.plot_dist(exp_model["posterior"]["turn_syllables"],
                 label="Exponential Model",
                 color=sns.color_palette()[2])
    plt.xlim(1, 50)
    plt.tick_params(axis="y", labelleft=False, left=False)
    plt.title("Turn Syllable Count")
    plt.xticks(ticks=list(range(1, 50, 10)),
               labels=list(range(1, 50, 10)))
    plt.xlabel("Number of Syllables")
    plt.ylabel("Density")
    plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close()

    return figname


def make_tcu_length_model(data, exponential=False):
    """Create the pymc3 model for the tcu lengths."""
    observed_tcu_length = data["tcu_lengths"]
    if exponential:
        with pm.Model() as tcu_length_model:
            rate = pm.Gamma("rate",
                            mu=0.0005,
                            sd=0.1)
            tcu_lengths = pm.Exponential("tcu_lengths",
                                        lam=rate,
                                        observed=observed_tcu_length)
            tcu_length_trace = pm.sample(10000,
                                         tune=4000,
                                         target_accept=0.9,
                                         return_inferencedata=False)
            tcu_length_posterior = \
                pm.sampling.sample_posterior_predictive(tcu_length_trace,
                                                        keep_size=True)
            model = tcu_length_model
    else:
        with pm.Model() as tcu_model:
            rate = pm.Gamma("rate",
                            mu=0.0005,
                            sd=0.1)
            shape = pm.Gamma("shape",
                             mu=1,
                             sd=0.7)
            tcu_lengths = pm.Gamma("tcu_lengths",
                                   alpha=shape,
                                   beta=rate,
                                   observed=observed_tcu_length)
            tcu_length_trace = pm.sample(10000,
                                         tune=4000,
                                         target_accept=0.9,
                                         return_inferencedata=False)
            tcu_length_posterior = \
                pm.sampling.sample_posterior_predictive(tcu_length_trace,
                                                        keep_size=True)
            model = tcu_model

    return {"trace": tcu_length_trace,
            "posterior": tcu_length_posterior,
            "model": model}


def make_tcu_length_graph(data, gamma_model, exp_model,
                          figname="tcu_lengths.png", show=False):
    """Create visuals for the tcu length data."""
    az.style.use("arviz-darkgrid")

    tcu_lengths = list(filter(lambda x: 0 < x < 10000, data["tcu_lengths"]))
    az.plot_dist(tcu_lengths, label="Data", kind="hist",
                 hist_kwargs={"bins": 20, "align": "right"},
                 color=sns.color_palette()[1])
    az.plot_dist(gamma_model["posterior"]["tcu_lengths"],
                 label="Gamma Model",
                 color=sns.color_palette()[0])
    az.plot_dist(exp_model["posterior"]["tcu_lengths"],
                 label="Exponential Model",
                 color=sns.color_palette()[2])
    plt.xlim(1, 10500)
    plt.title("Turn Construction Unit Duration")
    plt.xlabel("Seconds (s)")
    plt.xticks(ticks=list(range(1000, 10001, 1000)),
               labels=list(range(1, 11, 1)))
    plt.tick_params(axis="y", labelleft=False, left=False)
    plt.ylabel("Density")
    plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close()

    return figname


def make_tcu_words_model(data, exponential=False):
    """Create the pymc3 model for the tcu lengths."""
    observed_tcu_words = data["tcu_words"]
    if exponential:
        with pm.Model() as tcu_words_model:
            rate = pm.Gamma("rate",
                            mu=0.13,
                            sd=0.4)
            tcu_words = pm.Exponential("tcu_words",
                                        lam=rate,
                                        observed=observed_tcu_words)
            tcu_words_trace = pm.sample(10000,
                                        tune=4000,
                                        target_accept=0.9,
                                        return_inferencedata=False)
            tcu_words_posterior = \
                pm.sampling.sample_posterior_predictive(tcu_words_trace,
                                                        keep_size=True)
            model = tcu_words_model
    else:
        with pm.Model() as tcu_words_model:
            rate = pm.Gamma("rate",
                            mu=0.13,
                            sd=0.4)
            shape = pm.Gamma("shape",
                             mu=1,
                             sd=0.4)
            tcu_words = pm.Gamma("tcu_words",
                                 alpha=shape,
                                 beta=rate,
                                 observed=observed_tcu_words)
            tcu_words_trace = pm.sample(10000,
                                        tune=4000,
                                        target_accept=0.9,
                                        return_inferencedata=False)
            tcu_words_posterior = \
                pm.sampling.sample_posterior_predictive(tcu_words_trace,
                                                        keep_size=True)
            model = tcu_words_model

    return {"trace": tcu_words_trace,
            "posterior": tcu_words_posterior,
            "model": model}


def make_tcu_words_graph(data, gamma_model, exp_model,
                         figname="tcu_words.png", show=False):
    """Create visuals for the tcu words data."""
    az.style.use("arviz-darkgrid")

    tcu_words = list(filter(lambda x: 0 < x < 30, data["tcu_words"]))
    az.plot_dist(tcu_words, label="Data", kind="hist",
                 hist_kwargs={"bins": 15, "align": "mid"},
                 color=sns.color_palette()[1])
    az.plot_dist(gamma_model["posterior"]["tcu_words"],
                 label="Gamma Model",
                 color=sns.color_palette()[0])
    az.plot_dist(exp_model["posterior"]["tcu_words"],
                 label="Exponential Model",
                 color=sns.color_palette()[2])
    plt.xlim(1, 30)
    plt.title("Turn Construction Unit Word Count")
    plt.xlabel("Number of Words")
    plt.xticks(ticks=list(range(1, 30, 5)),
               labels=list(range(1, 30, 5)))
    plt.tick_params(axis="y", labelleft=False, left=False)
    plt.ylabel("Density")
    plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close()

    return figname


def make_tcu_syllables_model(data, exponential=False):
    """Create the pymc3 model for the tcu syllable counts."""
    observed_tcu_syllables = data["tcu_syllables"]
    if exponential:
        with pm.Model() as tcu_syllables_model:
            rate = pm.Gamma("rate",
                            mu=0.07,
                            sd=0.2)
            tcu_syllables = pm.Exponential("tcu_syllables",
                                           lam=rate,
                                           observed=observed_tcu_syllables)
            tcu_syllables_trace = pm.sample(10000,
                                            tune=4000,
                                            target_accept=0.9,
                                            return_inferencedata=False)
            tcu_syllables_posterior = \
                pm.sampling.sample_posterior_predictive(tcu_syllables_trace,
                                                        keep_size=True)
            model = tcu_syllables_model
    else:
        with pm.Model() as tcu_syllables_model:
            shape = pm.Gamma("shape",
                             mu=0.07,
                             sd=0.2)
            rate = pm.Gamma("rate",
                            mu=1,
                            sd=0.4)
            tcu_syllables = pm.Gamma("tcu_syllables",
                                     alpha=shape,
                                     beta=rate,
                                     observed=observed_tcu_syllables)
            tcu_syllables_trace = pm.sample(10000,
                                            tune=4000,
                                            target_accept=0.9,
                                            return_inferencedata=False)
            tcu_syllables_posterior = \
                pm.sampling.sample_posterior_predictive(tcu_syllables_trace,
                                                        keep_size=True)
            model = tcu_syllables_model

    return {"trace": tcu_syllables_trace,
            "posterior": tcu_syllables_posterior,
            "model": model}


def make_tcu_syllables_graph(data, gamma_model, exp_model,
                             figname="tcu_syllables.png", show=False):
    """Create visuals for the tcu syllable data."""
    az.style.use("arviz-darkgrid")

    tcu_syllables = list(filter(lambda x: 0 < x < 30, data["tcu_syllables"]))
    az.plot_dist(tcu_syllables, label="Data", kind="hist",
                 hist_kwargs={"bins": 15, "align": "mid"},
                 color=sns.color_palette()[1])
    az.plot_dist(gamma_model["posterior"]["tcu_syllables"],
                 label="Gamma Model",
                 color=sns.color_palette()[0])
    az.plot_dist(exp_model["posterior"]["tcu_syllables"],
                 label="Exponential Model",
                 color=sns.color_palette()[2])
    plt.xlim(1, 30)
    plt.title("Turn Construction Unit Syllable Count")
    plt.xlabel("Number of Syllables")
    plt.xticks(ticks=list(range(1, 30, 5)),
               labels=list(range(1, 30, 5)))
    plt.tick_params(axis="y", labelleft=False, left=False)
    plt.ylabel("Density")
    plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close()

    return figname


def make_tcus_per_turn_model_geometric(data):
    """Fit a pymc3 geometric model to the tcus per turn data."""
    tcus_per_turn = data["tcus_per_turn"]
    with pm.Model() as tcu_geometric_model:
        prob_success = pm.Beta("prob_success",
                               alpha=1,
                               beta=1)

        tcus_per_turn = pm.Geometric("tcus_per_turn",
                                     p=prob_success,
                                     observed=tcus_per_turn)

        tcu_trace = pm.sample(10000,
                              tune=6000,
                              target_accept=0.9,
                              return_inferencedata=False)
        tcu_posterior = pm.sampling.sample_posterior_predictive(tcu_trace,
                                                                keep_size=True)
        model = tcu_geometric_model

    return {"trace": tcu_trace,
            "posterior": tcu_posterior,
            "model": model}


def make_tcus_per_turn_model_negative_binomial(data):
    """Fit a pymc3 negative binomial model to the tcus per turn data."""
    tcus_per_turn = data["tcus_per_turn"]
    with pm.Model() as tcu_neg_bin_model:
        prob_success = pm.Beta("prob_success",
                               alpha=1,
                               beta=1)
        num_trials = pm.Gamma("num_trials",
                              mu=6.0,
                              sd=2.0)
        tcus_per_turn = pm.NegativeBinomial("tcus_per_turn",
                                            p=prob_success,
                                            n=num_trials,
                                            observed=tcus_per_turn)

        tcu_trace = pm.sample(10000,
                              tune=6000,
                              target_accept=0.9,
                              return_inferencedata=False)
        tcu_posterior = pm.sampling.sample_posterior_predictive(tcu_trace,
                                                                keep_size=True)
        model = tcu_neg_bin_model

    return {"trace": tcu_trace,
            "posterior": tcu_posterior,
            "model": model}


def make_tcus_per_turn_graph(data, geo_model, bin_model,
                             figname="tcus_per_turn.png", show=False):
    """ Create graphs of TCU per Turn data."""
    az.style.use("arviz-darkgrid")

    geo_posterior_samples = []
    for posterior_list in geo_model["posterior"]["tcus_per_turn"][0]:
        for i in range(50):
            geo_posterior_samples.append(posterior_list[i])
    bin_posterior_samples = []
    for posterior_list in bin_model["posterior"]["tcus_per_turn"][0]:
        for i in range(50):
            bin_posterior_samples.append(posterior_list[i])

    sns.histplot([bin_posterior_samples, data["tcus_per_turn"],
                      geo_posterior_samples],
                     binwidth=1, binrange=(1, 10), discrete=True,
                     stat="density", multiple="dodge", common_norm=False)
    plt.legend(labels=["Geometric Model", "Data", "Negative Binomial Model"])
    plt.title("TCUs per Turn")
    plt.xlabel("Number of TCUs per Turn")
    plt.ylabel("Percent")
    plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close()

    return figname

def display_corpus_params(fname_list, data):
    """Print corpus stats to the console."""
    print("|---------|-------|")
    print(f"|Conversation Count | \t{len(fname_list)}|")
    print(f"|Turn Count         | \t{len(data['turn_lengths'])}|")
    print(f"|TCU Count          | \t{len(data['tcu_lengths'])}|")
    print("|---------|-------|")
    return True


def display_gamma_model_stats(data, model, key):
    """Print model stats for the key in data to the console."""
    data_mean = np.array(data[key]).mean()
    data_std = np.array(data[key]).std()
    model_mean = model["posterior"][key].mean()
    model_std = model["posterior"][key].std()
    rate_mean = model["trace"]["rate"].mean()
    rate_std = model["trace"]["rate"].std()
    shape_mean = model["trace"]["shape"].mean()
    shape_std = model["trace"]["shape"].std()
    print(f"| {key} Data Mean     |\t{data_mean:.2E} |")
    print(f"| {key} Model Mean    |\t{model_mean:.2E} |")
    print(f"| {key} Data Std Dev  |\t{data_std:.2E} |")
    print(f"| {key} Model Std Dev |\t{model_std:.2E} |")
    print(f"| {key} Rate Mean     |\t{rate_mean:.2E} |")
    print(f"| {key} Rate Std Dev  |\t{rate_std:.2E} |")
    print(f"| {key} Shape Mean    |\t{shape_mean:.2E} |")
    print(f"| {key} Shape Std Dev |\t{shape_std:.2E} |")
    print("|--------------|---------|")

    return True


def display_exp_model_stats(data, model, key):
    """Print model stats for the key in data to the console."""
    data_mean = np.array(data[key]).mean()
    data_std = np.array(data[key]).std()
    model_mean = model["posterior"][key].mean()
    model_std = model["posterior"][key].std()
    rate_mean = model["trace"]["rate"].mean()
    rate_std = model["trace"]["rate"].std()
    print(f"| {key} Data Mean     |\t{data_mean:.2E} |")
    print(f"| {key} Model Mean    |\t{model_mean:.2E} |")
    print(f"| {key} Data Std Dev  |\t{data_std:.2E} |")
    print(f"| {key} Model Std Dev |\t{model_std:.2E} |")
    print(f"| {key} Rate Mean     |\t{rate_mean:.2E} |")
    print(f"| {key} Rate Std Dev  |\t{rate_std:.2E} |")
    print("|--------------|---------|")

    return True


def main():
    """Create models, graphs, and data csv. Report findings."""
    fname_list = filter_data.get_data()
    data = get_turn_length_data(fname_list, refresh=True)

    show = True
    turns = False
    tcus = False
    tcus_per_turn = True

    if turns:
        turn_length_gamma_model = make_turn_length_model(data)
        turn_length_exp_model = make_turn_length_model(data, exponential=True)
        make_turn_length_graph(data, turn_length_gamma_model,
                               turn_length_exp_model, show=show)

        turn_words_gamma_model = make_turn_words_model(data)
        turn_words_exp_model = make_turn_words_model(data, exponential=True)
        make_turn_words_graph(data, turn_words_gamma_model,
                              turn_words_exp_model, show=show)

        turn_syllables_gamma_model = make_turn_syllables_model(data)
        turn_syllables_exp_model = make_turn_syllables_model(data,
                                                             exponential=True)
        make_turn_syllables_graph(data, turn_syllables_gamma_model,
                                  turn_syllables_exp_model, show=show)

        display_corpus_params(fname_list, data)
        print(az.summary(turn_length_gamma_model["trace"]))
        print(az.summary(turn_length_exp_model["trace"]))
        print(az.summary(turn_words_gamma_model["trace"]))
        print(az.summary(turn_words_exp_model["trace"]))
        print(az.summary(turn_syllables_gamma_model["trace"]))
        print(az.summary(turn_syllables_exp_model["trace"]))

        display_gamma_model_stats(data, turn_length_gamma_model,
                                  "turn_lengths")
        display_exp_model_stats(data, turn_length_exp_model,
                                "turn_lengths")
        display_gamma_model_stats(data, turn_words_gamma_model,
                                  "turn_words")
        display_exp_model_stats(data, turn_words_exp_model,
                                "turn_words")
        display_gamma_model_stats(data, turn_syllables_gamma_model,
                                  "turn_syllables")
        display_exp_model_stats(data, turn_syllables_exp_model,
                                "turn_syllables")

        print(az.compare({"Gamma": turn_length_gamma_model["trace"],
                          "Exponential": turn_length_exp_model["trace"]},
                         ic="waic", scale="deviance"))
        print(az.compare({"Gamma": turn_words_gamma_model["trace"],
                          "Exponential": turn_words_exp_model["trace"]},
                         ic="waic", scale="deviance"))
        print(az.compare({"Gamma": turn_syllables_gamma_model["trace"],
                          "Exponential": turn_syllables_exp_model["trace"]},
                         ic="waic", scale="deviance"))
    elif tcus:
        tcu_length_gamma_model = make_tcu_length_model(data)
        tcu_length_exp_model = make_tcu_length_model(data, exponential=True)
        make_tcu_length_graph(data, tcu_length_gamma_model,
                              tcu_length_exp_model, show=show)

        tcu_words_gamma_model = make_tcu_words_model(data)
        tcu_words_exp_model = make_tcu_words_model(data, exponential=True)
        make_tcu_words_graph(data, tcu_words_gamma_model,
                             tcu_words_exp_model, show=show)

        tcu_syllables_gamma_model = make_tcu_syllables_model(data)
        tcu_syllables_exp_model = make_tcu_syllables_model(data,
                                                           exponential=True)
        make_tcu_syllables_graph(data, tcu_syllables_gamma_model,
                                 tcu_syllables_exp_model, show=show)

        display_corpus_params(fname_list, data)
        print(az.summary(tcu_length_gamma_model["trace"]))
        print(az.summary(tcu_length_exp_model["trace"]))
        print(az.summary(tcu_words_gamma_model["trace"]))
        print(az.summary(tcu_words_exp_model["trace"]))
        print(az.summary(tcu_syllables_gamma_model["trace"]))
        print(az.summary(tcu_syllables_exp_model["trace"]))
        display_gamma_model_stats(data, tcu_length_gamma_model,
                                  "tcu_lengths")
        display_exp_model_stats(data, tcu_length_exp_model,
                                "tcu_lengths")
        display_gamma_model_stats(data, tcu_words_gamma_model,
                                  "tcu_words")
        display_exp_model_stats(data, tcu_words_exp_model,
                                "tcu_words")
        display_gamma_model_stats(data, tcu_syllables_gamma_model,
                                  "tcu_syllables")
        display_exp_model_stats(data, tcu_syllables_exp_model,
                                "tcu_syllables")

        print(az.compare({"Gamma": tcu_length_gamma_model["trace"],
                          "Exponential": tcu_length_exp_model["trace"]},
                         ic="waic", scale="deviance"))
        print(az.compare({"Gamma": tcu_words_gamma_model["trace"],
                          "Exponential": tcu_words_exp_model["trace"]},
                         ic="waic", scale="deviance"))
        print(az.compare({"Gamma": tcu_syllables_gamma_model["trace"],
                          "Exponential": tcu_syllables_exp_model["trace"]},
                         ic="waic", scale="deviance"))

    elif tcus_per_turn:
        tcus_per_turn_geo_model = make_tcus_per_turn_model_geometric(data)
        tcus_per_turn_bin_model = make_tcus_per_turn_model_negative_binomial(data)
        #return tcus_per_turn_geo_model, tcus_per_turn_bin_model
        make_tcus_per_turn_graph(data, tcus_per_turn_geo_model,
                                 tcus_per_turn_bin_model, show=show)

        display_corpus_params(fname_list, data)
        print(az.summary(tcus_per_turn_geo_model["trace"]))
        print(az.summary(tcus_per_turn_bin_model["trace"]))

        tcu_per_turn_data_mean = np.array(data["tcus_per_turn"]).mean()
        tcu_per_turn_geo_model_mean = \
            tcus_per_turn_geo_model["posterior"]["tcus_per_turn"].mean()
        tcu_per_turn_bin_model_mean = \
            tcus_per_turn_bin_model["posterior"]["tcus_per_turn"].mean()
        tcu_per_turn_data_std = np.array(data["tcus_per_turn"]).std()
        tcu_per_turn_geo_model_std = \
            tcus_per_turn_geo_model["posterior"]["tcus_per_turn"].std()
        tcu_per_turn_bin_model_std = \
            tcus_per_turn_bin_model["posterior"]["tcus_per_turn"].std()
        tcu_per_turn_geo_p_mean = tcus_per_turn_geo_model["trace"]["prob_success"].mean()
        tcu_per_turn_bin_p_mean = tcus_per_turn_bin_model["trace"]["prob_success"].mean()
        tcu_per_turn_geo_p_std = tcus_per_turn_geo_model["trace"]["prob_success"].std()
        tcu_per_turn_bin_p_std = tcus_per_turn_bin_model["trace"]["prob_success"].std()
        tcu_per_turn_bin_n_mean = tcus_per_turn_bin_model["trace"]["num_trials"].mean()
        tcu_per_turn_bin_n_std = tcus_per_turn_bin_model["trace"]["num_trials"].std()
        print(f"| TCUs per Turn Data Mean                 |", end="")
        print(f"{tcu_per_turn_data_mean:.2E} |")
        print("| TCUs per Turn Geometric Model Mean       |", end="")
        print(f"{tcu_per_turn_geo_model_mean:.2E} |")
        print("| TCUs per Turn Neg Binomial Model Mean    |", end="")
        print(f"{tcu_per_turn_bin_model_mean:.2E} |")
        print("| TCUs per Turn Data Std Dev               |", end="")
        print(f"{tcu_per_turn_data_std:.2E} |")
        print("| TCUs per Turn Geometric Model Std Dev    |", end="")
        print(f"{tcu_per_turn_geo_model_std:.2E} |")
        print("| TCUs per Turn Neg Binomial Model Std Dev |", end="")
        print(f"{tcu_per_turn_bin_model_std:.2E} |")
        print("| TCUs per Turn Geometric p Mean           |", end="")
        print(f"{tcu_per_turn_geo_p_mean:.2E} |")
        print("| TCUs per Turn Neg Binomial p Mean        |", end="")
        print(f"{tcu_per_turn_bin_p_mean:.2E} |")
        print("| TCUs per Turn Geometric p Std Dev        |", end="")
        print(f"{tcu_per_turn_geo_p_std:.2E} |")
        print("| TCUs per Turn Neg Binomial p Std Dev     | ", end="")
        print(f"{tcu_per_turn_bin_p_std:.2E} |")
        print("| TCUs per Turn Neg Binomial n Std Dev     | ", end="")
        print(f"{tcu_per_turn_bin_n_mean:.2E} |")
        print("| TCUs per Turn Neg Binomial n Std Dev     | ", end="")
        print(f"{tcu_per_turn_bin_n_std:.2E} |")

        print(az.compare({"NegativeBinomial": tcus_per_turn_bin_model["trace"],
                          "Geometric": tcus_per_turn_geo_model["trace"]},
                         ic="waic", scale="deviance"))

if __name__ == "__main__":
    pass

from pymc3.model import modelcontext
import scipy.stats as st
from numpy import dot
from scipy.linalg import cholesky as chol
import warnings
import arviz as az
import numpy as np

def marginal_lik(mtrace, model=None, logp=None, maxiter=10000):
    """The Bridge Sampling Estimator of the Marginal Likelihood.

    Parameters
    ----------
    mtrace : MultiTrace, result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    logp : Model log-probability function, read from the model by default
    maxiter : Maximum number of iterations

    Returns
    -------
    marg_lik : Estimated Marginal log-likelihood
    """

    r0, tol1, tol2 = 0.5, 1e-10, 1e-4

    model = modelcontext(model)
    if logp is None:
        logp = model.logp_array
    vars = model.free_RVs

    # Split the samples into two parts
    # Use the first 50% for fitting the proposal distribution and the second 50%
    # in the iterative scheme.
    len_trace = len(mtrace)
    nchain = mtrace.nchains

    N1_ = len_trace // 2
    N1 = N1_*nchain
    N2 = len_trace*nchain - N1

    neff_list = dict() # effective sample size, a dict of ess for each var

    arraysz = model.bijection.ordering.size
    samples_4_fit = np.zeros((arraysz, N1))
    samples_4_iter = np.zeros((arraysz, N2))

    # matrix with already transformed samples.
    for var in vars:
        varmap = model.bijection.ordering.by_name[var.name]
        # for fitting the proposal
        x = mtrace[:N1_][var.name] # Getting N1 samples from each chain
        samples_4_fit[varmap.slc, : ] = x.reshape((x.shape[0],
                                                   np.prod(x.shape[1:], dtype=int))).T # What is .slc? I guess gives indices for the variables??
        # for the iterative scheme
        x2 = mtrace[N1_:][var.name]
        samples_4_iter[varmap.slc, :] = x2.reshape((x2.shape[0],
                                                    np.prod(x2.shape[1:], dtype=int))).T
        # effective sample size of samples_4_iter, scalar
        neff_list.update({var.name: az.ess(x)})

    # median effective sample size (scalar)
    neff = np.median(list(neff_list.values()))

    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    V = np.cov(samples_4_fit)
    L = chol(V, lower=True)

    # Draw N2 samples from the proposal distribution
    gen_samples = m[:, None] + dot(L, st.norm.rvs(0, 1,
                                                  size=samples_4_iter.shape))

    # Evaluate proposal distribution for posterior & generated samples
    q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, V)
    q22 = st.multivariate_normal.logpdf(gen_samples.T, m, V)

    # Evaluate unnormalized posterior for posterior & generated sampels
    q11 = np.asarray([logp(point) for point in samples_4_iter.T])
    q21 = np.asarray([logp(point) for point in gen_samples.T])

    # Iterative scheme as proposed in Meng and Wong (1996) to estimate
    # the marginal likelihood
    def iterative_scheme(q11, q12, q21, q22, r0, neff, tol, maxiter, criterion):
        l1 = q11 - q12
        l2 = q21 - q22
        lstar = np.median(l1) # To increase numerical stability
                              # subtracting the median of l1 from l1 & l2 later
        s1 = neff/(neff + N2)
        s2 = N2/(neff + N2)

        r = r0
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol

        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            rold = r
            logmlold = logml
            numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
            deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
            if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                warnings.warn("""Infinite value in iterative scheme, returning NaN.
                Try rerunning with more samples.""")
            r = (N1/N2) * np.sum(numi)/np.sum(deni)
            r_vals.append(r)
            logml = np.log(r) + lstar
            i += 1
            if criterion == 'r':
                criterion_val = np.abs((r - rold)/r)
            elif criterion== 'logml':
                criterion_val = np.abs((logml - logmlold)/logml)

        if i >= maxiter:
            return dict(logml = np.Nan, niter = i, r_vals = np.asarray(r_vals))
        else:
            return dict(logml = logml, niter = i)

    # Run iterative scheme:
    tmp = iterative_scheme(q11, q12, q21, q22, r0, neff, tol1, maxiter, 'r')
    if ~np.isfinite(tmp['logml']):
        warnings.warn("""logml could not be estimated within maxiter, rerunning with
                      adjusted starting value. Estimate might be more variable than usual.""")
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'logml')

    return dict(logml = tmp['logml'], niter = tmp['niter'], method = "normal",
                q11 = q11, q12 = q12, q21 = q21, q22 = q22)


def bayes_factor(trace1, trace2, model1, model2):
    """
    A bridge sampling estimate of the Bayes factor p(y|model1) / p(y|model2)

    Parameters:
    -----------
    trace1 : an InferenceData object
        Posterior samples from model1
    trace2 : an InferenceData object
        Posterior samples from model2
    model1 : PyMC model
    model2 : PyMC model

    Returns:
    --------
    The estimated Bayes Factor (scalar)

    Notes:
    ------
    The user needs to ensure the traces are sampled using the same observed data
    """

    log_marginal_likelihood1 = marginal_lik(model=model1, mtrace=trace1)['logml']
    log_marginal_likelihood2 = marginal_lik(model=model2, mtrace=trace2)['logml']
    return np.exp(log_marginal_likelihood1 - log_marginal_likelihood2)

fname_list = filter_data.get_data()
data = get_turn_length_data(fname_list, refresh=True)
# turn_length_gamma_model = make_turn_length_model(data)
# turn_length_exp_model = make_turn_length_model(data, exponential=True)
# BF = bayes_factor(turn_length_gamma_model["trace"],
#                   turn_length_exp_model["trace"],
#                   turn_length_gamma_model["model"],
#                   turn_length_exp_model["model"])
# print(f"The estimated Bayes factor for turn length in ms is {BF}")
# => 2.88e72

# turn_words_gamma_model = make_turn_words_model(data)
# turn_words_exp_model = make_turn_words_model(data, exponential=True)
# BF = bayes_factor(turn_words_gamma_model["trace"],
#                   turn_words_exp_model["trace"],
#                   turn_words_gamma_model["model"],
#                   turn_words_exp_model["model"])
# print(f"The estimated Bayes factor for turn length in words is {BF}")
# => 1.99e99

# turn_syllables_gamma_model = make_turn_syllables_model(data)
# turn_syllables_exp_model = make_turn_syllables_model(data, exponential=True)
# BF = bayes_factor(turn_syllables_gamma_model["trace"],
#                   turn_syllables_exp_model["trace"],
#                   turn_syllables_gamma_model["model"],
#                   turn_syllables_exp_model["model"])
# print(f"The estimated Bayes factor for turn length in syllables is {BF}")
# => 2.66e156

# tcu_length_gamma_model = make_tcu_length_model(data)
# tcu_length_exp_model = make_tcu_length_model(data, exponential=True)
# BF = bayes_factor(tcu_length_gamma_model["trace"],
#                   tcu_length_exp_model["trace"],
#                   tcu_length_gamma_model["model"],
#                   tcu_length_exp_model["model"])
# print(f"The estimated Bayes factor for tcu length in ms is {BF}")
# => 0.0301

# tcu_syllables_gamma_model = make_tcu_syllables_model(data)
# tcu_syllables_exp_model = make_tcu_syllables_model(data, exponential=True)
# BF = bayes_factor(tcu_length_gamma_model["trace"], tcu_length_exp_model["trace"],tcu_length_gamma_model["model"],tcu_length_exp_model["model"])
# print(f"The estimated Bayes factor for tcu length in syllables is {BF}")
# => 0.0258

# tcu_words_gamma_model = make_tcu_words_model(data)
# tcu_words_exp_model = make_tcu_words_model(data, exponential=True)
# BF = bayes_factor(tcu_words_gamma_model["trace"],
#                   tcu_words_exp_model["trace"],
#                   tcu_words_gamma_model["model"],
#                   tcu_words_exp_model["model"])
# print(f"The estimated Bayes factor for tcu length in words is {BF}")
# => 3.39e40

# tcus_per_turn_geometric_model = make_tcus_per_turn_model_geometric(data)
# tcus_per_turn_neg_bin_model = make_tcus_per_turn_model_negative_binomial(data)
# BF = bayes_factor(tcus_per_turn_geometric_model["trace"],
#                   tcus_per_turn_neg_bin_model["trace"],
#                   tcus_per_turn_geometric_model["model"],
#                   tcus_per_turn_neg_bin_model["model"])
# print(f"The estimated Bayes factor for tcus per turn is {BF}")
# => inf



tcu_length_gamma_model = make_tcu_length_model(data)
tcu_length_exp_model = make_tcu_length_model(data, exponential=True)
