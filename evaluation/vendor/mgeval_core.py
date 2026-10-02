# coding: utf-8
"""Vendored subset of MGEval's feature/metric code.

Upstream:
  * original (Python 2): https://github.com/RichardYang40148/mgeval
  * Python 3 port:       https://github.com/atsukoba/mgeval-python3

Paper:
  Li-Chia Yang and Alexander Lerch, "On the evaluation of generative models in
  music", Neural Computing and Applications, 2020.

Neither upstream repository carries a LICENSE file, so there are no explicit
redistribution terms. This copy exists for internal research use only; cite the
paper above. Do not ship it in an open-source release without first checking
with the original authors.

Trimmed to the ``Feature`` container and the ``MgevalMetrics`` methods this
project actually calls; ``extract_feature`` (which reads MIDI from disk) and the
per-bar metrics were dropped, since the adapter builds features in memory.
"""

import numpy as np
import python3_midi as midi
from pretty_midi import Instrument, PrettyMIDI
from python3_midi import Pattern


class Feature:
    def __init__(self, pretty_midi: PrettyMIDI, midi_pattern: Pattern, **kwargs):
        self.pretty_midi = pretty_midi
        self.midi_pattern = midi_pattern
        for k, v in kwargs.items():
            self.__setattr__(k, v)


# musically informed objective measures.
class MgevalMetrics(object):

    def __init__(self):
        pass

    def total_used_pitch(self, feature: Feature, track_num: int = 0):
        """
        total_used_pitch (Pitch count): The number of different pitches within a sample.

        Returns:
        "used_pitch": pitch count, scalar for each sample.
        """
        inst: Instrument = feature.pretty_midi.instruments[track_num]
        piano_roll = inst.get_piano_roll(fs=100)
        sum_notes = np.sum(piano_roll, axis=1)
        used_pitch = np.sum(sum_notes > 0)
        return used_pitch

    def total_pitch_class_histogram(self, feature: Feature, track_num: int = 0):
        """
        total_pitch_class_histogram (Pitch class histogram):
        The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
        In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

        Returns:
        "histogram": histrogram of 12 pitch, with weighted duration shape 12
        """
        inst: Instrument = feature.pretty_midi.instruments[track_num]
        piano_roll = inst.get_piano_roll(fs=100)
        histogram = np.zeros(12)
        for i in range(0, 128):
            pitch_class = i % 12
            histogram[pitch_class] += np.sum(piano_roll, axis=1)[i]
        histogram = histogram / sum(histogram)
        return histogram

    def pitch_class_transition_matrix(self, feature: Feature, normalize=0, track_num: int = 0):
        """
        pitch_class_transition_matrix (Pitch class transition matrix):
        The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
        The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.

        Args:
        "normalize" : If set to 0, return transition without normalization.
                      If set to 1, normalizae by row.
                      If set to 2, normalize by entire matrix sum.
        Returns:
        "transition_matrix": shape of [12, 12], transition_matrix of 12 x 12.
        """
        inst: Instrument = feature.pretty_midi.instruments[track_num]
        transition_matrix: np.ndarray = inst.get_pitch_class_transition_matrix()

        if normalize == 0:
            return transition_matrix

        elif normalize == 1:
            sums: np.ndarray = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1, 1)

        elif normalize == 2:
            return transition_matrix / sum(sum(transition_matrix))

        else:
            print("invalid normalization mode, return unnormalized matrix")
            return transition_matrix

    def pitch_range(self, feature: Feature, track_num: int = 0):
        """
        pitch_range (Pitch range):
        The pitch range is calculated by subtraction of the highest and lowest used pitch in semitones.

        Returns:
        "p_range": a scalar for each sample.
        """
        inst: Instrument = feature.pretty_midi.instruments[track_num]
        piano_roll = inst.get_piano_roll(fs=100)
        pitch_index = np.where(np.sum(piano_roll, axis=1) > 0)
        p_range = np.max(pitch_index) - np.min(pitch_index)
        return p_range

    def avg_pitch_shift(self, feature: Feature, track_num: int = 0):
        """
        avg_pitch_shift (Average pitch interval):
        Average value of the interval between two consecutive pitches in semitones.

        Args:
        "track_num" : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        "pitch_shift": a scalar for each sample.
        """
        pattern = feature.midi_pattern
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        total_used_note = self.total_used_note(feature, track_num=track_num)
        d_note = np.zeros((max(total_used_note - 1, 0)))
        # if total_used_note == 0:
        # return 0
        # d_note = np.zeros((total_used_note - 1))
        current_note = 0
        counter = 0
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if counter != 0:
                    d_note[counter - 1] = current_note - \
                        pattern[track_num][i].data[0]
                    current_note = pattern[track_num][i].data[0]
                    counter += 1
                else:
                    current_note = pattern[track_num][i].data[0]
                    counter += 1
        pitch_shift = np.mean(abs(d_note))
        return pitch_shift

    def avg_IOI(self, feature: Feature, track_num: int = 0):
        """
        avg_IOI (Average inter-onset-interval):
        To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

        Returns:
        "avg_ioi": a scalar for each sample.
        """

        inst: Instrument = feature.pretty_midi.instruments[track_num]
        onset = inst.get_onsets()
        ioi = np.diff(onset)
        avg_ioi = np.mean(ioi)
        return avg_ioi

    def total_used_note(self, feature: Feature, track_num: int = 0):
        """
        total_used_note (Note count): The number of used notes.
        As opposed to the pitch count, the note count does not contain pitch information but is a rhythm-related feature.

        Args:
        "track_num" : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        "used_notes": a scalar for each sample.
        """
        pattern = feature.midi_pattern
        used_notes = 0
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                used_notes += 1
        return used_notes

    def note_length_hist(self, feature: Feature, track_num: int = 0, normalize=True, pause_event=False):
        """
        note_length_hist (Note length histogram):
        To extract the note length histogram, we first define a set of allowable beat length classes:
        [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet].
        The pause_event option, when activated, will double the vector size to represent the same lengths for rests.
        The classification of each event is performed by dividing the basic unit into the length of (barlength)/96, and each note length is quantized to the closest length category.

        Args:
        "track_num" : specify the track number in the midi pattern, default is 1 (the second track).
        "normalize" : If true, normalize by vector sum.
        "pause_event" : when activated, will double the vector size to represent the same lengths for rests.

        Returns:
        "note_length_hist": The output vector has a length of either 12 (or 24 when pause_event is True).
        """

        pattern = feature.midi_pattern
        if pause_event is False:
            note_length_hist = np.zeros((12))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * \
                        resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    if "time_sig" not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                                 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find next note off
                    for j in range(i, len(pattern[track_num])):
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - \
                                    current_tick
                                distance = np.abs(
                                    np.array(hist_list) - note_length)
                                idx = distance.argmin()
                                note_length_hist[idx] += 1
                                break
        else:
            note_length_hist = np.zeros((24))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * \
                        resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    check_previous_off = True
                    if "time_sig" not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    tol = 3. * unit
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                                 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find next note off
                    for j in range(i, len(pattern[track_num])):
                        # find next note off
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - \
                                    current_tick
                                distance = np.abs(
                                    np.array(hist_list) - note_length)
                                idx = distance.argmin()
                                note_length_hist[idx] += 1
                                break
                            else:
                                if pattern[track_num][j].tick == current_tick:
                                    check_previous_off = False

                    # find previous note off/on
                    if check_previous_off is True:
                        for j in range(i - 1, 0, -1):
                            if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] != 0:
                                break

                            elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):

                                note_length = current_tick - \
                                    pattern[track_num][j].tick
                                distance = np.abs(
                                    np.array(hist_list) - note_length)
                                idx = distance.argmin()
                                if distance[idx] < tol:
                                    note_length_hist[idx + 12] += 1
                                break

        if normalize is False:
            return note_length_hist

        elif normalize is True:

            return note_length_hist / np.sum(note_length_hist)

    def note_length_transition_matrix(self, feature: Feature, track_num: int = 0, normalize=0, pause_event=False):
        """
        note_length_transition_matrix (Note length transition matrix):
        Similar to the pitch class transition matrix, the note length tran- sition matrix provides useful information for rhythm description.

        Args:
        "track_num" : specify the track number in the midi pattern, default is 1 (the second track).
        "normalize" : If true, normalize by vector sum.
        "pause_event" : when activated, will double the vector size to represent the same lengths for rests.

        "normalize" : If set to 0, return transition without normalization.
                      If set to 1, normalizae by row.
                      If set to 2, normalize by entire matrix sum.

        Returns:
        "transition_matrix": The output feature dimension is 12 × 12 (or 24 x 24 when pause_event is True).
        """
        pattern = feature.midi_pattern
        if pause_event is False:
            transition_matrix = np.zeros((12, 12))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            idx = None
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * \
                        resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    if "time_sig" not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                                 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find note off
                    for j in range(i, len(pattern[track_num])):
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:
                                note_length = pattern[track_num][j].tick - \
                                    current_tick
                                distance = np.abs(
                                    np.array(hist_list) - note_length)

                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx is not None:
                                    transition_matrix[last_idx][idx] += 1
                                break
        else:
            transition_matrix = np.zeros((24, 24))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            idx = None
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * \
                        resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    check_previous_off = True
                    if "time_sig" not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    tol = 3. * unit
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                                 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find next note off
                    for j in range(i, len(pattern[track_num])):
                        # find next note off
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - \
                                    current_tick
                                distance = np.abs(
                                    np.array(hist_list) - note_length)
                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx is not None:
                                    transition_matrix[last_idx][idx] += 1
                                break
                            else:
                                if pattern[track_num][j].tick == current_tick:
                                    check_previous_off = False

                    # find previous note off/on
                    if check_previous_off is True:
                        for j in range(i - 1, 0, -1):
                            if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] != 0:
                                break

                            elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):

                                note_length = current_tick - \
                                    pattern[track_num][j].tick
                                distance = np.abs(
                                    np.array(hist_list) - note_length)

                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx is not None:
                                    if distance[idx] < tol:
                                        idx = last_idx
                                        transition_matrix[last_idx][idx + 12] += 1
                                break
        if normalize == 0:
            return transition_matrix
        elif normalize == 1:
            sums: np.ndarray = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1, 1)
        elif normalize == 2:
            return transition_matrix / sum(sum(transition_matrix))
        else:
            print("invalid normalization mode, return unnormalized matrix")
            return transition_matrix
