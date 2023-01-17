import os

LINE_FOLDER = "data\lines"
GT_FOLDER = "data\GT" 

OUT_FOLDER = "out"
OUT_STATE_FILENAME = "state.als"
OUT_GT_FILENAME = "bounding_boxes.ngl"

OUT_NGRAMS_FOLDER = os.path.join(OUT_FOLDER,"ngram_imgs")

TIME_BASEFOLDER = os.path.join(OUT_FOLDER,"time")
TIME_WORDCORRECTION_FILENAME = "time_label_ngram.txt"

PERFORMANCE_BASEFOLDER = "performance"
PERFORMANCE_FILENAME = "performance.txt"

#reference height
H = 115
Y0 = 0

NGRAMS_TOLABEL = [3]

INVALID_CHARACTERS = set("'`%$&_\"/\,.:;-?!()[]{}+*")

