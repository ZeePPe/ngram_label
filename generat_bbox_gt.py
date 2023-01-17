import os
import cv2
from utils import load_aligments
import configs
from tqdm import tqdm

STATE_FILE = os.path.join(configs.OUT_FOLDER, configs.OUT_STATE_FILENAME)
OUT_GT_FILE = os.path.join(configs.OUT_FOLDER, configs.OUT_GT_FILENAME)

OUT_FOLDER = configs.OUT_NGRAMS_FOLDER
LINE_FOLDER = configs.LINE_FOLDER

def save_bbox_gt(state, dst_file="ngrams.ngl"):
    
    with open(dst_file, "w") as out_file:
        for doc_folder, all_lines in tqdm(state.items()):
            for line_filename, (boxes, transcriptions) in all_lines.items():
                img_path = os.path.join("lines",doc_folder,line_filename)
                line_img = cv2.imread(os.path.join(LINE_FOLDER, doc_folder, line_filename), cv2.IMREAD_GRAYSCALE)
                h, _ = line_img.shape

                for box, trans in zip(boxes, transcriptions):
                    if box[1] <= box[0]:
                        continue
                    
                    out_line = f"{img_path},{box[0]},{configs.Y0},{box[1]},{h},{trans}"
                    out_file.write(out_line+"\n")



if __name__ == "__main__":
    state = load_aligments(STATE_FILE)

    save_bbox_gt(state, dst_file=OUT_GT_FILE)

    print("Done!")