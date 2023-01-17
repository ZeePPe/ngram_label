import os
import cv2
import re
from PIL import Image, ImageDraw,ImageFont
from utils import load_aligments, save_alignments
import numpy as np
import time
from datetime import datetime
import configs

BBOX_FILE = os.path.join(configs.OUT_FOLDER, configs.OUT_STATE_FILENAME)
OUT_FOLDER = configs.OUT_NGRAMS_FOLDER 
LINE_FOLDER = configs.LINE_FOLDER

H = configs.H


# all constants

TEST_PREV_LINE = -25
TEST_PREV_DOC  = -25

FONT_REGULAR_PIL = ImageFont.truetype('assets/font/AlteHaasGroteskRegular.ttf', 25)
FONT_BOLD_PIL = ImageFont.truetype('assets/font/AlteHaasGroteskBold.ttf', 30)
FONT_CV = cv2.FONT_HERSHEY_SIMPLEX

def load_bbs(ngram_file):
    if os.path.exists(ngram_file):
        all_ngrams = load_aligments(ngram_file)
    
    else:
        all_ngrams = {}
        for n in configs.NGRAMS_TOLABEL:
            for doc_foldername in os.listdir(configs.LINE_FOLDER):
                if doc_foldername not in all_ngrams.keys():
                    all_ngrams[doc_foldername] = {}
                for line_filename in os.listdir(os.path.join(LINE_FOLDER, doc_foldername)):

                    if line_filename not in all_ngrams[doc_foldername].keys():
                        list_bbox = []
                        list_transcripts = []
                    else:
                        list_bbox = all_ngrams[doc_foldername][line_filename][0]
                        list_transcripts = all_ngrams[doc_foldername][line_filename][1]
                    
                    gt_line_filename = f"gt_{line_filename.split('.')[0]}_{doc_foldername}.txt"
                    with open(os.path.join(configs.GT_FOLDER, doc_foldername,gt_line_filename), "r") as gt_file:
                        for line in gt_file.readlines():
                            for not_ch in configs.INVALID_CHARACTERS:
                                line = line.replace(not_ch, "")
                            line = line.rstrip()
                            line = re.sub(' +', ' ',line)
                            for word in line.split(" "):
                                if len(word) > 0:
                                    for cr_ind in range(len(word)-n+1):
                                        ngram = word[cr_ind:cr_ind+n]
                                        list_transcripts.append(ngram)
                                        list_bbox.append([0,0])
                    all_ngrams[doc_foldername][line_filename] = (list_bbox,list_transcripts)
        os.makedirs(os.path.dirname(ngram_file))
        save_alignments(all_ngrams, ngram_file)
    return all_ngrams  

def label_ngrams(aligns, outfile="out"):
    count = 1
    all_aligns = _get_aligns_number(aligns)

    from_next_line = False
    from_next_doc = False

    curr_inddoc = 0
    keys_list_docs = list(aligns)

    while curr_inddoc < len(keys_list_docs):
        doc_folder = keys_list_docs[curr_inddoc]
        all_lines = aligns[doc_folder]

    #for doc_folder, all_lines in aligns.items():
        keys_list_lines = list(all_lines)
        curr_indline = 0
        if from_next_doc:
            curr_indline = len(keys_list_lines)-1
            from_next_doc = False
            from_next_line = True

        while 0 <= curr_indline < len(keys_list_lines):
            line_filename = keys_list_lines[curr_indline]
            (boxes, transcriptions) = all_lines[line_filename]

            # leggi immagine
            line_img = cv2.imread(os.path.join(LINE_FOLDER, doc_folder, line_filename))

            H = line_img.shape[0]

            curr_indword = 0
            if from_next_line:
                curr_indword = len(boxes)-1
                from_next_line = False

            while 0 <= curr_indword < len(boxes):
                box = boxes[curr_indword]
                trans = transcriptions[curr_indword]

                #Mouse CLick callback
                def click_event(event, x, y, flags, params):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        # displaying the coordinates
                        new_start = x
                        print(f"[{new_start}")

                        cv2.line(curr_img, (new_start, 0), (new_start, H), (0,0,255), 2) 
                        cv2.imshow('image', curr_img)
                        
                        # write file!!
                        params[0] = new_start
                    
                    # checking for right mouse clicks    
                    if event==cv2.EVENT_RBUTTONDOWN:
                        new_end = x
                        print(f"[   ,{new_end}]")
                
                        cv2.line(curr_img, (new_end, 0), (new_end, H), (255,0,255), 2) 
                        cv2.imshow('image', curr_img)

                        params[1] = new_end
                
                print(f"{box} {trans.ljust(15)}\t---| {count}/{all_aligns} |    \t..{os.path.join(doc_folder, line_filename)}")

                #curr_img = line_img.copy()
                curr_img = np.zeros((line_img.shape[0]+100, line_img.shape[1], line_img.shape[2]), dtype=np.uint8)
                curr_img[0:line_img.shape[0], 0:line_img.shape[1], 0:line_img.shape[2]] = line_img

                cv2.rectangle(curr_img,(box[0],1),(box[1],H),(0,255,0),1)

                #text under box
                #cv2.putText(curr_img, trans, (box[0],line_img.shape[0]+40), FONT_CV, 1, (128, 240, 128), 2) #box
                img = Image.new('RGB', (len(trans)*20, 40), color = (0, 0, 0))
                d = ImageDraw.Draw(img)
                d.text((0,0), trans, font=FONT_BOLD_PIL,  fill=(128, 240, 128))
                img = np.asarray(img)
                bottom_margin = 55
                y_start = curr_img.shape[0]-(40+bottom_margin)
                y_end = curr_img.shape[0]-bottom_margin
                x_start = box[0]
                x_end = box[0]+len(trans)*20
                a = (x_end-x_start)
                b = curr_img.shape[1]
                if x_end >curr_img.shape[1]:
                    x_end = curr_img.shape[1]
                    
                curr_img[y_start:y_end,x_start:x_end, : ] = img[:,:(x_end-x_start),:]
               
                # text bottom trans
                #cv2.putText(curr_img, trans, (10,line_img.shape[0]+90), font, 1, (255, 200, 200), 2) 
                img = Image.new('RGB', (len(trans)*25, 30), color = (0, 0, 0))
                d = ImageDraw.Draw(img)
                d.text((10,0), trans, font=FONT_REGULAR_PIL,  fill=(255,200,200))
                img = np.asarray(img)
                bottom_margin = 1
                curr_img[curr_img.shape[0]-(30+bottom_margin):curr_img.shape[0]-bottom_margin,0:len(trans)*25, : ] = img

                
                # state of progression
                str_curr_position =  f"{count}/{all_aligns}"
                cv2.putText(curr_img, str_curr_position, (line_img.shape[1]-15*len(str_curr_position),line_img.shape[0]+90), FONT_CV, 0.6, (255, 200, 200), 1)
                
                cv2.imshow('image', curr_img)
                cv2.setMouseCallback('image', click_event, param=box)
                key_pressed = cv2.waitKey(0)
                
                if key_pressed == 13:
                    #  ENTER
                    curr_indword += 1
                    count += 1
                elif key_pressed == 8:
                    # backspace
                    count -= 1
                    curr_indword -= 1
                    if curr_indword <0:
                        #curr_indword = 0
                        curr_indword = TEST_PREV_LINE
                        count += 1
                    
                elif key_pressed == 83:
                    # ALT+s
                    # VA RICHIAMATA LA FUNZIONE mesure_performnace()
                    print("-------- SAVE STATE -------")
                    save_alignments(aligns, outfile)
                elif key_pressed == 17:
                    #CTRL+q
                    #quit and save
                    print("-------- SAVE STATE -------")
                    save_alignments(aligns, outfile)
                    return
                    #exit()

                #save modification
                save_alignments(aligns, outfile)

            # new line
            if curr_indword > 0:
                #next line
                curr_indline += 1
            elif curr_indword == TEST_PREV_LINE and curr_indline>0:
                #prev line
                from_next_line = True
                curr_indline -= 1
                count -= 1
            elif curr_indword == TEST_PREV_LINE and curr_indline==0 and curr_inddoc>0:
                #prev doc
                curr_indline = TEST_PREV_DOC
                
        #("new_doc")
        if curr_indline > 0:
            curr_inddoc += 1
        elif curr_indline == TEST_PREV_DOC:
            from_next_doc = True
            curr_inddoc -= 1
            count -= 1

def _get_aligns_number(aligns):
    num_of_aligns = 0

    for doc_res in aligns:
        cur_doc_res = aligns[doc_res]
        for line_res in  cur_doc_res:
            num_of_aligns += len(cur_doc_res[line_res][0])

    return num_of_aligns


if __name__ == "__main__":
    
    data_bbox_statefile = os.path.join(configs.OUT_FOLDER, configs.OUT_STATE_FILENAME)

    bbox = load_bbs(data_bbox_statefile)
    

    start_time = time.time()
    label_ngrams(bbox, outfile=data_bbox_statefile)
    total_time = time.time() - start_time

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S__")

    #save ngram list bbox GT

    # save performance and time file
    if not os.path.exists(configs.TIME_BASEFOLDER):
        os.mkdir(configs.TIME_BASEFOLDER)
    time_filepath = os.path.join(configs.TIME_BASEFOLDER, dt_string+configs.TIME_WORDCORRECTION_FILENAME)
    with(open(time_filepath, "w") as timefile):
        timefile.write(f"Correction has required {total_time} seconds")


    print("Done!")


