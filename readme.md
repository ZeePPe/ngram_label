# NGram labelled tool
The tool allows to manually label ngrams in handwritten text rows

## Define enviroment
You need to have the anaconda environment manager installed on your computer.
If so, run the command
```conda env create -f environment.yml```
and acrivate the enciroment: 
```conda activate MiMalign```


## Preparation
1. Create a ```"data/lines"``` folder which contains the images of the text lines. The folder is organized into subfolders, one for each document.

2. Create the ```"data/GT"``` folder which contains the transcript txt files. The folder is organized into subfolders, one for each document.

3. Set all the input folders in the file ```configs.py```.


## Start Laelling

1. To start a process take care to delete the `out` folder. If the folder exists, the tool tries to continue a previus process.

2. define in ```configs.py``` file the N-grams that you eant to compute in the constant `NGRAMS_TOLABEL`

3. Run the ```GUI_tool.py``` file to start the interface and label your rows. The tool will display all the ngrams one at a time.
   With the ENTER key you can move to the next ngram.
   With the BACKSPACE key you go back to the previous ngram.
   With the cntrl+s keys you save the state
   With the cntrl+q keys you can close the GUI
   
   To label the ngrams you can use the mouse:
      with a click with the left button a new left segmentation boundary is set
      a right click sets a new right segmentation boundary
    
   finally the tool generate the alignment file `bounding_boxes.als` that save the state of the process and generates in the ```time``` folder a file where the total time spent on the process is reported.

4. Running the file ```crop_all_ngrams.py``` it is possible to generate all the images of labelled ngrams.

5. Running the file ```generat_bbox_gt.py``` it is possible to generate a txt file with the bbox of all the lines
