import os


# Base directory
BASE_DIR = os.path.abspath(".")
# base directory for data files
BASE_DIR_DATA = os.path.join(BASE_DIR, "data")

for CITY in ['london','rome']: 
    
    BASE_DIR_CITY = os.path.join(BASE_DIR_DATA, CITY)   
    SEQUENCES_DIR = os.path.join(BASE_DIR_DATA, CITY, 'words')
    OUTPUT_DIR = os.path.join(BASE_DIR_CITY, 'output-skip')
#     create_output_dir(OUTPUT_DIR)
     

    for SEQUENCE in os.listdir(SEQUENCES_DIR):
        print(SEQUENCES_DIR)
        INPUT_SEQUENCE = os.path.join(SEQUENCES_DIR,SEQUENCE)

        strategy = SEQUENCE.split('sequences_')[1].split('.')[0]
        prefix = "skip_"+ strategy
        print(INPUT_SEQUENCE)
        os.system("python GeoL/scripts/word2vec_embeddings.py -i {} -o {} -p {} -plt  -s 50 100 200 -ws 3 5 7 10 -c 2 5 10 -m -v 2".format(INPUT_SEQUENCE, OUTPUT_DIR, prefix))
