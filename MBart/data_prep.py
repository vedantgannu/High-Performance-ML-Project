from datasets import load_dataset, DatasetDict
import csv
import sys
import subprocess

NUMBER_OF_PAIRS = 200

def create_datasets():
    # subprocess.call([sys.executable, './../LASER/tasks/WikiMatrix/extract.py', '--tsv', './../WikiMatrix.en-fr.tsv.gz', '--bitext', './../WikiMatrix.en-fr.txt', 
    #                 '--src-lang', 'en', '--trg-lang', 'fr', '--threshold', '1.04', '--nb-sents', NUMBER_OF_PAIRS], shell=True)

    # subprocess.call([sys.executable, './../LASER/tasks/WikiMatrix/extract.py', '--tsv', './../WikiMatrix.en-es.tsv.gz', '--bitext', './../WikiMatrix.en-es.txt', 
    #                 '--src-lang', 'en', '--trg-lang', 'es', '--threshold', '1.04', '--nb-sents', NUMBER_OF_PAIRS], shell=True)

    # subprocess.call([sys.executable, './../LASER/tasks/WikiMatrix/extract.py', '--tsv', './../WikiMatrix.en-zh.tsv.gz', '--bitext', './../WikiMatrix.en-zh.txt', 
    #                 '--src-lang', 'en', '--trg-lang', 'zh', '--threshold', '1.04', '--nb-sents', NUMBER_OF_PAIRS], shell=True)
    
    print("Preparing English to French")

    with open("./../WikiMatrix.en-fr.txt.en", "r", encoding="utf8") as english_sentences:
        with open("./../WikiMatrix.en-fr.txt.fr", "r", encoding="utf8") as french_sentences:
            with open("./../WikiMatrix.en-fr.csv", "w", newline='', encoding="utf8") as reduced_csv_file:
                csvwriter = csv.writer(reduced_csv_file)
                csvwriter.writerow(["en", "fr"])
                for english_sentence in english_sentences:
                    csvwriter.writerow([english_sentence.rstrip("\n"), french_sentences.readline().rstrip("\n")])
            
    print("Preparing English to Spanish")
    
    with open("./../WikiMatrix.en-es.txt.en", "r", encoding="utf8") as english_sentences:
        with open("./../WikiMatrix.en-es.txt.es", "r", encoding="utf8") as spanish_sentences:
            with open("./../WikiMatrix.en-es.csv", "w", newline='', encoding="utf8") as reduced_csv_file:
                csvwriter = csv.writer(reduced_csv_file)
                csvwriter.writerow(["en", "es"])
                for english_sentence in english_sentences:
                    csvwriter.writerow([english_sentence.rstrip("\n"), spanish_sentences.readline().rstrip("\n")])

    print("Preparing English to Chinese")

    with open("./../WikiMatrix.en-zh.txt.en", "r", encoding="utf8") as english_sentences:
        with open("./../WikiMatrix.en-zh.txt.zh", "r", encoding="utf8") as chinese_sentences:
            with open("./../WikiMatrix.en-zh.csv", "w", newline='', encoding="utf8") as reduced_csv_file:
                csvwriter = csv.writer(reduced_csv_file)
                csvwriter.writerow(["en", "zh"])
                for english_sentence in english_sentences:
                    csvwriter.writerow([english_sentence.rstrip("\n"), chinese_sentences.readline().rstrip("\n")])
    
    #Creating the test, validation, and test datasets per language
    data_files = [r'./../WikiMatrix.en-fr.csv']
    ds = load_dataset('csv', data_files=data_files)
    
    train_test = ds["train"].train_test_split(train_size=0.7)
    train_validation = train_test["train"].train_test_split(train_size=0.7)
    train_test_valid_dataset = DatasetDict({
        'train': train_validation['train'],
        'test': train_test['test'],
        'valid': train_validation['test']})
    train_test_valid_dataset.save_to_disk(r'./data/en-fr')
    
    data_files = [r'./../WikiMatrix.en-es.csv']
    ds = load_dataset('csv', data_files=data_files)
    
    train_test = ds["train"].train_test_split(train_size=0.7)
    train_validation = train_test["train"].train_test_split(train_size=0.7)
    train_test_valid_dataset = DatasetDict({
        'train': train_validation['train'],
        'test': train_test['test'],
        'valid': train_validation['test']})
    train_test_valid_dataset.save_to_disk(r'./data/en-es')
    
    data_files = [r'./../WikiMatrix.en-zh.csv']
    ds = load_dataset('csv', data_files=data_files)
    
    train_test = ds["train"].train_test_split(train_size=0.7)
    train_validation = train_test["train"].train_test_split(train_size=0.7)
    train_test_valid_dataset = DatasetDict({
        'train': train_validation['train'],
        'test': train_test['test'],
        'valid': train_validation['test']})
    train_test_valid_dataset.save_to_disk(r'./data/en-zh')
