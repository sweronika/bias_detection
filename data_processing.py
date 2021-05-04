## nlp
import re
from nltk import word_tokenize
from stop_words import get_stop_words
from wordcloud import WordCloud
import string

## plots
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
random.seed(10)

import os
from collections import Counter
import math
import pandas as pd
from collections import defaultdict
import numpy as np

######################### READING DATA ############################################################
###################################################################################################
def read_data(path):
    """ 
    Function reads sentences in its raw txt. format and return list of pairs of sentences  
    """  
    with open(path, 'r', encoding = 'utf8') as f:
        data = f.read().rstrip()
        
    data = data.split("\n")
    return data

def split_data(data):
    """ 
    Split pairs of sentences 
    """ 
    data = [sent.split("\t") for sent in data]
    return data

def unify_structure(LANG):
    
    """
    Reads data from raw format and transform it to a proper structure -
    - list of sentences for every subset
    """ 
    
    path = 'data' + LANG
    
    ### READ ###############
    biased_all = split_data(read_data(os.path.join(path,'biased')))
    unbiased = split_data(read_data(os.path.join(path,'unbiased')))
    featured = split_data(read_data(os.path.join(path,'featured')))
                          
    if LANG == 'PL':
        b,r,u,f = 1,2,1,0
    elif LANG == 'EN':
        b,r,u,f = 3,4,3,0
                          
    biased = [sent[b] for sent in biased_all]
    reviewed = [sent[r] for sent in biased_all]
    unbiased = [sent[u] for sent in unbiased]
    featured = [sent[f] for sent in featured]
    
    d = {'biased':biased, 'reviewed': reviewed, 'unbiased': unbiased, 'featured': featured}
                          
    return d

def senteneces_to_tuples(sentences):
    
    sentences = [(sent.split('\t')[0], int(sent.split('\t')[1])) for sent in sentences]
    
    return sentences
    
######################### DUPLICATES #############################################################
##################################################################################################

def drop_biased_dups(d):
    
    """ In case of duplicates leaves the first sentence for biased sentences 
    and last sentence for reviewed sentences"""

    biased = pd.DataFrame(d['biased'])
    reviewed = pd.DataFrame(d['reviewed'])

    biased_dedup = biased.drop_duplicates(keep = 'first')
    reviewed_dedup = reviewed.drop_duplicates(keep = 'last')

    print("Biased deleted:",  len(biased) - len(biased_dedup))
    print("Reviewed deleted:", len(reviewed) - len(reviewed_dedup))
    
    return d

def BLEU(hyp, ref):
    
    """ Calulcate BLEU score between sentences """
    # get ngram stats
    stats = []
    stats.append(len(hyp))
    stats.append(len(ref))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hyp[i:i + n]) for i in range(len(hyp) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(ref[i:i + n]) for i in range(len(ref) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hyp) + 1 - n, 0]))

    # get bleu from stats
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    bleu = math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

    return 100 * bleu


def similar_sent(sentences, BLEU_threshold):
    
    """ Iterate over sentences and check bleu score between sentence and 10 following sentences.
    """
    
    duplicates = []
    duplicated_indices = []
    for i, sentence in enumerate(sentences):
        if i > len(sentences) - 11:
            break
        dups = [i]
        for n in range(1,11):
            if BLEU(sentence, sentences[i+n]) > BLEU_threshold:
                duplicates.append((sentence, sentences[i + n]))
                dups.append(i+n)
        
        if len(dups) > 1:
            duplicated_indices.append(dups)
    
    return duplicates, duplicated_indices


def clean_biased(d, biased_indices):

    """ Leave initial sentence from duplicates """
    
    ### Clean biased
    duplicates = []
    for sublist in biased_indices:
        duplicates = duplicates + sublist[1:]
        
    duplicates = set(duplicates)
    print("Number of duplicates:", len(duplicates))
    d['biased'] = [sent for i,sent in enumerate(d['biased']) if i not in duplicates]
    
    return d


def clean_reviewed(d, reviewed_indices):

    """ Leave the newest sentence from duplicates """
    
    ### Clean reviewed
    duplicates = []
    for sublist in reviewed_indices:
        duplicates = duplicates + sublist[:-1]
        
    duplicates = set(duplicates)
    print("Number of duplicates:", len(duplicates))
    d['reviewed'] = [sent for i,sent in enumerate(d['reviewed']) if i not in duplicates]
    
    return d

######################### CLEANING ###############################################################
##################################################################################################

def in_brackets_examples(d):
    """
    Look for examples of in-brackets text.
    """
    i = 0
    
    for name, sentences in d.items():
        print(name.upper())
        for sentence in sentences:
            if re.search("\((.*?)\)", sentence):
                print(re.search("\((.*?)\)", sentence)[0])
                i+=1
            if i > 5:
                break
        i = 0 

def delete_in_brackets(d):
    """ Delete text that is in brackets"""
    for file_name, sentences in d.items():
        d[file_name] = [re.sub("\((.*?)\)", "", sent) for sent in sentences]
    return d


def check_punc(d):   
    """
    Analyse types of punctuation.
    """    
    punc = defaultdict(int)
    
    for name, sentences in d.items():
        text = "\n".join(sentences)
        tokens = text.split()
        for n, token in enumerate(tokens):
            if re.match("\W+", token):
                punc[re.match("\W+", token)[0]] +=1
                
    punc = {k: v for k, v in sorted(punc.items(), key = lambda item: item[1], reverse=True)}
    
    return punc


def remove_punc(d):
    """
    Leave only basic punctuation
    """
    deleted = defaultdict(int)
    
    for name, sentences in d.items():
        
        init_length = len(sentences)           
        text = " \n ".join(sentences)
        tokens = text.split(" ")
        new_text = []
        
        for n, token in enumerate(tokens):
            if re.match("\W+", token):
                non_word = re.match("\W+", token)[0]
                if non_word in string.punctuation + '),)."[",."("\")(,"),?"\n':
                    new_text.append(token)
                else:
                    deleted[non_word] +=1
            else:
                new_text.append(token)
            
        new_text = " ".join(new_text)
        sentences = new_text.split("\n")
        d[name] = sentences
        
        
        assert init_length == len(sentences)
        
    print("Deleted signs:", deleted.keys())
    
    return d


def non_printable(d):
    """
    Analyse non-printable signs.
    """
    printable = set(string.printable + 'ćńóśźżąęł')
    all_non_pr = []
    
    for name, sentences in d.items():
        text = "\n".join(sentences)
        non_pr = list(filter(lambda x: x not in printable, text))
        all_non_pr = all_non_pr + non_pr
        
    all_non_pr = Counter(all_non_pr)    
    all_non_pr = {k: v for k, v in sorted(all_non_pr.items(), key=lambda item: item[1], reverse = True)}
    
    return all_non_pr


def replace_signs(d, to_drop):
    
    """ Replace most common non-printable signs with printable version. Drop rest."""
    
    replace = {'é': 'e','о': 'o', 'е': 'e','ö': 'o','а': 'a','ü': 'u','ä': 'a','á':'a','è':'e', 'ă':'a',
               'å': 'a','ñ': 'n','ô': 'o','ё': 'e','ú': 'u','ë': 'e','â': 'a','à':'a','ê': 'e',
               'с': 'c', 'р': 'p', 'у':'y', 'ο':'o', 'ō':'o', 'š':'s', 'ç':'c', 'ν':'v', 'č':'c',
               'ệ':'e', 'š':'s', 'ā':'a','ū': 'u', 'ã': 'a', 'č': 'c','ả':'a'}
    
    to_drop = {k:'' for k,v in to_drop.items()}
    
    for name, sentences in d.items():
        text = "\n".join(sentences)
        text = text.translate(str.maketrans(replace))
        text = text.translate(str.maketrans(to_drop))
        sentences = text.split('\n')
        d[name] = sentences
        
    return d

def delete_empty_spaces(d):
    """ Delete tokens that are empty spaces"""
    for file_name, sentences in d.items():
        for i, sentence in enumerate(sentences):     
            sentence = [token for token in sentence.split() if len(token) > 0]
            sentence = " ".join(sentence)
            sentences[i] = sentence
            
        d[file_name] = sentences
        
    return d

def clean_by_length(d, min_tokens, max_tokens):
   
    """ 
    Delete sentences with less or min_tokens or more than max_tokens
    """
    for file_name, sentences in d.items():
        cleaned = []
        deleted = []
        for sentence in sentences:
            length = len(sentence.split())
            if max_tokens >= length > min_tokens:
                cleaned.append(sentence)
            else:
                deleted.append(sentence)
    
        d[file_name] = cleaned
        print(file_name, "dropped sentences", len(deleted))
        
    return d

def clean_nw_ratios(d, ratio_threshold = 0.12):
    
    """ Drop sentences with too high ratio of non-word signs
    """
    nw_ratios = defaultdict(list)
    
    for file_name, sentences in d.items():
        initial_length = len(sentences)
        clean_sentences = []
        for sent in sentences:           
            non_word_signs = len(re.findall('[^\\w\s]', sent))
            all_signs = len(sent)
            
            try:
                wnn_ratio = non_word_signs / all_signs 
            except ZeroDivisionError:
                wnn_ratio = 1
            
            if wnn_ratio <= ratio_threshold:
                clean_sentences.append(sent)
                
        d[file_name] = clean_sentences
        print(file_name.upper(), 'dropped:', initial_length - len(clean_sentences))  
        
    return d

######################### ANALYSE ################################################################
##################################################################################################

def count_sent_lengths(d, short_threshold = 1, long_threshold = 50):
    
    """ Count lengths of sentences 
    long_threshold - how many tokens is a long sentence 
    """
    
    sent_lengths = {}
    shorts = []
    longs = []
    
    for file_name, sentences in d.items():
        lengths = []
        for i, sentence in enumerate(sentences):
            sent_len = len(sentence.split(" "))
            
            # examples of outliers
            if sent_len >= long_threshold:
                longs.append(sentence)
            elif sent_len <= short_threshold:
                shorts.append(sentence)
            
            lengths.append(sent_len)
            
        sent_lengths[file_name] = lengths
        print(file_name, "| minimum length:", min(sent_lengths[file_name]), "max_length:", max(sent_lengths[file_name]))

    return sent_lengths, shorts, longs

def count_nw_ratios(d, ratio_threshold = 0.12):
    
    """ Count ratio of non-word to word signs in a sentence
    ratio_threshold - at which ratio threshold sentence is considered to be too noisy
    """
    nw_ratios = defaultdict(list)
    high_ratios = []
    
    for file_name, sentences in d.items():
        for sent in sentences:           
            non_word_signs = len(re.findall('[^\\w\s]', sent))
            all_signs = len(sent)
            
            try:
                wnn_ratio = non_word_signs / all_signs 
            except ZeroDivisionError:
                wnn_ratio = 1
                
            if wnn_ratio > ratio_threshold:
                high_ratios.append(sent)
                
            nw_ratios[file_name].append(wnn_ratio)
            
    return nw_ratios, high_ratios
    
# percentile of word counts
def count_percentiles(word_cnt):    
    for i in [10, 25, 50, 75, 90, 95, 99]:        
        word_counts = sorted(word_cnt.values())
        perc = np.percentile(word_counts, i)
        print(i, '% of all words appear less often than', int(perc), 'time\s in text')

def analyse_labels(sentences, title):
    
    pos, neg, all_l, pos_ratio = count_labels(sentences)
    
    print('Positive label: {}%'.format(pos_ratio*100) )
    print('Positive no.: {:,}'.format(pos))
    print('Negative no.: {:,}'.format(neg))
    print('All sentences: {:,}'.format(all_l))
    print('\n')
    dist_plot_small(pos, neg, title)
    
######################### PLOT ###################################################################
##################################################################################################

def plot_lengths(lengths_dict, title = ""):
    
    """ 
    Input: dictionary where key = file name, 
    value = list of sentences lengths
    """
    
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.5)

    boxplots = pd.DataFrame({'length':[], "set":[]})
    
    for k, v in lengths_dict.items():
        boxplots = boxplots.append(pd.DataFrame({'length': v, 'set':[k]*len(v)}))
    
    PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
    
    plt.figure(figsize=(15,4))
    sns.boxplot(data = boxplots, x = 'set', y = 'length', **PROPS)
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Tokens no.')
    plt.show()
    
def wnw_distributions(lengths_dict, title = ""):
    
    "Input: dictionary where key = file name, value = list of sentences lengths"
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.5)
    
    PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}}
        
    boxplots = pd.DataFrame({'ratio':[], "set":[]})
    
    for k, v in lengths_dict.items():
        boxplots = boxplots.append(pd.DataFrame({'ratio': v, 'set':[k]*len(v)}))
        
    plt.figure(figsize=(15,4))
    sns.boxplot(data=boxplots, x='set', y ='ratio',**PROPS)
    plt.ylabel('Non-word ratio')
    plt.xlabel('')
    plt.title(title)   
    plt.show()
    
def dist_plot(d, title):
    
    """ 
    Plot number of sentences per dataset
    """
    
    plt.figure(figsize = (7,3))
    d = d.copy()
    
    for name, sentences in d.items():
        d[name] = len(sentences)
    
    x = list(d.keys())
    y = list(d.values())
    
    ax = sns.barplot(x = x, y = y, color = 'white', edgecolor='black')
    hatches = ['||', '\\\\', 'x', '-']

    # Loop over the bars
    for i, thisbar in enumerate(ax.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])
        
    for p in ax.patches:
        height = round(p.get_height())
        height = f"{height:,}"
        ax.annotate(height, (p.get_x() + 0.15, 2000 + p.get_height()))
        
    ax.margins(0.1)    
    ax.set_title(title, y = 0.8, x = 0.05)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    
def black_color_func(word, font_size, position,orientation, random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")

def plot_wordclouds(d, LANG, title = "", stopwords = []):
    
    """ 
    Plot word clouds.
    """
    wordclouds={}
        
    for file_name, sentences in d.items():
        text = " ".join(sentences)
        
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", ).generate(text)
        wordclouds[file_name] = wordcloud
        wordcloud.recolor(color_func = black_color_func)
        
    fig, axes = plt.subplots(2,2, figsize=(15,7))
    axes[0,0].imshow(wordclouds['biased'], interpolation='bilinear')
    axes[0,0].axis("off")
    axes[0,0].set_title('biased')
    
    axes[0,1].imshow(wordclouds['reviewed'], interpolation='bilinear')
    axes[0,1].axis("off")
    axes[0,1].set_title('revised')
    
    axes[1,0].imshow(wordclouds['unbiased'], interpolation='bilinear')
    axes[1,0].axis("off")
    axes[1,0].set_title('unbiased')
    
    axes[1,1].imshow(wordclouds['featured'], interpolation='bilinear')
    axes[1,1].axis("off")
    axes[1,1].set_title('featured')
    
    #plt.subplots_adjust(wspace=-0.6, hspace=0.2)
    plt.suptitle(title)
    
def dist_plot_small(pos, neg, title):
    
    plt.figure(figsize = (7,3))
    
    ax = sns.barplot(x = ['biased', 'unbiased'], y = [pos, neg], color = 'white', edgecolor='black')
    hatches = ['||', '\\\\', 'x', '-']

    # Loop over the bars
    for i, thisbar in enumerate(ax.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])
        
    for p in ax.patches:
        height = round(p.get_height())
        height = f"{height:,}"
        ax.annotate(height, (p.get_x() + 0.3, 2000 + p.get_height())) 
    
    ax.margins(0.1)  
    ax.set_title(title, y = 0.8, x = 0.05)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
######################### MERGING AND SPLITTING ##################################################
################################################################################################## 

def merge_datasets(d):
    
    """
    Merges datasets and appends labels.
    """
    
    pos = d['biased']
    neg = d['featured'] + d['unbiased'] + d['reviewed']
    
    pos = [sent +  "\t" + str(1) for sent in pos]
    neg = [sent +  "\t" + str(0) for sent in neg]

    all_sent = pos + neg

    random.shuffle(all_sent)

    return all_sent

def get_sent_labels(sentences):
    
    """ unzip tuples"""
    
    unzipped = list(zip(*sentences))
    sentences, labels = list(unzipped[0]), list(unzipped[1])
    
    return sentences, labels

def count_labels(sentences):
        
    sentences = senteneces_to_tuples(sentences)
    sentences, labels = get_sent_labels(sentences)
    
    cnt = Counter(labels)
    pos = cnt[1]
    neg = cnt[0]
    all_l = pos+neg
    
    pos_ratio = round(pos/all_l, 2)
    
    return pos, neg, all_l, pos_ratio
    
def train_test_val_split(path, sentences):
    
    """
    Splits data intro train, test and validation sets and save.
    """
    random.shuffle(sentences)
       
    test_split = round(0.7 * len(sentences))
    val_split = round(0.9 * len(sentences))
    
    train = sentences[:test_split]
    test = sentences[test_split:val_split]
    val = sentences[val_split:]
    
    # check if label ratios are similar     
    _, _, all_train, pos_ratio_train = count_labels(train)
    _, _, all_test, pos_ratio_test = count_labels(test)
    _, _, all_val, pos_ratio_val = count_labels(val)
    
    print("\nPositive ratios:\n", "train:", pos_ratio_train, "test:", pos_ratio_test, 'val:', pos_ratio_val)
    print("\nNumber of examples:\n", "train: {:,} test: {:,} val: {:,}".format(all_train, all_test, all_val))
    
    train = "\n".join(train)
    test = "\n".join(test)
    val = "\n".join(val)
    
    d = {'train': train, "test": test, "val": val}
    
    for key, value in d.items():        
        save_path = os.path.join(path, key)
        with open(save_path, 'w', encoding = 'utf8') as f:
            f.write(value)  