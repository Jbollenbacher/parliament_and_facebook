import pandas as pd
import numpy as np
import sys
from urllib.parse import urlparse
import re
from collections import Counter
import time
from multiprocessing import Pool

############################################################
# config params
############################################################

data_path = './data/'

n_workers = 22 # for parellelization

use_all_data = True
n_samples = 20 #if sampling.

response_types = ['tweet','commons_speech','url']

dist_metric = 'BC' # BC or cosine
how_many_resamples_of_text = 1 #TODO: refactor
run_control = False

how_to_aggregate_texts = 'by_text' #takes mean of individual texts' deltas
# how_to_aggregate_texts = 'by_source' #takes mean of individual source' deltas. e.g. treat MP or url domain as single text.
# how_to_aggregate_texts = 'as_single_text' #treats all texts as single text by summing wordcounts

post_event_gap_length = 48 # X hrs between when event happens and when we start looking for responses

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

parties_to_analyze = sys.argv[1:]
print('analyzing responses from party:', parties_to_analyze)
parties_to_analyze_str = ('_'+(' '.join(parties_to_analyze))) if isinstance(parties_to_analyze,list) else ''
print('for response text types:', response_types)
print('using dist metric:',dist_metric)
print('aggregating',how_to_aggregate_texts)
print('with post event time gap',post_event_gap_length)
print('with this many random resamples of texts:',how_many_resamples_of_text)
print('running', 'control' if run_control else 'test (i.e. not the control)')
print('')


############################################################
# load data
############################################################


print('loading data')
all_texts = pd.read_parquet(data_path + 'all_texts_cleaned.parquet')


print('filtering texts by type/party/etc')

#combine Lab and Lab/Co-op parties
all_texts['party'] = all_texts['party'].apply(lambda x: 'Lab' if x=='Lab/Co-op' else x)

#sample tweets uniformly. keep only ~200,000 tweets
non_tweets_df = all_texts[all_texts['text_type'].isin(['url','commons_speech'])]
tweets_df = all_texts[all_texts['text_type'].isin(['tweet'])]
tweets_df = tweets_df.sample(n = int(len(non_tweets_df)/2), random_state=0)

all_texts = pd.concat([non_tweets_df, tweets_df]).reset_index(drop=True)
del non_tweets_df; del tweets_df


#initialize results df
if use_all_data:  results = all_texts.copy().sample(frac=1).reset_index(drop=True)
else: results = all_texts.sample(n_samples).reset_index(drop=True).copy()
        
# filter by party
if parties_to_analyze is not None and len(parties_to_analyze) > 0:
    all_texts = all_texts[all_texts['party'].isin(parties_to_analyze)]
    
all_texts = all_texts[['text','mp_name','text_type', 'time']]


texts_dict_by_type = {'url':all_texts[all_texts['text_type']=='url'],
                 'commons_speech':all_texts[all_texts['text_type']=='commons_speech'],
                 'tweet':all_texts[all_texts['text_type']=='tweet']}




############################################################
#helper functions for getting wordcount distances
############################################################
from collections.abc import Iterable

def get_word_freq_dict(text):
    if isinstance(text, str):
        wordlist = text.split(' ')
    else: #accomodate pre-split texts for speed optimization after text resampling
        wordlist = text
    wordfreq = Counter(wordlist)
    return wordfreq


def resample_text_to_length(text, length=1000):
    wordlist = text.split(' ')
    resampled_wordlist = np.random.choice(wordlist, size=length, replace=True)
#     resampled_text = ' '.join(resampled_wordlist) 
    return resampled_wordlist #return raw wordlist for speed optimization


def get_braycurtis_similarity(freqs_1, freqs_2):
    if ((len(freqs_1)==0) and (len(freqs_2)==0)): return None
    items_both = [x for x in freqs_1 if x in freqs_2] #faster than set union
    
    total_freq_in_both = sum([min(freqs_1[item], freqs_2[item]) for item in items_both])
    BC_sim = 2*(float(total_freq_in_both) / (sum(freqs_1.values()) + sum(freqs_2.values())))
    
    return BC_sim


def get_cosine_similarity(freqs_1, freqs_2):
    if ((len(freqs_1)==0) and (len(freqs_2)==0)): return None
    items_both = [x for x in freqs_1 if x in freqs_2] #faster than set union
    
    sum_1_times_2 = sum([freqs_1[item]*freqs_2[item] for item in items_both])
    sum_1_squared = sum([val*val for val in freqs_1.values()])
    sum_2_squared = sum([val*val for val in freqs_2.values()]) 
    cosine_sim = sum_1_times_2/(np.sqrt(sum_1_squared) * np.sqrt(sum_2_squared))
    
    return cosine_sim


def get_shared_word_frac_sim(ref_freqs, freqs):
    if ((len(ref_freqs)==0) and (len(freqs)==0)): return None
    items_both = [x for x in freqs if x in ref_freqs] #faster than set union
    
    total_freq_in_both = sum([min(ref_freqs[item], freqs[item]) for item in items_both])
    sim = float(total_freq_in_both) / sum(freqs.values())
    return sim


def get_aggregated_texts(texts, how):
    if how=='as_single_text':
        texts = [' '.join(texts['text'])]
    elif how=='by_text':
        texts = texts['text']
    elif how=='by_source':
        response_type = texts['text_type'].iloc[0]
        if response_type == 'commons_speech' or response_type == 'tweet':
            texts = texts.groupby('mp_name')['text'].agg(lambda x: ' '.join(x))
        elif response_type == 'url': #agg urls by text
            texts = texts['text']
    return list(texts)


def get_distances(ref_embedding, embeddings, dist_func):
    dists = [dist_func(ref_embedding, embedding) for embedding in embeddings]
    return pd.Series(dists)

def set_dist_func(dist_metric):
    if dist_metric == 'cosine': 
        dist_func = get_cosine_similarity
    elif dist_metric == 'BC': 
        dist_func = get_braycurtis_similarity
    return dist_func
    
def get_change_in_distance(ref_text, before_texts, after_texts, how_aggregate, 
                           n_resample = how_many_resamples_of_text, dist_func = None):
    if dist_func is None: dist_func = set_dist_func(dist_metric)
    
    #get aggragated texts
    before_texts = get_aggregated_texts(before_texts, how=how_aggregate)
    after_texts = get_aggregated_texts(after_texts, how=how_aggregate)
    
    #resample texts to uniform length, if desired
    if n_resample>0:
        before_texts = [resample_text_to_length(x) for x in before_texts*n_resample]
        after_texts = [resample_text_to_length(x) for x in after_texts*n_resample]
    
    #get embeddings
    before_embeddings = [get_word_freq_dict(x) for x in before_texts]
    after_embeddings = [get_word_freq_dict(x) for x in after_texts]
    ref_embedding = get_word_freq_dict(ref_text['text'])

    #get before and after distances
    before_dists = get_distances(ref_embedding, before_embeddings, dist_func)
    after_dists = get_distances(ref_embedding, after_embeddings, dist_func)

    #compute difference
    change_in_distance = after_dists.mean() - before_dists.mean()
    
    return change_in_distance


####################
#main function
####################

def get_delta(ref_text, texts_dict_by_type, response_text_type, 
              period_in_days=14, how_aggregate = how_to_aggregate_texts,
              gap_length = post_event_gap_length, 
              control = run_control): 
    
    #define reference text embedding and time intervals
    begin = ref_text['time'] - pd.to_timedelta(period_in_days, unit='days')
    mid_left = ref_text['time']
    mid_right = ref_text['time'] + pd.to_timedelta(gap_length, unit='hours')
    end = ref_text['time'] + pd.to_timedelta(period_in_days, unit='days')
    
    #for control, swap the ref text content. 
    if control:
        stimulus_text_type = ref_text['text_type']
        if len(texts_dict_by_type[stimulus_text_type])>0:
            ref_text = texts_dict_by_type[stimulus_text_type].sample()
        else:
            return None

    #get before and after texts. slow queries.
    texts_of_response_type = texts_dict_by_type[response_text_type]
    before_texts = texts_of_response_type[texts_of_response_type['time'].between(begin,mid_left,'left')]
    after_texts = texts_of_response_type[texts_of_response_type['time'].between(mid_right,end,'right')]  

    if len(before_texts)>0 and len(after_texts)>0:
        change_in_distance = get_change_in_distance(ref_text, before_texts, after_texts, how_aggregate)
    else:
        change_in_distance = None
    
    return change_in_distance

########################################
#wrapper functions for parallelization
########################################

def add_deltas_to_df(results, texts_dict_by_type, response_text_type):
    results[response_text_type+'_sim_delta'] = results.apply( 
                          lambda x: get_delta(x, texts_dict_by_type, response_text_type, 14),
                          axis=1)
    return results

def add_commons_deltas_to_df(results):
    return add_deltas_to_df(results, texts_dict_by_type, 'commons_speech')

def add_url_deltas_to_df(results):
    return add_deltas_to_df(results, texts_dict_by_type, 'url')

def add_tweet_deltas_to_df(results):
    return add_deltas_to_df(results, texts_dict_by_type, 'tweet')


########################################
# run script below
########################################

print('running')
start = time.time()

if n_workers>1: #run parallel
    results = [pd.DataFrame(x) for x in np.array_split(results, n_workers*6)] #split into chunks for workers
    with Pool(processes=n_workers) as pool: #create mp pool and map results 
        if 'commons_speech' in response_types: results = pool.map(add_commons_deltas_to_df, results)
        if 'tweet' in response_types: results = pool.map(add_tweet_deltas_to_df, results)
        if 'url' in response_types: results = pool.map(add_url_deltas_to_df, results)
    results = pd.concat(results) #reduce
    
else:  #run sequential. mostly for easier debugging.  
    if 'commons_speech' in response_types: results = add_commons_deltas_to_df(results)
    if 'tweet' in response_types: results = add_tweet_deltas_to_df(results)
    if 'url' in response_types: results = add_url_deltas_to_df(results)

#count words
results['n_words_in_text'] = results['text'].apply(lambda x: len(x.split(' ')))

end = time.time()
print('time elapsed', end-start, 'seconds')


print('dropping duplicates and saving')
results = results.drop(columns = ['text']).reset_index(drop=True)
results.to_parquet(data_path+'results_'+
                   how_to_aggregate_texts+
                   parties_to_analyze_str+"_"+
                   str(post_event_gap_length)+'hr_'+
                   dist_metric+
                   '_r'+str(how_many_resamples_of_text)+
                   ('_control' if run_control else '')+
                   '.parquet')

eprint('done\n\n\n')
