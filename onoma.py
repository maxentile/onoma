# construct a markov model for letter-by-letter transitions within
# words in the linux/unix system dictionary and use it to score
# proposed names

import numpy as np

def transition_counts(n_symbols,sequence_list):
    counts = np.zeros((n_symbols,n_symbols))
    for seq in sequence_list:
        for i in range(len(seq)-1):
            counts[seq[i],seq[i+1]] += 1
    return counts

def triple_counts(n_symbols,sequence_list):
    counts = np.zeros((n_symbols,n_symbols,n_symbols))
    for seq in sequence_list:
        for i in range(len(seq)-2):
            counts[seq[i],seq[i+1],seq[i+2]] += 1
    return counts

def row_normalized_transition_matrix(counts,smoothing=1):
    return ((counts.T+smoothing) / counts.sum(1)).T

def word_to_intlist(word):
    return [ord(letter)-ord('a') for letter in word]

def intlist_to_word(intlist):
    return [chr(num+ord('a')) for num in intlist]

def score_sequence(sequence,log_transmat):
    '''
    assuming log_transmat contains log-probabilities of transitions T_{j,i}

    Pr(sequence) = \prod_i(Pr(sequence[i+1] | sequence[i]))
    \log(Pr(sequence)) = \sum_i(\log(Pr(sequence[i+1] | sequence[i])))

    if the transition matrix models higher-order transitions (e.g. T_{k,j,i}),
    then update accordingly

    '''
    score = 0.0
    if len(log_transmat.shape)==2:
        for i in range(len(sequence) - 1):
            score += log_transmat[sequence[i],sequence[i+1]]
    elif len(log_transmat.shape)==3:
        for i in range(len(sequence) - 2):
            score += log_transmat[sequence[i],sequence[i+1],sequence[i+2]]
    return score

def sample_names(parts_list, n_samples=10):
        ''' generate names sampled uniformly from the cartesian product of parts_list '''
        from random import choice
        return [''.join([choice(parts) for parts in parts_list]) for _ in range(n_samples)]


def repair_word(word,transmat, allow_deletions=False):
    ''' find the highest cost transition (a,b) in the word and replace the (a,b)
    transition with (a,c,b), where c maximizes the likelihood of the word
    given the transition matrix'''

    intlist = word_to_intlist(word)
    transition_costs = [transmat[intlist[i],intlist[i+1]] for i in range(len(word)-1)]
    worst_transition = np.argmin(transition_costs)
    candidates = [intlist[:worst_transition+1]+[i]+intlist[1+worst_transition:] for i in range(len(transmat))]

    # also consider staying the same
    candidates = candidates + [intlist]

    if allow_deletions:
        # also consider removal of a single letter
        candidates = candidates + [intlist.remove(i) for i in range(len(intlist))]

    # score all candidates
    scores = [score_sequence(c,np.log(transmat)) for c in candidates]

    # return the best one
    return ''.join(intlist_to_word(candidates[np.argmax(scores)]))

def learn_dictionary():
    wordlist = open('/usr/share/dict/words').read().split()
    wordlist = [w for w in wordlist if len(w) >= 3 and
                str.isalpha(w) and str.islower(w)]
    count_mat = transition_counts(26,[word_to_intlist(word) for word in wordlist])
    row_normalized = row_normalized_transition_matrix(count_mat)
    return wordlist,row_normalized

if __name__ == '__main__':
    # model the dictionary
    wordlist,row_normalized = learn_dictionary()
    shortwords = [w for w in wordlist if len(w) <= 4]

    # generate candidate names
    prefixes = ['','nano','micro','iso','thermo','meta','chem','multi',
                'ex','post','inter','net','gen','auto','proto','trans',
                'syn','bio','exa','dyna','proteo','mol','evo','over',
                'uber','insta']
    calorimetry_word_stems = ['cal','calor','met','metr','enthalp','phase',
                              'array','heat','kelvin','react','ensembl',
                              'boltz','space','densi','solu','stat','state']
    web2point0_suffixes = ['','ry','ly','lr','ble','nix','ic','er',
                           #'yo','zio','zu','izzle','tini',
                           'gen','tech','inc','gram',
                           'ful','dyn','co','bit']
    parts_list = [prefixes,
                #shortwords + ['']*len(shortwords),
                calorimetry_word_stems,
                #shortwords + ['']*len(shortwords),
                web2point0_suffixes]
    num_possibilities = np.prod([len(l) for l in parts_list])
    print('There are {0} possible names given the inputs'.format(num_possibilities))

    n_samples = 1000

    print('Sampling {0} of these...'.format(n_samples))
    company_names = list(set(sample_names(parts_list,n_samples=n_samples)))

    # remove any company names that are in the dictionary or inputs
    raw_parts = prefixes+calorimetry_word_stems+web2point0_suffixes
    company_names = [name for name in company_names if name not in wordlist
                     and name not in raw_parts]

    highest_probability_transitions = [(chr(i+ord('a')),chr(np.argmax(row)+ord('a')))
                                for i,row in enumerate(row_normalized)]
    lowest_probability_transitions = [(chr(i+ord('a')),chr(np.argmin(row)+ord('a')))
                                for i,row in enumerate(row_normalized)]
    print('Most likely transitions:',highest_probability_transitions)
    print('\n\n')
    print('Least likely transitions:',lowest_probability_transitions)

    # print the highest scoring names
    def print_names(company_names,transmat=row_normalized,all_names=False):
        print('\n\nThe top-100 names:')
        print(sorted(company_names,key=lambda i:-score_sequence(word_to_intlist(i),np.log(transmat)))[:100])

        print('\n\nThe bottom-100 names:')
        print(sorted(company_names,key=lambda i:score_sequence(word_to_intlist(i),np.log(transmat)))[:100])

        print('\n\n100 random names:')
        print(company_names[:100])

        if all_names:
            print('\n\nAll names')
            print(sorted(company_names,key=lambda i:-score_sequence(word_to_intlist(i),np.log(transmat))))


    print_names(company_names)

    #barely different from 2nd-order results...
    #print('\n\nNow according to 3rd-order statistics...')
    #tri_transmat = triple_counts(26,[word_to_intlist(word) for word in wordlist]) + 1
    #tri_transmat /= np.sum(tri_transmat)
    #print_names(company_names,tri_transmat)

    repairing=False
    if repairing:
        print('\n\nRepairing words...\n')
        all_repaired = [repair_word(name,row_normalized) for name in company_names]
        repaired = [name for i,name in enumerate(all_repaired) if name!=company_names[i]]
        num_repaired = len(repaired)
        print('{0} of {1} words were repaired...'.format(len(repaired),len(all_repaired)))

        print_names(repaired)

        iter_count = 2
        while num_repaired > 0 and iter_count < 10:
            print('\n\nRepairing words: round {0}...\n'.format(iter_count))
            all_repaired = [repair_word(name,row_normalized) for name in repaired]
            repaired = [name for i,name in enumerate(all_repaired) if name!=repaired[i]]
            num_repaired = len(repaired)
            print('{0} of {1} words were repaired...'.format(len(repaired),len(all_repaired)))
            iter_count+=1
            if len(repaired)>0: print_names(repaired)
