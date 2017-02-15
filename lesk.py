# importing essential libraries and modules.
import string
from itertools import chain

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

from cosine import cosine_similarity as cos_sim
from utils import lemmatize, porter, lemmatize_sentence, synset_properties

EN_STOPWORDS = stopwords.words('english')

def compare_overlaps(context, synsets_signatures, \
                     nbest=False, keepscore=False, normalizescore=False):
    """ 
    Calculates overlaps between the context sentence and the synset_signture
    and returns a ranked list of synsets from highest overlap to lowest.
    """
    overlaplen_synsets = [] # a tuple of (len(overlap), synset).
    for ss in synsets_signatures:
        overlaps = set(synsets_signatures[ss]).intersection(context)
        overlaplen_synsets.append((len(overlaps), ss))
    
    # Rank synsets from highest to lowest overlap.
    ranked_synsets = sorted(overlaplen_synsets, reverse=True)
    
    # Normalize scores such that it's between 0 to 1. 
    if normalizescore:
        total = float(sum(i[0] for i in ranked_synsets))
        ranked_synsets = [(i/total,j) for i,j in ranked_synsets]
      
    if not keepscore: # Returns a list of ranked synsets without scores
        ranked_synsets = [i[1] for i in sorted(overlaplen_synsets, \
                                               reverse=True)]
      
    if nbest: # Returns a ranked list of synsets.
        return ranked_synsets
    else: # Returns only the best sense.
        return ranked_synsets[0]

def simple_signature(ambiguous_word, pos=None, lemma=True, stem=False, \
                     hyperhypo=True, stop=True):
    """ 
    Returns a synsets_signatures dictionary that includes signature words of a 
    sense from its:
    (i)   definition
    (ii)  example sentences
    (iii) hypernyms and hyponyms
    """
    synsets_signatures = {}
    for ss in wn.synsets(ambiguous_word):
        try: # If POS is specified.
            if pos and str(ss.pos()) != pos:
                continue
        except:
            if pos and str(ss.pos) != pos:
                continue
        signature = []
        # Includes definition.
        ss_definition = synset_properties(ss, 'definition')
        signature+=ss_definition
        # Includes examples
        ss_examples = synset_properties(ss, 'examples')
        signature+=list(chain(*[i.split() for i in ss_examples]))
        # Includes lemma_names.
        ss_lemma_names = synset_properties(ss, 'lemma_names')
        signature+= ss_lemma_names
        
        # Optional: includes lemma_names of hypernyms and hyponyms.
        if hyperhypo == True:
            ss_hyponyms = synset_properties(ss, 'hyponyms')
            ss_hypernyms = synset_properties(ss, 'hypernyms')
            ss_hypohypernyms = ss_hypernyms+ss_hyponyms
            signature+= list(chain(*[i.lemma_names() for i in ss_hypohypernyms]))
        
        # Optional: removes stopwords.
        if stop == True: 
            signature = [i for i in signature if i not in EN_STOPWORDS]
        # Lemmatized context is preferred over stemmed context.
        if lemma == True: 
            signature = [lemmatize(i) for i in signature]
        # Matching exact words may cause sparsity, so optional matching for stems.
        if stem == True: 
            signature = [porter.stem(i) for i in signature]
        synsets_signatures[ss] = signature
        
    return synsets_signatures

def simple_lesk(context_sentence, ambiguous_word, \
                pos=None, lemma=True, stem=False, hyperhypo=True, \
                stop=True, context_is_lemmatized=False, \
                nbest=False, keepscore=False, normalizescore=False):
    """
    Simple Lesk is somewhere in between using more than the 
    original Lesk algorithm (1986) and using less signature 
    words than adapted Lesk (Banerjee and Pederson, 2002)
    """
    # Ensure that ambiguous word is a lemma.
    ambiguous_word = lemmatize(ambiguous_word) 
    # If ambiguous word not in WordNet return None
    if not wn.synsets(ambiguous_word):
        return None
    # Get the signatures for each synset.
    ss_sign = simple_signature(ambiguous_word, pos, lemma, stem, hyperhypo)
    # Disambiguate the sense in context.
    if context_is_lemmatized:
        context_sentence = context_sentence.split()
    else:
        context_sentence = lemmatize_sentence(context_sentence)
    best_sense = compare_overlaps(context_sentence, ss_sign, \
                                    nbest=nbest, keepscore=keepscore, \
                                    normalizescore=normalizescore)  
    return best_sense
