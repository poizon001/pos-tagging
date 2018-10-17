#!/bin/python
import nltk
from nltk.stem.porter import PorterStemmer

abbreviations = {"AFAIK", "BCOZ","B/C","BFN","BR","BTW","DM","EM","FB","FF","FFS","FML","FTF","FTL",\
                    "FTW","FWD","FWIW","HT","HTH","IMHO","IMO","IRL","JV","J/K","LI","LMAO",\
                    "LMK","LOL","MT","NSFW","OH","OMFG","OMG","PRT","RE","RR","RT","RTF","RTFM",\
                    "RTHX","SNAFU","SOB","STFU","TMB","TMI","WTF","WTH","YMMV","YW","TIL","NB",\
                    "ICYMI","CX","RTQ","STFW","TL","TL;DR","TT", "YOYO", "OH", "FTL", "FTW", "F2F"\
                    "DM", "TMB", "YOLO", "MOFO", "NP", "TTYL", "GTG", "CMB", "BRB", "ASAP", "PF", "MSG"\
                    "LMK", "WTH", "TWEETUP", "ICUMI", "CBH", "SM", "GRAND", "ASN", "YL", "DINK", "CC"}

conjunctions = {"WHAT","FOR","SINCE","WHEN","TILL","UNTIL","UNLESS","THOUGH","AFTER","BUT","WHENEVER",\
                "ALTHOUGH","AS LONG AS","NOR","EVEN IF","WHETHER","WHEREVER","WHILE","SO","EVEN THOUGH",\
                "OR","ONCE","AND","SO THAT","BECAUSE","AS IF","AS","YET","BEFORE"}

quntifiers = {"LITTLE", "NOT MANY ", "FEW", "LOTS", "MANY", "LESS", "SOME", "MUCH", "FEW", "LOT", "LARGE NUMBER",\
                 "NOT ANY", "PLENTY", "BIT", "SEVERAL", "NUMEROUS", "MORE", "SOMETHING"}

contractions = {"n't","'t", "'cause", "'ve", "'d", "'ll", "'s", "'m", "'am", "'clock", "'n", "'re", "'all"}

personlaity_adj = {"agreeable","brave","calm","delightful","eager","faithful","gentle","happy","jolly","kind",\
                "lively","nice","obedient","proud","relieved","silly","thankful","victorious","witty","zealous",\
                "angry","bewildered","clumsy","defeated","embarrassed","fierce","grumpy","helpless","itchy","jealous",\
                "lazy","mysterious","nervous","obnoxious","panicky","repulsive","scary","thoughtless","uptight","worried"}

shape_size_adj = {"big", "colossal", "fat", "gigantic", "great", "huge", "immense", "large", "little", "mammoth",\
                 "massive", "miniature", "petite", "puny", "scrawny", "short", "small", "tall", "teeny", "teeny-tiny",\
                 "tiny", "broad", "chubby", "crooked", "curved", "deep", "flat", "high", "hollow", "low", "narrow",\
                  "round", "shallow", "skinny", "square", "steep", "straight", "wide"}

noun_suffix = {"acy", "al", "ance", "ence", "dom", "er", "or", "ism", "ist", "ity", "ty", "ment", "ness", "ship",\
                 "ness", "sion", "tion"}

prefix = {"ante", "anti", "circum", "co", "de", "dis", "em", "en", "epi", "ex", "extra", "fore", "homo", "hyper", "il",\
             "im", "in", "infra", "inter", "intra", "macro", "micro", "mid", "mis", "mono", "non", "omni", "para", "pre",\
             "post", "re", "semi", "sub", "super", "therm", "trans", "tri", "un", "uni"}

verb_suffix = {"ate", "en", "ify", "fy", "ise", "ize", "ing", "ed"}

adj_suffix = {"able", "ible", "al", "esque", "ful", "ic", "ical", "ious", "ous", "ive", "less", "y"}

adverb_suffix = {"ly", "ward", "wards", "wise"}

transition_words = {"indeed","further","either", "neither","also", "moreover","furthermore", "besides","actually",\
                    "too","let", "additionally","nor","alternatively"}
global common_adjs

porter = PorterStemmer()

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """

    global common_adjs
    with open("common_adj.csv", "r") as file:
        for line in file:
            common_adjs = set(line.split(","))

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())

    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # Feature added
    #1. Stemmed word
    ftrs.append("STEMMED=" + porter.stem(word))
    
    #2. word is abbreviation
    if word.upper() in abbreviations:
        ftrs.append("IS_ABVT")

    #3. word is quantifier
    if word.upper() in quntifiers:
        ftrs.append("IS_QUNTF")

    #4. contains hyphen
    if word.find("-") != -1:
        ftrs.append("IS_HAVING_HYPHN")

    #5. word starts with  
    if word.startswith("#"):
        ftrs.append("IS_HASHTAG")

    #6. word is a mention
    if word.startswith("@"):
        ftrs.append("IS_MENTION")

    #7. word is contraction
    if word.lower() in contractions:
        ftrs.append("IS_CONTRCN")

    #8. word is personlaity adjacetive
    if word.lower() in personlaity_adj:
        ftrs.append("IS_PADJ")

    #9. word is shape-size adjacetive
    if word.lower() in shape_size_adj:
        ftrs.append("IS_SSADJ")

    #10. word is shape-size adjacetive
    if word.lower() in common_adjs:
        ftrs.append("IS_CMADJ")

    #11. word is noun_suffix
    for x in noun_suffix:
        if x in word.lower():
            ftrs.append("IS_NOUN_SFX")
            break

    #12. word is verb_suffix
    for x in verb_suffix:
        if x in word.lower():
            ftrs.append("IS_VRB_SFX")
            break

    #13. word is adj_suffix
    for x in adj_suffix:
        if x in word.lower():
            ftrs.append("IS_ADJ_SFX")
            break

    #14. word is adverb_suffix
    for x in adverb_suffix:
        if x in word.lower():
            ftrs.append("IS_ADV_SFX")
            break

    #14. word is having prefix
    for x in prefix:
        if x in word.lower():
            ftrs.append("IS_PREFX")
            break

    #15. word is having prefix
    for x in prefix:
        if x in word:
            ftrs.append("IS_PREFX")
            break

    if word.lower() in transition_words:
        ftrs.append("IS_TNSN")


    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ],
    [ "Rony", "is", "saying", "stfu", "LITTLE", "shocked"]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)

