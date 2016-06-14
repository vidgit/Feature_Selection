def sim(word1, word2, lch_threshold=4.15, verbose=False):
    from nltk.corpus import wordnet as wn
    results = []
    kek=10000000    # This exists to give the best matching link
    for net1 in wn.synsets(word1):
        for net2 in wn.synsets(word2):
            try:
                lch = net1.lch_similarity(net2)
            except:
                continue
            # The value to compare the LCH to was found empirically.
            # (The value is very application dependent. Experiment!)
            if lch >= lch_threshold:
                results.append((net1, net2))
                kek=min(kek,lch)
    # print kek
    if not results:
        return (False,kek)
    if verbose:
        for net1, net2 in results:
            print net1
            print net1.definition
            print net2
            print net2.definition
            print 'path similarity:'
            print net1.path_similarity(net2)
            print 'lch similarity:'
            print net1.lch_similarity(net2)
            print 'wup similarity:'
            print net1.wup_similarity(net2)
            print '-' * 79
    return (True,kek)