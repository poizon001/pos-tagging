import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """

    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    #T - Score matrix same as in assignement pdf
    T = np.zeros(shape=(L,N))
    #Back pointers - to store the previous best tag for word at (i-1)th position
    #that resulted into current best tag for (i)th word 
    back_pointer = np.full((L,N), -1)

    for i in xrange(L):
        emission = emission_scores[0][i]
        combined = emission + start_scores[i]
        T[i][0] = combined

    # Loop over all the words in a sequesnce
    for i in xrange(1, N):
        # Loop over all the tags for the word at index i 
        for j in xrange(L):
            # Varibale for maximum tag score from previous word (word at i-1)
            tmp_max = float('-inf')
            tmp_max_idx = -1
            #Emission value of word at idx i from state (i.e tag) j
            emission = emission_scores[i][j]
            #Loop over all the possibile tags for previous word T[tag (1..L), word at i-1]
            #and get max among them. Store the corresponding back pointer for there T[tag (1..L), word at i-1]
            for k in xrange(L):
                transition = trans_scores[k][j]
                prev_path = T[k][i-1]
                combined = transition + prev_path
                if (tmp_max < combined):
                    tmp_max = combined
                    tmp_max_idx = k

            back_pointer[j][i] = tmp_max_idx
            T[j][i] = tmp_max + emission

    # Doing this step outside because if N == 1 then above loop will not run
    # Variable for maximum tag score
    tag_max = float('-inf')
    # Variable for back pointer(previous T[tag, word])
    tag_max_idx = -1
    for i in xrange(L):
        T[i][N-1] = T[i][N-1] + end_scores[i]
        if (tag_max < T[i][N-1]):
            tag_max = T[i][N-1]
            tag_max_idx = i
    # print("Max tag -> " + str(tag_max_idx))

    #Variable to track the path length - should be equal to N
    path_length = 0
    #Variable to back track on the tags
    tag_idx = tag_max_idx
    #Varibale to track the word index in N
    word_idx = N-1 
    #Path strored using backtracking
    y = []

    #Getting the best path using backtracking on back_pointers
    while path_length != N-1:
        y.append(back_pointer[tag_idx][word_idx])
        tag_idx = back_pointer[tag_idx][word_idx]
        word_idx = word_idx - 1
        path_length = path_length + 1

    #Reversing the backtracked path
    y = y[::-1]
    #Adding the tag for the last word idx in N
    y.append(tag_max_idx)
    # print("Path -> " + str(y))

    return (tag_max, y)
