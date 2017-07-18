import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords

#make sure to run
#nltk.download()
punc_tokens = {'.': '||period||',
               ',': '||comma||',
               '"': '||quotation||',
               ';': '||semicolon||',
               '!': '||exclamation||',
               '?': '||question||',
               '(': '||left_paren||',
               ')': '||right_paren||',
               '--': '||hyphen||',
               '-': '||hyphen||',
               ':': '||colon||',
               '[': '||left_brack||',
               ']': '||right_brack||',
               '{': '||left_curly||',
               '}': '||right_curly||',
               '\n': '||newline||'}


# This takes in a string and replaces all of the punctuation with tokens
# from punc_tokens.  It also removes any stop words in the string
# it returns a list of the words in the string
def _split_remove_stopwords_tokenize_punc(combined_string):
    words = word_tokenize(combined_string)
    words = [word for word in words if word not in stopwords.words('english')]
    return [word.lower() if word not in punc_tokens.keys() else punc_tokens[word] for word in words]


# _tokenize_word_lists takes a list of words and a dictionary of words to integer
# ids, and it returns a string of the ids corresponding to the input list of words,
# separated by spaces
def _tokenize_word_lists(row_list, words_to_int):
    return ' '.join([str(words_to_int[word]) for word in row_list])


# do_split() opens the csv file containing the combined headline strings,
# and calls the above two methods on each row to remove stopwords, tokenize punctuation, and
# then create strings of word ids for the combined headline strings.  it outputs these
# strings to a new csv file
def do_split():
    data = pd.read_csv('../data/headlines_combined.csv', header=0)
    data['Combined'] = data.apply(lambda row: _split_remove_stopwords_tokenize_punc(row['Combined']), axis=1)

    words = set()
    for index, row in data.iterrows():
        words.update(row[3])

    words_to_int = {word: index for index, word in enumerate(words)}

    data['Combined'] = data.apply(lambda row: _tokenize_word_lists(row['Combined'], words_to_int), axis=1)

    data.to_csv('../data/tokenized_headlines.csv')
