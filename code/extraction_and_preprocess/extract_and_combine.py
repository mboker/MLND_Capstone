import pandas as pd


# do_extract_and_combine() opens the original data csv file, and for each
# row, it combines all 25 headlines into one string.  It inserts a "new headline"
# token at the start of each headline in the string, and it removes some
# undesired characters found at the beginnings and ends of many headlines
def do_extract_and_combine():
    data = pd.read_csv('../data/Combined_News_DJIA.csv', header=0, na_values=[])
    data.fillna('', inplace=True)
    data['Combined'] = ''
    for i in range(1, 26):
        data['Combined'] += ' ||new_headline|| ' + \
                            data['Top'+str(i)].str.strip('b\\\'" ')
        data = data.drop('Top'+str(i), axis=1)

    data.to_csv('../data/headlines_combined.csv')
