import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import random
from wordcloud import WordCloud, STOPWORDS
from collections import Counter


def binary_to_rise_or_fall_string(num):
    return "Rose/Steady" if num == '1' else 'Fell'


# This function reads in the data set, and creates a bar chart, showing the
# number of up days next to the number of down days, for each year represented
# in the dataset.  The name is a misnomer
def high_level_histogram():
    data = pd.read_csv('../../data/headlines_combined.csv',
                       header=0,
                       usecols=['Date', 'Label'],
                       converters={'Label': binary_to_rise_or_fall_string},
                       parse_dates=['Date'])
    data = data.groupby([data['Date'].dt.year, 'Label']).count().unstack('Label')
    data.plot(kind='bar', mark_right=False)
    plt.legend(['Fell', 'Rose/Steady'])
    plt.title('Up and Down Days by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Days')
    plt.show()


def num_words(string):
    return len(string.split())


# length_histogram() creates a histogram of the lengths of the combined
# headline strings
def length_histogram():
    data = pd.read_csv('../../data/tokenized_headlines.csv',
                       header=0,
                       usecols=['Combined'],
                       converters={'Combined': num_words})
    data.hist()
    plt.xlabel('Number of Words')
    plt.ylabel('Number of Points')
    plt.title('')
    plt.show()


# distribution_props() was used to calculate the mean, standard deviation,
# 1st quartile, and 3rd quartile values of the lengths of the combined
# headline strings
def distribution_props():
    data = pd.read_csv('../../data/tokenized_headlines.csv',
                       header=0,
                       usecols=['Combined'],
                       converters={'Combined': num_words})
    print('mean: ', data.Combined.mean())
    print('stddev: ', data.Combined.std())
    print('75%tile: ', data.Combined.quantile(q=0.75))
    print('25%tile: ', data.Combined.quantile(q=0.25))


# build_cloud() creates a word cloud out of the supplied text in <text>
# using <color_func> to create the base color for the cloud, and <img_path>
# leads to a png file which is used to form the outline of the cloud.
# the cloud is output to a png file at <output_path>
def build_cloud(text, color_func, img_path,  output_path):
    mask = np.array(Image.open(img_path))
    stopwords = set(STOPWORDS)
    stopwords.add('||new_headline||')
    stopwords.add('new_headline')
    wc = WordCloud(max_words=2000,
                   mask=mask,
                   scale=3,
                   stopwords=stopwords,
                   prefer_horizontal=0.8,
                   margin=1,
                   background_color=None,
                   mode='RGBA',
                   random_state=1).generate(text)
    # plt.title(title)
    plt.figure(figsize=(5, 5))
    plt.imshow(wc.recolor(color_func=color_func, random_state=3),
                interpolation="bilinear")
    plt.axis('off')
    plt.show()
    wc.to_file(output_path)


# word_clouds() uses the build_cloud() method to create two word word clouds:
# 1 for days where the DJIA rose or stayed the same, and 1 for days where the
# DJIA fell.  It uses a png of an arrow pointing up, with a green color function
# for the days where the DJIA rose/stayed, and it uses a png of an arrow pointing
# down with a red color function for the days where the DJIA fell
def word_clouds():
    data = pd.read_csv('../../data/headlines_combined.csv',
                       header=0,
                       usecols=['Combined', 'Label'])

    positive_lines = data[data['Label'] == 1]['Combined']
    positive_text = positive_lines.str.cat(sep=' ')
    negative_lines = data[data['Label'] == 0]['Combined']
    negative_text = negative_lines.str.cat(sep=' ')

    def red_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
        return "hsl(0, 100%%, %d%%)" % random.randint(30, 80)

    def green_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
        return "hsl(100, 100%%, %d%%)" % random.randint(30, 80)

    build_cloud(negative_text,
                red_func,
                '../../documents/visuals/img_files/down_arrow.png',
                '../../documents/visuals/down_days_cloud.png')

    build_cloud(positive_text,
                green_func,
                '../../documents/visuals/img_files/up_arrow.png',
                '../../documents/visuals/up_days_cloud.png')
