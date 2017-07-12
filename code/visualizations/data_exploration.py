import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import random
from wordcloud import WordCloud, STOPWORDS
from collections import Counter


def binary_to_rise_or_fall_string(num):
    return "Rose/Steady" if num == '1' else 'Fell'

def high_level_histogram():
    data = pd.read_csv('../../data/headlines_combined.csv',
                       header=0,
                       usecols=['Date', 'Label'],
                       converters={'Label': binary_to_rise_or_fall_string},
                       parse_dates=['Date'])
    data = data.groupby([data['Date'].dt.year, 'Label']).count().unstack('Label')
    data.plot(kind='bar', mark_right=False)
    # data['Label'].apply(pd.value_counts).plot(kind='bar')
    plt.legend(['Fell', 'Rose/Steady'])
    plt.title('Up and Down Days by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Days')
    plt.show()


def num_words(string):
    return len(string.split())


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


def distribution_props():
    data = pd.read_csv('../../data/tokenized_headlines.csv',
                       header=0,
                       usecols=['Combined'],
                       converters={'Combined': num_words})
    print('mean: ', data.Combined.mean())
    print('stddev: ', data.Combined.std())
    print('75%tile: ', data.Combined.quantile(q=0.75))
    print('25%tile: ', data.Combined.quantile(q=0.25))


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
