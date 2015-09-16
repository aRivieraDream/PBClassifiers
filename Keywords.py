import pypyodbc

# return a list of stories in one particular set from sql
#conn = pypyodbc.connect('Driver=FreeTDS;Server=ext.pitchbook.com;uid=pierce.young;pwd=fas44aca;database=dbd_copy')
#print conn.cursor().execute('select top 10 * from news').fetchone()[0]


def get_words(story_list):
    """
    Break up words from stories in story_list into counts
    return dictionary of counts
    """
    word_count = {}
    for story in story_list:
        split = story.split()
        for word in split:
            occurences = 1
            if word in word_count:
                occurences = word_count[word] + 1
            word_count[word] = occurences
    return word_count

if __name__ == '__main__':
    # sample list of stories to use as placeholder
    sample_mt = ['PitchBook raised $500 million in series A',
        'SugarCRM raised 200 million in series b']
    sample_ot = ['Datafox is a dumb companay', 'News is out of control',
        'Kelly Clarkson wins american idol']

    print get_words(sample_mt)

    
