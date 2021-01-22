import utils

def text_processing(dataset, Y=None):
  def count_punct(text):
    try:
        count = sum([1 for char in text if char in string.punctuation])
        return round(count/(len(text) - text.count(" ")), 3)*100
    except:
        return 0

  def count_numr(text):
    try:
        count = sum([1 for char in text.split() if char.isnumeric()])
        return count
    except:
        return 0
  
  dataset['title_len'] = len(dataset['title']) - dataset['title'].count(" ")
  dataset['body_len'] = len(dataset['text']) - dataset['text'].count(" ")
  dataset['punct%'] = count_punct(dataset['title'])

  dataset['title'] = re.sub('[^A-Za-z0-9 ]+', '', dataset['title'])
  dataset['body'] = re.sub('[^A-Za-z0-9 ]+', '', dataset['text'])


  dataset['title_num'] = count_numr(dataset['title'])
  dataset['body_num'] = count_numr(dataset['text'])

  #def cleaning(dataset, Y=None):
  ### Dataset Preprocessing
  #from nltk.corpus import stopwords
  #from nltk.stem.porter import PorterStemmer
  #import re
  ps = PorterStemmer()
  review = re.sub('[^a-zA-Z]', ' ', dataset['title'])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  dataset['title_clean']  =  review


  return dataset
if __name__ == '__main__':
