from flask import Flask, render_template, request, jsonify, url_for
import json
import requests
import pickle
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from imdb import Cinemagoer

ia = Cinemagoer()

headers = {
    "X-RapidAPI-Key": "ef33758c26msh90b77f1145da547p18115fjsnb013a57e8018",
    "X-RapidAPI-Host": "imdb8.p.rapidapi.com"
}

filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

tfidffile = 'tfidf.sav'
tf_idf = pickle.load(open(tfidffile, 'rb'))


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


def search(input):
    url = "https://imdb8.p.rapidapi.com/title/v2/find"
    querystring = {"title": input, "titleType": "movie,tvSeries,tvMiniSeries,tvMovie,tvSpecial,tvShort",
                   "limit": "20", "sortArg": "moviemeter,asc"}
    response = requests.request(
        "GET", url, headers=headers, params=querystring)
    response = json.loads(response.text)
    response = response["results"]

    results = []
    for x in response:
        year = imgurl = None
        id = x["id"].replace('/title/', '')[:-1]
        if ("image" in x):
            imgurl = x["image"]["url"]
        title = x["title"]
        if ("year" in x):
            year = x["year"]
        type = x["titleType"][0].upper() + x["titleType"][1:]
        if len(type) > 5:
            type = type[:2] + ' ' + type[2:]
        results.append([id, imgurl, title, year, type])

    return results


def getReviews(titleId):
    start_url = 'https://www.imdb.com/title/%s/reviews?ref_=tt_urv' % titleId
    link = 'https://www.imdb.com/title/%s/reviews/_ajax' % titleId

    params = {
        'ref_': 'undefined',
        'paginationKey': ''
    }
    reviews = []
    with requests.Session() as s:
        s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
        res = s.get(start_url)

        while True:
            soup = BeautifulSoup(res.text, "lxml")
            for item in soup.select(".review-container"):
                review = item.select_one("div.show-more__control")
                reviews.append(review)

            try:
                pagination_key = soup.select_one(
                    ".load-more-data[data-key]").get("data-key")
            except AttributeError:
                break
            params['paginationKey'] = pagination_key
            res = s.get(link, params=params)
    return reviews


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input = request.form['title']

        output = search(input)

        return render_template('index.html', test=output)
    else:
        return render_template('index.html', test='')


@app.route('/reviews/<title>', methods=['GET', 'POST'])
def reviews(title):

    # clean_reviews.append(good_reviews + bad_reviews)
    # clean_reviews.append(good_reviews)
    # clean_reviews.append(bad_reviews)
    if request.method == 'POST':
        movie = ia.get_movie(title[2:])
        plot = movie['plot'][0]

        genres = movie['genres']

        image = request.form['image']
        movieTitle = request.form['title']
        year = request.form['year']

        reviews = getReviews(title)
        reviews = list(map(str, reviews))
        good_reviews = 0
        bad_reviews = 0
        for i in reviews:
            test_processes = [review_to_words(i)]
            test_input = tf_idf.transform(test_processes)
            res = model.predict(test_input)[0]
            if res == 1:
                good_reviews += 1
            else:
                bad_reviews += 1

        total_reviews = good_reviews + bad_reviews
        good_reviews_perc = round((good_reviews/total_reviews) * 100)
        return render_template('reviews.html', image=image, plot=plot, title=movieTitle, year=year, genres=genres, total=total_reviews, perc=good_reviews_perc)
