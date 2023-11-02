from flask import Flask, render_template, request
import os
import csv
from googleapiclient.discovery import build
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import re
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Style
from typing import Dict
from flask import send_file

app = Flask(__name__)


DEVELOPER_KEY = "AIzaSyCDTKQJOD4aCJP6WdQtnC2PM7zlkUnfCSA"
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
# Create a client object to interact with the YouTube API
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

def delete_non_matching_csv_files(directory_path, video_id):
    for file_name in os.listdir(directory_path):
        if not file_name.endswith('.csv'):
            continue
        if file_name == f'{video_id}.csv':
            continue
        os.remove(os.path.join(directory_path, file_name))

def extract_video_id(youtube_link):
    video_id_regex = r"^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(video_id_regex, youtube_link)
    if match:
        video_id = match.group(1)
        return video_id
    else:
        return None

def analyze_sentiment(csv_file):
    sid = SentimentIntensityAnalyzer()

    comments = []
    with open(csv_file, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            comments.append(row['Comment'])

    num_neutral = 0
    num_positive = 0
    num_negative = 0
    for comment in comments:
        sentiment_scores = sid.polarity_scores(comment)
        if sentiment_scores['compound'] == 0.0:
            num_neutral += 1
        elif sentiment_scores['compound'] > 0.0:
            num_positive += 1
        else:
            num_negative += 1

    results = {'num_neutral': num_neutral, 'num_positive': num_positive, 'num_negative': num_negative}
    return results

def bar_chart(csv_file: str) -> None:
    results: Dict[str, int] = analyze_sentiment(csv_file)

    num_neutral = results['num_neutral']
    num_positive = results['num_positive']
    num_negative = results['num_negative']

    df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Number of Comments': [num_positive, num_negative, num_neutral]
    })

    fig = px.bar(df, x='Sentiment', y='Number of Comments', color='Sentiment', 
                 color_discrete_sequence=['#87CEFA', '#FFA07A', '#D3D3D3'],
                 title='Sentiment Analysis Results')
    fig.update_layout(title_font=dict(size=20))

    return fig

def plot_sentiment(csv_file: str) -> None:
    results: Dict[str, int] = analyze_sentiment(csv_file)

    num_neutral = results['num_neutral']
    num_positive = results['num_positive']
    num_negative = results['num_negative']

    labels = ['Neutral', 'Positive', 'Negative']
    values = [num_neutral, num_positive, num_negative]
    colors = ['yellow', 'green', 'red']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 marker=dict(colors=colors))])
    fig.update_layout(title={'text': 'Sentiment Analysis Results', 'font': {'size': 20, 'family': 'Arial', 'color': 'grey'},
                              'x': 0.5, 'y': 0.9},
                      font=dict(size=14))

    return fig

def get_channel_id(video_id):
    response = youtube.videos().list(part='snippet', id=video_id).execute()
    channel_id = response['items'][0]['snippet']['channelId']
    return channel_id


def save_video_comments_to_csv(video_id):
    comments = []
    results = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText'
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            username = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comments.append([username, comment])
        if 'nextPageToken' in results:
            nextPage = results['nextPageToken']
            results = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=nextPage
            ).execute()
        else:
            break

    filename = video_id + '.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Username', 'Comment'])
        for comment in comments:
            writer.writerow([comment[0], comment[1]])

    return filename

def get_video_stats(video_id):
    try:
        response = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()

        return response['items'][0]['statistics']

    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def get_channel_info(youtube, channel_id):
    try:
        response = youtube.channels().list(
            part='snippet,statistics,brandingSettings',
            id=channel_id
        ).execute()

        channel_title = response['items'][0]['snippet']['title']
        video_count = response['items'][0]['statistics']['videoCount']
        channel_logo_url = response['items'][0]['snippet']['thumbnails']['high']['url']
        channel_created_date = response['items'][0]['snippet']['publishedAt']
        subscriber_count = response['items'][0]['statistics']['subscriberCount']
        channel_description = response['items'][0]['snippet']['description']

        channel_info = {
            'channel_title': channel_title,
            'video_count': video_count,
            'channel_logo_url': channel_logo_url,
            'channel_created_date': channel_created_date,
            'subscriber_count': subscriber_count,
            'channel_description': channel_description
        }

        return channel_info

    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_comments/<video_id>', methods=['GET'])
def download_comments(video_id):
    csv_file = f'{video_id}.csv'
    return send_file(csv_file, as_attachment=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    youtube_link = request.form['youtube_link']
    directory_path = os.getcwd()


    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            channel_id = get_channel_id(video_id)
            csv_file = save_video_comments_to_csv(video_id)
            delete_non_matching_csv_files(directory_path, video_id)
            results = analyze_sentiment(csv_file)
            bar_chart_fig = bar_chart(csv_file)
            plot_sentiment_fig = plot_sentiment(csv_file)
            channel_info = get_channel_info(youtube, channel_id)
            video_stats = get_video_stats(video_id)
            
            return render_template('results.html', 
                               video_id=video_id,
                               results=results,
                               channel_title=channel_info['channel_title'],
                               video_count=channel_info['video_count'],
                               channel_logo_url=channel_info['channel_logo_url'],
                               channel_created_date=channel_info['channel_created_date'],
                               subscriber_count=channel_info['subscriber_count'],
                               total_views=video_stats['viewCount'],
                               like_count=video_stats['likeCount'],
                               comment_count=video_stats['commentCount'],
                               youtube_link=youtube_link,
                               bar_chart=bar_chart_fig.to_html(full_html=False),
                               plot_sentiment=plot_sentiment_fig.to_html(full_html=False))
        else:
            return "Invalid YouTube link"
    return "No YouTube link provided"

if __name__ == '__main__':
    app.run(debug=True)
