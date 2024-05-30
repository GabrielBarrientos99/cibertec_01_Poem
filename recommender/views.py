from django.shortcuts import render, get_object_or_404, redirect, HttpResponse
from .models import Poem
from .recommender import PoemRecommender
from django.conf import settings
import os
import gdown
import pandas as pd
from textblob import TextBlob

def index(request):
    recommendations = None
    theme = None
    if request.method == 'POST':
        theme = request.POST.get('theme')
        if theme:
            data_path = os.path.join(settings.BASE_DIR, 'recommender', 'data', 'dataPoemasTokenized.csv')

            if not os.path.exists(data_path):
                file_id = '1Jbf5WIkgtAxl65fnCxDIz2uNgyv6NLZP'
                url = f'https://drive.google.com/uc?export=download&id={file_id}'
                gdown.download(url, data_path, quiet=False)

            data = pd.read_csv(data_path)
            recommender = PoemRecommender(data)
            recommendations = recommender.recommend_poems(theme).to_dict(orient='records')
           
    return render(request, 'index.html', {'recommendations': recommendations, 'theme': theme})

def poem_detail(request, poem_id):
    poem = get_object_or_404(Poem, pk=poem_id)
    return render(request, 'poem_detail.html', {'poem': poem})

def user_profile(request):
    return render(request, 'user_profile.html')

def save_poem(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        poet = request.POST.get('poet')
        content = request.POST.get('content')
        tags = request.POST.get('tags')
        
        Poem.objects.create(title=title, poet=poet, content=content, tags=tags)
        
        return HttpResponse(status=204)

def saved_poems(request):
    poems = Poem.objects.all()
    
    for poem in poems:
        analysis = TextBlob(poem.content)
        poem.sentiment = analysis.sentiment.polarity
        poem.sentiment_percentage = round((analysis.sentiment.polarity + 1) * 50, 2)
                
        if poem.sentiment_percentage < 33:
            poem.color_class = 'bg-red-500'  # Rojo
        elif 33 <= poem.sentiment_percentage < 61:
            poem.color_class = 'bg-yellow-500'  # Amarillo
        else:
            poem.color_class = 'bg-green-400'  # Verde
         
    return render(request, 'saved_poems.html', {'poems': poems})



def analyze_poem(request):
    sentiment = None
    sentiment_label = None
    color_class = None
    sentiment_percentage = None
    poem_content = ''

    if request.method == 'POST':
        if 'analyze' in request.POST:
            poem_content = request.POST.get('poem_content', '')
            if poem_content:
                analysis = TextBlob(poem_content)
                sentiment = analysis.sentiment.polarity
                sentiment_percentage = round((sentiment + 1) * 50, 2)
                
                if sentiment_percentage < 33:
                    sentiment_label = 'Negativo'
                    color_class = 'bg-red-700'
                elif 33 <= sentiment_percentage < 61:
                    sentiment_label = 'Neutral'
                    color_class = 'bg-yellow-400'
                elif 62 <= sentiment_percentage < 75:
                    sentiment_label = 'Positivo'
                    color_class = 'bg-green-400'
                else:
                    sentiment_label = 'Positivo'
                    color_class = 'bg-green-400'
        elif 'clear' in request.POST:
            poem_content = ''

    return render(request, 'analyze_poem.html', {
        'sentiment': sentiment,
        'sentiment_label': sentiment_label,
        'color_class': color_class,
        'sentiment_percentage': sentiment_percentage,
        'poem_content': poem_content
    })


def delete_poem(request, poem_id):
    poem = get_object_or_404(Poem, id=poem_id)
    if request.method == 'POST':
        poem.delete()
        return redirect('saved_poems')
    return render(request, 'saved_poems.html', {'poems': Poem.objects.all()})
