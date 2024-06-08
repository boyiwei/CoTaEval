singers = ['Taylor Swift',
          'Morgan Wallen',
          'Beyonce',
          'Drake',
          'Yeat',
          'SZA',
          'Luke Combs',
          'Zach Bryan',
          'Kanye West',
          'Noah Kahan',
          'Toby Keith',
          'Dua Lipa',
          'Tate McRae',
          'Olivia Rodrigo',
          '21 Savage',
          'Doja Cat',
          'Travis Scott',
          'Jennifer Lopez',
          'Jelly Roll',
          'Chris Stapleton',
          'Jack Harlow',
          'Usher',
          'Ty Dolla $ign',
          'Teddy Swims',
          'Bob Marley And The Wailers',
          'Miley Cyrus',
          'The Weeknd',
          'Ariana Grande',
          'Benson Boone',
          'Kendrick Lamar',
          'Lainey Wilson',
          'Billie Eilish',
          'Metallica',
          'Fleetwood Mac',
          'Blackberry Smoke',
          'Lana Del Rey',
          'Future',
          'Peso Pluma',
          'Tyla',
          'Stray Kids',
          'Jung Kook',
          'J. Cole',
          'Bruno Mars',
          'Nicki Minaj',
          'Tyler, The Creator',
          'Post Malone',
          'Michael Jackson',
          'Bailey Zimmerman',
          'Ed Sheeran',
          'Nirvana'
          ]

from lyricsgenius import Genius
import pandas as pd



genius = Genius('Vu9FhINzap3csALK7aOJXbakEmgauRW9tUsFoy3NmKlNP8PXwTFxw19MPzlXcdfp')


singers_all = []
singer_ranks_all = []
song_titles_all = []
song_lyrics_all = []
pageviews_all = []
for i, singer in enumerate(singers):
    try:
        artist = genius.search_artist(singer, sort='popularity', max_songs=10)
    except:
        continue
    song_titles = [s.title for s in artist.songs]
    song_lyrics = [s.lyrics for s in artist.songs]
    pageviews = [s.stats.pageviews for s in artist.songs]
    song_titles_all.extend(song_titles)
    song_lyrics_all.extend(song_lyrics)
    singer_ranks_all.extend([i]*len(song_titles))
    singers_all.extend([singer]*len(song_titles))
    pageviews_all.extend(pageviews)
    # import pdb 
    # pdb.set_trace()
    df = pd.DataFrame()
    df['singer'] = singers_all
    df['singer_rank'] = singer_ranks_all
    df['pageviews'] = pageviews_all
    df['song_title'] = song_titles_all
    df['song_lyrics'] = song_lyrics_all
    df.to_csv('lyrics_popular.csv', index=False)
    print(len(df))