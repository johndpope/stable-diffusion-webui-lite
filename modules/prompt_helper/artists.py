import os
import csv
from collections import namedtuple


Artist = namedtuple('Artist', ['name', 'weight', 'category'])


class ArtistsDatabase:
    
    def __init__(self, fp):
        self.cats = set()
        self.artists = []

        if not os.path.exists(fp):
            return

        with open(fp, newline='') as fn:
            reader = csv.DictReader(fn)
            for row in reader:
                artist = Artist(row['artist'], float(row['score']), row['category'])
                self.artists.append(artist)
                self.cats.add(artist.category)

        self.artists.sort()

    @property
    def categories(self):
        return sorted(self.cats)
