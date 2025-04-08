from enum import Enum, auto

class Genre(Enum):
    ACTION = auto()
    ADVENTURE = auto()
    ANIMATION = auto()
    CHILDRENS = auto()
    COMEDY = auto()
    CRIME = auto()
    DOCUMENTARY = auto()
    DRAMA = auto()
    FANTASY = auto()
    FILM_NOIR = auto()
    HORROR = auto()
    MUSICAL = auto()
    MYSTERY = auto()
    ROMANCE = auto()
    SCIFI = auto()
    THRILLER = auto()
    WAR = auto()
    WESTERN = auto()
    UNKNOWN = auto()

genre_map = {
    "Action": Genre.ACTION,
    "Adventure": Genre.ADVENTURE,
    "Animation": Genre.ANIMATION,
    "Children's": Genre.CHILDRENS,
    "Comedy": Genre.COMEDY,
    "Crime": Genre.CRIME,
    "Documentary": Genre.DOCUMENTARY,
    "Drama": Genre.DRAMA,
    "Fantasy": Genre.FANTASY,
    "Film-Noir": Genre.FILM_NOIR,
    "Horror": Genre.HORROR,
    "Musical": Genre.MUSICAL,
    "Mystery": Genre.MYSTERY,
    "Romance": Genre.ROMANCE,
    "Sci-Fi": Genre.SCIFI,
    "Thriller": Genre.THRILLER,
    "War": Genre.WAR,
    "Western": Genre.WESTERN
}