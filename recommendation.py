import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from response_model import RecommendationResponse
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Recommendation:
    def __init__(self):
        self.original_data = pd.read_csv("data/food_data.csv")
        self.original_data = self.original_data.dropna()
        self.df = pd.read_csv("data/food_data.csv")
        nltk.download("wordnet")
        nltk.download("stopwords")
        self.df = self.df.dropna()
        self.df = self.df[
            [
                "Srno",
                "TranslatedRecipeName",
                "TranslatedIngredients",
                "TranslatedInstructions",
                "TotalTimeInMins",
                "Course",
                "Diet",
                "URL",
            ]
        ]
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: re.sub(r"[\d+]", "", x)
        )
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: re.sub(r"[^\w\s]", "", x)
        )
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: x.lower()
        )
        words = set(nltk.corpus.words.words())
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: " ".join(
                w
                for w in nltk.wordpunct_tokenize(x)
                if w.lower() in words or not w.isalpha()
            )
        )
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: " ".join(w for w in x.split() if len(w) > 2)
        )
        stop_words = stopwords.words("english")
        wordnet_lemmatizer = WordNetLemmatizer()
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: " ".join(wordnet_lemmatizer.lemmatize(w) for w in x.split())
        )
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: " ".join(w for w in x.split() if w not in stop_words)
        )
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: " ".join(w for w in x.split() if len(w) > 2)
        )
        custom_stopwords = [
            "cup",
            "teaspoon",
            "tablespoon",
            "soaked",
            "overnight",
            "chopped",
            "per",
            "chop",
            "slice",
            "tablesp",
            "sliced",
            "cunim",
            "chopped",
            "pound",
            "ounce",
            "inch",
            "pound",
            "pint",
            "quart",
            "gallon",
            "milliliter",
            "liter",
            "gram",
            "kilogram",
            "small",
            "medium",
            "large",
            "finely",
            "fresh",
            "dried",
            "ground",
            "cut",
            "thinly",
            "sliced",
            "peeled",
            "seeded",
            "grated",
            "crushed",
            "minced",
            "whole",
            "halved",
            "quartered",
            "clove",
            "head",
            "stalk",
            "bunch",
            "package",
            "can",
            "jar",
            "bottle",
            "container",
            "bag",
            "box",
            "carton",
            "envelope",
            "packet",
            "pouch",
            "bar",
            "stick",
            "drop",
            "dash",
            "pinch",
            "sprinkle",
            "to",
            "taste",
            "as",
            "needed",
            "for",
            "serving",
            "optional",
        ]
        self.df["TranslatedIngredients"] = self.df["TranslatedIngredients"].apply(
            lambda x: " ".join(w for w in x.split() if w not in custom_stopwords)
        )

    def recommend(self, given_ingred):
        self.filtered_df = self.df[
            self.df["TranslatedIngredients"].apply(
                lambda x: any(ingred in x for ingred in given_ingred)
            )
        ]
        ingred_list = self.filtered_df["TranslatedIngredients"]
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(ingred_list)
        ip_vector = tfidf.transform([" ".join(given_ingred)])
        sim_cos = cosine_similarity(tfidf_matrix, ip_vector)
        top_n = 5
        top_n_idx = sim_cos.flatten().argsort()[-top_n:][::-1]
        return top_n_idx

    def getdishes(self, ingredients: list):
        idx = self.recommend(ingredients)
        response = []
        for i in idx:
            row = self.filtered_df.iloc[i]
            response.append(
                RecommendationResponse(
                    name=row["TranslatedRecipeName"].replace("\xa0", " "),
                    ingredients=row["TranslatedIngredients"].replace("\xa0", " "),
                    instructions=row["TranslatedInstructions"].replace("\xa0", " "),
                    link=row["URL"],
                    diet=row["Diet"],
                    course=row["Course"],
                    time=row["TotalTimeInMins"],
                )
            )
        return response


if __name__ == "__main__":
    rec = Recommendation()
    print(rec.getdishes(["milk", "sugar", "ghee"]))
