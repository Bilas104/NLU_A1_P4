import os
import glob
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

class NewsClassifierPipeline:
    def __init__(self):
        self.dataset_slug = "kishanyadav/inshort-news"
        self.data_path = "news_data"
        self.df = None
        self.results = {}
        
        # TF-IDF Configuration
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2), 
            max_features=5000
        )

    def download_data(self):
        # Check if data path exists and contains CSV files
        if os.path.exists(self.data_path):
            existing_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            if existing_files:
                n_files = len(existing_files)
                if n_files == 7:
                    print(f"Data found in '{self.data_path}' {n_files} files). Skipping download.")
                    return True

                else:
                    print(f"Data found in '{self.data_path}' but expected 7 files, found {n_files}.")
                    return False
            
        os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
        
        if not os.path.exists('kaggle.json'):
            print("'kaggle.json' not found in the current directory.")
            print("Please place your API key file here to proceed.")
            return False

        # 2. Download and Unzip
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            print(f"Authenticated successfully. Downloading {self.dataset_slug}...")

            api.dataset_download_files(
                self.dataset_slug, 
                path=self.data_path, 
                unzip=True
            )

            print("Download and extraction complete.")

            return True
        
        except Exception as e:

            print(f"API Error: {e}")
            return False

    def merge_and_filter(self):        
        # The dataset is split into 7 csv files, so we are reading and combining them into one dataframe.
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        if not csv_files:
            print("No CSV files found.")
            return False
            
        print(f"Found {len(csv_files)} file shards.")
        
        dfs = []

        for f in csv_files:
            try:
                temp = pd.read_csv(f)
                
                if {'news_headline', 'news_article', 'news_category'}.issubset(temp.columns):
                    dfs.append(temp)

            except:
                pass
        
        if not dfs:
            return False
        
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Standardization
        full_df['news_category'] = full_df['news_category'].str.lower().str.strip()
        
        # Filter for Sports & Politics
        target_cats = ['sports', 'politics']
        self.df = full_df[full_df['news_category'].isin(target_cats)].copy()
        
        # Deduplication
        before = len(self.df)
        self.df.drop_duplicates(subset=['news_article'], inplace=True)
        
        print(f"Data Ready: {len(self.df)} samples (Dropped {before - len(self.df)} duplicates).")
        print(f"Distribution:\n{self.df['news_category'].value_counts()}")
        return True

    def prepare_features(self):
        # Combining headline and article for richer text representation
        self.df['text'] = self.df['news_headline'] + " " + self.df['news_article']
        
        X = self.vectorizer.fit_transform(self.df['text'])
        y = self.df['news_category']
        
        # We are performing a stratified split as the two labels are imabalanced.
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def run_experiment(self, X_train, X_test, y_train, y_test):
        
        models = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(max_iter=1000),
                'params': {'C': [1, 10]}
            },

            {
                'name': 'SVM',
                'model': SVC(),
                'params': {'C': [0.1, 1, 10],
                           'kernel': ['linear', 'rbf']}
            },

            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(),
                'params': {'n_estimators': [50, 100],
                           'max_depth': [None, 20, 50]}
            },

            {
                'name': 'Multinomial NB',
                'model': MultinomialNB(),
                'params': {'alpha': [0.1, 0.5, 1.0]}
            }
        ]

        for config in models:
            print(f"\nTuning {config['name']}...")
            t0 = time.time()
            
            # 3-Fold Cross Validation
            clf = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1)
            clf.fit(X_train, y_train)
            
            best_model = clf.best_estimator_
            preds = best_model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            self.results[config['name']] = {
                'accuracy': acc,
                'best_params': clf.best_params_,
                'time': time.time() - t0,
                'confusion': confusion_matrix(y_test, preds)
            }

            print(f"    Best Params: {clf.best_params_}")
            print(f"    Test Accuracy: {self.results[config['name']]['accuracy']:.4f}")
            print(f"    Time Taken: {self.results[config['name']]['time']:.2f} seconds")

    def generate_plots(self):
        if not self.results:
            return
        
        # Performance Bar Chart
        names = list(self.results.keys())
        accs = [self.results[n]['accuracy'] for n in names]
        
        plt.figure(figsize=(10, 6))

        sns.barplot(x=names, y=accs, hue=names, palette='viridis', legend=False)

        plt.title('Classifier Accuracy: Sports vs Politics')
        plt.ylim(0.9, 1.0)
        plt.ylabel('Accuracy Score')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig('comparison.png')
        print("Saved 'comparison.png'")
        
        # Confusion Matrix of the best model
        best_performing_model = max(self.results, key=lambda x: self.results[x]['accuracy'])
        cm = self.results[best_performing_model]['confusion']
        
        plt.figure(figsize=(6, 5))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Politics', 'Sports'], yticklabels=['Politics', 'Sports'])
        
        plt.title(f'Confusion Matrix: {best_performing_model}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plt.savefig('confusion_matrix.png')
        print(f"Saved 'confusion_matrix.png' (Best: {best_performing_model})")


if __name__ == "__main__":
    pipeline = NewsClassifierPipeline()
    
    if pipeline.download_data():
        
        if pipeline.merge_and_filter():
            
            X_train, X_test, y_train, y_test = pipeline.prepare_features()
            pipeline.run_experiment(X_train, X_test, y_train, y_test)
            
            pipeline.generate_plots()