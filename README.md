# IMDb Movie Reviews Sentiment Analysis

This project focuses on analyzing movie reviews from IMDb to classify them into **positive** or **negative** sentiments. The primary goal is to understand public perception, audience reactions, and feedback trends using Natural Language Processing (NLP) and Machine Learning techniques.

For a complete overview, please refer to the file **"Sentiment analysis presentation .pdf"** included in this repository.

## Project Objectives & Significance
* **Enhance User Experience:** By providing quick summaries of audience opinions, it helps new users make informed viewing decisions.
* **Support Investment Decisions:** It enables production companies to identify strengths and weaknesses in their work based on audience feedback, reducing financial risks in future projects.
* **Automation and Resource Efficiency:** It replaces time-consuming and costly manual reviews with an intelligent system that operates around the clock with high accuracy.

##  Dataset Description
* **Source:** The dataset was collected from [Kaggle - IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/mahmoudshaheen1134/imdp-data).
* **Size:** It consists of **50,000 rows** and 2 columns:
  * `review`: The written text of the movie review.
  * `sentiment`: The target label (Positive / Negative).
* **Distribution:** The dataset is fully balanced with 25,000 positive and 25,000 negative reviews.

##  Getting Started
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mahmoudshaheen1134/imdp-data).
2. Place the downloaded `IMDB Dataset.csv` file inside the project root directory.
3. Run the notebook/script to start preprocessing and training!

##  Project Pipeline

### 1. Exploratory Data Analysis (EDA) & Data Cleaning
* Examined the dataset structure and checked for missing values (none were found).
* Detected and removed **418 duplicate rows** to ensure data integrity and prevent model bias.
* Text normalization by converting all text to lowercase.
* Removed URLs, punctuation, special characters, and extra whitespaces.

### 2. Advanced Text Preprocessing
* **Tokenization:** Splitting the text reviews into individual words.
* **Optimized Custom Stop Words Removal:** Instead of traditional stop words removal which strips out crucial context, a customized approach was implemented. Essential negation terms (e.g., *not, no, never, didn't, wasn't, haven't*) were explicitly excluded from the removal list to preserve the semantic meaning of sentences.
* **Stemming:** Applying `PorterStemmer` to reduce words to their base form.
* **Lemmatization:** Applying `WordNetLemmatizer` to return words to their proper dictionary root forms.

### 3. Advanced Feature Engineering & Vectorization
To maximize model accuracy, the pipeline combines text vectorization with advanced engineered metadata and lexicon features:
* **Text Vectorization:** Converted text into numerical features using `CountVectorizer` and `TfidfVectorizer` with **N-grams**.
* **Lexicon-Based Sentiment Scores:** Leveraged NLTK's VADER `SentimentIntensityAnalyzer` to calculate explicit positive (`pos_score`) and negative (`neg_score`) distribution scores for each review text.
* **Contextual Features:** Extracted structural and semantic properties:
  * Text length (`text_len`)
  * Total word count (`word_count`)
  * Punctuation count (`punct_count`)
  * Negation flag (`has_negation`): A binary indicator tracking the presence of immediate context shifts like the word 'not'.
* **Feature Scaling:** Applied `MinMaxScaler` to seamlessly normalize the combined numerical meta-features (`text_len`, `word_count`, `punct_count`, `has_negation`) before horizontally stacking them with the sparse TF-IDF matrix.

##  Model Training & Evaluation
The data was split using a stratified **80% training** and **20% testing** approach to ensure class balance across splits:

###  Logistic Regression
* Chosen for its efficiency with high-dimensional text data and excellent interpretability.
* **Accuracy:** **88.34%**.
* **F1-Score:** **0.88** for both positive and negative classes, demonstrating a highly balanced performance.

###  Random Forest Classifier
* Built using **100 decision trees** to capture complex non-linear interactions within the data.
* **Accuracy:** **84.42%**.
* **F1-Score:** **0.84** across both sentiment classes.

### Optimization & Validation
* **K-Fold Cross-Validation:** Utilized 5-fold cross-validation to guarantee the model's stability and generalizability.
* **Hyperparameter Tuning:** Conducted using `GridSearchCV` to optimize parameters (such as the regularization strength `C` and penalty type).
* **Confusion Matrix:** Evaluated model performance deeply by analyzing True Positives, False Positives, True Negatives, and False Negatives.

## Tech Stack & Libraries
* **Python** (Core Language)
* **Pandas & NumPy** (Data Manipulation)
* **NLTK** (Natural Language Toolkit, including VADER `SentimentIntensityAnalyzer`)
* **Scikit-Learn** (Feature extraction, scaling, model building, and evaluation)
* **Regex (`re`)** (Text cleaning and regex patterns)

