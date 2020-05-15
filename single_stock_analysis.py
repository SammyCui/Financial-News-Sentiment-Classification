import pickle
import pandas as pd
from sentiment_analysis_nltk import analyze
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tweets_processing import text_processing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp


# Naive Base, Decision Tree. xgboost


def smoothen(l, size):
    l_smoothen = []
    for i in range(len(l)):
        if i < size - 1:
            window = l[:i+1]
            average = sum(window)/(i+1)
            l_smoothen.append(average)
        else:
            window = l[i - size + 1: i + 1]
            average = sum(window)/ size
            l_smoothen.append(average)
    return l_smoothen





def add_prev(df, col, days, inplace = True):
    if inplace:
        df['prev_{}_{}'.format(col, str(days))] = df[col].shift(days)
    else:
        new_df = df.copy()
        new_df['prev_{}_{}'.format(col,days)] = new_df[col].shift(days)
        return new_df

if __name__ == "__main__":
    # TODO: average of a few day

    AAPL = pickle.load(open('tweets/AAPL_tweets.pickle', 'rb'))
    WMT = pickle.load(open('tweets/WMT_tweets.pickle', 'rb'))
    AMZN = pickle.load(open("tweets/AMZN_tweets.pickle", 'rb'))
    FB = pickle.load(open("tweets/FB_tweets.pickle", 'rb'))

    AAPL_price = pd.read_csv("prices/AAPL.csv")
    AAPL_price['Date'] = pd.to_datetime(AAPL_price['Date'])
    AAPL['timestamp'] = pd.to_datetime(AAPL['timestamp'])
    AAPL.text = AAPL['text'].apply(text_processing)

    WMT_price = pd.read_csv("prices/WMT.csv")
    WMT_price['Date'] = pd.to_datetime(WMT_price['Date'])
    WMT['timestamp'] = pd.to_datetime(WMT['timestamp'])
    WMT['text'].apply(text_processing)

    AMZN_price = pd.read_csv("prices/AMZN.csv")
    AMZN_price['Date'] = pd.to_datetime(AAPL_price['Date'])
    AMZN['timestamp'] = pd.to_datetime(AMZN['timestamp'])
    AMZN['text'].apply(text_processing)

    FB_price = pd.read_csv("prices/FB.csv")
    FB_price['Date'] = pd.to_datetime(FB_price['Date'])
    FB['timestamp'] = pd.to_datetime(FB['timestamp'])
    FB['text'].apply(text_processing)

    AAPL = pickle.load(open('tweets/AAPL_tweets.pickle', 'rb'))
    AAPL_price = pd.read_csv("prices/AAPL.csv")
    AAPL_price['Date'] = pd.to_datetime(AAPL_price['Date'])
    AAPL['timestamp'] = pd.to_datetime(AAPL['timestamp'])

    AAPL_analyzed = analyze(AAPL)
    AAPL_analyzed = AAPL_analyzed.groupby('timestamp').mean()
    AAPL_price.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)
    AAPL_joined = AAPL_price.set_index("Date").join(AAPL_analyzed).reset_index()
    AAPL_joined['label'] = AAPL_joined['Close'] - AAPL_joined['Close'].shift(1)
    AAPL_joined['label'] = AAPL_joined['label'].apply(lambda x: 0 if x < 0 else 1)
    AAPL_joined['label'] = AAPL_joined['label'].shift(-1)
    AAPL_joined.dropna(inplace=True)

    AMZN_analyzed = analyze(AMZN)
    AMZN_analyzed = AMZN_analyzed.groupby('timestamp').mean()

    AMZN_price.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)
    AMZN_joined = AMZN_price.set_index("Date").join(AMZN_analyzed).reset_index()
    AMZN_joined['label'] = AMZN_joined['Close'] - AMZN_joined['Close'].shift(1)
    AMZN_joined['label'] = AMZN_joined['label'].apply(lambda x: 0 if x < 0 else 1)
    AMZN_joined['label'] = AMZN_joined['label'].shift(-1)
    AMZN_joined.dropna(inplace=True)

    WMT_analyzed = analyze(WMT)
    WMT_analyzed = WMT_analyzed.groupby('timestamp').mean()

    WMT_price.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)
    WMT_joined = WMT_price.set_index("Date").join(WMT_analyzed).reset_index()
    WMT_joined['label'] = WMT_joined['Close'] - WMT_joined['Close'].shift(1)
    WMT_joined['label'] = WMT_joined['label'].apply(lambda x: 0 if x < 0 else 1)
    WMT_joined['label'] = WMT_joined['label'].shift(-1)
    WMT_joined.dropna(inplace=True)

    FB_analyzed = analyze(FB)
    FB_analyzed = FB_analyzed.groupby('timestamp').mean()

    FB_price.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)
    FB_joined = FB_price.set_index("Date").join(FB_analyzed).reset_index()
    FB_joined['label'] = FB_joined['Close'] - FB_joined['Close'].shift(1)
    FB_joined['label'] = FB_joined['label'].apply(lambda x: 0 if x < 0 else 1)
    FB_joined['label'] = FB_joined['label'].shift(-1)
    FB_joined.dropna(inplace=True)

    add_prev(AAPL_joined, 'label', 1)
    add_prev(AAPL_joined, 'label', 2)
    add_prev(AAPL_joined, 'label', 3)
    add_prev(AAPL_joined, 'text_positive', 1)
    add_prev(AAPL_joined, 'text_positive', 2)
    add_prev(AAPL_joined, 'text_positive', 3)
    add_prev(AAPL_joined, 'text_negative', 1)
    add_prev(AAPL_joined, 'text_negative', 2)
    add_prev(AAPL_joined, 'text_negative', 3)
    AAPL_joined.dropna(inplace=True)

    add_prev(AMZN_joined, 'label', 1)
    add_prev(AMZN_joined, 'label', 2)
    add_prev(AMZN_joined, 'label', 3)
    add_prev(AMZN_joined, 'text_positive', 1)
    add_prev(AMZN_joined, 'text_positive', 2)
    add_prev(AMZN_joined, 'text_positive', 3)
    add_prev(AMZN_joined, 'text_negative', 1)
    add_prev(AMZN_joined, 'text_negative', 2)
    add_prev(AMZN_joined, 'text_negative', 3)
    AMZN_joined.dropna(inplace=True)

    add_prev(WMT_joined, 'label', 1)
    add_prev(WMT_joined, 'label', 2)
    add_prev(WMT_joined, 'label', 3)
    add_prev(WMT_joined, 'text_positive', 1)
    add_prev(WMT_joined, 'text_positive', 2)
    add_prev(WMT_joined, 'text_positive', 3)
    add_prev(WMT_joined, 'text_negative', 1)
    add_prev(WMT_joined, 'text_negative', 2)
    add_prev(WMT_joined, 'text_negative', 3)
    WMT_joined.dropna(inplace=True)

    add_prev(FB_joined, 'label', 1)
    add_prev(FB_joined, 'label', 2)
    add_prev(FB_joined, 'label', 3)
    add_prev(FB_joined, 'text_positive', 1)
    add_prev(FB_joined, 'text_positive', 2)
    add_prev(FB_joined, 'text_positive', 3)
    add_prev(FB_joined, 'text_negative', 1)
    add_prev(FB_joined, 'text_negative', 2)
    add_prev(FB_joined, 'text_negative', 3)
    WMT_joined.dropna(inplace=True)

    data = pd.concat([WMT_joined, AMZN_joined, AAPL_joined]).reset_index(drop=True).drop('Date', axis=1)

    X = data[['text_positive', 'text_negative', 'text_neutral', 'text_compound']]
    y = data['label']

    # print(len(y), len(AMZN_price))

    # print(stats.pearsonr(y.iloc[:503], AAPL_price.Close))




    gnb = GaussianNB()
    X = data.drop(['text_compound', 'label'], axis=1)
    print(X)
    y = data['label']
    scores = cross_val_score(gnb, X, y, cv=5)
    print(np.mean(scores), np.std(scores))

    lr = LogisticRegression()
    scores_lr = cross_val_score(lr, X, y, cv=5)
    print(np.mean(scores_lr), np.std(scores_lr))

    grb = GradientBoostingClassifier()
    scores_grb = cross_val_score(grb, X, y, cv=5)
    print(np.mean(scores_grb), np.std(scores_grb))

    rf = RandomForestClassifier()
    scores_rf = cross_val_score(rf, X, y, cv=5)
    print(np.mean(scores_rf), np.std(scores_rf))

    """
    
    xgb = XGBClassifier()
    scores_xgb = cross_val_score(xgb, X, y, cv=5)
    print(scores_xgb)
    """
    add_prev(data, 'label', 4)
    add_prev(data, 'label', 5)
    add_prev(data, 'label', 6)
    add_prev(data, 'text_positive', 4)
    add_prev(data, 'text_positive', 5)
    add_prev(data, 'text_positive', 6)
    add_prev(data, 'text_negative', 4)
    add_prev(data, 'text_negative', 5)
    add_prev(data, 'text_negative', 6)
    data.dropna(inplace=True)


    cv = StratifiedKFold(n_splits=5, shuffle=True)
    lw = 2
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for (train, test) in cv.split(X, y):
        probas_ = gnb.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    # Feature extraction
    test = SelectKBest(score_func=chi2, k=3)
    fit = test.fit(X, y)

    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)

    features = fit.transform(X)
    # Summarize selected features
    print(features[0:5, :])
    print(X.head())

