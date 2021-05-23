from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import gzip, pickle


app = Flask(__name__)
df=pd.read_csv('dataset/sample30.csv')
# XGBoost Model for final prediction
xg_pickle_model= xgb.XGBClassifier()
xg_pickle_model.load_model('pickle/xg_best.bin')
#Tfidf vectorizer
with gzip.open('pickle/tfidf.pkl', 'rb') as f:
    p = pickle.Unpickler(f)
    tfidf = p.load()
# load reco model
with gzip.open('pickle/reco_best.pkl', 'rb') as f:
    p = pickle.Unpickler(f)
    reco_pickle_model = p.load()
# cleaned df
with gzip.open('pickle/sent_df.pkl', 'rb') as f:
    p = pickle.Unpickler(f)
    sent_df = p.load()


df = df.dropna(subset=['user_sentiment'])


# Putting feature variable to X
X = sent_df.drop(['user_sentiment'], axis=1)
# Putting sentiment variable to y
y = sent_df['user_sentiment']

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

xg_pickle_model._le = LabelEncoder().fit(y_test)



## Filtering 20 to 5 recommendations based on % of positive predictions
def best_5(user_name):
    ratings=df.loc[:,['reviews_username','name','reviews_rating']]
    # giving user id and item id
    ratings['user_id'] = ratings.groupby(ratings.loc[:,'reviews_username'].tolist(), sort=False).ngroup() + 1
    ratings['item_id'] = ratings.groupby(ratings.loc[:,'name'].tolist(), sort=False).ngroup() + 1
    # dropping rows with null values and dropping duplicates
    ratings.dropna(inplace=True)
    ratings.drop_duplicates(subset=['reviews_username','name'],inplace=True)
    user_input=ratings.loc[ratings['reviews_username']==user_name]['user_id'].reset_index(drop=True)[0]
    d = reco_pickle_model.loc[user_input].sort_values(ascending=False)[0:20]
    # merge with main dataframe to get review_text and review_title
    d = pd.merge(d,ratings.loc[:,['item_id','name']],left_on='item_id',right_on='item_id', how = 'left')
    d.drop_duplicates(inplace=True)
    d = pd.merge(d,sent_df,left_on=['name'],right_on=['name'],how = 'inner')
    d.drop_duplicates(inplace=True)
    # Applying tfidf
    tv_reviews=tfidf.transform(d['reviews_text_title'].to_list())
    return list(tv_reviews.shape)
    # predicting using XGBoost
    #d['final_pred']=xg_pickle_model.predict(tv_reviews)
    #return d['final_pred'].to_list()[:5]
    # calculation % of 1's in each recommended item
    #final_df=d1.loc[:,['name','final_pred']].groupby(by='name').agg(['sum','count'])
    #final_df['%']=final_df['final_pred']['sum']/final_df['final_pred']['count']
    #Sorting it based on %
    #final_df=final_df.sort_values(by='%',ascending=False)
    # final filtered top5 recommendation
    #return final_df.index[:5].to_list()


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if (request.method == "POST"):
        user_input = [x for x in request.form.values()][0]
        # final filtered top5 recommendation
        output=best_5(user_input)
        return render_template('index.html', prediction_text='Top 5 Recommendation', my_list=output)
    else :
        return render_template('index.html')

if __name__ == "__main__":
    app.run()
