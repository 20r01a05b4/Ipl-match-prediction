from django.shortcuts import render, redirect,HttpResponse
from django.contrib.auth.models import User, auth
from django.contrib import messages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
matplotlib.use('Agg')
def test(request):
    data = pd.read_csv("ml/week10matches (1).csv")
    col = (data.columns)
    imp = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']

    c = []
    for i in col:
        if i in imp:
            c.append(i)
    d=[]
    for i in imp:
        d.append(request.POST[i])
    if request.method == "POST":
        d = {}
        for i in imp:
            if i in request.POST:
                d[i] = request.POST[i]
       
        data = data.drop(['umpire3'], axis=1)
        data = data[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue','winner']]
        data = data.dropna()
        data_numerical = data.select_dtypes('int64')
        data_categorical = data.select_dtypes('object')
        cat_columns = data_categorical.columns
        
        enc = OrdinalEncoder()
        enc_data = enc.fit_transform(data_categorical)
        enc_data = pd.DataFrame(enc_data, columns=cat_columns)

        df = pd.concat([data_numerical, enc_data], axis=1)
        df = df.dropna()
        label = df['winner']
        df = df.drop(['winner'], axis=1)

        X, X_test, y, y_test = train_test_split(df, label, random_state=32, test_size=0.05)
       
        rfc=RandomForestClassifier(n_estimators=100, max_depth=6)
        #preprocessing of input data
        user_df = pd.DataFrame(d, index=[0])
        user_categorical = user_df.select_dtypes('object')
        user_categorical = enc.transform(user_categorical)
        user_df.update(user_categorical)
        pred = rfc.predict(X_test)

       
        predictions = rfc.predict(user_df)
        rfc.fit(X, y)
        predictions = rfc.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
       

        return HttpResponse(predictions)
    else:
        return render(request, "test.html", {"columns": c})


def ipl(request):
    df = pd.read_csv("ml/week10matches (1).csv")
    df = df.drop(['umpire3'], axis=1).dropna()

    venue_plot = sns.countplot(y='venue', data=df, order=df['venue'].value_counts().index[:10])
    venue_plot.figure.set_size_inches(10, 6)
    venue_plot.figure.savefig("static/venue_countplot.png")
    plt.close(venue_plot.figure)

    toss_plot = sns.countplot(x="toss_decision", data=df)
    toss_plot.set(xlabel='Toss Decision', ylabel='Count', title='Toss Decision')
    toss_plot.figure.set_size_inches(6, 4)
    toss_plot.figure.savefig("static/toss.png")
    plt.close(toss_plot.figure)

    df['team1_win'] = np.where(df['winner'] == df['team1'], 1, 0)
    df = df[['season', 'city', 'team1', 'team2', 'winner', 'team1_win']]

    X = pd.get_dummies(df.drop('team1_win', axis=1))
    y = df['team1_win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rfc = RandomForestClassifier(n_estimators=200, max_depth=6)
    rfc.fit(X_train, y_train)


  

    classification_rep = classification_report(y_test, pred)
    confusion_mat = confusion_matrix(y_test, pred)
    return render(request, 'ipl.html', {'classification_rep': classification_rep, 'confusion_mat': confusion_mat})
def register(request):
    if request.method == "POST":
        f_n = request.POST["first_name"]
        l_n = request.POST["last_name"]
        name = request.POST["username"]
        pass1 = request.POST["password"]
        pass2 = request.POST["password1"]
        email = request.POST["email"]
        
        if pass1 == pass2:
            user = User.objects.create_user(username=name, password=pass1, email=email, first_name=f_n, last_name=l_n)
            user.save()
            messages.success(request, "Registration successful")
            return redirect("/login")
        else:
            messages.error(request, "Passwords do not match")
            return redirect("/register")
    else:
        return render(request, "register.html")

def login(request):
    if request.method == "POST":
        name = request.POST["username"]
        password = request.POST["password"]
        user = auth.authenticate(username=name, password=password)
        if user is not None:
            auth.login(request, user)
            messages.success(request, "Logged in successfully")
            return redirect("/ipl")
        else:
            messages.error(request, "Wrong credentials")
            return redirect("/register")
    else:
        return render(request, "login.html")

def logout(request):
    auth.logout(request)
    messages.success(request, "Logged out successfully")
    return redirect("/")

def home(request):
    return render(request, "home1.html")
