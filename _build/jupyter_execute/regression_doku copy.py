#!/usr/bin/env python
# coding: utf-8

# # Regression

# ## Plan

# ### Use Case
# 
# Das Unternehmen, welches bisher vor allem als Marktplatz für Immobilien von Kunden agiert hat, möchte in den Immobilienhandel einstiegen. Den Kunden sollen Angebote für Immobilien gemacht werden, wenn diese ihre Immobilie auf dem Handelsplatz registieren. Somit kann das Unternehmen als erstes eine potenziell gewinnbringende Immobilie entdecken und erwerben. Der Kunde spart sich Zeit und Aufwand beim Verkauf der Immobilie. Um die Immobilien marktkorrekt einschätzen zu können sollen gesammelte Daten zu vergleichbaren Immobilien und deren Standorten verwendet werden, um besonders rentable Immobilien zu identifizieren. 

# ### Problematik
# 
# Anhand von Hausdaten wie Raumanzahl, Quadratmeter, ... sowie Daten über die Umgebung (Standort) der Immobilie wie die Menge an Schulen in unmittelbarer Umgebung oder ide Durchschnittliche Lärmbelästigung sollen Preise vorhergesagt werden. Somit kann das Unternehmen überteuerte oder günstige Immobilien finden und den Besitzern bessere Angebote machen. Dies soll die Erfolgschance von Angeboten erhöhen wodurch das Unternehmen günstiger rentable Immobilien erwerben kann und den Kunden Zeit und Aufwand bei der Interessentensuche und der Kaufabwicklung erspart wird.

# ### Variablen
# 
# Es werden strukturierte Immobiliendaten verwendet. Für den Use Case soll außerdem verglichen werden, wie viele verschiedene Variablen tatsächlich benötigt werden, um zuverlässig den Preis einer Immobilie vorherzusagen, weshalb versucht wird, zuerst ein Modell zu erstellen, welches nur mit den relevantesten Faktoren den tatsächlichen Wert der Immobilie ermittelt. 

# ### Metriken
# 
# Als Erfolg wird verbucht, wenn es durch das Modell möglich ist, schnell und ohne menschliches Eingreifen Immobilienpreise zuverlässig vorherzusagen.
# Die Schlüsselergebnisse für den Erfolg ist eine Modellgenauigkeit (hier die erklärbare Streuung der Zielvariable durch das Modell) von mindestens 66%, wodurch Zeit der Mitarbeiter und Experten eingespart wird. Eine höhere Genauigkeit kann zukünftig dann mit präziseren Daten erreicht werden und somit menschliches Eingreifen obsolet machen.
# Bei schlechterer Genauigkeit gilt das Modell als zu ungenau und damit nicht einsetzbar. Daraufhin müsste analysiert werden, welche Daten zusätzlich benötigt werden, wie diese zu präzisieren sind oder welches Modell eventuell besser funktionieren könnte.

# ## Data

# ### Datenimport
# 
# Importieren aller libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Daten einfügen und Spaltennamen vergeben

# In[2]:


# Datensatz hat keine Zeilenbeschreibung! 
names = ['CRIM', 'ZN','INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']
df = pd.read_csv('housing.csv', header=None, delimiter='\s+', names=names)


# Da verglichen werden soll, wie viel genauer unsere Preisvorhersage mit vielen Variablen ist, wird ein weiteres DataFrame benötigt. Hier könnte man das DataFrame einfach kopieren oder per sklearn.datasets das Original Dataset importieren

# In[3]:


from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.head()
#target variable 
data['PRICE'] = boston.target 
# Durchschnittspreis der Häuser in $1000
data.shape


# In[4]:


df.describe()


# In[5]:


data.describe()


# Beide Datasets sind identisch!

# ### Datenexploration und Erkenntnisgewinn

# In[6]:


X = data.drop(columns='PRICE') # target variable!
Y = data.PRICE


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)


# In[8]:


df_train = pd.DataFrame(X_train.copy())
df_train = df_train.join(pd.DataFrame(y_train))


# In[9]:


df_train.info()


# Alle Variablen sind vom Datentyp Float oder int und damit numerisch, somit keine Anpassung am Datentyp notwendig.

# In[10]:


df_train.head()


# Erklärung aller Variablen: 
# CRIM - Pro-Kopf-Verbrechensrate nach Stadt
# ZN - Anteil der Wohnbauflächen, die für Grundstücke über 25.000 m² ausgewiesen sind.
# INDUS - Anteil der Flächen für Nicht-Einzelhandelsunternehmen pro Stadt.
# CHAS - Charles River Dummy-Variable (1, wenn das Gebiet an den Fluss grenzt; sonst 0)
# NOX - Konzentration von Stickstoffoxiden (Teile pro 10 Millionen)
# RM - durchschnittliche Anzahl der Zimmer pro Wohnung
# AGE - Anteil der Eigentumswohnungen, die vor 1940 gebaut wurden
# DIS - gewichtete Entfernungen zu fünf Beschäftigungszentren/Arbeitsämter
# RAD - Index der Erreichbarkeit von Radialautobahnen
# TAX - Vollwertiger Grundsteuersatz pro 10.000 \$
# PTRATIO - Schüler-Lehrer-Verhältnis nach Stadt
# B - 1000(Bk - 0,63)^2, wobei Bk der Anteil der Schwarzen in der Stadt ist 
# LSTAT - % niedrigerer Status der Bevölkerung
# PRICE - Medianwert der Eigenheime in 1000 \$

# In[11]:


plt.figure(figsize=(20,15))
sns.heatmap(df_train.corr(), annot=True)


# Erkenntnisgewinn: 
# Da wir den Preis präzise vorhersagen möchten, interessieren uns vor allem die Variablen mit der stärksten negativen oder positiven Korrelation zum Preis.Direkt fällt auf, dass LSTAT eine sehr starke negative Korrelation zu PRICE hat während RM hat eine starke positive Korrelation zu PRICE. Außerdem wird klar, dass viele Variablen eine mittelstarke Korrelation mit der Preisvariable aufweisen. Somit wird interessant zu sehen sein, wie genau das Modell bei der Verwendung aller Variablen werden kann.

# ### Datenbereinigung und -transformation

# In[12]:


print(df_train.isnull().sum())


# Keine fehlenden Werte

# In[13]:


df_train.drop_duplicates(inplace=True)
df_train.info() # Vergleich mit vorherigem data.info() un zu sehen, ob Values fehlen


# Keine Duplikate!

# Datensatz verkleinern, um zu überprüfen, ob die zwei relevantesten Variablen bereits ausreichen, um zufriedenstellende Genauigkeit zu erreichen.

# ### Datenaufbereitung

# In[14]:


data = data.drop(data.columns[[0, 1, 2, 3, 4, 6,7,8,9,10,11]], axis=1)
sns.pairplot(data)


# Variablen scheinen nahezu normalverteilt zu sein und wenig Ausreißer zu besitzen. 

# In[15]:


X = data.drop(columns='PRICE')
Y = data.PRICE


# Data splitting

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)


# ## Modell

# ### Modellauswahl
# 
# Da es sich um eine recht einfache Regressionsaufgabe handelt, ist der Einsatz eines linearen Modells zu empfehlen. Durch Ensemble-Methoden können daraufhin die Modelle noch genauer analysieren.

# ### Modellerstellung

# Modelltraining mit Trainingsdaten

# In[17]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_train)


# Modelevaluation

# In[18]:


from sklearn import metrics
from math import sqrt
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# Bestimmtheitsmaß R^2 (genauer adjusted R^2) beschreibt die Anpassungsgüte der Regression und dient für uns hier als Kennzahl für die "Genauigkeit" des Modells.

# In[19]:


px.scatter(x=X_train['RM'], y=y_train, opacity=0.65, 
                trendline='ols', trendline_color_override='red')


# Regressionsgerade für RM (durchschnittliche Anzahl der Zimmer pro Wohnung)

# In[17]:


px.scatter(x=X_train['LSTAT'], y=y_train, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred')


# Regressionsgerade für LSTAT (% niedrigerer Status der Bevölkerung)

# In[18]:


sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80})


# Residplot zeigt den Fehler zwischen einem vorhergesagten Wert und dem tatsächlichen Wert an. (Je näher an der Gerade auf 0, desto besser)

# Modellevaluation mit Testdaten

# In[19]:


y_pred = linreg.predict(X_test)


# In[20]:


print('R^2:',metrics.r2_score(y_test, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Erreichte Erfolgsmetrik von mindestens 66% erreicht.

# In[21]:


px.scatter(x=X_test['RM'], y=y_test, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred')


# Regressionsgerade für RM (durchschnittliche Anzahl der Zimmer pro Wohnung)

# In[22]:


px.scatter(x=X_test['LSTAT'], y=y_test, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred')


# Regressionsgerade für LSTAT (% niedrigerer Status der Bevölkerung)

# In[23]:


sns.residplot(x=y_pred, y=y_test, scatter_kws={"s": 80})


# Weitaus weniger Punkte, da Testset wesentlich kleiner als Trainingset ist.

# Modell mit allen Features/Variablen trainieren, um damit die Genauigkeit/Bestimmtheitsmaß zu erhöhen

# In[24]:


X= df.drop(columns='PRICE')
Y= df.PRICE


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)


# Modelltraining mit Trainingsdaten

# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_train)


# In[27]:


print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# Weitaus bessere Metriken erreicht. Circa 0.11 höherer Adjusted R^2 Wert.

# In[28]:


sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80})


# Im Residplot spiegelt sich die Verbesserung des Modells auch wider.

# ### Modellevaluation
# 
# Modellevaluation mit Testdaten

# In[29]:


y_pred = linreg.predict(X_test)


# In[30]:


print('R^2:',metrics.r2_score(y_test, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Auch bei der Anwendung auf Testdaten höherer Adjusted R^2 Wert erreicht.

# In[31]:


sns.residplot(x=y_pred, y=y_test, scatter_kws={"s": 80})


# Residplot wirkt bis auf einen Ausreißer sehr zufriedenstellend.

# Weiteres Regressionsmodell zum Vergleichen

# Mit Hilfe von Machine Learning Ansätzen kann das Modell noch weiter verbessert werden. Hierzu wird aus dem Bereich der ensemble Methods (Bagging) der Random Forest Regressor verwendet.

# In[33]:


from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators=50)
# n_estimators gibt die Menge an Bäumen an
forest_regressor.fit(X_train, y_train)
y_pred_forest = forest_regressor.predict(X_test)


# In[34]:


print('R^2:',metrics.r2_score(y_test, y_pred_forest))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred_forest))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred_forest))
print('MSE:',metrics.mean_squared_error(y_test, y_pred_forest))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred_forest)))


# Hier wird ein weitaus höherer Adjusted R^2 Wert und ein weitaus geringerer RMSE Wert erreicht. Dieses Modell ist um einiges besser geeignet.

# In[35]:


sns.residplot(x=y_pred_forest, y=y_test, scatter_kws={"s": 80})


# Sehr geringe Fehler/Abweichung!

# ### Interpretation
# 
# Die definierten Erfolgsmetriken wurden bereits mit dem Linear Regression Modell erreicht. Bei der Anwendung auf Testdaten ist die Genauigkeit des Modells mit allen Variablen bereits um einiges höher als das Modell mit nur den relevantesten Variablen. Im Vergleich zur Linearen Regression ist die Preisvorhersage mit dem Random Forest Regressor am effektivsten/genauesten. Mit einem Bestimmtheitsmaß von 0.88 kann das Modell 88% der Streuung der Daten "erklären". Dies ist ein sehr gutes Ergebnis wenn man bedenkt, wie enorm hilfreich ein solches Modell sein kann. Wenn solch ein gutes Ergebnis bereits bei (meist) wenig korrelierenden Daten zu Stande kommt, können durch den Einsatz von noch mehr und präziseren Daten eventuell bald Immobilienmarktexperten obsolet für das Unternehmen werden. Außerdem hat das Dataset ein ethisches Problem durch die Niederstellung von Immobilien an Standorten mit einem hohen Anteil an schwarzer Bevölkerung. Solch eine Auffassung ist nicht mehr zeitgemäß und spielt hoffentlich in der Realität keine Rolle mehr. Stattdessen sollten mehr Faktoren zur Preisermittlung einbezogen werden wie die Grundstücksfläche, Lärmbelästigung oder Nähe zu Schulen und Kindertagesstätten. Solche reellen Faktoren können die Preisvorhersage noch weiter optimieren und damit das Unternehmen bei der Entscheidungsfindung noch besser unterstützen. Ein solch komplexes Thema wie die Ermittlung vom Wert einer Immobilie ist jedoch nicht mit der Analyse von Daten erledigt, sondern bedarf auch weiterhin Strategie und Marktanalysen, welche wiederum neue Daten liefern könnten, die das Modell noch weiter optimieren können.
