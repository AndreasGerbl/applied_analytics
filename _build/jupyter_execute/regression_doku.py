#!/usr/bin/env python
# coding: utf-8

# # Regression

# ## Plan

# ### Vorgehensweise:
# 
# Use Case definieren durch Business Model Canvas </br>
# Problemidentifikation </br>
# Variablenidentifikation </br>
# Erfolgsmetriken identifizieren (und Misserfolg) </br>
# 

# ### Use Case
# 
# Das Unternehmen, welches bisher vor allem als Marktplatz für Immobilien von Kunden agiert hat, möchte in den Immobilienhandel einstiegen. Den Kunden sollen Angebote für Immobilien gemacht werden, wenn diese ihre Immobilie auf dem Handelsplatz registrieren. Somit kann das Unternehmen als erstes eine potenziell gewinnbringende Immobilie entdecken und erwerben. Der Kunde spart sich Zeit und Aufwand beim Verkauf der Immobilie. Um die Immobilien marktkorrekt einschätzen zu können sollen gesammelte Daten zu vergleichbaren Immobilien und deren Standorten verwendet werden, um besonders rentable Immobilien zu identifizieren. 

# ### Problematik
# 
# Anhand von Hausdaten wie Raumanzahl, Quadratmeter, ... sowie Daten über die Umgebung (Standort) der Immobilie wie die Menge an Schulen in unmittelbarer Umgebung oder die durchschnittliche Lärmbelästigung sollen Preise vorhergesagt werden. Somit kann das Unternehmen überteuerte oder günstige Immobilien finden und den Besitzern bessere Angebote machen. Dies soll die Erfolgschance von Angeboten erhöhen wodurch das Unternehmen günstiger rentable Immobilien erwerben kann und den Kunden Zeit und Aufwand bei der Interessentensuche und der Kaufabwicklung erspart wird.

# ### Variablen
# 
# Es werden strukturierte Immobiliendaten verwendet. Für den Use Case soll außerdem verglichen werden, wie viele verschiedene Variablen tatsächlich benötigt werden, um zuverlässig den Preis einer Immobilie vorherzusagen, weshalb versucht wird, zuerst ein Modell zu erstellen, welches nur mit den relevantesten Faktoren den tatsächlichen Wert der Immobilie ermittelt. Daher ist die Zielvariable der Preis der Immobilie und die erklärende Variablen sind Informationen zur Immobilie (Grundstücksgröße, Anzahl an Schlafzimmer) sowie Informationen zum Standort der Immobilie.

# ### Metriken
# 
# Als Erfolg wird verbucht, wenn es durch das Modell möglich ist, schnell und ohne menschliches Eingreifen Immobilienpreise zuverlässig vorherzusagen.
# Die Schlüsselergebnisse für den Erfolg ist eine Modellgenauigkeit (hier die erklärbare Streuung der Zielvariable durch das Modell) von mindestens 66%, wodurch Zeit der Mitarbeiter und Experten eingespart wird. Eine höhere Genauigkeit kann zukünftig dann mit präziseren Daten erreicht werden und somit menschliches Eingreifen obsolet machen.
# Bei schlechterer Genauigkeit gilt das Modell als zu ungenau und damit nicht einsetzbar. Daraufhin müsste analysiert werden, welche Daten zusätzlich benötigt werden, wie diese zu präzisieren sind oder welches Modell eventuell besser funktionieren könnte.

# ## Data

# ### Vorgehensweise:
# 
# Datenimport mit allen nötigen libraries</br>
# Datensatz kopieren und anpassen </br>
# data splitting </br>
# Datenexploration anhand des Trainingsdatensatz (Nicht testdaten verwenden!)</br>
# Korrelation zu Zielvariable per Heatmap analysieren</br>
# Datenbereinigung (Duplikate und fehlende Werte)</br>
# Datenverteilung aufzeigen</br>
# stark korrelierende Variablen/Features auswählen</br>
# Später: RobustScaling!</br>

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
data = pd.read_csv('housing.csv', header=None, delimiter='\s+', names=names)


# Da verglichen werden soll, wie viel genauer unsere Preisvorhersage mit vielen Variablen ist, wird ein weiteres DataFrame benötigt.

# In[3]:


X = data.drop(columns='PRICE') # Zielvariable!
Y = data.PRICE


# Data Splitting für Datenexploration

# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=10)


# In[45]:


df_train = pd.DataFrame(X_train.copy())
df_train = df_train.join(pd.DataFrame(y_train))


# In[46]:


df_train.describe()


# ### Datenexploration und Erkenntnisgewinn

# In[47]:


df_train.info()


# Alle Variablen sind vom Datentyp Float oder int und damit numerisch, somit keine Anpassung am Datentyp notwendig.

# In[48]:


df_train.head()


# Erklärung aller Variablen: </br>
# CRIM - Pro-Kopf-Verbrechensrate nach Stadt</br>
# ZN - Anteil der Wohnbauflächen, die für Grundstücke über 25.000 m² ausgewiesen sind.</br>
# INDUS - Anteil der Flächen für Nicht-Einzelhandelsunternehmen pro Stadt.</br>
# CHAS - Charles River Dummy-Variable (1, wenn das Gebiet an den Fluss grenzt; sonst 0)</br>
# NOX - Konzentration von Stickstoffoxiden (Teile pro 10 Millionen)</br>
# RM - durchschnittliche Anzahl der Zimmer pro Wohnung</br>
# AGE - Anteil der Eigentumswohnungen, die vor 1940 gebaut wurden</br>
# DIS - gewichtete Entfernungen zu fünf Beschäftigungszentren/Arbeitsämter</br>
# RAD - Index der Erreichbarkeit von Radialautobahnen</br>
# TAX - Vollwertiger Grundsteuersatz pro 10.000 \$</br>
# PTRATIO - Schüler-Lehrer-Verhältnis nach Stadt</br>
# B - 1000(Bk - 0,63)^2, wobei Bk der Anteil der Schwarzen in der Stadt ist </br>
# LSTAT - % niedrigerer Status der Bevölkerung</br>
# PRICE - Medianwert der Eigenheime in 1000 \$</br>

# In[49]:


plt.figure(figsize=(20,15))
sns.heatmap(df_train.corr(), annot=True)


# Erkenntnisgewinn: </br>
# Da wir den Preis präzise vorhersagen möchten, interessieren uns vor allem die Variablen mit der stärksten negativen oder positiven Korrelation zum Preis. Direkt fällt auf, dass LSTAT eine sehr starke negative Korrelation zu PRICE hat während RM hat eine starke positive Korrelation zu PRICE. Außerdem wird klar, dass viele Variablen eine mittelstarke Korrelation mit der Preisvariable aufweisen. Somit wird interessant zu sehen sein, wie genau das Modell bei der Verwendung aller Variablen werden kann.

# ### Datenbereinigung und -transformation

# In[50]:


print(df_train.isnull().sum())


# Keine fehlenden Werte

# In[51]:


df_train.drop_duplicates(inplace=True)
df_train.info() # Vergleich mit vorherigem data.info() un zu sehen, ob Values fehlen


# Keine Duplikate!

# In[52]:


df_train.hist(bins = 50, figsize = (20,20))
plt.show()


# Die Histogramme zeigen die Verteilung der Daten auf. Es wird bewusst darauf verzichtet, Ausreißer zu elimieren und Feature Scaling zu betreiben, um zu sehen wie die Modelle ohne diese Schritte klar kommen. (RobustScaler später relevant)

# Datensatz verkleinern, um zu überprüfen, ob die zwei relevantesten Variablen bereits ausreichen, um zufriedenstellende Genauigkeit zu erreichen.

# In[53]:


df_train = df_train.drop(df_train.columns[[0, 1, 2, 3, 4, 6,7,8,9,10,11]], axis=1)
sns.pairplot(df_train)


# Variablen scheinen nahezu normalverteilt zu sein und wenig Ausreißer zu besitzen. 

# Nun wird das Dataset auf die ausgewählten Variablen beschränkt und für die Evaluation gesplittet.

# In[54]:


data = data.drop(data.columns[[0, 1, 2, 3, 4, 6,7,8,9,10,11]], axis=1)


# In[55]:


X = data.drop(columns='PRICE')
Y = data.PRICE


# In[56]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=1)


# ## Modell

# ### Vorgehensweise: 
# 
# Regressionsmodell mit ausgewählten Variablen anhand von Trainingsdaten trainieren sowie Regressionsgeraden einzeichnen</br>
# Trainiertes Modell evaluieren anhand Metriken und Residplot</br>
# Regressionsmodell mit Testdaten evaluieren (Metriken, Regressionsgeraden und Residplot)</br>
# Regressionsmodell mit allen Variablen/Features trainieren und evaluieren (Residplot und Metriken)</br>
# Trainiertes Modell mit Testdaten evaluieren (Residplot und Metriken)</br>
# Weiteres Regressionsmodell mit anderem Algorithmus erstellen (Training + Test durch Metriken udn Residplot)</br>
# Metriken der Modelle vergleichen</br>
# Lineares Regressionsmodell mit Scaler verbessern (Evaluation durch Metriken und Residplot)</br>
# Interpretation</br>

# ### Modellauswahl
# 
# Da es sich um eine recht einfache Regressionsaufgabe handelt, ist der Einsatz eines linearen Modells zu empfehlen. Für einen adäquaten Vergleich werden beide Modelle auf den LinearRegression Algorithmus beruhen. Durch Ensemble-Methoden können daraufhin die Modelle noch genauer analysieren und durch RobustScaler kann die Genauigkeit (Bestimmtheitsmaß) weiter verbessert werden.

# ### Modellerstellung

# Modelltraining mit Trainingsdaten

# In[57]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_train)


# Modelevaluation

# In[58]:


from sklearn import metrics
from math import sqrt
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# Bestimmtheitsmaß R^2 (genauer adjusted R^2) beschreibt die Anpassungsgüte der Regression und dient für uns hier neben dem RMSE (Wurzel der mittleren Fehlerquadratsumme) als Hauptbewertungskriterium für die Modelle.

# In[59]:


px.scatter(x=X_train['RM'], y=y_train, opacity=0.65, 
                trendline='ols', trendline_color_override='red')


# Regressionsgerade für RM (durchschnittliche Anzahl der Zimmer pro Wohnung)

# In[60]:


px.scatter(x=X_train['LSTAT'], y=y_train, opacity=0.65, 
                trendline='ols', trendline_color_override='red')


# Regressionsgerade für LSTAT (% niedrigerer Status der Bevölkerung)

# In[61]:


sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80})


# Residplot zeigt den Fehler zwischen einem vorhergesagten Wert und dem tatsächlichen Wert an. (Je näher an der Gerade auf 0, desto besser)

# Modellevaluation mit Testdaten

# In[62]:


y_pred = linreg.predict(X_test)


# In[63]:


print('R^2:',metrics.r2_score(y_test, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Erreichte Erfolgsmetrik von mindestens 66% erreicht.

# In[64]:


px.scatter(x=X_test['RM'], y=y_test, opacity=0.65, 
                trendline='ols', trendline_color_override='red')


# Regressionsgerade für RM (durchschnittliche Anzahl der Zimmer pro Wohnung)

# In[65]:


px.scatter(x=X_test['LSTAT'], y=y_test, opacity=0.65, 
                trendline='ols', trendline_color_override='red')


# Regressionsgerade für LSTAT (% niedrigerer Status der Bevölkerung)

# In[66]:


sns.residplot(x=y_pred, y=y_test, scatter_kws={"s": 80})


# Weitaus weniger Punkte, da Testset wesentlich kleiner als Trainingset ist.

# Modell mit allen Features/Variablen trainieren, um damit die Genauigkeit/das Bestimmtheitsmaß zu erhöhen.

# In[67]:


X= df.drop(columns='PRICE')
Y= df.PRICE


# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=1)


# Modelltraining mit Trainingsdaten

# In[69]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_train)


# In[70]:


print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# Weitaus bessere Metriken erreicht. Circa 0.10 höherer Adjusted R^2 Wert.

# In[71]:


sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80})


# Im Residplot spiegelt sich die Verbesserung des Modells auch wider.

# ### Modellevaluation
# 
# Modellevaluation mit Testdaten

# In[72]:


y_pred = linreg.predict(X_test)


# In[73]:


print('R^2:',metrics.r2_score(y_test, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Auch bei der Anwendung auf Testdaten höherer Adjusted R^2 Wert erreicht.

# In[74]:


sns.residplot(x=y_pred, y=y_test, scatter_kws={"s": 80})


# Residplot spiegelt die Verbesserung des Modells im Vergleich zum Modell mit weniger Variablen wider.

# #### Weiteres Regressionsmodell zum Vergleichen

# Mit Hilfe von Machine Learning Ansätzen kann das Modell noch weiter verbessert werden. Hierzu wird aus dem Bereich der ensemble Methods (Bagging) der Random Forest Regressor verwendet. Dieses Modell dient dazu, herauszufinden wie genau wir mit den aktuellen Daten werden können.

# In[93]:


from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators=100)
# n_estimators gibt die Menge an Bäumen an
forest_regressor.fit(X_train, y_train)
y_pred_forest = forest_regressor.predict(X_test)


# In[94]:


print('R^2:',metrics.r2_score(y_test, y_pred_forest))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred_forest))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred_forest))
print('MSE:',metrics.mean_squared_error(y_test, y_pred_forest))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred_forest)))


# Hier wird ein weitaus höherer Adjusted R^2 Wert und ein weitaus geringerer RMSE Wert erreicht. Dieses Modell ist um einiges besser geeignet.

# In[77]:


sns.residplot(x=y_pred_forest, y=y_test, scatter_kws={"s": 80})


# Sehr geringe Fehler/Abweichung (alle Abweichungen unter 10 und RMSE von 2.8)!

# Eine Möglichkeit, das LinearRegression Modell zu verbessern besteht darin, die Daten von einigen Ausreißern zu befreien. Hierzu wird ein RobustScaler angewendet.

# In[81]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
x = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(x,Y, test_size=0.2)


# In[82]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_train)


# In[83]:


print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


# In[84]:


y_pred = linreg.predict(X_test)


# In[85]:


print('R^2:',metrics.r2_score(y_test, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Durch den Einsatz eines RobustScalers zur Eliminierung von Ausreißern wird das LinearRegression Modell nur marginal besser. Dies liegt vermutlich daran, dass das Dataset keine allzu starken Ausreißer hat.

# Weiterhin ist das RandomForestRegressor Modell die mit Abstand beste Wahl für den Use Case und mit hohem Bestimmtheitsmaß (R^2) und geringem Fehler (RMSE).

# ### Interpretation
# 
# Die definierten Erfolgsmetriken wurden bereits mit dem Linear Regression Modell erreicht. Bei der Anwendung auf Testdaten ist die Genauigkeit des Modells mit allen Variablen bereits um einiges höher als das Modell mit nur den relevantesten Variablen. Im Vergleich zur Linearen Regression ist die Preisvorhersage mit dem Random Forest Regressor am effektivsten/genauesten. Mit einem Bestimmtheitsmaß von knapp über 0.90 kann das Modell über 90% der Streuung der Daten "erklären". Dies ist ein sehr gutes Ergebnis, wenn man bedenkt, wie enorm hilfreich ein solches Modell sein kann. Wenn solch ein gutes Ergebnis bereits bei (meist) wenig korrelierenden Daten zu Stande kommt, können durch den Einsatz von noch mehr und präziseren Daten eventuell bald Immobilienmarktexperten obsolet für das Unternehmen werden. Außerdem hat das Dataset ein ethisches Problem durch die Niederstellung von Immobilien an Standorten mit einem hohen Anteil an schwarzer Bevölkerung. Solch eine Auffassung ist nicht mehr zeitgemäß und spielt hoffentlich in der Realität keine Rolle mehr. Stattdessen sollten mehr Faktoren zur Preisermittlung einbezogen werden wie die Grundstücksfläche, Lärmbelästigung oder Nähe zu Schulen und Kindertagesstätten. Solche reellen Faktoren können die Preisvorhersage noch weiter optimieren und damit das Unternehmen bei der Entscheidungsfindung noch besser unterstützen. Ein solch komplexes Thema wie die Ermittlung vom Wert einer Immobilie ist jedoch nicht mit der Analyse von Daten erledigt, sondern bedarf auch weiterhin Strategie und Marktanalysen, welche wiederum neue Daten liefern könnten, die das Modell noch weiter optimieren können.
