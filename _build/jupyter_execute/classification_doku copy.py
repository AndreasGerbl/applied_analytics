#!/usr/bin/env python
# coding: utf-8

# # Classification

# ## Plan

# ### Use Case
# 
# Über die Website des Unternehmens sollen Kunden Immobilien finden. Hierzu sind einige Filter zu setzen, die die Menge an Immobilienanzeigen reduziert und nur diejenigen anzeigt, welche für den Kunden interessant sind. Hierzu muss die Immobilie anhand ihrer Eigenschaften in verschiedene Klassen klassifiziert werden. Das Unternehmen möchte ihren Kunden die Möglichkeit geben, speziell nach luxuriösen Wohnungen zu filtern. Stand jetzt müssen Mitarbeiter manuell Immobilien als luxuriös klassifizieren, was zukünftig durch ein Modell ersetzt werden soll.

# ### Problematik
# 
# Wir benötigen ein Modell welches Immobilien anhand von Eigenschaften in eine von zwei Klassen klassifiziert. Das Ziel ist, diese Klassen ohne menschliches Zutun zu erstellen und dies mit einer hohen Genauigkeit. Im Speziellen wollen wir die Immobilie als luxuriös oder basic (normal) klassifizieren. Diese Klassifikation soll anhand von relevanten Faktoren erfolgen, welche die Immobilie genauer beschreiben. </br>
# Das Modell soll Experten aus dem Unternehmen bei der Immobilienanzeigenerstellung Arbeit abnehmen und den Kunden weitere Möglichekeiten liefern, Anzeigen nach ihren Wünschen zu präzisieren.

# ### Variablen
# 
# Es werden strukturierte Immobiliendaten verwendet. Die relevanten Variablen zur Identifikation vom Immobilienstatus (luxoriös oder basic) müssen identifiziert werden und dessen Aussagekraft analysiert werden, um herauszufinden, was die Faktoren sind, welche Luxusimmobilien von anderen unterscheiden.
# 
# 

# ### Metriken
# 
# Als Erfolg wird das Modell beurteilt, wenn eine Modellgenauigkeit von mindestens 90% erreicht wird. Unser Modell gilt als gescheitert, wenn das Modell zu ungenau ist bzw. zu viele False Negative/Positive. 

# ## Data

# ### Datenimport
# 
# Importieren aller libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Daten einfügen und Spaltennamen vergeben

# In[2]:


data=pd.read_csv("ParisHousingClass.csv")


# ### Datenexploration und Erkenntnisgewinn

# In[3]:


data.info()


# Auffällig ist, dass der Datentyp von category object ist.

# In[4]:


x = data.drop("category" , axis = 1)
y = data['category']


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=42)


# In[6]:


df_train = pd.DataFrame(X_train.copy())
df_train = df_train.join(pd.DataFrame(y_train))


# In[7]:


df_train.describe()


# Es fällt auf, dass category hier fehlt

# In[8]:


df_train.head()


# Category scheint das ideale Zielattribut (target feature) zu sein, welches vom Modell zukünftig klassifiziert werden soll.

# In[9]:


df_train['category'].unique()


# Hier sehen wir, dass category Basic und Luxury als values besitzt. 

# ### Datenbereinigung und -transformation

# In[10]:


df_train.isnull().sum()


# Keine fehlenden Werte

# In[11]:


df_train.drop_duplicates(inplace=True)
df_train.info() # Vergleich mit vorherigem data.info() un zu sehen, ob Values fehlen


# Keine Duplikate!

# In[12]:


df_train.hist(bins = 50, figsize = (20,20))
plt.show()


# Es scheinen keine Unregelmäßigkeiten im Datensatz zu bestehen. 

# In[13]:


plt.figure(figsize=(20,15))
sns.heatmap(df_train.corr(),annot = True)


# Price und squareMeters korrelieren sehr stark, jedoch fehlt unser Target Feature Category!

# Damit wir mit Category arbeiten können, muss das Feature normalisiert werden (Values müssen in numerische Values verändert werden).

# In[14]:


df_train['category'].replace("Basic", 0 , inplace = True)
df_train['category'].replace("Luxury", 1 , inplace = True)
df_train['category'].unique() # category values anzeigen lassen


# In[15]:


plt.figure(figsize=(20,15))
sns.heatmap(df_train.corr(), annot = True)


# Diesmal wird auch unser Zielattribut (target feature) category angezeigt. hasYard, hasPool und isNewBuilt scheinen eine starke Korrelation mit category zu haben. Die restlichen Features scheinen kaum korrelation mit category zu haben. </br>
# Aufgrund der hohen Korrelation von Price und squareMeters müssen wir eines davon entfernen.

# In[16]:


df_train = df_train.drop(["price"], axis = 1)


# ### Datenaufbereitung

# In[17]:


sns.countplot( x = df_train['category'])


# Es gibt weitaus weniger Luxuswohnungen als Basic Wohnungen. 0 = Basic, 1 = Luxury

# In[18]:


sns.countplot(x = df_train["hasPool"], hue=data["category"])


# Luxuswohnungen müssen einen Pool besitzen, jedoch ist nicht jede Wohnung mit Pool eine Luxuswohnung

# In[19]:


sns.countplot(x = df_train["isNewBuilt"], hue=data["category"])


# Luxuswohnungen müssen Neubauten sein, jedoch ist nicht jede neugebaute Wohnung eine Luxuswohnung

# In[20]:


sns.countplot(x = df_train["hasYard"], hue=data["category"])


# Luxuswohnungen müssen einen Garten besitzen, jedoch ist nicht jede Wohnung mit Garten eine Luxuswohnung

# Von den Visulisierungen stellen wir fest, dass Luxuswohnungen immer einen Garten und einen Pool haben, sowie immer ein Neubau sind. Um sicher zu gehen, dass dies nicht bei allen Variablen der Fall ist, testen wir dies einmal mit hasStormProtector.

# In[21]:


sns.countplot(x = df_train["hasStormProtector"], hue=data["category"])


# Vergleichen wir die Variable hasStormProtector fällt auf, dass es auch Luxuswohnungen ohne Blitzableiter gibt.

# Auf den Einsatz von Feature Scaling (z.B. StandardScaler) wird bewusst verzichtet um dieses später zu verwenden!

# ## Modell

# ### Modellauswahl
# 
# Ensemble Methoden wie Random Forest Classifier Modelle kombinieren mehrere Klassifiaktionsmodellvorhersagen und eine besonders hohe Genauigkeit zu erreichen, was für unseren Use Case eine Vorraussetzung ist. Außerdem eignet sich das Modell für große Datensätze und overfittet nur selten.

# ### Modellerstellung

# Modelltraining mit Trainingsdaten

# In[22]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50) # n_estimators gibt die Menge an Bäumen an
rf.fit(X_train, y_train)
rf.score(X_train,y_train)


# Anwendung mit Testdaten

# In[23]:


y_pred = rf.predict(X_test)
rf.score(X_test,y_test)


# ### Modellevaluation 

# In[24]:


from sklearn.metrics import confusion_matrix, classification_report
plt.figure(figsize = (8,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True)
confusion_matrix(y_test, y_pred) # Um Zahlen genau anzuzeigen!


# True Positive: 1744 der Wohnungen wurden als Basic (0) klassifiziert und waren auch Basic (0)
# True Negative: 256 der Wohnungen wurden als Luxus (1) klassifiziert und waren auch Luxus (1)
# Keine False Positive/Negative Fälle durch den RandomForestClassifier --> 100% Genauigkeit!

# In[25]:


print(classification_report(y_test, y_pred))


# Extra Umsetzung mit K Nearest Neighbors als Vergleich zu RandomForestClassifier

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(X_train, y_train)
knc.score(X_train,y_train)


# In[27]:


y_pred = knc.predict(X_test)
knc.score(X_test,y_test)


# In[28]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

confusion_matrix(y_test, y_pred)


# Genauigkeit von KNN Modell sehr viel schlechter!

# Um die Genauigkeit zu erhöhen könnte eine Standardisierung zur Skalierung unserer Features helfen.

# In[29]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.2)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=3 )
knc.fit(X_train, Y_train)
knc.score(X_train,Y_train)


# In[31]:


y_pred = knc.predict(X_test)
knc.score(X_test,Y_test)


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

confusion_matrix(Y_test, y_pred)


# Nun erreicht auch der KNeighborsClassifier eine sehr gute Genauigkeit (nahezu perfekt)

# ### Interpretation
# 
# Anhand der Daten wird schnell klar, was die Gemeinsamkeiten von Luxusimmobilien zueinander sind und was diese von normalen (basic) Immobilien unterscheidet. Durch das Random Forest Classifer Modell kann wird eine 100%ige Genauigkeit erreicht, wenn es darum geht, die Kategorie einer Immobilie bestimmt werden. Die hohe Genauigkeit hängt stark mit den zugrundeliegenden Daten zusammen, welche strukturiert und fehlerfrei sind und sich optimal für Klassifikationsmodelle eignen, da besonders 3 Variablen ausschlaggebend für einen Luxusimmobilie sind. Im Vergleich mit dem zweiten Modell fällt auf, wie gut das Random Forest Modell wirklich ist. Besonders die Genauigkeit bei der Klassifikation von class 1 (Luxusimmobilien) mit dem KNearestClassifier Modell ist untragbar, glücklicherweiße funktioniert das Random Forest Classifier Modell einwandfrei.
