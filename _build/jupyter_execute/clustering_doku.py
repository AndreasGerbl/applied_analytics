#!/usr/bin/env python
# coding: utf-8

# # Clustering

# ## Plan

# ### Vorgehensweise:
# 
# Use Case definieren durch Business Model Canvas </br>
# Problemidentifikation </br>
# Variablenidentifikation </br>
# Erfolgs- und Misserfolgsmetriken identifizieren </br>

# ### Use Case
# 
# Der Use Case umfasst die Kundensegmentierung. Bei der Kundensegmentierung wird der potenzielle Kundenstamm des Unternehmens auf Grundlage seiner Bedürfnisse, Kaufmerkmale usw. in einzelne Gruppen eingeteilt. Ziel ist es, Kundengruppen zu identifizieren, welche empfänglich für bestimmte Preisniveaus von Immobilien sind. Durch diese Identifikation können Kunden besser angesprochen werden bzw. Anzeigen präziser geschalten werden, was die Kaufwahrscheinlichkeit und damit den Umsatz für das Unternehmen steigert. Cluster müssen gebildet werden die Kunden möglichst sicher in für das Unternehmen relevante Gruppen einzuordnen. Clustering basiert auf der Gruppierung von Datenpunkten basierend auf Ähnlichkeiten zwischen ihnen und Unterschieden zu anderen. Die Ausgabe, die wir erhalten, ist etwas, das wir uns selbst benennen müssen. 
# 

# ### Problematik
# 
# Um Kundengruppen zu identifizieren, benötigen wir ein Modell welches das Kaufverhalten unserer Kunde analysiert und herausfindet, welche Kunden empfänglich für bestimmte Preisniveaus sind um diese Kunden gezielter mit Angeboten anzusteuern. Das Ziel ist, Immobilienanzeigen gezielt auf bestimmte Kundengruppen zu schalten, was die Effektivität der Anzeigen enorm steigert und die Kaufwahrscheinlichkeit erhöht.

# ### Variablen
# 
# Es werden strukturierte Kundenstammdaten verwendet. Anhand von Daten wie Alter, Geschlecht, Einkommen, Ausbildungsstand, ... soll jeder Kunde einem bestimmten Kaufverhalten zugeordnet werden. Zielvariable ist die Kaufbereitschaft eines Kunden.

# ### Metriken
# 
# Als Erfolg wird das Modell verbucht, wenn es Kunden zuverlässig in Cluster einordnen kann und sich klare Grenzen zwischen den Kundengruppen ergeben. Unser Modell gilt als gescheitert, wenn diese Grenzen nicht erkennbar sind oder nicht aussagekräftig genug sind, um wirtschaftliche Vorteile daraus zu generieren.
# 

# ## Data

# ### Vorgehensweise:
# 
# Datenimport + Import Libraries </br>
# Datenexloration/Analyse</br>
# Datavorbereitung</br>
# Datenbereinigung (Fehlende Werte und Duplikate)</br>
# Analyse nahc Ausreißern und Datenverteilung (Häufigkeitsverteilung)</br>
# Korrelation der Variablen aufzeigen</br>
# Zielvariablen definieren und analysieren</br>
# 

# ### Datenimport
# 
# Importieren aller libraries

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# Daten einfügen und Spaltennamen vergeben

# In[2]:


dataset = pd.read_csv('Spending_Score.csv',index_col='CustomerID')


# ### Datenexploration und Erkenntnisgewinn

# In[3]:


dataset.head()


# Gender nicht numerisch.

# In[4]:


dataset.info()


# CustomerID: Kundenidentifikationsnummer (hier irrelevant) </br>
# Gender: kategorialen Datentyp; Geschlecht des Kunden.</br>
# Age: numerischen Datentyp; Alter des Kunden in Jahren.</br>
# Annual Income(k$): numerischen Datentyp; Jahreseinkommen der Kunden in 1000 Dollar</br>
# Spending Score: numerischen Datentyp; Punktzahl zwischen 1 und 100 für einen Kunden auf der Grundlage seines Ausgabeverhaltens.</br>

# Gender muss vor dem Einsatz im Modell vorverarbeitet und in numerische Werte umgewandelt werden. 

# In[5]:


gender= {'Male':0, 'Female':1}
dataset['Gender']= dataset['Gender'].map(gender)


# In[6]:


dataset.describe()


# ### Datenbereinigung und -transformation

# In[7]:


dataset.isnull().sum()


# Keine fehlenden Werte

# In[8]:


dataset.drop_duplicates(inplace=True)
dataset.info() # Vergleich mit vorherigem data.info() un zu sehen, ob Values fehlen


# Keine Duplikate!

# Mit Hilfe eines Streudiagramms können wir den Datensatz auf Unregelmäßigkeiten überprüfen.

# In[9]:


sns.pairplot(dataset)
plt.show()


# Es scheinen keine Unregelmäßigkeiten im Datensatz zu bestehen. </br>
# Mit Hilfe einer Heatmap können zusätzlich Korrelationen zwischen Variablen aufgezeigt werden.

# In[10]:


sns.heatmap(dataset.corr(), annot = True)
plt.show()


# Das Geschlecht korreliert nur in sehr geringem Maße mit dem "Spending Score" und etwas stärker mit dem Jahreseinkommen.</br>
# Das Alter der Kunden korreliert recht stark negativ mit "Spending Score".</br>
# Das Jahreseinkommen korreliert nur sehr gering mit dem Alter wie auch das Jahreseinkommen und der "Spending Score".</br>
# Da für uns die Korrelation mit dem Kaufverhalten der Kunden also dem Spending Score am wichtigsten ist, wird folglich analysiert, welche Altersgruppen eventuell besonders empfänglich für bestimmte Preisniveaus sind.

# In[11]:


sns.distplot(dataset['Age'], bins=30)


# In[12]:


sns.distplot(dataset['Spending_Score'], bins=30)


# Age und Spending Score enthalten Werte, die nahezu normal verteilt sind.

# ### Datenaufbereitung

# Für die Modellerstellung sind nur Alter und Spending Score relevant.

# In[13]:


x = dataset.iloc[:, [1, 3]].values


# ## Modell

# ### Vorgehensweise:
# 
# Modellauswahl</br>
# K (optimale Gruppenanzahl) bestimmen durch Ellbow Method</br>
# Modellerstellung</br>
# Modellvisualisierung + Label definieren</br>
# Modellevaluation durch Silhoutte Analyse</br>
# Interpretation</br>
# 

# ### Modellauswahl
# 
# Modelle, welche auf den k-Means-Algorithmus aufbauen sind simpel zu implementieren, skalieren gut mit großen Datensätzen und sind einfach zu interpretieren. Probleme durch viele Ausreißer sind bei unseren Daten ohnehin nicht zu befürchten. Lediglich die Bestimmung eines optimalen K (Anzahl an zu bildenden Gruppen) muss manuell erfolgen, wobei es hierzu einige Hilfsmittel gibt.

# Um die am besten geeignete Anzahl an Gruppen zu finden, wenden wir die "Ellbow Method" an.

# In[14]:


from sklearn.cluster import KMeans
# Anzahl an Clustern durch Elbow Method bestimmen
# Summe der quadrierten Abstände zwischen jedem Punkt und dessen nächsten Clusterzentrums
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x)
    distortions.append(kmeanModel.inertia_)


# In[36]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 10), distortions, color='red', marker='o')
plt.title('The Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Distortion')
plt.show()


# Wir sehen ab 4 Clustern nurnoch geringe Änderungen an der Verzerrung (Distortion). Somit haben wir die optimale Anzahl an zu definierten Clustern bestimmt.

# ### Modellerstellung

# In[37]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
ymeans = kmeans.fit_predict(x)


# ### Modellvisualisierung

# Die einzelnen Cluster werden passende Namen/Label zugeordnet

# In[38]:


plt.figure(figsize=(10,15))
plt.title('Zielgruppenidentifikation')

plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 50, c = 'orange', label = 'sparsame Kunden' )
plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 50, c = 'red', label = 'Zielgruppe')
plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 50, c = 'green', label = 'junge Kunden')
plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 50, c = 'blue', label = 'alte Kunden')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100,c = 'black')

plt.xlabel('Alter')
plt.ylabel('Kaufbereitschaft (1-100)')
plt.legend()
plt.grid(False)
plt.show()


# Label/Namen passend für den Use Case definiert. k=4 scheint die richtige Entscheidung gewesen zu sein. Um sicher zu gehen kann dies noch evaluiert werden. (Interpretation folgt)

# ### Modellevaluation
# 
# Modelevaluation (Silhoutte Analysis)
# 
# 
# Um sicher zu gehen, dass die optimale Anzahl an Clustern wirklich 4 ist, können wir das ganze nochmals mit Hilfe der Silhoutte Methode überprüfen.

# In[17]:


import sklearn.cluster as cluster
import sklearn.metrics as metrics
for i in range(2,8):
    labels=cluster.KMeans(n_clusters=i,init="k-means++").fit(x).labels_
    print ("Clusters = "+ str(i) +" is "
           +str(metrics.silhouette_score(x,labels,metric="euclidean")))


# Höchster Wert bei k(clusters) = 4, damit bereits die beste Anzahl an Clustern gefunden

# ### Interpretation
# 
# Durch die Clusteranalyse lässt sich erkennen, je niedriger das Alter, desto höher ist die Ausgabenquote. Daraus schließen wir, dass folglich nur Angebote für hochpreisige Immobilien an unter 40-Jährige gesendet werden sollten, da ältere Kunden eine geringere Kaufbereitschaft für hochpreisige Immobilien haben. Gründe hierfür gibt es viele. So kann es Unterschiede zwischen alt zu jung in den Finanzierungsmöglichkeiten oder auch im Lebensstil und den damit einhergehenden Entscheidungen geben. Ältere Kunden erhalten eventuell schwieriger Immobilienkredite in einem ohnehin inflationären Markt und benötigen eventuell gar keine teurere und damit verbunden auch oft größere Immobilie, da die Familienplanung möglicherweise schon abgeschlossen ist oder die Prioritäten an die Wunschimmobilie eine andere geworden ist. Junge Kunden sind möglicherweise auch einfach etwas risikobereiter beim Immobilienkauf oder planen mit der gekauften Immobilie noch lange Zeit. Das Modell präsentiert uns anhand der Daten die Gruppen, welche nun vom Unternehmen für Marktanalysen weiterverwendet werden können, um eine noch genauere Zielgruppe für Immobilienpreisniveaus zu finden.
