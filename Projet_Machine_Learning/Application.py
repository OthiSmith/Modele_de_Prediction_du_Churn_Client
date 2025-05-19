# Import des modules et packages

import streamlit as st
import pickle
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score

#-------------------------- Configuration de la page ----------------------
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Customer churn",
    page_icon="🧊",
    #layout="wide",
    initial_sidebar_state="expanded"
)
# Ajouter des sélecteurs dans une barre sur le côté
add_tile = st.sidebar.subheader("Sélection de filtres")


#--------------------------------- Head
## ===> Header

st.markdown("<h1 style='text-align: center;'>PROJET MACHINE LEARNING</h1>", unsafe_allow_html=True)


st.image('https://www.sikafinance.com/api/image/ImageNewsGet?id=7D392CA2-0B1A-4EF4-9F3E-284C045F7C3B')

st.markdown("""
**Le but de ce projet est de prédire si un client va résilier son abonnement à partir de certaines de ses caractéristiques.** """
)

#-------------------- Identification des éditeurs -----------------------------------
st.markdown(""" Ce projet est réalisé par:
*   **ALLOU Mardoché Jacques**
*   **ARRA Othniel Emmanuel**
*   **NAGALO Bouma Zakaria** """)

st.markdown('Sous la supervision de **M. Raymond COULIBALY**')

st.markdown(""" ==> Pré requis :
*   **Python :** pandas, sklearn, streamlit.
""")


#------------------  Importation des données  --------------------
#@st.cache

def load_data():
    data = pd.read_csv("C:/Users/HP/Desktop/ISE_2/Machine_Learning/PROJET/DataSet_FDD.csv", sep=";", index_col="ID")
    return data
data = load_data()

# Afficher les données uniquement si l'utilisateur souhaite les voir
if st.checkbox('Afficher la base de données'):
    st.subheader('Base de données')
    st.write(data)
    
    
    st.write("La base décrit le churn ou non des clients d'une société de télécommunication à partir de certaines caractéristiques de ceux-ci. Elle contient", data.shape[0], "lignes et", data.shape[1], "colonnes." , unsafe_allow_html=True)

    
    st.write("La variable CATEGORIE_OFFRE_M1 n'est pas dans la base de donnée et la variable ALEA n'est pas dans le dictionnaire des données. De plus, la variable MONTANT_OM_HT_M1 contient une seule modalité. Elle ne contient aucune information. Nous allons donc supprimer les variables MONTANT_OM_HT_M1 et ALEA de notre jeu de données pour la suite de notre analyse.")
    
    data = data.drop (["MONTANT_OM_HT_M1", "ALEA"], axis=1)
    
add_selectbox = st.sidebar.selectbox(
    "Que voulez-vous afficher ? ",
    ("Visualisation des données", "Performances du modèle initial", "Prédiction")
)


#-------------------        Visualisation       ---------------------

if add_selectbox=="Visualisation des données" :
        add_selectbox = st.sidebar.selectbox(
    "Analyses",
    ("Analyse Univariée", "Analyse Bivariée")
            
)
        if add_selectbox=="Analyse Univariée" :
            # Premier titre
            st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Visualisation des données </h1>", unsafe_allow_html=True)

                # Deuxième titre
            st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:26px;color:SlateBlue;text-align: left;'> Analyse univariée</h1>", unsafe_allow_html=True)

                # Troisième titre

            st.markdown("<h3 style='font-family:Lucida Caligraphy;font-size:26px;color:Red;text-align: center;'> Diagrammes circulaires des variables catégorielles</h3>", unsafe_allow_html=True)

                 # Faire trois colonnes 
            gauche, milieu, droite = st.columns(3)

                # Créer le graphique
            with gauche:
                    #----- Premier graphique de gauche 
                    fig, ax = plt.subplots()
                    data["CHURN"].value_counts().plot(kind="pie", ax=ax, autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("Il y a plus de clients qui n'ont pas churné (89.5%) que des clients qui ont churné (moins de 10.5%).")


                    #----- Deuxième graphique de gauche

                    fig, ax = plt.subplots()
                    data["STATUT_TRAFIC_M1"].value_counts().plot(kind="pie", ax=ax, autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("Il y a plus de clients( 90.7%) ont ouvert au moins une session internet le mois precedent que de clients qui ne l'ont pas fait(moins de 9.3%)")


                    #----- Troisième graphique de gauche

                    fig, ax = plt.subplots()
                    data["STATUT_RC_M2"].value_counts().plot(kind="pie", ax=ax, autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("Les clients qui se sont rechargés il ya 2 mois ont une proportion de 77.7% ,tandis que ceux qui sont restés sans recharge il y a deux mois representent 22.3%")

            with milieu:

                    fig, ax = plt.subplots()
                    data["AUTRE_ABONNEMENT"].value_counts().plot(kind="pie", ax=ax,autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("La variable AUTRE_ABONNEMENT indique si le client détient une autre technologie(1), 0 sinon.")

                    #----- Deuxième graphique du milieu
                    fig, ax = plt.subplots()
                    data["STATUT_FACT_M1"].value_counts().plot(kind="pie", ax=ax, autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("La quasi totalité soit 98.6% des clients ont payé leur facture du mois dernier. Cela laisse une bonne appréhension sur la ponctualité des clients")


                    #----- Troisième graphique du milieu

                    fig, ax = plt.subplots()
                    data["STATUT_FACT_M2"].value_counts().plot(kind="pie", ax=ax, autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("Un grand nombre de clients soit 98.5% a payé sa facture d'il y a deux mois.")

            with droite:

                    #----- Premier graphique de droite
                    fig, ax = plt.subplots()
                    data["STATUT_RC_M1"].value_counts().plot(kind="pie", ax=ax, autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("Les clients qui se sont rechargés le mois precedent representent 86.4% contre 13.6% qui ne l'ont pas fait")

                    #----- Deuxième graphique de droite
                    fig, ax = plt.subplots()
                    data["STATUT_TRAFIC_M2"].value_counts().plot(kind="pie", ax=ax, autopct=lambda x: str(round(x, 1)) + '%')
                    st.pyplot(fig)
                    st.write("Les clients n'ayant pas ouvert au moins une session internet il y a deux mois representent 84.4% contre ceux qui l'ont fait 15.6%")



            st.markdown("<h2 style='font-family:Lucida Caligraphy;font-size:26px;color:Black;text-align: left;'> Analyse de la variable ANCIENNETE</h2>", unsafe_allow_html=True) 

            fig, ax = plt.subplots(figsize=(3,3))
            ax.bar(x=data["ANCIENNETE"].unique(), height=data["ANCIENNETE"].value_counts(), color='r')
            ax.set_xlabel(xlabel='Ancienneté' ,color='b', size=20)
            ax.set_ylabel(ylabel='Effectifs' ,color='b', size=20)
            plt.xticks(rotation = 45)
            plt.title("Effectif par Tranche d'ancienneté")
            plt.show()
            st.pyplot(fig)

            st.write("Les clients ayant une ancienneté de 1 à 3 mois sont les plus nombreux. Ils sont environ 1750 soit 35% des clients. Les plus anciens c'est à dire ceux qui ont une ancienneté de 6 à 24 mois sont les moins nombreux environ 400 soit 8%.")

            st.markdown("<h2 style='font-family:Lucida Caligraphy;font-size:26px;color:Black;text-align: left;'> Description des variables quantitatives</h2>", unsafe_allow_html=True)

                   # Faire deux colonnes 
            gauche, droite = st.columns(2)

                    # Créer le graphique
            with gauche:
                #----- Premier graphique de gauche 
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data["VOL_DATA_M1"], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution du volume de data consommé par la ligne le mois précédent ")

                    #----- Deuxième graphique de gauche

                fig, ax = plt.subplots()
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['MONTANT_FACT_HT_M1'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution de la valeur totale de la facture payée le mois précédent (en Fcfa)")



                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['VOL_DATA_M2'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution du volume de data consommé par la ligne il y a deux mois (en Mo)")

                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data[ 'VOL_TOT_DATA_KO_M2'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution du Volume de data consommé par la ligne mobile du client il y a deux mois (en Ko)")


                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['MONTANT_OM_HT_M2'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution de la valeur totale du rechargement effectué via Orange Money (en Fcfa)")



                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['CONSO_M2'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution des dépenses totales effectuées par la ligne mobile du client  il y a deux mois (en Fcfa)")


            with droite:


                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['MONTANT_FACT_HT_M2'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution de la valeur totale de la facture payée le mois précédent (en Fcfa)")



                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['VOL_TOT_DATA_KO_M1'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution du Volume de data consommé par la ligne mobile du client le mois précédent (en Ko)")




                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data[ 'CONSO_M1'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution des dépenses totales effectuées par la ligne mobile du client le mois précédent (en Fcfa)")



                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['VOL_TOT_VOIX_M1'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution du nombre total de minute d'appel effectué par la ligne mobile du client le mois précédent (en min)")


                fig, ax = plt.subplots(figsize=(12, 10))
                ax.hist(data['VOL_TOT_VOIX_M2'], bins=50)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                st.pyplot(fig)
                st.write("Distribution du Nombre total de minute d'appel effectué par la ligne mobile du client il y a deux mois (en min)")



            st.markdown("<h2 style='font-family:Lucida Caligraphy;font-size:26px;color:Black;text-align: left;'> On observe donc en général une très forte asymétrie positive dans la distribution des données.</h2>", unsafe_allow_html=True)   

            st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:26px;color:Red;text-align: center;'> Description des variables quantitatives</h1>", unsafe_allow_html=True) 

            df1 = pd.read_csv("C:/Users/HP/Desktop/ISE_2/Machine_Learning/PROJET/DataSet_FDD.csv", sep=";", index_col="ID",dtype={'CHURN':'category', 'AUTRE_ABONNEMENT':'category', 'STATUT_RC_M1':'category', 'STATUT_TRAFIC_M1':'category', 'STATUT_FACT_M1':'category', 'STATUT_TRAFIC_M2':'category', 'STATUT_RC_M2':'category', 'STATUT_FACT_M2':'category'})

            # Suppression de la colonne "MONTANT_OM_HT_M1"
            df1 = df1.drop("MONTANT_OM_HT_M1", axis = 1)

            # Affichage du dataframe dans Streamlit
            st.dataframe(df1.describe())
            st.markdown("**Ci-dessus le tableau présentant les caractéristiques de tendances centrales et dispersions des variables quantitatives.**")
        if add_selectbox=="Analyse Bivariée" :

            st.markdown("<h3 style='font-family:Lucida Caligraphy;font-size:26px;color:SlateBlue;text-align: left;'> Analyse Bivariée</h3>", unsafe_allow_html=True)

            df = pd.read_csv("C:/Users/HP/Desktop/ISE_2/Machine_Learning/PROJET/DataSet_FDD.csv", sep=";", index_col="ID")
            df = df.drop('MONTANT_OM_HT_M1', axis = 1)

            corr = df.corr(method='spearman')
            corr.sort_values(['CHURN'], ascending=False, inplace=True)

            # Affichage des coefficients de corrélation de Spearman
            st.write("--- Affichage des coefficients de corrélations de Spearman entre la variable CHURN et les autres variables. ---")
            st.write(corr['CHURN'])

            st.markdown(""" Nous constatons une faible corrélation entre la variable CHURN et certaines variables. Les variables les plus corrélées au CHURN sont :
               **STATUT_TRAFIC_M1,**
               **STATUT_FACT_M2,**
               **STATUT_FACT_M1,** 
               **ANCIENNETE.**""")


            st.markdown("Affichage du heatmap")
            fig, ax = plt.subplots(figsize=(18,10))
            sns.heatmap(df.corr(method='spearman'), vmin=-1, vmax=1, annot=True, ax=ax)
            plt.title("Heatmap")
            st.pyplot(fig)

# ---------------------------------- Performances du modèle initial  ---------------------------
if add_selectbox=="Performances du modèle initial" :
    # Premier titre
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Performances du modèle initial </h1>", unsafe_allow_html=True)
    
    #-------------------  Traitement des données   ----------------------

    # Récupérer la liste des variables qui comportent des valeurs manquantes
    null_variables = data.columns[data.isnull().any()].tolist()
    
    # Imputation des valeurs manquantes

    impute_value = -1 # Imputer les valeurs manquantes avec la valeur -1

    for col in null_variables:
        data[col].fillna(impute_value, inplace=True) # Remplacer les valeurs manquantes par la valeur imputée

    data = data.replace(",", ".", regex=True)
    recodage = {'1- 03Mois': 1, '5- 18Mois': 2, '3- 09Mois' : 3,'2- 06Mois': 4, '4- 12Mois': 5, '7- 25Mois+': 6, '6- 24Mois': 7}
    data['ANCIENNETE'] = data['ANCIENNETE'].map(recodage)

    
    #------- Split des données
    
    
    X = data.drop('CHURN', axis = 1)
    y = data['CHURN']
    seed = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = seed, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, 
                                                    random_state = seed, stratify =y_test)
    
    #-------- Sur échantillonnage
    X2 = X_train
    X2['CHURN'] = y_train.values

    minority = X2[X2.CHURN == 1]
    majority = X2[X2.CHURN == 0]

    minority_upsampled = resample(minority, replace=True, n_samples = len(majority), random_state = seed)

    upsampled = pd.concat([majority, minority_upsampled])

    X_train_up = upsampled.drop('CHURN', axis=1)
    y_train_up = upsampled['CHURN']
    
    train_features = X_train_up
    train_labels = y_train_up
    
    # Convertir les colonnes en catégorie
    cat_cols = ['AUTRE_ABONNEMENT', 'ANCIENNETE', 'STATUT_RC_M1', 'STATUT_RC_M2', 'STATUT_TRAFIC_M1', 'STATUT_TRAFIC_M2',  'STATUT_FACT_M1', 'STATUT_FACT_M2']
    train_features[cat_cols] = train_features[cat_cols].astype('str')

    # Sélectionner les variables numériques
    num_cols = train_features.select_dtypes(include=['int', 'float']).columns.tolist()

    # Instanciation de l'objet
    scaler = preprocessing.MinMaxScaler()

    # Entrainement
    mod_scaler = scaler.fit(train_features[num_cols])

    # Application à l'ensemble des échantillons
    train_features[num_cols] = mod_scaler.transform(train_features[num_cols])
    X_val[num_cols] = mod_scaler.transform(X_val[num_cols])
    X_test[num_cols] = mod_scaler.transform(X_test[num_cols])
    
    # Remise des données dans un dataFrame
    train_features = pd.DataFrame(train_features, columns = X.columns)
    X_val = pd.DataFrame(X_val, columns = X.columns)
    X_test = pd.DataFrame(X_test, columns = X.columns)
    
    
    #------ Sélection de variables 
    vars_selected = ['VOL_DATA_M1', 'VOL_DATA_M2', 'STATUT_TRAFIC_M1', 'VOL_TOT_VOIX_M1', 'CONSO_M1', 'VOL_TOT_VOIX_M2', 'CONSO_M2', 'VOL_TOT_DATA_KO_M1', 'VOL_TOT_DATA_KO_M2', 'ANCIENNETE', 'STATUT_TRAFIC_M2', 'STATUT_RC_M1', 'STATUT_FACT_M2', 'STATUT_FACT_M1', 'AUTRE_ABONNEMENT', 'MONTANT_OM_HT_M2', 'STATUT_RC_M2']
    
    train_features = train_features[vars_selected]
    X_val = X_val[vars_selected]
    X_test = X_test[vars_selected]
    
    #-------------------------------------------------   Entrainement du modèle --------------------------------------------
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(train_features, train_labels)
    
    
    st.write("Les performances ci-dessous sont issues d'un modèle de RandomFrorest effectué avec notre jeu de données.")
    
    
    # Calcul des scores d'importance des variables
    vars_imp = pd.Series(rf_model.feature_importances_, index=train_features.columns).sort_values(ascending=False)

    # Création du graphique
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(x=vars_imp.index, y=vars_imp, ax=ax)
    ax.set_xticklabels(vars_imp.index, rotation=90)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Score d'importance de la variable")
    ax.set_title("Importance des variables prédictrices")

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)
    
    accuracy = round(rf_model.score(X_test, y_test), 2)
    st.write("Accuracy sur l'ensemble d'entrainement ---->", accuracy)
    
    y_pred = rf_model.predict(X_test)
    
    precision = round(precision_score (y_test, y_pred, average='weighted'),2)
    
    st.write("Précision  ---->", precision)
    
    #-----------------------------------------------------------------------------------------------------------------------
    
    #-------------------------------------------------   Chargement du modèle (pickle) --------------------------------------------
if add_selectbox=="Prédiction" :
    # Chargement du modèle à partir du fichier pickle
    with open('C:/Users/HP/Desktop/ISE_2/Machine_Learning/PROJET/rf_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    
    
    df=st.file_uploader("Choisir un fichier", type="csv")
    if df is not None:
        dd=pd.read_csv(df, sep=';', index_col="ID")
        dd = dd.drop (["MONTANT_OM_HT_M1", "ALEA"], axis=1)
        #st.dataframe(dd)
        if st.button("Valider"):
            if 'CHURN' in dd.columns:
          
                X=dd.drop("CHURN", axis=1)
                y_true=dd["CHURN"]


                #-------------------  Traitement des données   ----------------------

                # Récupérer la liste des variables qui comportent des valeurs manquantes
                null_variables = X.columns[X.isnull().any()].tolist()

                # Imputation des valeurs manquantes

                impute_value = -1 # Imputer les valeurs manquantes avec la valeur -1

                for col in null_variables:
                    X[col].fillna(impute_value, inplace=True) # Remplacer les valeurs manquantes par la valeur imputée

                X = X.replace(",", ".", regex=True)

                recodage = {'1- 03Mois': 1, '5- 18Mois': 2, '3- 09Mois' : 3,'2- 06Mois': 4, '4- 12Mois': 5, '7- 25Mois+': 6, '6- 24Mois': 7}
                X['ANCIENNETE'] = X['ANCIENNETE'].map(recodage)
                vars_selected = ['VOL_DATA_M1', 'VOL_DATA_M2', 'STATUT_TRAFIC_M1', 'VOL_TOT_VOIX_M1', 'STATUT_TRAFIC_M1', 'CONSO_M1', 'VOL_TOT_VOIX_M2', 'CONSO_M2', 'VOL_TOT_DATA_KO_M1', 'VOL_TOT_DATA_KO_M2', 'ANCIENNETE', 'STATUT_TRAFIC_M2', 'STATUT_RC_M1', 'STATUT_FACT_M2', 'STATUT_FACT_M1', 'AUTRE_ABONNEMENT', 'MONTANT_OM_HT_M2', 'STATUT_RC_M2',]
                X = X[vars_selected]
                y_pred=loaded_model.predict(X)
                score=accuracy_score(y_true,y_pred)
                st.write("Le score est :",score)
                precision = round(precision_score (y_true, y_pred, average='weighted'),2)

                st.write("La précision est : ", precision)

            if 'CHURN' not in dd.columns:
                 # Récupérer la liste des variables qui comportent des valeurs manquantes
                null_variables = dd.columns[dd.isnull().any()].tolist()

                # Imputation des valeurs manquantes

                impute_value = -1 # Imputer les valeurs manquantes avec la valeur -1

                for col in null_variables:
                    dd[col].fillna(impute_value, inplace=True) # Remplacer les valeurs manquantes par la valeur imputée

                dd = dd.replace(",", ".", regex=True)

                recodage = {'1- 03Mois': 1, '5- 18Mois': 2, '3- 09Mois' : 3,'2- 06Mois': 4, '4- 12Mois': 5, '7- 25Mois+': 6, '6- 24Mois': 7}
                dd['ANCIENNETE'] = dd['ANCIENNETE'].map(recodage)
                vars_selected = ['VOL_DATA_M1', 'VOL_DATA_M2', 'STATUT_TRAFIC_M1', 'VOL_TOT_VOIX_M1', 'STATUT_TRAFIC_M1', 'CONSO_M1', 'VOL_TOT_VOIX_M2', 'CONSO_M2', 'VOL_TOT_DATA_KO_M1', 'VOL_TOT_DATA_KO_M2', 'ANCIENNETE', 'STATUT_TRAFIC_M2', 'STATUT_RC_M1', 'STATUT_FACT_M2', 'STATUT_FACT_M1', 'AUTRE_ABONNEMENT', 'MONTANT_OM_HT_M2', 'STATUT_RC_M2',]
                dd = dd[vars_selected]
                y_pred=loaded_model.predict(dd)
                st.write(y_pred)