import streamlit as st
import streamlit.components.v1 as components
import pickle 
import pandas as pd
import shap
import numpy as np
import PIL
import seaborn as sns
from zipfile import ZipFile
import matplotlib.pyplot as plt
from sklearn.externals import six
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


z = ZipFile("X_test.zip")
X_test = pd.read_csv(z.open('X_test.csv'))

# loading the trained model
classifier_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(classifier_in)

# target
X_test_target = X_test.copy()
prob = classifier.predict_proba(X_test)[:,1]
targetpred = []
for i in range(0,len(prob)) :
  if prob[i] >= 0.27 :
    targetpred.append(1)
  else :
    targetpred.append(0)
X_test_target['TARGET']=targetpred

# loading shap_values
shap_in = open('shap.pkl', 'rb') 
shap_values = pickle.load(shap_in)
explainer = shap.KernelExplainer(classifier.predict_proba, X_test, link = 'logit') 

# Most important features
vals = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame(list(zip(X_test.columns,sum(vals))),columns=['col_name','feature_importance_vals'])
importance_df = importance_df.sort_values('feature_importance_vals', ascending=False)
most_important_features = importance_df['col_name'][importance_df['feature_importance_vals']>5].tolist()

# defining the function which will make the prediction using the data which the user inputs 
def prediction(client):    
    prediction = classifier.predict_proba(X_test)[X_test[X_test['SK_ID_CURR']==client].index.item()][1]
    if prediction >= 0.27:
        pred = 'Refusé'
    else:
        pred = 'Autorisé'
    return pred



# to plot shap
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# main function  
def main():       
    st.image(PIL.Image.open('/content/drive/My Drive/Projet 7/logo.png'), width = 150)
    st.title('Home Credit Default Risk')
    st.write('## Présentation des informations personnelles du client et décision d\'octroi du crédit')
    
    # select client
    st.sidebar.markdown('## Sélectionner le numéro du client')
    client = st.sidebar.selectbox('Numéro du client', X_test.iloc[0:100,:]['SK_ID_CURR'])
    result =""
      
    # shap features
    st.write('## Visualisation des features les plus importantes et leur impact sur le défault de paiement')
    st.image(PIL.Image.open('/content/drive/My Drive/Projet 7/shap features.png'), width = 600)
    st.write('Exemple d\'interprétation: Plus la valeur de "Ext_source_2" est importante, plus la probabilité de faillite diminue')

    # print prediction 
    st.write('## Prêt {}'.format(prediction(client)))
    st.write('## Probabilité de faillite :', classifier.predict_proba(X_test)[X_test[X_test['SK_ID_CURR']==client].index.item()][1])
    
    # shap
    st.write('## Visualisation de l\'impact des features les plus importantes sur la probabilité de faillite du client')
    st_shap(shap.force_plot(explainer.expected_value[1], 
                            shap_values[1][X_test[X_test['SK_ID_CURR']==client].index.item(),:], 
                            X_test.iloc[X_test[X_test['SK_ID_CURR']==client].index.item(),:], 
                            link='logit'), 300)
    st.write('La couleur bleue signifie que la valeur de l\'information fait baisser la probabilité de faillite')
    st.write('La couleur rouge signifie que la valeur de l\'information fait augmenter la probabilité de faillite')

    # select info client à visualiser
    st.write('## Comparaison avec les autres clients')
    option = st.selectbox('Informations sur le client', most_important_features)
    
    if (X_test[option].dtypes!='object')==True:
      data = pd.DataFrame({'Moyenne des clients remboursant leur prêt': [X_test_target[X_test_target['TARGET']==0][option].mean()],
                        'Moyenne': [X_test_target[option].mean()],
                        'Moyenne des clients faisant faillite': [X_test_target[X_test_target['TARGET']==1][option].mean()],
                        'Le client':X_test_target[X_test_target['SK_ID_CURR']==client][option].tolist()}, index=[0])
      values = data.values.flatten()
      fig1 = plt.figure()
      ax = sns.barplot(x = data.columns,y=values)
      plt.xticks(rotation=45, horizontalalignment='right')
      st.pyplot(fig1)
    else :
      st.write('Le client est dans la catégorie :', X_test.loc[X_test[X_test['SK_ID_CURR']==client].index.item(),option])
      fig2 = plt.figure()
      ax = sns.countplot(x = option, data = X_test_target, hue = 'TARGET')
      plt.xticks(rotation=45, horizontalalignment='right')
      st.pyplot(fig2)

if __name__=='__main__': 
    main()
