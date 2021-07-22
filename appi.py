{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copie de api.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyPNNM2ts57HaBnE23ctAmNr",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/charlottesei/OCP7_Home-Credit/blob/main/Copie_de_api.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oBXPjYUEJ41f",
        "outputId": "a4031ad0-53ec-438c-f60b-f0cae7cdadd5"
      },
      "source": [
        "import streamlit as st\n",
        "import streamlit.components.v1 as components\n",
        "import pickle \n",
        "import pandas as pd\n",
        "import shap\n",
        "import numpy as np\n",
        "import PIL\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "X_test = pd.read_csv('/content/drive/My Drive/Projet 7/X_test.csv')\n",
        "X_test = X_test.iloc[0:2000,:]\n",
        "\n",
        "# loading the trained model\n",
        "classifier_in = open('/content/drive/My Drive/Projet 7/classifier.pkl', 'rb') \n",
        "classifier = pickle.load(classifier_in)\n",
        "\n",
        "# target\n",
        "X_test_target = X_test.copy()\n",
        "prob = classifier.predict_proba(X_test)[:,1]\n",
        "targetpred = []\n",
        "for i in range(0,len(prob)) :\n",
        "  if prob[i] >= 0.27 :\n",
        "    targetpred.append(1)\n",
        "  else :\n",
        "    targetpred.append(0)\n",
        "X_test_target['TARGET']=targetpred\n",
        "\n",
        "# loading shap_values\n",
        "shap_in = open('/content/drive/My Drive/Projet 7/shap.pkl', 'rb') \n",
        "shap_values = pickle.load(shap_in)\n",
        "explainer = shap.KernelExplainer(classifier.predict_proba, X_test, link = 'logit') \n",
        "\n",
        "# Most important features\n",
        "vals = np.abs(shap_values).mean(axis=0)\n",
        "importance_df = pd.DataFrame(list(zip(X_test.columns,sum(vals))),columns=['col_name','feature_importance_vals'])\n",
        "importance_df = importance_df.sort_values('feature_importance_vals', ascending=False)\n",
        "most_important_features = importance_df['col_name'][importance_df['feature_importance_vals']>5].tolist()\n",
        "\n",
        "# defining the function which will make the prediction using the data which the user inputs \n",
        "def prediction(client):    \n",
        "    prediction = classifier.predict_proba(X_test)[X_test[X_test['SK_ID_CURR']==client].index.item()][1]\n",
        "    if prediction >= 0.27:\n",
        "        pred = 'Refusé'\n",
        "    else:\n",
        "        pred = 'Autorisé'\n",
        "    return pred\n",
        "\n",
        "\n",
        "\n",
        "# to plot shap\n",
        "def st_shap(plot, height=None):\n",
        "    shap_html = f\"<head>{shap.getjs()}</head><body>{plot.html()}</body>\"\n",
        "    components.html(shap_html, height=height)\n",
        "\n",
        "# main function  \n",
        "def main():       \n",
        "    st.image(PIL.Image.open('/content/drive/My Drive/Projet 7/logo.png'), width = 150)\n",
        "    st.title('Home Credit Default Risk')\n",
        "    st.write('## Présentation des informations personnelles du client et décision d\\'octroi du crédit')\n",
        "    \n",
        "    # select client\n",
        "    st.sidebar.markdown('## Sélectionner le numéro du client')\n",
        "    client = st.sidebar.selectbox('Numéro du client', X_test.iloc[0:100,:]['SK_ID_CURR'])\n",
        "    result =\"\"\n",
        "      \n",
        "    # shap features\n",
        "    st.write('## Visualisation des features les plus importantes et leur impact sur le défault de paiement')\n",
        "    st.image(PIL.Image.open('/content/drive/My Drive/Projet 7/shap features.png'), width = 600)\n",
        "    st.write('Exemple d\\'interprétation: Plus la valeur de \"Ext_source_2\" est importante, plus la probabilité de faillite diminue')\n",
        "\n",
        "    # print prediction \n",
        "    st.write('## Prêt {}'.format(prediction(client)))\n",
        "    st.write('## Probabilité de faillite :', classifier.predict_proba(X_test)[X_test[X_test['SK_ID_CURR']==client].index.item()][1])\n",
        "    \n",
        "    # shap\n",
        "    st.write('## Visualisation de l\\'impact des features les plus importantes sur la probabilité de faillite du client')\n",
        "    st_shap(shap.force_plot(explainer.expected_value[1], \n",
        "                            shap_values[1][X_test[X_test['SK_ID_CURR']==client].index.item(),:], \n",
        "                            X_test.iloc[X_test[X_test['SK_ID_CURR']==client].index.item(),:], \n",
        "                            link='logit'), 300)\n",
        "    st.write('La couleur bleue signifie que la valeur de l\\'information fait baisser la probabilité de faillite')\n",
        "    st.write('La couleur rouge signifie que la valeur de l\\'information fait augmenter la probabilité de faillite')\n",
        "\n",
        "    # select info client à visualiser\n",
        "    st.write('## Comparaison avec les autres clients')\n",
        "    option = st.selectbox('Informations sur le client', most_important_features)\n",
        "    \n",
        "    if (X_test[option].dtypes!='object')==True:\n",
        "      data = pd.DataFrame({'Moyenne des clients remboursant leur prêt': [X_test_target[X_test_target['TARGET']==0][option].mean()],\n",
        "                        'Moyenne': [X_test_target[option].mean()],\n",
        "                        'Moyenne des clients faisant faillite': [X_test_target[X_test_target['TARGET']==1][option].mean()],\n",
        "                        'Le client':X_test_target[X_test_target['SK_ID_CURR']==client][option].tolist()}, index=[0])\n",
        "      values = data.values.flatten()\n",
        "      fig1 = plt.figure()\n",
        "      ax = sns.barplot(x = data.columns,y=values)\n",
        "      plt.xticks(rotation=45, horizontalalignment='right')\n",
        "      st.pyplot(fig1)\n",
        "    else :\n",
        "      st.write('Le client est dans la catégorie :', X_test.loc[X_test[X_test['SK_ID_CURR']==client].index.item(),option])\n",
        "      fig2 = plt.figure()\n",
        "      ax = sns.countplot(x = option, data = X_test_target, hue = 'TARGET')\n",
        "      plt.xticks(rotation=45, horizontalalignment='right')\n",
        "      st.pyplot(fig2)\n",
        "\n",
        "if __name__=='__main__': \n",
        "    main()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Overwriting app.py\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}
