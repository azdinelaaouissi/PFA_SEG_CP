# 🏥 Segmentation d'Images Médicales pour le Diagnostic du Cancer du Poumon et des Maladies Cardiaques

## 📌 Description du Projet
Ce projet a été réalisé au sein de la **Faculté des Sciences Ben M'Sik** dans le cadre du projet de fin d'année pour l'obtention du **Master en Data Science et Big Data**. Il vise à développer une solution efficace pour la **segmentation d'images médicales**, avec une application spécifique sur :
- 🫁 **Le cancer du poumon**  
- ❤️ **Les maladies cardiaques**  

Nous avons utilisé **U-Net** comme architecture principale, en comparant **PyTorch** et **TensorFlow/Keras** pour la segmentation d'images.  

📊 Une interface interactive développée avec **Streamlit** permet la visualisation des résultats.

---

## 🚀 Objectifs
- Développer une solution de **segmentation d’images médicales** pour aider les cliniciens.  
- Comparer les performances de **PyTorch** et **TensorFlow/Keras** en termes de **précision et de rapidité**.  
- Offrir une interface utilisateur interactive pour faciliter l’interprétation des résultats.  

---

## 🖼️ Résultats Visuels
### 🔬 Segmentation Médicale  
<img src="./img/7.gif" width="400px">

### 🏗️ Architecture du Modèle U-Net  
<img src="./u-net-architecture.png" style="width: 100%; height: auto;">

### 📉 Courbes de Performance  

**Courbe de Calibration**  
<img src="./calibration_curve.png" style="width: 100%; height: auto;">

**Courbe Precision-Recall**  
<img src="./precision_recall_curve.png" style="width: 100%; height: auto;">

**Courbe ROC**  
<img src="./roc_curve.png" style="width: 100%; height: auto;">

---

## 📚 Données Utilisées
Deux ensembles de données ont été utilisés :

- **Cancer du poumon** 🫁 : [Dataset NCI](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- **Maladies cardiaques** ❤️ : [Dataset Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation)

---

## 🏗️ Méthodologie

### 🔹 Modèle utilisé : **U-Net**
Nous avons adopté **U-Net**, connu pour son efficacité dans la segmentation d’images médicales.

Deux implémentations ont été développées :
1. **U-Net avec PyTorch** : Pour la segmentation des images CT du cancer du poumon.  
2. **U-Net avec TensorFlow/Keras** : Pour la segmentation des images des maladies cardiaques.  

Les modèles ont été **entraînés avec data augmentation** et des **techniques d’optimisation adaptées**.

---

## 🎯 Performances des Modèles  


<div style="width: 100%; overflow-x: auto;">
  <table>
    <tr>
      <th>📊 **Métrique**</th>
      <th>**Valeur**</th>
    </tr>
    <tr>
      <td>IoU (Intersection over Union)</td>
      <td>0.8190</td>
    </tr>
    <tr>
      <td>Précision</td>
      <td>0.9673</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>0.8424</td>
    </tr>
    <tr>
      <td>F1 Score</td>
      <td>0.9005</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td>0.99995</td>
    </tr>
    <tr>
      <td>Dice Score</td>
      <td>0.8575</td>
    </tr>
  </table>
</div>

---

## 🖥️ Interface Utilisateur
Nous avons développé une interface interactive avec **Streamlit** qui permet de :
- **Charger des fichiers d’imagerie médicale** (`.npy`, `.nii.gz`).
- **Visualiser les prédictions en superposant les masques segmentés**.
- **Comparer les images normales et segmentées**.

### 📌 Exemple de Visualisation de l'Application Streamlit

**Page d'Accueil**  
![Page d'Accueil](./img/1.png)

**Visualisation des Résultats**  
![Visualisation](img/2.png)

**Chargement des Fichiers**  
![Chargement](img/3.png)

**Affichage des Prédictions**  
![Affichage](img/4.png)

**Superposition des Masques**  
![Superposition](img/5.png)

**Analyse des Performances**  
![Analyse](img/6.png)

---

## ⚙️ Installation et Exécution

### 🔽 1. Cloner le projet  
```bash
git clone https://github.com/votre-repo/segmentation-medicale.git
cd segmentation-medicale
```

### 📦 2. Installer les dépendances  
```bash
pip install -r requirements.txt
```

### 🚀 3. Lancer l'application Streamlit  
```bash
streamlit run app.py
```

---

## 📬 Contact  
📧 Email : laaouissi.azdine@gmail.com  
📌 GitHub : [Votre Profil](https://github.com/azdinelaaouissi)  

Si ce projet vous est utile, n’hésitez pas à laisser une ⭐ sur GitHub ! 🚀
