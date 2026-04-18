import pandas as pd
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
from thefuzz import process, fuzz
from scipy.stats import poisson


# 1. CONFIGURATION ET CHARGEMENT
FILE_ASSURANCE = 'dataset_final_v2_corrige_20260418_054155.xlsx' 
FILE_COORDS = 'dzaadmin_boundaries.xlsx'

print("Chargement des fichiers...")
df_assurance = pd.read_excel(FILE_ASSURANCE)
# Note : Vérifie bien le nom de l'onglet ici (dza_admin2 ou dzaadmin_boundaries)
df_coords = pd.read_excel(FILE_COORDS, sheet_name='dza_admin2')

# 2. NETTOYAGE DES DONNÉES FINANCIÈRES (Correction du TypeError)
print("Nettoyage des colonnes numériques...")

def clean_numeric(col):
    # Enlève les espaces, les "DA", "DZD" et force le format numérique
    col = col.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(col, errors='coerce').fillna(0)

df_assurance['CAPITAL_ASSURE'] = clean_numeric(df_assurance['CAPITAL_ASSURE'])
df_assurance['PRIME_NETTE'] = clean_numeric(df_assurance['PRIME_NETTE'])
df_assurance['FACTEUR_A'] = pd.to_numeric(df_assurance['FACTEUR_A'], errors='coerce').fillna(0.1)

# 3. NETTOYAGE GÉOGRAPHIQUE

FIX_WILAYAS = {
    "EL TAREF": "EL TARF", "EL_TAREF": "EL TARF",
    "TIPAZA": "TIPAZA", "AIN TEMOUCHENT": "AIN TEMOUCHENT"
}

MANUAL_COMMUNE_FIXES = {
    "SOUMAA_BLIDA": "SOUMAA", "OULED_YAICH_BLIDA": "OULED YAICH",
    "CHORFA_BOUIRA": "CHORFA", "EL_OUELDJA_SETIF": "EL OUELDJA",
    "CHERAGA_ALGER": "CHERAGA", "LARBAA_BLIDA": "LARBAA",
    "AIN_BENIAN_(ALGER)": "AIN BENIAN", "MOHAMMADIA_MASCARA": "MOHAMMADIA",
    "ARZEW_ORAN": "ARZEW", "BESBES_EL_TARF": "BESBES"
}

def clean_text(s):
    if pd.isna(s) or str(s).strip() in ["", "|"]: 
        return "VIDE"
    s = str(s).upper().strip()
    if s in MANUAL_COMMUNE_FIXES: s = MANUAL_COMMUNE_FIXES[s]
    s = re.sub(r'(_.*|\(.*\))', '', s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace("-", " ").replace("'", " ").replace("_", " ")
    return " ".join(s.split()).strip()

print("Application du nettoyage géographique...")
df_assurance['WILAYA_CLEAN'] = df_assurance['WILAYA'].apply(clean_text).replace(FIX_WILAYAS)
df_coords['WILAYA_REF'] = df_coords['adm1_name'].apply(clean_text).replace(FIX_WILAYAS)

df_assurance['COMMUNE_CLEAN'] = df_assurance['COMMUNE'].apply(clean_text)
df_coords['COMMUNE_REF'] = df_coords['adm2_ref_name'].apply(clean_text)

df_assurance['JOIN_KEY'] = df_assurance['COMMUNE_CLEAN'] + "|" + df_assurance['WILAYA_CLEAN']
df_coords['JOIN_KEY_REF'] = df_coords['COMMUNE_REF'] + "|" + df_coords['WILAYA_REF']

# 4. MATCHING ET FUSION
liste_ref = df_coords['JOIN_KEY_REF'].unique()
cles_assurance = df_assurance['JOIN_KEY'].unique()
mapping_final = {cle: cle for cle in cles_assurance if cle in liste_ref}

print("Fuzzy Matching pour les communes restantes...")
restant = [c for c in cles_assurance if c not in mapping_final and c != "VIDE|VIDE"]
for cle in restant:
    match, score = process.extractOne(cle, liste_ref, scorer=fuzz.token_sort_ratio)
    if score > 80:
        mapping_final[cle] = match

df_assurance['KEY_MATCHED'] = df_assurance['JOIN_KEY'].map(mapping_final)
df = pd.merge(
    df_assurance,
    df_coords[['JOIN_KEY_REF', 'center_lat', 'center_lon']],
    left_on='KEY_MATCHED', right_on='JOIN_KEY_REF', how='left'
)

# On filtre les données non localisées pour la simulation
df_simu = df.dropna(subset=['center_lat']).copy()
print(f"Bilan : {len(df_simu)} lignes prêtes pour Monte Carlo.")


# 5. SIMULATION DE MONTE CARLO
NB_SIMULATIONS = 10000 
PROBA_ANNUELLE = 0.08 
RAYON_IMPACT = 70      

poids_map = {'III': 100, 'IIb': 50, 'IIa': 20, 'I': 5, '0': 1}
df_simu['POIDS'] = df_simu['ZONE'].astype(str).map(poids_map).fillna(1)

def dist_haversine(lat1, lon1, lat2_vec, lon2_vec):
    lat1, lon1, lat2_vec, lon2_vec = map(np.radians, [lat1, lon1, lat2_vec, lon2_vec])
    a = np.sin((lat2_vec-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2_vec)*np.sin((lon2_vec-lon1)/2)**2
    return 6371 * (2 * np.arcsin(np.sqrt(a)))

pertes = []
print(f"\nSimulation en cours pour {NB_SIMULATIONS} ans...")

# Pré-extraction des arrays numpy pour la vitesse
lats = df_simu['center_lat'].values
lons = df_simu['center_lon'].values
capitaux = df_simu['CAPITAL_ASSURE'].values
facteurs = df_simu['FACTEUR_A'].values

for _ in range(NB_SIMULATIONS):
    p_an = 0.0
    nb_events = poisson.rvs(PROBA_ANNUELLE)
    
    for _ in range(nb_events):
        idx_epi = np.random.choice(len(df_simu), p=df_simu['POIDS'] / df_simu['POIDS'].sum())
        lat_e, lon_e = lats[idx_epi], lons[idx_epi]
        
        # Calcul des distances
        d = dist_haversine(lat_e, lon_e, lats, lons)
        mask = d <= RAYON_IMPACT
        
        if np.any(mask):
            # Intensité avec atténuation
            accel = facteurs[mask] / (1 + (d[mask]/15)**2)
            mu = np.clip(accel, 0.01, 0.9)
            # Sévérité aléatoire (Beta)
            mdr = np.random.beta(mu * 8, (1 - mu) * 8)
            p_an += np.sum(capitaux[mask] * mdr)
            
    pertes.append(p_an)


# 6. RÉSULTATS
pertes = np.array(pertes)
aal = pertes.mean()
pml_99 = np.percentile(pertes, 99)
primes = df_simu['PRIME_NETTE'].sum()

print("\n" + "="*40)
print(" RÉSULTATS FINANCIERS")
print("="*40)
print(f"Perte Moyenne (AAL)  : {aal:,.2f} DZD")
print(f"Perte Max (PML 99%)  : {pml_99:,.2f} DZD")
print(f"Indice de Risque     : {(aal/primes)*100:.2f}%")
print("-" * 40)

plt.figure(figsize=(10, 5))
plt.hist(pertes[pertes > 0], bins=50, color='firebrick', alpha=0.7)
plt.axvline(pml_99, color='black', linestyle='--', label=f'PML 99% : {pml_99:,.0f}')
plt.title('Distribution des Pertes Catastrophiques (Algérie)')
plt.xlabel('Pertes (DZD)')
plt.ylabel('Fréquence (Années)')
plt.legend()
plt.show()

# Calcul du Capital Exposé par Wilaya (Phase II.B)
points_chauds = df_simu.groupby('WILAYA_CLEAN').agg({
    'CAPITAL_ASSURE': 'sum',
    'NUMERO_POLICE': 'count'
}).rename(columns={'CAPITAL_ASSURE': 'Cumul_Capitaux', 'NUMERO_POLICE': 'Nombre_Polices'})

points_chauds = points_chauds.sort_values(by='Cumul_Capitaux', ascending=False)
print("\n--- ANALYSE DES POINTS CHAUDS (CONCENTRATION) ---")
print(points_chauds.head(10))

# Export pour ton rapport
points_chauds.to_excel("tableau_bord_cumuls.xlsx")

# Analyse de la contribution au risque par Wilaya
# On calcule l'Espérance de perte (AAL local) par Wilaya
df_simu['AAL_Local'] = df_simu['CAPITAL_ASSURE'] * df_simu['FACTEUR_A'] * (df_simu['POIDS'] / df_simu['POIDS'].sum())

profil_wilaya = df_simu.groupby('WILAYA_CLEAN').agg({
    'AAL_Local': 'sum',
    'CAPITAL_ASSURE': 'sum'
}).sort_values(by='AAL_Local', ascending=False)

# On calcule le % de contribution au risque total
total_risk = profil_wilaya['AAL_Local'].sum()
profil_wilaya['Contribution_Risque_%'] = (profil_wilaya['AAL_Local'] / total_risk) * 100

print("\n--- PROFIL DE PERTE PAR WILAYA (TOP 10) ---")
print(profil_wilaya[['CAPITAL_ASSURE', 'Contribution_Risque_%']].head(10))

# Graphique du profil de risque
profil_wilaya['Contribution_Risque_%'].head(10).plot(kind='bar', color='darkorange', figsize=(10,5))
plt.title('Top 10 Wilayas contribuant au risque Sismique')
plt.ylabel('% de contribution à la perte totale')
plt.show()


