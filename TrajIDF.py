import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import xml.etree.ElementTree as ET
from shapely.geometry import Point
from geopy.geocoders import Nominatim
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from tqdm import tqdm
import warnings
import re
import folium
from folium import plugins
import time
from math import radians, sin, cos, sqrt, atan2
import os
import pickle

warnings.filterwarnings('ignore')

# ========================================
# CONFIG
# ========================================
print("=" * 60)
print("PROJET ML - PRÉDICTION DU TRAFIC EN ÎLE-DE-FRANCE")
print("=" * 60)

# params globaux
PARIS_DATA_API = "https://opendata.paris.fr/api/records/1.0/search/"
SYTADIN_BASE = 'https://www.sytadin.fr/diffusion'

# codes dept + bbox
DEPARTEMENTS_IDF = ['75', '77', '78', '91', '92', '93', '94', '95']
VIEWBOX_IDF = ((48.12, 1.45), (49.24, 3.56))

# villes principales (pas exhaustif)
COMMUNES_IDF = {
    'Paris', 'Boulogne-Billancourt', 'Saint-Denis', 'Argenteuil',
    'Montreuil', 'Nanterre', 'Vitry-sur-Seine', 'Créteil',
    'Versailles', 'Colombes', 'Asnières-sur-Seine', 'Aulnay-sous-Bois',
    'Rueil-Malmaison', 'Aubervilliers', 'Champigny-sur-Marne',
    'Saint-Maur-des-Fossés', 'Drancy', 'Issy-les-Moulineaux',
    'Levallois-Perret', 'Noisy-le-Grand', 'Antony', 'Neuilly-sur-Seine',
    'Clichy', 'Ivry-sur-Seine', 'Villejuif', 'Épinay-sur-Seine',
    'Pantin', 'Bondy', 'Fontenay-sous-Bois', 'Maisons-Alfort',
    'Clamart', 'Sartrouville', 'Évry-Courcouronnes', 'Meaux',
    'Sevran', 'Villepinte', 'Chelles', 'Cergy', 'Massy',
    'Vincennes', 'Suresnes', 'Montrouge', 'Cachan', 'Pontoise',
    'Meudon', 'Puteaux', 'Rosny-sous-Bois', 'Gennevilliers',
    'Saint-Cloud', 'Sceaux', 'Fontainebleau', 'Rambouillet',
    'Athis-Mons', 'Brunoy', 'Corbeil-Essonnes', 'Draveil',
    'Montgeron', 'Orsay', 'Ris-Orangis', 'Sainte-Geneviève-des-Bois',
    'Savigny-sur-Orge', 'Viry-Châtillon', 'Yerres', 'Corbeil-Essonnes'
}

# ========================================
# GEOCODAGE
# ========================================

def geocoder_adresse_idf(adresse):
    """Geocode une adresse avec restriction IDF"""
    geolocator = Nominatim(user_agent="traffic_predictor_idf_m2")
    
    def nettoyer_commune(nom):
        """normalize nom ville"""
        return nom.strip().replace('-', ' ').lower()
    
    def extraire_commune(adresse):
        """extrait ville de l'adresse si possible"""
        if ',' in adresse:
            parties = adresse.split(',')
            return parties[-1].strip()
        return None
    
    # tentative 1: avec viewbox
    try:
        location = geolocator.geocode(
            adresse,
            viewbox=VIEWBOX_IDF,
            bounded=True,
            country_codes=['fr'],
            timeout=10
        )
        
        if location:
            # check si vraiment en IDF
            lat, lon = location.latitude, location.longitude
            if (VIEWBOX_IDF[0][0] <= lat <= VIEWBOX_IDF[1][0] and 
                VIEWBOX_IDF[0][1] <= lon <= VIEWBOX_IDF[1][1]):
                print(f"✓ Adresse trouvée en IDF: {location.address}")
                return (lat, lon)
    except Exception as e:
        print(f"Erreur géocodage tentative 1: {e}")
    
    # tentative 2: check commune
    commune = extraire_commune(adresse)
    if commune:
        commune_norm = nettoyer_commune(commune)
        
        # verif si commune dans liste
        for commune_idf in COMMUNES_IDF:
            if nettoyer_commune(commune_idf) == commune_norm:
                # retry avec commune explicite
                adresse_complete = f"{adresse}, Île-de-France, France"
                try:
                    time.sleep(1)  # rate limit
                    location = geolocator.geocode(
                        adresse_complete,
                        viewbox=VIEWBOX_IDF,
                        bounded=True,
                        country_codes=['fr']
                    )
                    if location:
                        print(f"✓ Adresse trouvée avec commune IDF: {location.address}")
                        return (location.latitude, location.longitude)
                except:
                    pass
    
    # tentative 3: ajout IDF
    if "île-de-france" not in adresse.lower() and "ile de france" not in adresse.lower():
        adresse_idf = f"{adresse}, Île-de-France, France"
        try:
            time.sleep(1)
            location = geolocator.geocode(
                adresse_idf,
                viewbox=VIEWBOX_IDF,
                bounded=True,
                country_codes=['fr']
            )
            if location:
                lat, lon = location.latitude, location.longitude
                if (VIEWBOX_IDF[0][0] <= lat <= VIEWBOX_IDF[1][0] and 
                    VIEWBOX_IDF[0][1] <= lon <= VIEWBOX_IDF[1][1]):
                    print(f"✓ Adresse trouvée avec ajout IDF: {location.address}")
                    return (lat, lon)
        except:
            pass
    
    print(f" Adresse non trouvée en Île-de-France: {adresse}")
    print("   Vérifiez que l'adresse est bien située dans l'un des départements IDF:")
    print("   75 (Paris), 77 (Seine-et-Marne), 78 (Yvelines), 91 (Essonne),")
    print("   92 (Hauts-de-Seine), 93 (Seine-Saint-Denis), 94 (Val-de-Marne), 95 (Val-d'Oise)")
    return None

# ========================================
# CHARGEMENT DATA
# ========================================

def charger_comptages_routiers(nombre_lignes=10000, filtre_date=None):
    """recup data comptage depuis opendata paris"""
    print(f"\n[1/6] Chargement des données de comptage routier ({nombre_lignes} lignes)...")
    
    tous_les_enregistrements = []
    debut = 0
    
    while True:
        params = {
            "dataset": "comptages-routiers-permanents",
            "rows": min(nombre_lignes - len(tous_les_enregistrements), 1000),
            "start": debut
        }
        
        # filtre date si demandé
        if filtre_date:
            params["q"] = f"t_1h:{filtre_date}*"
        
        try:
            reponse = requests.get(PARIS_DATA_API, params=params)
            reponse.raise_for_status()
            donnees = reponse.json()
            enregistrements = donnees.get("records", [])
            
            if not enregistrements:
                break
                
            tous_les_enregistrements.extend(enregistrements)
            debut += len(enregistrements)
            
            if len(tous_les_enregistrements) >= nombre_lignes:
                break
                
        except Exception as e:
            if len(tous_les_enregistrements) > 0:
                print(f"   Arrêt du chargement à {len(tous_les_enregistrements)} enregistrements")
                break
            else:
                print(f"Erreur lors du chargement: {e}")
                break
    
    if not tous_les_enregistrements:
        print("  Aucune donnée chargée depuis l'API")
        return pd.DataFrame()
    
    # conversion df
    df = pd.json_normalize(tous_les_enregistrements)
    
    # rename colonnes
    colonnes_mapping = {
        "fields.t_1h": "datetime_str",
        "fields.iu_ac": "identifiant_arc",
        "fields.q": "debit",
        "fields.k": "occupation"
    }
    
    for old_col, new_col in colonnes_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # traitement dates
    if 'datetime_str' in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime_str"], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df["heure"] = df["datetime"].dt.hour
        df["jour_semaine"] = df["datetime"].dt.dayofweek
        df["date"] = df["datetime"].dt.date
        df["minute"] = df["datetime"].dt.minute
    
    # calcul vitesse
    if 'occupation' in df.columns and 'debit' in df.columns:
        df["occupation"] = pd.to_numeric(df["occupation"], errors='coerce')
        df["debit"] = pd.to_numeric(df["debit"], errors='coerce')
        
        # calcul vitesse adaptée contexte urbain
        def calculer_vitesse_realiste(row):
            if row["occupation"] > 0 and row["debit"] > 0:
                # formule empirique
                vitesse_base = (row["debit"] / row["occupation"]) * 1.8
                
                # ajustement rush hour
                if row["heure"] in [7,8,9,17,18,19] and row["jour_semaine"] < 5:
                    vitesse_base *= 0.7
                
                # bornes realistes
                return np.clip(vitesse_base, 5, 90)
            else:
                # defaut selon heure
                if row["heure"] in [7,8,9,17,18,19] and row["jour_semaine"] < 5:
                    return 25
                else:
                    return 45
        
        df["vitesse_kmh"] = df.apply(calculer_vitesse_realiste, axis=1)
    
    print(f"✓ {len(df)} enregistrements chargés depuis Paris Data")
    nombre_lignes = min(nombre_lignes, 10000)
    return df

def charger_referentiel_geographique():
    """charge ref geo des capteurs"""
    print("\n[1/6] Chargement du référentiel géographique...")
    
    params = {
        "dataset": "referentiel-comptages-routiers",
        "rows": 10000
    }
    
    try:
        reponse = requests.get(PARIS_DATA_API, params=params)
        reponse.raise_for_status()
        donnees = reponse.json()
        
        df_ref = pd.json_normalize(donnees.get("records", []))
        
        if 'fields.geo_shape.coordinates' in df_ref.columns:
            df_ref['coords'] = df_ref['fields.geo_shape.coordinates']
            df_ref['lat'] = df_ref['coords'].apply(
                lambda x: np.mean([pt[1] for pt in x]) if isinstance(x, list) and len(x) > 0 else None
            )
            df_ref['lon'] = df_ref['coords'].apply(
                lambda x: np.mean([pt[0] for pt in x]) if isinstance(x, list) and len(x) > 0 else None
            )
        
        if 'fields.iu_ac' in df_ref.columns:
            df_ref['identifiant_arc'] = df_ref['fields.iu_ac']
        
        print(f"✓ {len(df_ref)} arcs géographiques chargés")
        return df_ref[['identifiant_arc', 'lat', 'lon']].dropna()
        
    except Exception as e:
        print(f"  Erreur chargement référentiel: {e}")
        return pd.DataFrame()

def charger_donnees_sytadin_enrichies():
    # data temps reel sytadin
    print("\n[2/6] Chargement des données Sytadin enrichies...")
    
    try:
        # geometrie arcs
        url_arcs = f"{SYTADIN_BASE}/mifmid/modelisation/Arc.mif"
        gdf_arcs = gpd.read_file(url_arcs)
        print(f"  ✓ {len(gdf_arcs)} arcs Sytadin chargés")
        
        # temps parcours dyn
        url_temps = f"{SYTADIN_BASE}/xml/arcs_dyn.xml"
        response = requests.get(url_temps)
        response.raise_for_status()
        
        # parse xml
        root = ET.fromstring(response.content)
        temps_parcours = []
        
        for ArcDynamique in root.findall('.//ArcDynamique'):
            code_arc = ArcDynamique.get('ID_ARC')
            temps_sec = ArcDynamique.findtext('TPReference')
            vitesse = ArcDynamique.findtext('VitesseInstantanneeBridee')
            taux_occupation = ArcDynamique.findtext('TauxOccupation')
            etat_trafic = ArcDynamique.findtext('EtatTrafic')
            
            if code_arc and temps_sec:
                temps_parcours.append({
                    'ID_ARC': code_arc,
                    'temps_parcours_sec': float(temps_sec),
                    'vitesse_kmh': float(vitesse) if vitesse else None,
                    'taux_occupation': float(taux_occupation) if taux_occupation else None,
                    'etat_trafic': etat_trafic
                })
        
        df_temps = pd.DataFrame(temps_parcours)
        print(f"  ✓ {len(df_temps)} temps de parcours chargés")
        
        # merge geo
        gdf_arcs['ID_ARC'] = gdf_arcs['ID_ARC'].astype(str)
        gdf_sytadin = gdf_arcs.merge(df_temps, on='ID_ARC', how='left')
        
        return gdf_sytadin
        
    except Exception as e:
        print(f"  Erreur Sytadin: {e}")
        return None

# ========================================
# GRAPHE ROUTIER
# ========================================

def construire_graphe_routier(region="Île-de-France, France"):
    """build graph avec cache local"""
    
    # fichier cache
    cache_filename = "graphe_routier_idf.pkl"
    
    # check cache
    if os.path.exists(cache_filename):
        print(f"\n Chargement du graphe routier depuis le cache...")
        try:
            with open(cache_filename, 'rb') as f:
                cache_data = pickle.load(f)
                G = cache_data['graphe']
                nodes_gdf = cache_data['nodes']
                edges_gdf = cache_data['edges']
                
            print(f"✓ Graphe chargé depuis le cache: {len(nodes_gdf)} nœuds, {len(edges_gdf)} arêtes")
            print(f"  (Pour forcer le re-téléchargement, supprimez le fichier '{cache_filename}')")
            return G, nodes_gdf, edges_gdf
            
        except Exception as e:
            print(f"  Erreur lors du chargement du cache: {e}")
            print("  → Téléchargement d'un nouveau graphe...")
    
    # si pas cache, dl
    print(f"\n[3/6] Construction du graphe routier pour {region}...")
    print("   Cette opération peut prendre plusieurs minutes la première fois...")
    
    try:
        # dl graphe IDF
        G = ox.graph_from_place(region, network_type="drive", simplify=True)
        
        # convert gdfs
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        
        # add ids
        edges_gdf = edges_gdf.reset_index()
        edges_gdf['edge_id'] = edges_gdf.index
        
        print(f"✓ Graphe construit: {len(nodes_gdf)} nœuds, {len(edges_gdf)} arêtes")
        
        # save cache
        print("   Sauvegarde du graphe dans le cache...")
        cache_data = {
            'graphe': G,
            'nodes': nodes_gdf,
            'edges': edges_gdf,
            'date_creation': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(cache_filename, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"  ✓ Cache sauvegardé dans '{cache_filename}'")
        
        return G, nodes_gdf, edges_gdf
        
    except Exception as e:
        print(f"Erreur lors de la construction du graphe: {e}")
        print("Tentative avec une zone plus restreinte...")
        
        # fallback paris
        G = ox.graph_from_place("Paris, France", network_type="drive")
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        edges_gdf = edges_gdf.reset_index()
        edges_gdf['edge_id'] = edges_gdf.index
        
        print(f"✓ Graphe Paris construit: {len(nodes_gdf)} nœuds, {len(edges_gdf)} arêtes")
        
        # save fallback
        cache_data = {
            'graphe': G,
            'nodes': nodes_gdf,
            'edges': edges_gdf,
            'date_creation': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Paris seulement (fallback)'
        }
        
        with open("graphe_routier_paris.pkl", 'wb') as f:
            pickle.dump(cache_data, f)
        
        return G, nodes_gdf, edges_gdf

# ========================================
# NETTOYAGE DATA
# ========================================

def parse_lanes(x):
    """
    converti lanes en scalaire
    """
    # valeurs manquantes
    if x is None:
        return 1.0
    
    # gestion floats/nan
    if isinstance(x, float):
        if np.isnan(x):
            return 1.0
        else:
            return float(x)
    
    # int direct
    if isinstance(x, (int, np.integer)):
        return float(x)
    
    # uniformisation liste
    if isinstance(x, (list, tuple, np.ndarray)):
        tokens = [str(v) for v in x]
    else:
        tokens = re.split(r'[;,]', str(x))
    
    # extract nums
    nums = []
    for token in tokens:
        try:
            num = float(token.strip())
            nums.append(num)
        except (ValueError, AttributeError):
            continue
    
    # moyenne si plusieurs valeurs
    if nums:
        return float(np.mean(nums))
    else:
        return 1.0

def extraire_vitesse_max(x):
    """extrait vitesse max avec gestion arrays"""
    # arrays
    if isinstance(x, (list, np.ndarray)):
        if len(x) > 0:
            # premier elem valide
            for val in x:
                try:
                    return float(str(val).split()[0])
                except:
                    continue
        return 50
    
    # valeurs manquantes
    try:
        if pd.isna(x):
            return 50  # defaut ville
    except:
        # si pd.isna fail sur array
        return 50
    
    if isinstance(x, (int, float)):
        return float(x)
    
    # parse "50" ou "50 km/h"
    try:
        return float(str(x).split()[0])
    except:
        return 50

def detect_oneway(x):
    """detecte sens unique avec gestion arrays"""
    # si sequence, cherche True ou 'yes'
    if isinstance(x, (list, tuple, np.ndarray)):
        return int(any((val is True) or (str(val).lower() == 'yes') for val in x))
    # scalaire
    return int((x is True) or (str(x).lower() == 'yes'))

def to_bool_int(val):
    """bool to int quel que soit le type"""
    if isinstance(val, (list, tuple, np.ndarray)):
        return int(any((v is True) or (str(v).lower() == 'yes') for v in val))
    # scalaires/strings
    return int((val is True) or (str(val).lower() == 'yes'))

# ========================================
# MAPPING VITESSES
# ========================================

def mapper_vitesses_sur_graphe(donnees_vitesse, referentiel_geo, edges_gdf, donnees_sytadin=None):
    """associe vitesses aux edges"""
    print("\n[4/6] Association des vitesses au graphe...")
    
    # si ref geo et data
    if not referentiel_geo.empty and 'identifiant_arc' in donnees_vitesse.columns:
        # merge vitesse + coords
        donnees_avec_geo = donnees_vitesse.merge(
            referentiel_geo, 
            on='identifiant_arc', 
            how='left'
        )
        
        # filtre avec coords
        donnees_avec_geo = donnees_avec_geo.dropna(subset=['lat', 'lon'])
        
        if not donnees_avec_geo.empty:
            # gdf pour points
            gdf_comptages = gpd.GeoDataFrame(
                donnees_avec_geo,
                geometry=gpd.points_from_xy(donnees_avec_geo.lon, donnees_avec_geo.lat),
                crs="EPSG:4326"
            )
            
            # simplif edges pour join
            edges_simple = edges_gdf[['edge_id', 'geometry']].copy()
            
            # spatial join
            correspondances = gpd.sjoin_nearest(
                gdf_comptages, 
                edges_simple, 
                how='left', 
                max_distance=0.002
            )
            
            # agreg par edge/heure/jour
            vitesses_par_arete = (
                correspondances
                .groupby(['edge_id', 'heure', 'jour_semaine'])['vitesse_kmh']
                .agg(['mean', 'std', 'count'])
                .reset_index()
            )
            
            vitesses_par_arete.columns = ['edge_id', 'heure', 'jour_semaine', 
                                         'vitesse_moyenne', 'vitesse_std', 'nombre_obs']
            
            print(f"✓ Vitesses réelles mappées sur {vitesses_par_arete['edge_id'].nunique()} arêtes")
            
            # enrichir sytadin si dispo
            if donnees_sytadin is not None:
                vitesses_par_arete = enrichir_avec_sytadin(vitesses_par_arete, donnees_sytadin, edges_gdf)
            
            return vitesses_par_arete
    
    # sinon synthetique
    print("  Génération de données synthétiques réalistes...")
    return generer_donnees_synthetiques_realistes(edges_gdf)

def generer_donnees_synthetiques_realistes(edges_gdf):
    """genere data synthetiques selon type route"""
    n_edges_sample = min(2000, len(edges_gdf))
    edges_sample = edges_gdf.sample(n_edges_sample)
    
    vitesses_synthetiques = []
    
    for _, edge in edges_sample.iterrows():
        for heure in range(24):
            for jour in range(7):
                # vitesse base selon type
                highway_type = edge.get('highway', 'unclassified')
                
                if highway_type in ['motorway', 'motorway_link']:
                    vitesse_base = 90
                elif highway_type in ['trunk', 'trunk_link']:
                    vitesse_base = 70
                elif highway_type in ['primary', 'primary_link']:
                    vitesse_base = 50
                elif highway_type in ['secondary', 'secondary_link']:
                    vitesse_base = 40
                elif highway_type in ['tertiary', 'tertiary_link']:
                    vitesse_base = 35
                elif highway_type in ['residential']:
                    vitesse_base = 30
                else:
                    vitesse_base = 25
                
                # facteur heure pointe
                if jour < 5:  # semaine
                    if heure in [7, 8, 9]:  # matin
                        facteur = 0.4 + 0.2 * np.random.random()
                    elif heure in [17, 18, 19]:  # soir
                        facteur = 0.35 + 0.15 * np.random.random()
                    elif heure in [12, 13]:  # midi
                        facteur = 0.6 + 0.1 * np.random.random()
                    elif heure in [0, 1, 2, 3, 4, 5]:  # nuit
                        facteur = 0.9 + 0.1 * np.random.random()
                    else:
                        facteur = 0.7 + 0.2 * np.random.random()
                else:  # weekend
                    if heure in [11, 12, 13, 14, 15, 16]:
                        facteur = 0.6 + 0.2 * np.random.random()
                    else:
                        facteur = 0.8 + 0.15 * np.random.random()
                
                # calcul vitesse avec variation
                vitesse = vitesse_base * facteur
                vitesse += np.random.normal(0, 3)
                vitesse = max(5, min(vitesse, vitesse_base * 1.1))
                
                vitesses_synthetiques.append({
                    'edge_id': edge['edge_id'],
                    'heure': heure,
                    'jour_semaine': jour,
                    'vitesse_kmh': vitesse
                })
    
    df_synthetique = pd.DataFrame(vitesses_synthetiques)
    
    # agreg
    vitesses_par_arete = (
        df_synthetique
        .groupby(['edge_id', 'heure', 'jour_semaine'])['vitesse_kmh']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    
    vitesses_par_arete.columns = ['edge_id', 'heure', 'jour_semaine', 
                                 'vitesse_moyenne', 'vitesse_std', 'nombre_obs']
    
    vitesses_par_arete['vitesse_std'] = vitesses_par_arete['vitesse_std'].fillna(5)
    vitesses_par_arete['nombre_obs'] = vitesses_par_arete['nombre_obs'].fillna(10)
    
    return vitesses_par_arete

def enrichir_avec_sytadin(vitesses_par_arete, donnees_sytadin, edges_gdf):
    """enrichi avec info sytadin"""
    print("  → Enrichissement avec données Sytadin...")
    
    # convert sytadin en points pour match
    sytadin_points = donnees_sytadin.copy()
    sytadin_points['geometry'] = sytadin_points.geometry.centroid
    
    # join avec edges
    edges_simple = edges_gdf[['edge_id', 'geometry']].copy()
    
    correspondances_sytadin = gpd.sjoin_nearest(
        sytadin_points,
        edges_simple,
        how='left',
        max_distance=0.001
    )
    
    # appliquer facteurs correction
    for _, row in correspondances_sytadin.iterrows():
        if pd.notna(row['vitesse_kmh']) and pd.notna(row['edge_id']):
            # ratio sytadin aux vitesses histo
            mask = vitesses_par_arete['edge_id'] == row['edge_id']
            if mask.any():
                facteur_correction = 0.8  # poids sytadin
                vitesses_par_arete.loc[mask, 'vitesse_moyenne'] = (
                    vitesses_par_arete.loc[mask, 'vitesse_moyenne'] * (1 - facteur_correction) +
                    row['vitesse_kmh'] * facteur_correction
                )
    
    print(f"  ✓ {len(correspondances_sytadin)} correspondances Sytadin appliquées")
    return vitesses_par_arete

# ========================================
# FEATURES
# ========================================

def creer_features_temporelles(df):
    """ajout feat temporelles"""
    # basic
    df['heure_pointe_matin'] = ((df['heure'] >= 7) & (df['heure'] <= 9)).astype(int)
    df['heure_pointe_soir'] = ((df['heure'] >= 17) & (df['heure'] <= 19)).astype(int)
    df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
    
    # cycliques
    df['heure_sin'] = np.sin(2 * np.pi * df['heure'] / 24)
    df['heure_cos'] = np.cos(2 * np.pi * df['heure'] / 24)
    df['jour_sin'] = np.sin(2 * np.pi * df['jour_semaine'] / 7)
    df['jour_cos'] = np.cos(2 * np.pi * df['jour_semaine'] / 7)
    
    # avancees
    if 'minute' in df.columns:
        df['minute_jour'] = df['heure'] * 60 + df['minute']
    else:
        df['minute_jour'] = df['heure'] * 60
    
    # periodes journee
    df['periode_nuit'] = ((df['heure'] >= 0) & (df['heure'] < 6)).astype(int)
    df['periode_matin'] = ((df['heure'] >= 6) & (df['heure'] < 12)).astype(int)
    df['periode_apres_midi'] = ((df['heure'] >= 12) & (df['heure'] < 18)).astype(int)
    df['periode_soir'] = ((df['heure'] >= 18) & (df['heure'] < 24)).astype(int)
    
    return df

def creer_features_route(edges_gdf):
    """feat pour chaque type route"""
    features = pd.DataFrame()
    features['edge_id'] = edges_gdf['edge_id']
    features['longueur'] = edges_gdf['length']
    
    # clean lanes
    features['nombre_voies'] = edges_gdf['lanes'].apply(parse_lanes)
    
    # types route granulaire
    features['route_autoroute'] = edges_gdf['highway'].isin(['motorway', 'motorway_link']).astype(int)
    features['route_principale'] = edges_gdf['highway'].isin(['primary', 'trunk']).astype(int)
    features['route_secondaire'] = edges_gdf['highway'].isin(['secondary', 'tertiary']).astype(int)
    features['route_residentielle'] = edges_gdf['highway'].isin(['residential']).astype(int)
    features['route_service'] = edges_gdf['highway'].isin(['service', 'living_street']).astype(int)
    
    # vitesse max
    features['vitesse_max'] = edges_gdf['maxspeed'].apply(extraire_vitesse_max)
    
    # feat geometriques
    features['est_sens_unique'] = edges_gdf['oneway'].apply(detect_oneway)
    features['a_pont'] = edges_gdf['bridge'].apply(to_bool_int)
    features['a_tunnel'] = edges_gdf['tunnel'].apply(to_bool_int)
    
    return features

# ========================================
# TRAINING
# ========================================

def entrainer_modele_ameliore(vitesses_par_arete, features_routes):
    """entrainement ensemble optimise"""
    print("\n[5/6] Entraînement du modèle de prédiction amélioré...")
    
    # merge data
    donnees = vitesses_par_arete.merge(features_routes, on='edge_id', how='inner')
    
    # feat temporelles
    donnees = creer_features_temporelles(donnees)
    
    # interactions
    donnees['heure_x_type_route'] = donnees['heure'] * donnees['route_principale']
    donnees['weekend_x_longueur'] = donnees['est_weekend'] * donnees['longueur']
    donnees['voies_x_vitesse_max'] = donnees['nombre_voies'] * donnees['vitesse_max']
    
    # colonnes features
    colonnes_features = [
        'longueur', 'nombre_voies', 'vitesse_max',
        'route_autoroute', 'route_principale', 'route_secondaire', 
        'route_residentielle', 'route_service',
        'heure_pointe_matin', 'heure_pointe_soir', 'est_weekend',
        'heure_sin', 'heure_cos', 'jour_sin', 'jour_cos',
        'minute_jour', 'periode_nuit', 'periode_matin', 
        'periode_apres_midi', 'periode_soir',
        'est_sens_unique', 'a_pont', 'a_tunnel',
        'heure_x_type_route', 'weekend_x_longueur', 'voies_x_vitesse_max'
    ]
    
    # filtrer colonnes existantes
    colonnes_features = [col for col in colonnes_features if col in donnees.columns]
    
    X = donnees[colonnes_features]
    y = donnees['vitesse_moyenne']
    
    # split stratifie
    X['stratify_key'] = donnees['heure'].astype(str) + '_' + donnees['jour_semaine'].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X['stratify_key']
    )
    
    # retirer cle stratif
    X_train = X_train.drop('stratify_key', axis=1)
    X_test = X_test.drop('stratify_key', axis=1)
    
    # normalisation
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # MODELE ENSEMBLE
    print("  → Construction du modèle ensemble...")
    
    # xgb optimise
    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    
    # gradient boost
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.7,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    # random forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    # ensemble pondere
    ensemble = VotingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('gb', gb_model),
            ('rf', rf_model)
        ],
        weights=[0.5, 0.3, 0.2]
    )
    
    # train
    print("  → Entraînement en cours...")
    ensemble.fit(X_train_norm, y_train)
    
    # eval
    predictions = ensemble.predict(X_test_norm)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"\n Performance du modèle ensemble:")
    print(f"  MAE: {mae:.2f} km/h")
    print(f"  RMSE: {rmse:.2f} km/h")
    print(f"  R²: {r2:.3f}")
    
    # CALIBRATION
    print("\n  → Calibration du modèle...")
    calibrateur = IsotonicRegression(out_of_bounds='clip')
    calibrateur.fit(predictions, y_test)
    
    # test calib
    predictions_cal = calibrateur.transform(predictions)
    mae_cal = mean_absolute_error(y_test, predictions_cal)
    print(f"  MAE après calibration: {mae_cal:.2f} km/h")
    
    # fonction pred calibree
    def predire_calibre(X_norm, heure=None):
        pred = ensemble.predict(X_norm)
        pred_cal = calibrateur.transform(pred)
        
        # ajust heure pointe
        if heure is not None:
            if heure in [7,8,9,17,18,19]:
                pred_cal *= 0.85
        
        return pred_cal
    
    # analyse feat importantes
    print("\n Features les plus importantes:")
    importances = []
    for name, model in ensemble.estimators:
        if hasattr(model, 'feature_importances_'):
            importances.append(model.feature_importances_)
    
    if importances:
        mean_importances = np.mean(importances, axis=0)
        feature_importance = pd.DataFrame({
            'feature': colonnes_features,
            'importance': mean_importances
        }).sort_values('importance', ascending=False)
        
        for i, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return {
        'modele': ensemble,
        'scaler': scaler,
        'calibrateur': calibrateur,
        'predire_calibre': predire_calibre,
        'colonnes': colonnes_features,
        'performance': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_calibre': mae_cal
        }
    }

# ========================================
# GRAPHE TEMPOREL
# ========================================

class CacheGrapheTemporelAvance:
    """cache avance avec feat calculees"""
    
    def __init__(self):
        self.cache_features_statiques = None
        self.cache_features_routes = {}  # par type route
        self.derniere_extraction = None
    
    def get_features_statiques(self, G):
        """recup ou calcul feat statiques"""
    # si cache existe et graphe pas change
        if self.cache_features_statiques is not None and len(G.edges()) == self.derniere_extraction:
            print("  → Utilisation du cache des features statiques ✓")
            return self.cache_features_statiques.copy()
    
        print("  → Mise en cache des features statiques...")
    
    # extraction une fois
        edges_data = []
        for u, v, data in G.edges(data=True):
            edges_data.append({
                'u': u,
                'v': v,
                'key': data.get('key', 0),
                'length': data['length'],
                'highway': data.get('highway', 'unclassified'),
                'lanes': parse_lanes(data.get('lanes', 1)),
                'maxspeed': extraire_vitesse_max(data.get('maxspeed', 50)),
                'oneway': to_bool_int(data.get('oneway', False)),
                'bridge': to_bool_int(data.get('bridge', 'no')),
                'tunnel': to_bool_int(data.get('tunnel', 'no'))
            })
    
        self.cache_features_statiques = pd.DataFrame(edges_data)
        self.derniere_extraction = len(G.edges())
    
        return self.cache_features_statiques.copy()
    
    def _calculer_features_temporelles(self, df_edges, heure, jour_semaine):
        """calc feat temp depuis statiques"""
        # creation feat
        df_features = pd.DataFrame({
            'longueur': df_edges['length'],
            'nombre_voies': df_edges['lanes'],
            'vitesse_max': df_edges['maxspeed'],
            'route_autoroute': df_edges['highway'].isin(['motorway', 'motorway_link']).astype(int),
            'route_principale': df_edges['highway'].isin(['primary', 'trunk']).astype(int),
            'route_secondaire': df_edges['highway'].isin(['secondary', 'tertiary']).astype(int),
            'route_residentielle': (df_edges['highway'] == 'residential').astype(int),
            'route_service': df_edges['highway'].isin(['service', 'living_street']).astype(int),
            'heure': heure,
            'jour_semaine': jour_semaine,
            'minute': 0,
            'est_sens_unique': df_edges['oneway'].apply(to_bool_int),
            'a_pont': df_edges['bridge'].apply(to_bool_int),
            'a_tunnel': df_edges['tunnel'].apply(to_bool_int)
        })
        
        # ajout feat temp
        df_features = creer_features_temporelles(df_features)
        
        # interactions
        df_features['heure_x_type_route'] = df_features['heure'] * df_features['route_principale']
        df_features['weekend_x_longueur'] = df_features['est_weekend'] * df_features['longueur']
        df_features['voies_x_vitesse_max'] = df_features['nombre_voies'] * df_features['vitesse_max']
        
        return df_features
        
    def get_features_completes(self, G, heure, jour_semaine):
        """recup feat completes avec cache"""
        
        # cle cache pour heure/jour
        cache_key = (heure, jour_semaine)
        
        # si deja calcule
        if cache_key in self.cache_features_routes:
            print(f"  → Cache hit pour {heure}h jour {jour_semaine} ✓")
            return self.cache_features_routes[cache_key].copy(), self.cache_features_statiques.copy()
        
        # sinon calcul
        df_edges = self.get_features_statiques(G)
        
        # feat temp une fois
        df_features = self._calculer_features_temporelles(df_edges, heure, jour_semaine)
        
        # cache (garder 10 dernieres heures)
        self.cache_features_routes[cache_key] = df_features
        if len(self.cache_features_routes) > 10:
            # suppr plus ancienne
            oldest_key = list(self.cache_features_routes.keys())[0]
            del self.cache_features_routes[oldest_key]
        
        return df_features, df_edges

# instance globale cache
cache_graphe_avance = CacheGrapheTemporelAvance()

def construire_graphe_temporel_ameliore(G, modele_dict, features_routes, heure, jour_semaine):
    """version optimisee vectorisee"""
    import time
    start_time = time.time()
    
    print(f"\n  Calcul vectorisé des temps de trajet pour {heure}h (jour {jour_semaine})...")
    
    modele = modele_dict['modele']
    scaler = modele_dict['scaler']
    calibrateur = modele_dict['calibrateur']
    colonnes = modele_dict['colonnes']
    
    # 1. extraction data
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'u': u,
            'v': v,
            'length': data['length'],
            'highway': data.get('highway', 'unclassified'),
            'lanes': data.get('lanes', 1),
            'maxspeed': data.get('maxspeed', 50),
            'oneway': data.get('oneway', False),
            'bridge': data.get('bridge', 'no'),
            'tunnel': data.get('tunnel', 'no')
        })
    
    df_edges = pd.DataFrame(edges_data)
    n_edges = len(df_edges)
    print(f"  → Traitement vectorisé de {n_edges} arêtes...")
    
    # 2. creation feat vectorisee
    df_features = pd.DataFrame({
        'longueur': df_edges['length'],
        'nombre_voies': df_edges['lanes'].apply(parse_lanes),
        'vitesse_max': df_edges['maxspeed'].apply(extraire_vitesse_max),
        'route_autoroute': df_edges['highway'].isin(['motorway', 'motorway_link']).astype(int),
        'route_principale': df_edges['highway'].isin(['primary', 'trunk']).astype(int),
        'route_secondaire': df_edges['highway'].isin(['secondary', 'tertiary']).astype(int),
        'route_residentielle': (df_edges['highway'] == 'residential').astype(int),
        'route_service': df_edges['highway'].isin(['service', 'living_street']).astype(int),
        'heure': heure,
        'jour_semaine': jour_semaine,
        'minute': 0,
        'est_sens_unique': df_edges['oneway'].apply(to_bool_int),
        'a_pont': df_edges['bridge'].apply(to_bool_int),
        'a_tunnel': df_edges['tunnel'].apply(to_bool_int)
    })
    
    # 3. feat temporelles
    df_features = creer_features_temporelles(df_features)
    
    # 4. interactions vectorisees
    df_features['heure_x_type_route'] = df_features['heure'] * df_features['route_principale']
    df_features['weekend_x_longueur'] = df_features['est_weekend'] * df_features['longueur']
    df_features['voies_x_vitesse_max'] = df_features['nombre_voies'] * df_features['vitesse_max']
    
    # 5. prep matrice sklearn
    X = df_features[colonnes].values
    
    # 6. prediction vectorisee
    print("  → Prédiction vectorisée...")
    X_norm = scaler.transform(X)
    vitesses_predites = modele.predict(X_norm)
    
    # calib vectorisee
    vitesses_calibrees = calibrateur.transform(vitesses_predites)
    if heure in [7, 8, 9, 17, 18, 19]:
        vitesses_calibrees *= 0.85
    
    vitesses_calibrees = np.maximum(vitesses_calibrees, 5)
    
    # 7. calcul temps vectorise
    temps_minutes = (df_edges['length'].values / 1000) / vitesses_calibrees * 60
    
    # 8. construction graphe batch
    print("  → Construction du graphe...")
    graphe_temps = nx.DiGraph()
    
    edges_list = [
        (df_edges.iloc[i]['u'], df_edges.iloc[i]['v'], {
            'temps': temps_minutes[i],
            'longueur': df_edges.iloc[i]['length'],
            'vitesse': vitesses_calibrees[i],
            'type_route': df_edges.iloc[i]['highway']
        })
        for i in range(n_edges)
    ]
    
    graphe_temps.add_edges_from(edges_list)  # batch add
    
    elapsed = time.time() - start_time
    print(f"✓ Graphe temporel construit en {elapsed:.2f} secondes")
    
    return graphe_temps

# ========================================
# PREDICTION TRAJET
# ========================================

def predire_trajet_ameliore(adresse_depart, adresse_destination, heure_depart, 
                           G, modele_dict, features_routes, nodes_gdf):
    """pred temps trajet entre 2 adresses"""
    
    print("\n" + "="*60)
    print(" PRÉDICTION DE TRAJET AMÉLIORÉE")
    print("="*60)
    
    # geocode adresses IDF
    print("\n Géocodage des adresses...")
    coords_depart = geocoder_adresse_idf(adresse_depart)
    coords_destination = geocoder_adresse_idf(adresse_destination)
    
    if not coords_depart or not coords_destination:
        return None
    
    print(f"  Départ: {adresse_depart}")
    print(f"  → Coordonnées: {coords_depart[0]:.4f}, {coords_depart[1]:.4f}")
    print(f"  Destination: {adresse_destination}")
    print(f"  → Coordonnées: {coords_destination[0]:.4f}, {coords_destination[1]:.4f}")
    
    # distance vol oiseau
    distance_directe = calcul_distance_haversine(
        coords_depart[0], coords_depart[1],
        coords_destination[0], coords_destination[1]
    )
    print(f"  Distance à vol d'oiseau: {distance_directe:.1f} km")
    
    # noeuds proches
    noeud_depart = ox.nearest_nodes(G, coords_depart[1], coords_depart[0])
    noeud_destination = ox.nearest_nodes(G, coords_destination[1], coords_destination[0])
    
    # extract jour/heure
    jour_semaine = heure_depart.weekday()
    heure = heure_depart.hour
    
    # graphe avec temps
    graphe_temps = construire_graphe_temporel_ameliore(
        G, modele_dict, features_routes, heure, jour_semaine
    )
    
    try:
        # chemin rapide
        chemin_rapide = nx.shortest_path(
            graphe_temps, noeud_depart, noeud_destination, weight='temps'
        )
        
        # stats detaillees
        temps_total = 0
        distance_totale = 0
        vitesses = []
        segments = []
        
        for i, (u, v) in enumerate(zip(chemin_rapide[:-1], chemin_rapide[1:])):
            edge = graphe_temps[u][v]
            temps_segment = edge['temps']
            longueur_segment = edge['longueur']
            vitesse_segment = edge['vitesse']
            
            temps_total += temps_segment
            distance_totale += longueur_segment
            vitesses.append(vitesse_segment)
            
            segments.append({
                'numero': i + 1,
                'temps_min': temps_segment,
                'distance_m': longueur_segment,
                'vitesse_kmh': vitesse_segment,
                'type_route': edge.get('type_route', 'unknown')
            })
        
        # marge parking/imprevus
        temps_parking = np.random.uniform(2, 5)
        temps_total += temps_parking
        
        # resultats
        print(f"\nRÉSULTATS DE LA PRÉDICTION:")
        print(f"Heure de départ: {heure_depart.strftime('%A %d/%m/%Y à %H:%M')}")
        print(f"Temps estimé: {temps_total:.0f} minutes (dont {temps_parking:.0f} min parking)")
        print(f"Distance: {distance_totale/1000:.1f} km")
        print(f"Vitesse moyenne: {np.mean(vitesses):.0f} km/h")
        print(f"Nombre d'intersections: {len(chemin_rapide)-1}")
        print(f"Ratio distance réelle/directe: {(distance_totale/1000)/distance_directe:.2f}")
        
        # analyse trafic
        if heure in [7,8,9,17,18,19] and jour_semaine < 5:
            print(f"\n  HEURE DE POINTE détectée")
            print(f"  → Temps en conditions normales estimé: ~{temps_total*0.6:.0f} minutes")
            print(f"  → Ralentissement: +{((temps_total/(temps_total*0.6))-1)*100:.0f}%")
        
        # top 3 segments lents
        segments_lents = sorted(segments, key=lambda x: x['vitesse_kmh'])[:3]
        print(f"\n Segments les plus lents:")
        for seg in segments_lents:
            print(f"  - Segment {seg['numero']}: {seg['vitesse_kmh']:.0f} km/h ({seg['type_route']})")
        
        return {
            'temps_minutes': temps_total,
            'temps_sans_parking': temps_total - temps_parking,
            'distance_km': distance_totale / 1000,
            'distance_directe_km': distance_directe,
            'vitesse_moyenne': np.mean(vitesses),
            'vitesse_min': min(vitesses),
            'vitesse_max': max(vitesses),
            'chemin': chemin_rapide,
            'segments': segments,
            'heure_depart': heure_depart,
            'heure_arrivee': heure_depart + timedelta(minutes=temps_total)
        }
        
    except nx.NetworkXNoPath:
        print("\n Aucun itinéraire trouvé entre ces deux points")
        print("   Vérifiez que les deux adresses sont bien en Île-de-France")
        return None
    except Exception as e:
        print(f"\n Erreur lors du calcul: {e}")
        return None

# ========================================
# CARTE INTERACTIVE
# ========================================
def generer_itineraire_interactif(resultat_prediction, G, nodes_gdf):
    """
    genere carte interactive avec itineraire
    """
    
    if not resultat_prediction or 'chemin' not in resultat_prediction:
        print(" Pas d'itinéraire à afficher")
        return None
    
    chemin_nodes = resultat_prediction['chemin']
    
    # 1. EXTRACTION COORDS ET ETAPES
    def extraire_etapes_itineraire(chemin_nodes, G, nodes_gdf):
        """extrait coords et noms rues"""
        
        etapes = []
        coordonnees = []
        
        for i, node in enumerate(chemin_nodes):
            # coords noeud
            node_data = nodes_gdf.loc[node]
            lat, lon = node_data['y'], node_data['x']
            coordonnees.append([lat, lon])
            
            # nom rue si dispo
            if i < len(chemin_nodes) - 1:
                next_node = chemin_nodes[i + 1]
                edge_data = G.get_edge_data(node, next_node)
                
                if edge_data and 'name' in edge_data[0]:
                    nom_rue = edge_data[0]['name']
                    if isinstance(nom_rue, list):
                        nom_rue = nom_rue[0]
                    
                    # ajout etape si nouvelle rue
                    if not etapes or etapes[-1]['rue'] != nom_rue:
                        etapes.append({
                            'numero': len(etapes) + 1,
                            'rue': nom_rue,
                            'lat': lat,
                            'lon': lon,
                            'type': 'Continuer sur' if etapes else 'Départ',
                            'distance_cumul': 0
                        })
        
        # ajout destination
        if coordonnees:
            etapes.append({
                'numero': len(etapes) + 1,
                'rue': 'Destination',
                'lat': coordonnees[-1][0],
                'lon': coordonnees[-1][1],
                'type': 'Arrivée',
                'distance_cumul': resultat_prediction['distance_km']
            })
        
        return coordonnees, etapes
    
    # 2. CREATION CARTE FOLIUM
    def creer_carte_itineraire(coordonnees, etapes, resultat):
        """cree carte folium"""
        
        # centrage
        lat_centre = np.mean([c[0] for c in coordonnees])
        lon_centre = np.mean([c[1] for c in coordonnees])
        
        # init carte
        carte = folium.Map(
            location=[lat_centre, lon_centre],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # trace itineraire
        folium.PolyLine(
            coordonnees,
            color='blue',
            weight=5,
            opacity=0.8,
            popup=f"Trajet: {resultat['temps_minutes']:.0f} min, {resultat['distance_km']:.1f} km"

            ).add_to(carte)
        
        # Marqueur de départ
        folium.Marker(
            coordonnees[0],
            popup=folium.Popup(
                f"<b>Départ</b><br>{etapes[0]['rue'] if etapes else 'Départ'}",
                max_width=200
            ),
            icon=folium.Icon(color='green', icon='play')
        ).add_to(carte)
        
        # Marqueur d'arrivée
        folium.Marker(
            coordonnees[-1],
            popup=folium.Popup(
                f"<b>Arrivée</b><br>Temps: {resultat['temps_minutes']:.0f} min",
                max_width=200
            ),
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(carte)
        
        # Ajouter les étapes principales
        for i, etape in enumerate(etapes[1:-1], 1):
            if i % 3 == 0:  # Afficher 1 étape sur 3 pour ne pas surcharger
                folium.CircleMarker(
                    [etape['lat'], etape['lon']],
                    radius=5,
                    popup=f"Étape {etape['numero']}: {etape['rue']}",
                    color='blue',
                    fill=True
                ).add_to(carte)
        
        # Ajouter panneau de contrôle avec infos
        info_html = f"""
        <div style='position: fixed; 
                    top: 10px; right: 10px; 
                    width: 300px; 
                    background-color: white; 
                    border: 2px solid grey; 
                    z-index: 9999; 
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;'>
            <h4>Résumé du trajet</h4>
            <p><b>Temps estimé:</b> {resultat['temps_minutes']:.0f} minutes</p>
            <p><b>Distance:</b> {resultat['distance_km']:.1f} km</p>
            <p><b>Vitesse moyenne:</b> {resultat['vitesse_moyenne']:.0f} km/h</p>
            <p><b>Nombre d'étapes:</b> {len(etapes)}</p>
        </div>
        """
        carte.get_root().html.add_child(folium.Element(info_html))
        
        # Plugin pour mesurer les distances
        plugins.MeasureControl().add_to(carte)
        
        # Plugin fullscreen
        plugins.Fullscreen().add_to(carte)
        
        # Ajouter une mini-carte
        minimap = plugins.MiniMap(toggle_display=True)
        carte.add_child(minimap)
        
        return carte
    
    # 3. generation feuille de route
    def generer_feuille_route(etapes, resultat):
        """cree feuille de route detaillee"""
        
        feuille_route = []
        feuille_route.append("=" * 60)
        feuille_route.append(" FEUILLE DE ROUTE")
        feuille_route.append("=" * 60)
        feuille_route.append(f"Temps total: {resultat['temps_minutes']:.0f} minutes")
        feuille_route.append(f"Distance: {resultat['distance_km']:.1f} km")
        feuille_route.append(f"Vitesse moyenne: {resultat['vitesse_moyenne']:.0f} km/h")
        feuille_route.append("-" * 60)
        
        distance_cumul = 0
        for i, etape in enumerate(etapes):
            if i == 0:
                feuille_route.append(f" DÉPART: {etape['rue']}")
            elif i == len(etapes) - 1:
                feuille_route.append(f" ARRIVÉE: Vous êtes arrivé à destination")
            else:
                # Calculer la distance depuis la dernière étape
                if i > 0:
                    dist_etape = calcul_distance_etape(
                        etapes[i-1]['lat'], etapes[i-1]['lon'],
                        etape['lat'], etape['lon']
                    )
                    distance_cumul += dist_etape
                    
                    # Déterminer le type de direction
                    direction = determiner_direction(etapes[i-1], etape)
                    
                    feuille_route.append(
                        f"{i}. {direction} {etape['rue']} "
                        f"(dans {dist_etape:.1f} km)"
                    )
        
        feuille_route.append("=" * 60)
        return "\n".join(feuille_route)
    
    # 4. export et partage
    def exporter_itineraire(carte, etapes, resultat):
        """export l'itineraire dans differents formats"""
        
        # save carte HTML
        carte.save('itineraire_idf.html')
        print("✓ Carte sauvegardée dans 'itineraire_idf.html'")
        
        # export en GPX pour GPS
        gpx_content = generer_gpx(etapes, resultat)
        with open('itineraire.gpx', 'w') as f:
            f.write(gpx_content)
        print("✓ Fichier GPX sauvegardé dans 'itineraire.gpx'")
    
    # 5. fonction principale affichage
    # extract data
    coordonnees, etapes = extraire_etapes_itineraire(chemin_nodes, G, nodes_gdf)
    
    # cree carte
    carte = creer_carte_itineraire(coordonnees, etapes, resultat_prediction)
    
    # genere feuille route
    feuille_route = generer_feuille_route(etapes, resultat_prediction)
    print(feuille_route)
    
    # export
    exporter_itineraire(carte, etapes, resultat_prediction)
    
    # affiche carte dans jupyter (si applicable)
    try:
        from IPython.display import display
        display(carte)
    except:
        print("\n Ouvrez 'itineraire_idf.html' dans votre navigateur pour voir la carte")
    
    return {
        'carte': carte,
        'etapes': etapes,
        'feuille_route': feuille_route
    }

# fonctions utilitaires supplementaires
def calcul_distance_haversine(lat1, lon1, lat2, lon2):
    """
    calcul distance vol d'oiseau entre 2 points (km)
    avec formule haversine
    """
    R = 6371  # rayon terre km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def calcul_distance_etape(lat1, lon1, lat2, lon2):
    """calc distance entre 2 points GPS"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # rayon terre km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def determiner_direction(etape_avant, etape_apres):
    """determine direction a prendre"""
    # calc simplifie angle
    dx = etape_apres['lon'] - etape_avant['lon']
    dy = etape_apres['lat'] - etape_avant['lat']
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    if -45 <= angle < 45:
        return "↗ Tourner à droite sur"
    elif 45 <= angle < 135:
        return "↑ Continuer tout droit sur"
    elif angle >= 135 or angle < -135:
        return "↖ Tourner à gauche sur"
    else:
        return "↓ Faire demi-tour sur"

def generer_gpx(etapes, resultat):
    """genere fichier GPX pour nav GPS"""
    gpx_template = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Traffic Predictor IDF">
  <metadata>
    <name>Trajet IDF</name>
    <desc>Temps: {temps} min, Distance: {distance} km</desc>
    <time>{timestamp}</time>
  </metadata>
  <rte>
    <name>Itinéraire optimisé</name>
    {waypoints}
  </rte>
</gpx>"""
    
    waypoints = []
    for etape in etapes:
        waypoints.append(f"""    <rtept lat="{etape['lat']}" lon="{etape['lon']}">
      <name>{etape['rue']}</name>
      <desc>{etape['type']}</desc>
    </rtept>""")
    
    from datetime import datetime
    
    return gpx_template.format(
        temps=int(resultat['temps_minutes']),
        distance=round(resultat['distance_km'], 1),
        timestamp=datetime.now().isoformat(),
        waypoints="\n".join(waypoints)
    )

# integration dans fonction principale de prediction
def predire_trajet_avec_carte(adresse_depart, adresse_destination, 
                             heure_depart, G, modele, scaler, 
                             features_routes, nodes_gdf):
    """version amelioree avec generation carte"""
    
    # prediction standard
    resultat = predire_trajet_ameliore(
        adresse_depart, adresse_destination, heure_depart,
        G, modele, scaler, features_routes,nodes_gdf
    )
    
    if resultat:
        # generer itineraire interactif
        itineraire = generer_itineraire_interactif(resultat, G, nodes_gdf)
        
        # ajouter au resultat
        resultat['itineraire'] = itineraire
        
        print("\n Itinéraire généré avec succès!")
        print("   → Carte interactive: itineraire.html")
        print("   → Fichier GPS: itineraire.gpx")
    
    return resultat

# ========================================
# FONCTIONS SAUVEGARDE ET CHARGEMENT
# ========================================

def sauvegarder_modele(modele_dict, nom_fichier='modele_trafic_idf.pkl'):
    """save modele et composants"""
    print(f"\n Sauvegarde du modèle dans '{nom_fichier}'...")
    
    # creer dict simplifie pour save
    modele_save = {
        'modele': modele_dict['modele'],
        'scaler': modele_dict['scaler'],
        'calibrateur': modele_dict['calibrateur'],
        'colonnes': modele_dict['colonnes'],
        'performance': modele_dict['performance'],
        'date_entrainement': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(modele_save, nom_fichier)
    print(f"✓ Modèle sauvegardé avec succès!")
    print(f"  Performance: MAE={modele_save['performance']['mae']:.2f} km/h")
    
def charger_modele(nom_fichier='modele_trafic_idf.pkl'):
    """charge modele pre-entraine"""
    print(f"\n Chargement du modèle depuis '{nom_fichier}'...")
    
    try:
        modele_save = joblib.load(nom_fichier)
        
        # recreer fonction pred calibree
        def predire_calibre(X_norm, heure=None):
            pred = modele_save['modele'].predict(X_norm)
            pred_cal = modele_save['calibrateur'].transform(pred)
            
            if heure is not None:
                if heure in [7,8,9,17,18,19]:
                    pred_cal *= 0.85
            
            return pred_cal
        
        modele_dict = {
            'modele': modele_save['modele'],
            'scaler': modele_save['scaler'],
            'calibrateur': modele_save['calibrateur'],
            'predire_calibre': predire_calibre,
            'colonnes': modele_save['colonnes'],
            'performance': modele_save['performance']
        }
        
        print(f"✓ Modèle chargé avec succès!")
        print(f"  Date d'entraînement: {modele_save['date_entrainement']}")
        print(f"  Performance: MAE={modele_save['performance']['mae']:.2f} km/h")
        
        return modele_dict
        
    except FileNotFoundError:
        print(f" Fichier '{nom_fichier}' introuvable")
        return None
    except Exception as e:
        print(f" Erreur lors du chargement: {e}")
        return None

# ========================================
# ANALYSEUR TRAFIC TEMPS REEL
# ========================================

class AnalyseurTraficTempsReel:
    """classe pour analyser trafic temps reel"""
    
    def __init__(self, modele_dict, G, features_routes):
        self.modele_dict = modele_dict
        self.G = G
        self.features_routes = features_routes
        self.historique_predictions = []
        
    def analyser_conditions_actuelles(self):
        """analyse conditions trafic actuelles"""
        maintenant = datetime.now()
        heure = maintenant.hour
        jour = maintenant.weekday()
        
        print(f"\n ANALYSE DU TRAFIC EN TEMPS RÉEL")
        print(f"   {maintenant.strftime('%A %d/%m/%Y - %H:%M')}")
        print("="*50)
        
        # construire graphe temporel actuel
        graphe_temps = construire_graphe_temporel_ameliore(
            self.G, self.modele_dict, self.features_routes, heure, jour
        )
        
        # analyser differents types routes
        stats_par_type = {}
        
        for u, v, data in graphe_temps.edges(data=True):
            type_route = data.get('type_route', 'unknown')
            vitesse = data.get('vitesse', 0)
            
            if type_route not in stats_par_type:
                stats_par_type[type_route] = []
            stats_par_type[type_route].append(vitesse)
        
        # afficher stats
        print("\n Vitesses moyennes par type de route:")
        for type_route, vitesses in sorted(stats_par_type.items()):
            if vitesses:
                vitesse_moy = np.mean(vitesses)
                vitesse_min = np.min(vitesses)
                vitesse_max = np.max(vitesses)
                
                # determiner etat trafic
                if type_route in ['motorway', 'trunk']:
                    if vitesse_moy < 50:
                        etat = " Très chargé"
                    elif vitesse_moy < 70:
                        etat = " Chargé"
                    else:
                        etat = " Fluide"
                else:
                    if vitesse_moy < 20:
                        etat = " Très chargé"
                    elif vitesse_moy < 35:
                        etat = " Chargé"
                    else:
                        etat = " Fluide"
                
                print(f"  {type_route:20s}: {vitesse_moy:5.1f} km/h [{vitesse_min:.0f}-{vitesse_max:.0f}] {etat}")
        
        # identifier zones congestionnees
        self.identifier_zones_congestionnees(graphe_temps)
        
        return graphe_temps
    
    def identifier_zones_congestionnees(self, graphe_temps, seuil_congestion=20):
        """identifie zones avec vitesse < seuil"""
        zones_lentes = []
        
        for u, v, data in graphe_temps.edges(data=True):
            if data['vitesse'] < seuil_congestion:
                zones_lentes.append({
                    'edge': (u, v),
                    'vitesse': data['vitesse'],
                    'type': data.get('type_route', 'unknown')
                })
        
        if zones_lentes:
            print(f"\n  {len(zones_lentes)} zones congestionnées détectées (< {seuil_congestion} km/h)")
            # afficher 5 pires
            zones_lentes.sort(key=lambda x: x['vitesse'])
            for i, zone in enumerate(zones_lentes[:5]):
                print(f"   {i+1}. {zone['type']:15s} - {zone['vitesse']:.1f} km/h")
        else:
            print("\n Aucune zone fortement congestionnée")
    
    def predire_evolution_trafic(self, heures_futures=3):
        """predit evolution trafic pour prochaines heures"""
        print(f"\n PRÉVISION DU TRAFIC ({heures_futures}h)")
        print("="*50)
        
        maintenant = datetime.now()
        previsions = []
        
        for h in range(heures_futures + 1):
            heure_future = maintenant + timedelta(hours=h)
            heure = heure_future.hour
            jour = heure_future.weekday()
            
            # calc vitesse moy globale
            graphe_temps = construire_graphe_temporel_ameliore(
                self.G, self.modele_dict, self.features_routes, heure, jour
            )
            
            vitesses = [data['vitesse'] for _, _, data in graphe_temps.edges(data=True)]
            vitesse_moy = np.mean(vitesses)
            
            previsions.append({
                'heure': heure_future,
                'vitesse_moyenne': vitesse_moy,
                'congestion': 100 - (vitesse_moy / 50 * 100)  # indice congestion
            })
            
            print(f"  {heure_future.strftime('%H:%M')} - "
                  f"Vitesse moy: {vitesse_moy:.1f} km/h - "
                  f"Congestion: {previsions[-1]['congestion']:.0f}%")
        
        # recommandations
        self.generer_recommandations(previsions)
        
        return previsions
    
    def generer_recommandations(self, previsions):
        """genere recommandations basees sur previsions"""
        print("\n RECOMMANDATIONS:")
        
        # trouver meilleure heure
        meilleure_heure = max(previsions, key=lambda x: x['vitesse_moyenne'])
        pire_heure = min(previsions, key=lambda x: x['vitesse_moyenne'])
        
        if meilleure_heure['heure'].hour != pire_heure['heure'].hour:
            print(f"  ✓ Privilégiez un départ vers {meilleure_heure['heure'].strftime('%H:%M')}")
            print(f"  ✗ Évitez de partir vers {pire_heure['heure'].strftime('%H:%M')}")
        
        # analyse tendance
        if len(previsions) > 1:
            tendance = previsions[-1]['congestion'] - previsions[0]['congestion']
            if tendance > 10:
                print("    Le trafic va se densifier - Partez dès que possible")
            elif tendance < -10:
                print("  ✓ Le trafic va s'améliorer - Vous pouvez attendre")
            else:
                print("  → Conditions stables prévues")

# ========================================
# INTERFACE UTILISATEUR INTERACTIVE
# ========================================

def interface_interactive(G, modele_dict, features_routes, nodes_gdf):
    """interface ligne de commande pour utiliser systeme"""
    
    print("\n" + "="*60)
    print(" SYSTÈME DE PRÉDICTION DE TRAFIC ÎLE-DE-FRANCE")
    print("="*60)
    
    analyseur = AnalyseurTraficTempsReel(modele_dict, G, features_routes)
    
    while True:
        print("\n MENU PRINCIPAL:")
        print("1. Prédire un trajet")
        print("2. Analyser le trafic actuel")
        print("3. Prévisions de trafic")
        print("4. Comparer plusieurs itinéraires")
        print("5. Historique des trajets")
        print("6. Quitter")
        
        choix = input("\nVotre choix (1-6): ")
        
        if choix == "1":
            # prediction trajet
            print("\n  PRÉDICTION DE TRAJET")
            print("-"*40)
            
            adresse_depart = input("Adresse de départ (en Île-de-France): ")
            adresse_destination = input("Adresse de destination (en Île-de-France): ")
            
            # choix heure
            print("\nQuand souhaitez-vous partir?")
            print("1. Maintenant")
            print("2. À une heure précise")
            
            choix_heure = input("Votre choix (1-2): ")
            
            if choix_heure == "1":
                heure_depart = datetime.now()
            else:
                heure_str = input("Heure de départ (format HH:MM): ")
                try:
                    h, m = map(int, heure_str.split(':'))
                    heure_depart = datetime.now().replace(hour=h, minute=m)
                except:
                    print(" Format incorrect, utilisation de l'heure actuelle")
                    heure_depart = datetime.now()
            
            # prediction avec carte
            resultat = predire_trajet_ameliore(
                adresse_depart, adresse_destination, heure_depart,
                G, modele_dict, features_routes, nodes_gdf
            )
            
            if resultat:
                # generer carte
                print("\n  Génération de la carte interactive...")
                generer_itineraire_interactif(resultat, G, nodes_gdf)
                
                # proposer alternatives
                print("\n Souhaitez-vous voir des itinéraires alternatifs? (o/n)")
                if input().lower() == 'o':
                    generer_alternatives(
                        adresse_depart, adresse_destination, heure_depart,
                        G, modele_dict, features_routes, nodes_gdf
                    )
        
        elif choix == "2":
            # analyse temps reel
            analyseur.analyser_conditions_actuelles()
        
        elif choix == "3":
            # previsions
            heures = int(input("\nPrévisions pour combien d'heures? (1-6): "))
            analyseur.predire_evolution_trafic(min(heures, 6))
        
        elif choix == "4":
            # comparaison itineraires
            comparer_itineraires_multiples(G, modele_dict, features_routes, nodes_gdf)
        
        elif choix == "5":
            # historique
            afficher_historique_trajets()
        
        elif choix == "6":
            print("\n Au revoir! Bonne route!")
            break
        
        else:
            print(" Choix invalide")

def generer_alternatives(adresse_depart, adresse_destination, heure_depart,
                        G, modele_dict, features_routes, nodes_gdf):
    """genere itineraires alternatifs"""
    print("\n🔄 RECHERCHE D'ITINÉRAIRES ALTERNATIFS...")
    
    alternatives = []
    
    # alternative 1: depart 30 min plus tot
    heure_tot = heure_depart - timedelta(minutes=30)
    res_tot = predire_trajet_ameliore(
        adresse_depart, adresse_destination, heure_tot,
        G, modele_dict, features_routes, nodes_gdf
    )
    if res_tot:
        alternatives.append(("Départ 30 min plus tôt", res_tot))
    
    # alternative 2: depart 30 min plus tard
    heure_tard = heure_depart + timedelta(minutes=30)
    res_tard = predire_trajet_ameliore(
        adresse_depart, adresse_destination, heure_tard,
        G, modele_dict, features_routes, nodes_gdf
    )
    if res_tard:
        alternatives.append(("Départ 30 min plus tard", res_tard))
    
    # afficher comparaison
    if alternatives:
        print("\n COMPARAISON DES ALTERNATIVES:")
        print("-"*60)
        print(f"{'Option':30s} {'Départ':10s} {'Durée':10s} {'Vitesse':10s}")
        print("-"*60)
        
        # Notez: resultat n'est pas défini ici, il faudrait le passer en paramètre
        # pour l'instant j'utilise les données de res_tot comme proxy
        print(f"{'Original':30s} {heure_depart.strftime('%H:%M'):10s} "
              f"{'--'} min{' ':5s} "
              f"{'--'} km/h")
        
        for nom, alt in alternatives:
            print(f"{nom:30s} {alt['heure_depart'].strftime('%H:%M'):10s} "
                  f"{alt['temps_minutes']:.0f} min{' ':5s} "
                  f"{alt['vitesse_moyenne']:.0f} km/h")

def comparer_itineraires_multiples(G, modele_dict, features_routes, nodes_gdf):
    """compare plusieurs destinations depuis point depart"""
    print("\n COMPARAISON MULTI-DESTINATIONS")
    print("-"*40)
    
    adresse_depart = input("Adresse de départ commune: ")
    
    destinations = []
    print("\nEntrez les destinations (appuyez sur Entrée pour terminer):")
    i = 1
    while True:
        dest = input(f"Destination {i}: ")
        if not dest:
            break
        destinations.append(dest)
        i += 1
    
    if len(destinations) < 2:
        print(" Il faut au moins 2 destinations pour comparer")
        return
    
    heure_depart = datetime.now()
    resultats = []
    
    print("\n Calcul des trajets...")
    for dest in destinations:
        res = predire_trajet_ameliore(
            adresse_depart, dest, heure_depart,
            G, modele_dict, features_routes, nodes_gdf
        )
        if res:
            resultats.append((dest, res))
    
    # afficher tableau comparatif
    if resultats:
        print("\n TABLEAU COMPARATIF:")
        print("-"*80)
        print(f"{'Destination':40s} {'Temps':10s} {'Distance':10s} {'Vitesse':10s}")
        print("-"*80)
        
        # trier par temps
        resultats.sort(key=lambda x: x[1]['temps_minutes'])
        
        for dest, res in resultats:
            print(f"{dest[:40]:40s} {res['temps_minutes']:.0f} min{' ':5s} "
                  f"{res['distance_km']:.1f} km{' ':5s} "
                  f"{res['vitesse_moyenne']:.0f} km/h")
        
        print("-"*80)
        print(f" Destination la plus rapide: {resultats[0][0]}")

def afficher_historique_trajets():
    """affiche historique trajets (simulation)"""
    print("\n HISTORIQUE DES TRAJETS")
    print("-"*60)
    print("Cette fonctionnalité sera disponible dans une prochaine version")
    print("Elle permettra de:")
    print("  • Sauvegarder vos trajets favoris")
    print("  • Analyser vos habitudes de déplacement")
    print("  • Recevoir des suggestions personnalisées")

# ========================================
# PROGRAMME PRINCIPAL
# ========================================

def main():
    """fonction principale du programme"""
    
    print("\n" + "="*80)
    print(" " * 20 + " PRÉDICTION DU TRAFIC EN ÎLE-DE-FRANCE ")
    print(" " * 25 + "PROJET - Machine Learning")
    print("="*80)
    
    # config
    print("\n  CONFIGURATION")
    print("-"*40)
    
    # demander mode
    print("Choisissez le mode d'exécution:")
    print("1. Entraînement complet (recommandé pour première utilisation)")
    print("2. Chargement d'un modèle existant")
    print("3. Mode démonstration (données synthétiques)")
    
    mode = input("\nVotre choix (1-3): ")
    
    try:
        # etape 1: charger graphe routier
        print("\n  Chargement du réseau routier d'Île-de-France...")
        print("   (Cette opération peut prendre quelques minutes la première fois)")
        
        G, nodes_gdf, edges_gdf = construire_graphe_routier()
        
        if mode == "1":
            # mode 1: entrainement complet
            print("\n" + "="*60)
            print("MODE ENTRAÎNEMENT COMPLET")
            print("="*60)
            
            # charger donnees
            print("\n[Phase 1/3] Chargement des données...")
            
            # donnees comptage
            donnees_comptage = charger_comptages_routiers(nombre_lignes=10000)
            
            # referentiel geo
            referentiel_geo = charger_referentiel_geographique()
            
            # donnees sytadin (optionnel)
            donnees_sytadin = charger_donnees_sytadin_enrichies()
           
           # mapper vitesses
            vitesses_par_arete = mapper_vitesses_sur_graphe(
               donnees_comptage, referentiel_geo, edges_gdf, donnees_sytadin
           )
           
           # creer features routes
            features_routes = creer_features_route(edges_gdf)
           
           # entrainer modele
            print("\n[Phase 2/3] Entraînement du modèle...")
            modele_dict = entrainer_modele_ameliore(vitesses_par_arete, features_routes)
           
           # sauvegarder
            print("\n[Phase 3/3] Sauvegarde...")
            sauvegarder_modele(modele_dict)
           
            print("\n✅ Entraînement terminé avec succès!")
           
        elif mode == "2":
            # mode 2: chargement modele existant
            print("\n" + "="*60)
            print("MODE CHARGEMENT DE MODÈLE")
            print("="*60)
           
            modele_dict = charger_modele()
           
            if not modele_dict:
                print("\n  Aucun modèle trouvé, passage en mode démonstration...")
                mode = "3"
            else:
                # creer features pour graphe actuel
                features_routes = creer_features_route(edges_gdf)
               
        if mode == "3" or (mode == "2" and not modele_dict):
            # mode 3: demo rapide
            print("\n" + "="*60)
            print("MODE DÉMONSTRATION")
            print("="*60)
           
            print("\n  Génération d'un modèle de démonstration...")
            print("   (Les prédictions seront approximatives)")
           
            # creer donnees synthetiques
            features_routes = creer_features_route(edges_gdf)
            vitesses_synthetiques = generer_donnees_synthetiques_realistes(edges_gdf)
           
            # entrainer modele rapide
            modele_dict = entrainer_modele_ameliore(vitesses_synthetiques, features_routes)
           
            print("\n Modèle de démonstration prêt!")
       
        # interface interactive
        print("\n" + "="*60)
        print(" SYSTÈME PRÊT À L'EMPLOI")
        print("="*60)
       
        # lancer interface
        interface_interactive(G, modele_dict, features_routes, nodes_gdf)
       
    except KeyboardInterrupt:
        print("\n\n  Interruption par l'utilisateur")
        print(" Au revoir!")
       
    except Exception as e:
        print(f"\n Erreur: {e}")
        print("\nPour plus d'aide, vérifiez que:")
        print("  • Vous avez installé tous les packages requis")
        print("  • Vous êtes connecté à Internet")
        print("  • Vous avez suffisamment de mémoire disponible")
       
        # mode degrade
        print("\n Tentative de lancement en mode dégradé...")
        try:
            # essayer avec paris uniquement
            print("   → Chargement du graphe de Paris uniquement...")
            G = ox.graph_from_place("Paris, France", network_type="drive")
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
            edges_gdf = edges_gdf.reset_index()
            edges_gdf['edge_id'] = edges_gdf.index
           
            # donnees minimales
            features_routes = creer_features_route(edges_gdf)
            vitesses_synthetiques = generer_donnees_synthetiques_realistes(edges_gdf.head(500))
            modele_dict = entrainer_modele_ameliore(vitesses_synthetiques, features_routes)
           
            print("\n Mode dégradé activé (Paris uniquement)")
            interface_interactive(G, modele_dict, features_routes, nodes_gdf)
           
        except Exception as e2:
            print(f"\n Impossible de démarrer: {e2}")

def exemple_utilisation_directe():
    """exemple utilisation directe sans interface"""
    print("\n EXEMPLE D'UTILISATION DIRECTE")
    print("-"*40)
   
    # charger composants
    print("Chargement...")
    G, nodes_gdf, edges_gdf = construire_graphe_routier("Paris, France")
    features_routes = creer_features_route(edges_gdf)
   
    # modele simple pour demo
    vitesses_synthetiques = generer_donnees_synthetiques_realistes(edges_gdf.head(1000))
    modele_dict = entrainer_modele_ameliore(vitesses_synthetiques, features_routes)
   
    # exemple prediction
    print("\n Exemple de trajet:")
    resultat = predire_trajet_ameliore(
        "Tour Eiffel, Paris",
        "Arc de Triomphe, Paris",
        datetime.now(),
        G, modele_dict, features_routes, nodes_gdf
    )
   
    if resultat:
        print(f"\n Trajet calculé avec succès!")
        print(f"   Temps: {resultat['temps_minutes']:.0f} minutes")
        print(f"   Distance: {resultat['distance_km']:.1f} km")

# ========================================
# POINT D'ENTREE DU PROGRAMME
# ========================================

if __name__ == "__main__":
    # verifier dependances
    try:
        import osmnx
        import networkx
        import pandas
        import numpy
        import geopandas
        import sklearn
        import xgboost
        import folium
        print("✓ Toutes les dépendances sont installées")
    except ImportError as e:
        print(f" Dépendance manquante: {e}")
        print("\nInstallez les dépendances avec:")
        print("pip install osmnx networkx pandas numpy geopandas scikit-learn xgboost folium geopy requests joblib tqdm shapely")
        exit(1)
   
    # lancer programme principal
    main()