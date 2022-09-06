from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import folium
from folium.plugins import FastMarkerCluster

class dbscan_clustering_canada:
    def __init__(self, epsilon, min_samples,study_area_provinces,cluster_area_provinces):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.study_area_provinces = study_area_provinces 
        self.cluster_area_provinces = cluster_area_provinces

    def import_data(self):
        data_path = r".\canadacities.csv"
        #data_pathr"C:\Users\jua12849\Documents\GitHub\GeospatialDataAnalysis\canadacities.csv"
        canadian_cities = pd.read_csv(data_path)
        datum = "EPSG:4326"

        #create geodataframe containing data with all canadian cities and a point geometry column
        geometry = [Point(xy) for xy in zip(canadian_cities["lng"],canadian_cities["lat"])]
        gdf = gpd.GeoDataFrame(canadian_cities,crs=datum,geometry=geometry)

        return gdf


    def create_gdf_dictionary(self,gdf):
        #create dictionary with complete data for each province
        d = {}
        for province in gdf.province_id.unique():
            d["province_{}".format(province)] = gdf.loc[gdf["province_id"]==province]

        return d

    def obtain_provinces(self,d):
        provinces = list(d.keys())
            
        return provinces


    def create_numpy_dictionary(self,d,gdf,provinces):
        #obtain province names as well as list of dictionary keys.

        #obtain lat/long data for each province and the entire country as a numpy array.
        d_lat_lon_numpy = {}
        for province in provinces:
            d_lat_lon_numpy["{}".format(province)] = [d.get(province)[["lat","lng"]].to_numpy()]

        d_lat_lon_numpy["Canada"] = [gdf[["lat","lng"]].to_numpy()]

        return d_lat_lon_numpy

    def replace_dictionary(self,d_lat_lon_numpy,province,d):

        d_lat_lon_numpy["{}".format(province)] = [d.get(province)[["lat","lng"]].to_numpy()]
        
        return d_lat_lon_numpy


    def get_centermost_point(self,cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    def apply_DBSCAN(self,d_lat_lon_numpy,province,epsilon,min_samples,algorithm = 'ball_tree',metric='haversine'):
        return d_lat_lon_numpy["{}".format(province)].append(
                {"dbs_{}".format(province):DBSCAN(eps=epsilon, min_samples=min_samples,algorithm = algorithm,metric=metric).fit(np.radians(d_lat_lon_numpy.get(province)[0]))})

    def retrieve_labels(self,d_lat_lon_numpy,province):
        return d_lat_lon_numpy["{}".format(province)].append(
                {"{}_cluster_label".format(province):d_lat_lon_numpy.get(province)[1]["dbs_{}".format(province)].labels_})

    def obtain_cluster_labels(self,d_lat_lon_numpy,province):
        return d_lat_lon_numpy["{}".format(province)].append(
                {"{}_num_clusters".format(province):len(set(d_lat_lon_numpy.get(province)[2]["{}_cluster_label".format(province)]))})

    def obtain_clusters_numpy(self,d_lat_lon_numpy,province):
        return d_lat_lon_numpy["{}".format(province)].append(
                {"{}_clusters".format(province):
                pd.Series(d_lat_lon_numpy.get(province)[0][d_lat_lon_numpy.get(province)[2]["{}_cluster_label".format(province)] == n] for n in range(d_lat_lon_numpy["{}".format(province)][3]["{}_num_clusters".format(province)]))})


    def perform_dbscan(self,d_lat_lon_numpy,epsilon,min_samples,d):

        #perform DBSCAN algorithm to each province separately as well as the entire country
        for province in list(d_lat_lon_numpy.keys()):
            #Create DBSCAN object and apply to each latitude/longitude pair
            self.apply_DBSCAN(d_lat_lon_numpy=d_lat_lon_numpy,province=province,epsilon=epsilon,min_samples=min_samples,algorithm = 'ball_tree',metric='haversine')
            #Retrieve labels obtained from algorithm
            self.retrieve_labels(d_lat_lon_numpy=d_lat_lon_numpy,province=province)
            #Obtain cluster labels
            self.obtain_cluster_labels(d_lat_lon_numpy=d_lat_lon_numpy,province=province)
            #obtain clusters numpy
            self.obtain_clusters_numpy(d_lat_lon_numpy=d_lat_lon_numpy,province=province)

            # Check for empty clusters, DBSCAN function does not like them
            # We edrop any dictionaries with empty clusters to later run them again
            # With only one DBSCAN neighbour (no noise)
            final_clusters = d_lat_lon_numpy.get(province)[4].get("{}_clusters".format(province))
            if len(final_clusters.iloc[-1]) == 0:
                final_clusters.drop(final_clusters.tail(1).index,inplace=True)
            
        for province in list(d_lat_lon_numpy.keys()):

            final_clusters = d_lat_lon_numpy.get(province)[4].get("{}_clusters".format(province))
            if len(final_clusters) == 0:
                del d_lat_lon_numpy["{}".format(province)]
                self.replace_dictionary(d_lat_lon_numpy,province,d)
                #min_samples_final = 1
                #Create DBSCAN object and apply to each latitude/longitude pair
                self.apply_DBSCAN(d_lat_lon_numpy=d_lat_lon_numpy,province=province,epsilon=epsilon,min_samples=1,algorithm = 'ball_tree',metric='haversine')
                #Retrieve labels obtained from algorithm
                self.retrieve_labels(d_lat_lon_numpy=d_lat_lon_numpy,province=province)
                #Obtain cluster labels
                self.obtain_cluster_labels(d_lat_lon_numpy=d_lat_lon_numpy,province=province)
                #obtain clusters numpy
                self.obtain_clusters_numpy(d_lat_lon_numpy=d_lat_lon_numpy,province=province)

        for province in list(d_lat_lon_numpy.keys()):

            final_clusters = d_lat_lon_numpy.get(province)[4].get("{}_clusters".format(province))
            if len(final_clusters) > 0 :

                d_lat_lon_numpy["{}".format(province)].append(
                    {"{}_centermost_points".format(province):d_lat_lon_numpy.get(province)[4]["{}_clusters".format(province)].map(self.get_centermost_point)})

                #unzip the list of centermost points (lat,lon) tuples into separate lat/lon lists
                lats, lons = zip(*d_lat_lon_numpy.get(province)[5]["{}_centermost_points".format(province)])
                #create a pandas dataframe
                rep_points = pd.DataFrame({'lon':lons, 'lat':lats})

                d_lat_lon_numpy["{}".format(province)].append({"{}_centermost_points_numpy".format(province) : rep_points.to_numpy()})

                d_lat_lon_numpy["{}".format(province)].append(
                    {"{}_gdf_cluster_samples".format(province):gpd.GeoDataFrame(rep_points, geometry=gpd.points_from_xy(rep_points.lon, rep_points.lat),crs = "EPSG:4326" )})

        return d_lat_lon_numpy

    def calculate_mean_ontario_loc(self,d):    
        #mean location for ontario cities
        mean_lat_on = np.mean(d["province_ON"]["lat"])
        mean_lng_on = np.mean(d["province_ON"]["lng"])
        
        return mean_lat_on,mean_lng_on

    def calculate_mean_canada_loc(self,gdf):
        #mean location for canada cities
        gdf_mean_lat = np.mean(gdf.lat)
        gdf_mean_lng = np.mean(gdf.lng)

        return gdf_mean_lat,gdf_mean_lng

    def cities_dict(self,d_lat_lon_numpy):
        cities = {}
        for province in d_lat_lon_numpy.keys():
            cities["{}".format(province)] = d_lat_lon_numpy.get("{}".format(province))[0]
        
        return cities

    def clusters_dict(self,d_lat_lon_numpy):
        clusters={}
        
        for province in d_lat_lon_numpy.keys():
            #print(d_lat_lon_numpy.get("{}".format(province)))#[6])
            clusters["{}".format(province)] = d_lat_lon_numpy.get("{}".format(province))[6].get("{}_centermost_points_numpy".format(province))
        
        return clusters

    def study_area_numpy(self,cities,study_area_provinces):
        study_area = []
        
        for stdy_area in study_area_provinces:
            study_area.append(cities[stdy_area])

        study_area = np.concatenate(study_area)

        return study_area

    def cluster_area_numpy(self,clusters,cluster_area_provinces):
        cluster_area = []

        for clstr_area in cluster_area_provinces:
            cluster_area.append(clusters[clstr_area])

        cluster_area = np.concatenate(cluster_area)

        return cluster_area


    def create_map(self,gdf_mean_lat,gdf_mean_lng,study_area,study_clusters,zoom):

        my_map = folium.Map(location=[gdf_mean_lat,gdf_mean_lng], zoom_start=zoom)

        for point in study_clusters :
            loc = [point[1],point[0]]
            folium.Marker(location=loc,icon=folium.Icon(color="red")).add_to(my_map)
            #folium.Circle(radius=40000,location=[point[1],point[0]],color="red").add_to(my_map)

        for point in study_area :
            loc = [point[0],point[1]]
            #folium.Marker(location=loc,icon=folium.Icon(color="blue")).add_to(my_map)
            folium.Circle(radius=4000,location=loc,color="BLUE").add_to(my_map)
        
        #folium.GeoJson(data = gdf).add_to(my_map)    

        return my_map 



    def get_dict(self):
        gdf = self.import_data()
        d = self.create_gdf_dictionary(gdf)
        provinces = self.obtain_provinces(d)
        d_lat_lon_numpy = self.create_numpy_dictionary(d,gdf,provinces)
        d_lat_lon_numpy = self.perform_dbscan(d_lat_lon_numpy,epsilon = self.epsilon,min_samples=self.min_samples,d=d)   

        return d_lat_lon_numpy 




    def run_map(self):
        gdf = self.import_data()
        d = self.create_gdf_dictionary(gdf)
        provinces = self.obtain_provinces(d)
        d_lat_lon_numpy = self.create_numpy_dictionary(d,gdf,provinces)
        d_lat_lon_numpy = self.perform_dbscan(d_lat_lon_numpy,epsilon = self.epsilon,min_samples=self.min_samples,d=d)
        mean_lat_on,mean_lng_on = self.calculate_mean_ontario_loc(d)
        gdf_mean_lat, gdf_mean_lng = self.calculate_mean_canada_loc(gdf)
        clusters = self.clusters_dict(d_lat_lon_numpy)
        cities = self.cities_dict(d_lat_lon_numpy)
        study_area = self.study_area_numpy(cities,self.study_area_provinces)
        study_clusters = self.cluster_area_numpy(clusters,self.cluster_area_provinces)
        map = self.create_map(gdf_mean_lat,gdf_mean_lng,study_area,study_clusters,zoom=5)

        return map

    

    
