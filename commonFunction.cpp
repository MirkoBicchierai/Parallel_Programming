#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>

#define DIM 3

using namespace std;

struct Point {

    double coordinate[DIM]{};
    int actualCentroid{};

    Point(){
        for(double c : coordinate){
            c = 0;
        }
        actualCentroid = -1;
    }

    void operator+=(const Point& p){
        for (int i=0; i<DIM; i++)
            this->coordinate[i] += p.coordinate[i];
    }

    void operator/=(const int& n){
        for (double & i : this->coordinate)
            i /= (double)n;
    }

};

std::random_device rand_dev;
std::mt19937 gen(rand_dev());

vector<Point> randomCentroid(int k, vector<Point> &data) {

    vector<Point> centroids(k);

    vector<int> indexes(k);
    std::uniform_int_distribution<> distrib(0, data.size() - 1);

    int r;
    for (int i = 0; i < k; i++) {
        while(true) {
            r = distrib(gen);
            if(std::find(indexes.begin(), indexes.end(), r) == indexes.end())
                break;
        }
        indexes.push_back(r);
        centroids[i] = data[r];
    }
    return centroids;
}

double euclideanDistance(const Point& p1, const Point& p2) {
    double dist = 0;
    #pragma omp simd
    for (int i = 0; i < DIM; i++)
        dist += (p1.coordinate[i] - p2.coordinate[i]) * (p1.coordinate[i] - p2.coordinate[i]);
    return sqrt(dist);
}

double distance(const Point& p1, const Point& p2) {
    double dist = 0;
    #pragma omp simd
    for (int i = 0; i < DIM; i++)
        dist += (p1.coordinate[i] - p2.coordinate[i]) * (p1.coordinate[i] - p2.coordinate[i]);
    return dist;
}

bool areEqual(const std::vector<Point> &vec1, const std::vector<Point> &vec2) {
    for (int j = 0; j < vec1.size(); j++) {
        for (int i = 0; i < DIM; i++) {
            if (vec1[j].coordinate[i] != vec2[j].coordinate[i]) {
                return false;
            }
        }
    }
    return true;
}

vector<Point> loadDataset(const string &path) {

    vector<Point> data;
    ifstream file(path);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string temp;
        Point p{};
        for (double & i : p.coordinate) {
            getline(ss, temp, ',');
            i = stod(temp);
        }
        p.actualCentroid = -1;
        data.push_back(p);
    }

    return data;
}

void writeResult(const string &len, const string &k, int n, double time,
                 const string &filename) {
    std::ofstream file(filename, std::ios::app);
    file << len + " " + k + " " + std::to_string(n) + " " + std::to_string(time) << std::endl;
    file.close();
}

void writeCSV(const vector<Point> &data, const string &filename) {
    ofstream file(filename);
    file << "X";
    file << ",";
    file << "Y";
    file << ",";
    file << "Z";
    file << ",";
    file << "Actual";
    file << endl;
    for (const auto &point: data) {
        for (double c: point.coordinate) {
            file << c;
            file << ",";
        }
        file << point.actualCentroid;
        file << endl;
    }
    file.close();
}

Point next_centroid (const std::vector<double> &dist, const std::vector<Point> &data) {
    std::discrete_distribution<> distrib(dist.begin(), dist.end());
    return data[distrib(gen)];
}


std::vector<Point> initialization_kmean_par(const std::vector<Point> &data, const int k, const int t) {
    std::vector<Point> centroids;
    centroids.reserve(k);
    std::uniform_int_distribution<> distrib(0, data.size() - 1);
    centroids.push_back(data[distrib(gen)]);
    int chunk = ceil(data.size() / t);
    while (centroids.size() < k) {
        vector<double> distances_glob(data.size(), numeric_limits<double>::max());
        #pragma omp parallel for schedule(static, chunk) num_threads(t)
            for (int i = 0; i < data.size(); i++) {
                for (const Point &centroid : centroids) {
                    distances_glob[i] = std::min(distances_glob[i], euclideanDistance(data[i], centroid));
                }
            }
        centroids.push_back(next_centroid(distances_glob, data));
    }
    return centroids;
}

std::vector<Point> initialization_kmean_seq(const std::vector<Point> &data, const int k) {
    std::vector<Point> centroids;
    std::uniform_int_distribution<> distrib(0, data.size() - 1);
    centroids.push_back(data[distrib(gen)]);
    while (centroids.size() < k) {
        vector<double> distances_glob(data.size(), numeric_limits<double>::max());
        for (size_t i = 0; i < data.size(); i++) {
            for (const Point &centroid : centroids) {
                distances_glob[i] = std::min(distances_glob[i], euclideanDistance(data[i], centroid));
            }
        }
        centroids.push_back(next_centroid(distances_glob, data));
    }
    return centroids;
}

