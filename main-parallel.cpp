#include <iostream>
#include <vector>
#include <omp.h>
#include "commonFunction.cpp"
#include <string>

using namespace std;

vector<Point> kMeans(vector<Point> &data, vector<Point> &centroids, int k, int maxIterations, int threads) {
    int block = ceil(data.size() / threads);
    vector<Point> newCentroids(k, Point());
    vector<int> counts(k, 0);
    for (int iter = 0; iter < maxIterations; ++iter) {
        newCentroids = vector<Point>(k, Point());
        counts = vector<int>(k, 0);
        #pragma omp parallel num_threads(threads)
        {
            vector<int> tmp_counts(k, 0);
            vector<Point> tmp_newCentroids(k, Point());
            #pragma omp for nowait schedule(static, block)
            for (Point &pt: data) {
                double minDistance = distance(pt, centroids[0]);
                pt.actualCentroid = 0;
                for (int j = 1; j < k; j++) {
                    double dist = distance(pt, centroids[j]);
                    if (dist < minDistance) {
                        minDistance = dist;
                        pt.actualCentroid = j;
                    }
                }
                tmp_newCentroids[pt.actualCentroid] += pt;
                tmp_counts[pt.actualCentroid]++;
            }
            #pragma omp critical
            {
                for (int i = 0; i < k; i++) {
                    newCentroids[i] += tmp_newCentroids[i];
                    counts[i] += tmp_counts[i];
                }
            }
        }
        for (int i = 0; i < k; i++) {
            newCentroids[i] /= counts[i];
        }
        if (areEqual(centroids, newCentroids)) {
            return centroids;
        }
        centroids = newCentroids;
    }
    return centroids;
}

int runSingleTest(bool output, const int t, vector<Point> data, const int k, int n, int maxIterations) {

    vector<Point> centroids = initialization_kmean_par(data, k, t);
    double sum_time = 0;
    for (int i = 0; i < n; ++i) {
        double dtime = omp_get_wtime();
        centroids = kMeans(data, centroids, k, maxIterations, t);
        dtime = omp_get_wtime() - dtime;
        sum_time += dtime;
    }
    sum_time = sum_time / n;

    cout << "D:" + to_string(data.size()) + " K:" + to_string(k) + " T:" + to_string(t) +
            " Execution time parallel: " << sum_time << " seconds" << endl;

    writeResult(to_string(data.size()), to_string(k), t, sum_time, "../Times/Times_Parallel.txt");

    if (output) {
        writeCSV(centroids,
                 "../output/centroids/parallel_" + to_string(k) + "_" + to_string(data.size()) + "_" +
                 to_string(t) + ".csv");
        writeCSV(data, "../output/clusters/parallel_" + to_string(k) + "_" + to_string(data.size()) + "_" +
                       to_string(t) + ".csv");
    }
    return 0;
}

int runSingleTest_initialization(bool output, const int t, vector<Point> data, const int k, int n) {

    double sum_time = 0;
    for (int i = 0; i < n; ++i) {
        double dtime = omp_get_wtime();
        vector<Point> centroids = initialization_kmean_par(data, k, t);
        dtime = omp_get_wtime() - dtime;
        sum_time += dtime;
    }
    sum_time = sum_time / n;

    cout << "D:" + to_string(data.size()) + " K:" + to_string(k) + " T:" + to_string(t) +
            " Execution time parallel initialization: " << sum_time << " seconds" << endl;

    if (output)
        writeResult(to_string(data.size()), to_string(k), t, sum_time, "../Times/Times_Kmeans_Parallel.txt");

    return 0;

}

int runAllTest(bool output, bool type, int n) {

    std::vector<int> threads = {2, 4, 8, 16};
    std::vector<std::string> file_names = { "100", "1000", "10000", "100000", "1000000", "10000000"};
    std::vector<int> cen = {3, 5, 10, 15, 20, 25, 30, 40, 50};

    int maxIterations = 100;

    for (const std::string &name: file_names) {
        for (const auto k: cen) {
            vector<Point> data = loadDataset("../input/dataset_" + name + "_" + to_string(k) + ".csv");
            for (const auto t: threads) {
                if (type)
                    runSingleTest(output, t, data, k, n, maxIterations);
                else
                    runSingleTest_initialization(output, t, data, k, n);
            }
        }
    }

    return 0;

}

int main() {

    int ret;
    int n_test = 100;

    vector<Point> data = loadDataset("../input/dataset_1000_5.csv");
    ret = runSingleTest(true, 8, data, 5, n_test, 100);
    ret = runSingleTest(true, 16, data, 5, n_test, 100);
    //ret = runSingleTest_initialization(false, 16, data, 3, n_test);
    //ret = runSingleTest(false, 16, data, 5, n_test, 100);

    //ret = runAllTest(false,true, n_test);

    return ret;
}