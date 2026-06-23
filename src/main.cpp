#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;
// A function to read our CSV file and return a 2D grid of strings
vector<vector<string>> readCSV(const  string& filename) {
     vector< vector< string>> dataset;
    // 1. Open the file stream
     ifstream file(filename);
    // Check if the file successfully opened
    if (!file.is_open()) {
         cerr << "Error: Could not open the file " << filename <<  "\n";
        return dataset;
    }
     string line;
    // 2. Read the file line by line
    while ( getline(file, line)) {
         vector< string> row;
         stringstream lineStream(line);
         string cell;
        
        // 3. Break the line into individual cells every time there is a comma
        while ( getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }
        // Add the completed row to our main dataset
        dataset.push_back(row);
    }
    file.close();
    return dataset;
}
int main() {
    std::string filepath = "../data/matches.csv"; 
    std::vector<std::vector<std::string>> data = readCSV(filepath);

    // Creating our math vectors
    std::vector<double> X; // Features: Point Differential
    std::vector<double> y; // Labels: Win (1) or Loss (0)

    // Start at index 1 to skip the header row!
    for (size_t i = 1; i < data.size(); ++i) {
        
        // Convert strings to doubles (std::stod)
        double scoreA = std::stod(data[i][0]);
        double scoreB = std::stod(data[i][1]);
        double winLabel = std::stod(data[i][2]);

        // Feature Engineering: Calculate the difference
        double pointDifferential = scoreA - scoreB;

        // Store them in our math vectors
        X.push_back(pointDifferential);
        y.push_back(winLabel);
    }

    // Print to verify our math matrices are ready
    std::cout << "--- Feature Engineering Complete ---" << std::endl;
    for (size_t i = 0; i < X.size(); ++i) {
        std::cout << "Match " << i+1 
                  << " | Feature X (Point Diff): " << X[i] 
                  << "\t| Label y (Outcome): " << y[i] << std::endl;
    }

    return 0;
}
