#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath> // Added for  exp() and  log()
using namespace std;

// 1. Our Sigmoid Math Function
// Squashes any number into a probability between 0 and 1
double sigmoid(double z) {
    return 1.0 / (1.0 +  exp(-z));
}

// A function to read our CSV file and return a 2D grid of strings
vector< vector<string>> readCSV(const  string& filename) {
     vector< vector< string>> dataset;
    
    // 1. Open the file stream
     ifstream file(filename);
    
    // Check if the file successfully opened
    if (!file.is_open()) {
         cerr << "Error: Could not open the file " << filename << "\n";
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

// --- PHASE 5: PREDICTION FUNCTION ---
// Takes a new point differential and our learned parameters to output a win probability
void predict_match(double point_differential, double weight, double bias) {
    double z = (weight * point_differential) + bias;
    double probability = sigmoid(z);
    
     cout << "Point Differential: " << (point_differential > 0 ? "+" : "") << point_differential 
              << " | Win Probability: " << (probability * 100.0) << "%" <<  endl;
}

int main() {
    // Call our function and point it to the data folder
     string filepath = "../data/matches.csv"; 
     vector< vector< string>> data = readCSV(filepath);

    // Creating our math vectors
     vector<double> X; // Features: Point Differential
     vector<double> y; // Labels: Win (1) or Loss (0)

    // Start at index 1 to skip the header row!
    for (size_t i = 1; i < data.size(); ++i) {
        
        // Convert strings to doubles ( stod)
        double scoreA =  stod(data[i][0]);
        double scoreB =  stod(data[i][1]);
        double winLabel =  stod(data[i][2]);

        // Feature Engineering: Calculate the difference
        double pointDifferential = scoreA - scoreB;

        // Store them in our math vectors
        X.push_back(pointDifferential);
        y.push_back(winLabel);
    }

    // --- MACHINE LEARNING ENGINE: GRADIENT DESCENT ---
    
    // Initialize parameters (we start with 0 and let the model learn the right numbers)
    double weight = 0.0;
    double bias = 0.0;
    
    // Hyperparameters
    double learning_rate = 0.01; // The "step size" for our corrections
    int epochs = 1000;           // Number of training iterations

     cout << "\n--- Starting Training ---" <<  endl;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;

        // Loop through every match in our dataset
        for (size_t i = 0; i < X.size(); ++i) {
            
            // 1. Forward Pass: Make a prediction using current weight and bias
            double z = (weight * X[i]) + bias;
            double prediction = sigmoid(z);
            
            // 2. Calculate Error (Prediction vs Reality)
            double error = prediction - y[i];

            // Calculate Log Loss for tracking progress
            // Adding 1e-9 prevents math errors if prediction is exactly 0 or 1
            total_loss += -y[i] *  log(prediction + 1e-9) - (1 - y[i]) *  log(1 - prediction + 1e-9);

            // 3. Backward Pass: Calculate gradients and update parameters
            weight = weight - (learning_rate * error * X[i]);
            bias = bias - (learning_rate * error);
        }

        // Print progress every 100 epochs so we can watch it learn
        if (epoch % 100 == 0) {
             cout << "Epoch " << epoch 
                      << " | Loss: " << (total_loss / X.size()) 
                      << " | Weight: " << weight 
                      << " | Bias: " << bias <<  endl;
        }
    }

     cout << "\n--- Training Complete ---" <<  endl;
     cout << "Final Learned Weight: " << weight <<  endl;
     cout << "Final Learned Bias: " << bias <<  endl;

    // --- TESTING OUT OUR PREDICTIONS ---
     cout << "\n--- Making Future Predictions ---" <<  endl;
    
    // Let's test a few hypothetical matchups!
    predict_match(10.0, weight, bias);  // Team A is expected to win by 10
    predict_match(-5.0, weight, bias);  // Team A is expected to lose by 5
    predict_match(0.0, weight, bias);   // An exactly even matchup

    return 0;
}
