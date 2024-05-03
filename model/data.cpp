/**
 * @file data_processing.cpp
 * @brief Functions for loading, tokenizing, and batching text data.
 */

#include "data.h" // Include the necessary header file for declarations of data types and functions.

/**
 * @brief Loads text data from a file into a vector of strings.
 * 
 * @param filename The name of the file to load.
 * @return A vector of strings containing the lines of text from the file.
 */
vector<string> load_text_data(const string& filename) {
    vector<string> text_data; // Initialize an empty vector to store the lines of text.
    ifstream file(filename); // Open the file with the given filename for input.
    if (file.is_open()) { // Check if the file was successfully opened.
        string line; // Initialize a string to store each line of text.
        while (getline(file, line)) { // Read each line of text from the file.
            text_data.push_back(line); // Add the line of text to the vector.
        }
        file.close(); // Close the file after reading.
    } else {
        cerr << "Failed to open file: " << filename << endl; // Output an error message if the file could not be opened.
    }
    return text_data; // Return the vector containing the loaded text data.
}

/**
 * @brief Tokenizes the lines of text stored in a vector of strings.
 * 
 * @param text_data A vector of strings containing the lines of text to tokenize.
 * @return A vector of vectors of strings, where each inner vector represents the tokens of a single line of text.
 */
vector<vector<string>> tokenize_data(const vector<string>& text_data) {
    vector<vector<string>> tokenized_data; // Initialize an empty vector to store the tokenized data.
    for (const string& text : text_data) { // Iterate over each line of text in the input vector.
        vector<string> tokens; // Initialize an empty vector to store the tokens of the current line.
        istringstream iss(text); // Create an input string stream to tokenize the current line.
        string token; // Initialize a string to store each token.
        while (iss >> token) { // Tokenize the current line using stringstream.
            tokens.push_back(token); // Add each token to the vector of tokens.
        }
        tokenized_data.push_back(tokens); // Add the vector of tokens for the current line to the tokenized data.
    }
    return tokenized_data; // Return the vector containing the tokenized data.
}

/**
 * @brief Creates batches of tokenized data with a specified batch size.
 * 
 * @param tokenized_data A vector of vectors of strings containing the tokenized data.
 * @param batch_size The desired size of each batch.
 * @return A vector of vectors of vectors of strings, representing batches of tokenized data.
 */
vector<vector<vector<string>>> create_batches(const vector<vector<string>>& tokenized_data, int batch_size) {
    vector<vector<vector<string>>> batches; // Initialize an empty vector to store the batches of data.
    int num_batches = (tokenized_data.size() + batch_size - 1) / batch_size; // Calculate the number of batches needed.
    batches.reserve(num_batches); // Reserve memory for the batches vector to improve performance.

    for (int i = 0; i < num_batches; ++i) { // Iterate over the number of batches.
        int start_idx = i * batch_size; // Calculate the starting index of the current batch.
        int end_idx = min((i + 1) * batch_size, static_cast<int>(tokenized_data.size())); // Calculate the ending index of the current batch.
        vector<vector<string>> batch; // Initialize an empty vector to store the current batch.
        for (int j = start_idx; j < end_idx; ++j) { // Iterate over the indices of the current batch.
            batch.push_back(tokenized_data[j]); // Add the tokenized data corresponding to the current index to the batch.
        }
        batches.push_back(batch); // Add the current batch to the vector of batches.
    }
    return batches; // Return the vector containing the batches of tokenized data.
}
