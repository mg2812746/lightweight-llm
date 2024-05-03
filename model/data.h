#ifndef DATA_H
#define DATA_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
// Namespace
using std::vector;
using std::string;
using std::cerr;
using std::endl;
using std::istringstream;
using std::getline;
using std::ifstream;
using std::min;

// Function Declarations
vector<string> load_text_data(const string &filename);
vector<vector<string>> tokenize_data(const vector<string> &text_data);
vector<vector<vector<int>>> create_batches(const vector<vector<int>> &tokenized_data, int batch_size);


#endif // DATA_HPP
