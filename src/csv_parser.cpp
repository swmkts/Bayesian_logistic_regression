/*
 *@file CsvParser.cpp
 *@brief csv data into vector
 *@author Katsuhisa Takahashi
 *@date 2019/07/15
 */
#include "include/csv_parser.hpp"

CsvParser::CsvParser()
    : _input_filename("./file_pass")
    , _delimiter(','){
        std::cerr << "Input the file pass name!" << std::endl;
    }

CsvParser::CsvParser(const std::string filename, const char delimiter)
    : _input_filename(filename)
    , _delimiter(delimiter){
        this->readInputfile();
    }

std::vector<std::vector<double> >& CsvParser::readInputfile(){
    std::ifstream ifs(this->_input_filename);
    if (ifs.fail()){
        std::cerr << "Failed to open file '" << this->_input_filename << "'." << std::endl;
    }
    std::string columns;
    std::string column;
    getline(ifs, columns);
    std::stringstream ss(columns);
    while(getline(ss, column, this->_delimiter)){
        this->_input_column_names.push_back(column);
    }
    std::string row;
    while (getline(ifs, row)){
        std::vector<double> row_items;
        row_items = this->split(row, this->_delimiter);
        this->_input_data.push_back(row_items);
        std::cout << "#" << row << std::endl;
    }
    // for (auto v: this->_input_data){
    //     for (auto c: v){
    //         std::cout << c << " ";
    //     }
    //     std::cout << "\n";
    // }
    // _input_column_names
    return _input_data;
}

std::vector<double> CsvParser::split(const std::string &s, char delim){
    std::vector<double> factor;
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)){
        if(!item.empty()){
            factor.push_back(std::stod(item));
        }
    }
    return factor;
}

std::vector<std::string>& CsvParser::get_columns(){
    return this->_input_column_names;
}