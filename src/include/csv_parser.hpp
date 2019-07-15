/*
 *@file csv_parser.hpp
 *@brief csv data into vector
 *@author Katsuhisa Takahashi
 *@date 2019/07/15
 */
#ifndef CSV_PARSER_HPP
#define CSV_PARSER_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

class CsvParser {
    public:
        CsvParser();
        CsvParser(const std::string filename, const char delimiter);
        // ~CsvParser();
        std::vector<std::vector<double> >& readInputfile();
        std::vector<double> split(const std::string &s, char delim);
        std::vector<std::string>& get_columns();

    private:
        const std::string _input_filename;
        const char _delimiter;
        std::vector<std::string> _input_column_names;
        std::vector<std::vector<double> > _input_data;
        
};

#endif // CSV_PARSER_HPP