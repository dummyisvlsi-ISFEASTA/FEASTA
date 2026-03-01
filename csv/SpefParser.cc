#include "csv/SpefParser.hh"
#include <iostream>
#include <fstream>
#include <sstream>

namespace sta {

SpefParser::SpefParser() {}

bool SpefParser::parse(const std::string& spef_file) {
    std::ifstream file(spef_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open SPEF file: " << spef_file << std::endl;
        return false;
    }

    std::string line;
    bool in_conn_section = false;
    while (std::getline(file, line)) {
        if (line.find("*CONN") != std::string::npos) {
            in_conn_section = true;
            continue;
        }

        if (in_conn_section) {
            if (line.rfind("*D_NET", 0) == 0) {
                std::istringstream iss(line);
                std::string keyword, net_name;
                iss >> keyword >> net_name;

                // Continue reading lines for this net until we find *END
                while (std::getline(file, line) && line.find("*END") == std::string::npos) {
                    if (line.rfind("*P", 0) == 0) {
                        std::istringstream p_iss(line);
                        std::string p_keyword, pin_name;
                        char direction;
                        float x, y;

                        p_iss >> p_keyword >> pin_name >> direction;

                        // Now, look for the *C line
                        std::string next_line;
                        if (std::getline(file, next_line) && next_line.rfind("*C", 0) == 0) {
                            std::istringstream c_iss(next_line);
                            std::string c_keyword;
                            c_iss >> c_keyword >> x >> y;

                            pin_coordinates_[pin_name] = {x, y};
                        }
                    }
                }
            }
        }
    }
    return true;
}

const std::map<std::string, PinCoordinates>& SpefParser::getPinCoordinates() const {
    return pin_coordinates_;
}

}