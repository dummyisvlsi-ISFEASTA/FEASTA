#ifndef __SPEF_PARSER_HH__
#define __SPEF_PARSER_HH__

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace sta {

struct PinCoordinates {
    float x, y;
};

class SpefParser {
public:
    SpefParser();
    bool parse(const std::string& spef_file);
    const std::map<std::string, PinCoordinates>& getPinCoordinates() const;

private:
    std::map<std::string, PinCoordinates> pin_coordinates_;
    std::map<std::string, std::string> name_map_;  // Compressed name → actual name

    // Resolve compressed names like "*6" or "*347057:A1" to actual names
    std::string resolveName(const std::string& name) const {
        if (name.empty() || name[0] != '*') return name;
        
        // Handle instance:pin format like "*347057:A1"
        size_t colon_pos = name.find(':');
        if (colon_pos != std::string::npos) {
            std::string inst_part = name.substr(0, colon_pos);
            std::string pin_part = name.substr(colon_pos + 1);
            auto it = name_map_.find(inst_part);
            if (it != name_map_.end()) {
                // Use "/" as separator to match OpenSTA's naming convention
                return it->second + "/" + pin_part;
            }
            return name;  // No mapping found, return original
        }
        
        // Simple name like "*6"
        auto it = name_map_.find(name);
        return (it != name_map_.end()) ? it->second : name;
    }
};

// =================================================================//
// SpefParser Implementation
// =================================================================//

// Helper to tokenize a string
static inline std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(line);
    while (tokenStream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

inline SpefParser::SpefParser() {}

inline bool SpefParser::parse(const std::string& spef_file) {
    pin_coordinates_.clear();
    name_map_.clear();
    
    std::ifstream file(spef_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open SPEF file for reading: " << spef_file << std::endl;
        return false;
    }

    enum class SpefState { IDLE, IN_NAME_MAP, IN_D_NET, IN_CONN };
    SpefState state = SpefState::IDLE;

    std::string line;
    while (std::getline(file, line)) {
        // Trim leading/trailing whitespace from the line
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) {
            continue;
        }

        // Detect *NAME_MAP section
        if (line.rfind("*NAME_MAP", 0) == 0) {
            state = SpefState::IN_NAME_MAP;
            continue;
        }

        // Parse NAME_MAP entries
        if (state == SpefState::IN_NAME_MAP) {
            // NAME_MAP entries look like: "*6 i_dbg_reg_4_"
            if (line[0] == '*') {
                auto tokens = tokenize(line);
                if (tokens.size() >= 2) {
                    // Check if first token looks like a map entry (*number or *name)
                    // and second token is the actual name
                    bool is_map_entry = true;
                    for (size_t i = 1; i < tokens[0].size(); ++i) {
                        char c = tokens[0][i];
                        if (!std::isalnum(c) && c != '_') {
                            is_map_entry = false;
                            break;
                        }
                    }
                    if (is_map_entry && tokens[0].size() > 1) {
                        name_map_[tokens[0]] = tokens[1];
                        continue;
                    }
                }
                // If we hit a line starting with * that's not a map entry,
                // we've left the NAME_MAP section - don't continue, fall through
            }
        }

        // State transitions based on SPEF keywords
        if (line.rfind("*D_NET", 0) == 0) {
            state = SpefState::IN_D_NET;
            continue;
        }
        if (line.rfind("*END", 0) == 0) {
            state = SpefState::IDLE;
            continue;
        }

        if (state == SpefState::IN_D_NET) {
            if (line.rfind("*CONN", 0) == 0) {
                state = SpefState::IN_CONN;
                continue;
            }
        }
        
        if (state == SpefState::IN_CONN) {
            if (line.rfind("*P", 0) == 0 || line.rfind("*I", 0) == 0) {
                // This is a coordinate line, process it
                auto tokens = tokenize(line);
                if (tokens.size() < 2) {
                    continue;
                }

                std::string pin_name = tokens[1];

                // Find the *C token on the same line
                auto it = std::find(tokens.begin(), tokens.end(), "*C");

                if (it != tokens.end() && std::distance(it, tokens.end()) > 2) {
                    try {
                        float x = std::stof(*(it + 1));
                        float y = std::stof(*(it + 2));
                        // Resolve compressed name to actual name
                        std::string resolved_name = resolveName(pin_name);
                        pin_coordinates_[resolved_name] = {x, y};
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Could not parse coordinate on line: " << line << std::endl;
                    }
                }
            } else if (line[0] == '*') {
                // Any other section (like *CAP, *RES) ends the *CONN section
                // but keeps us within the *D_NET block.
                state = SpefState::IN_D_NET;
            }
        }
    }
    
    std::cerr << "SPEF Parser: Loaded " << name_map_.size() << " name mappings, "
              << pin_coordinates_.size() << " pin coordinates." << std::endl;
    
    // Return true even if empty - the file was parsed successfully
    return true;
}

inline const std::map<std::string, PinCoordinates>& SpefParser::getPinCoordinates() const {
    return pin_coordinates_;
}

} // namespace sta

#endif // __SPEF_PARSER_HH__