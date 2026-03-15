// OpenSTA, Static Timing Analyzer
// Copyright (c) 2025, Parallax Software, Inc.
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
// 
// The origin of this software must not be misrepresented; you must not
// claim that you wrote the original software.
// 
// Altered source versions must be plainly marked as such, and must not be
// misrepresented as being the original software.
// 
// This notice may not be removed or altered from any source distribution.

#include "csv/SpefParser.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

namespace sta {

static std::vector<std::string>
tokenize(const std::string &line)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream stream(line);
	while (stream >> token)
		tokens.push_back(token);
	return tokens;
}

SpefParser::SpefParser()
{
}

std::string
SpefParser::resolveName(const std::string &name) const
{
	if (name.empty() || name[0] != '*')
		return name;

	// Handle instance:pin format like "*347057:A1"
	size_t colon_pos = name.find(':');
	if (colon_pos != std::string::npos) {
		std::string inst_part = name.substr(0, colon_pos);
		std::string pin_part = name.substr(colon_pos + 1);
		auto it = name_map_.find(inst_part);
		if (it != name_map_.end())
			return it->second + "/" + pin_part;
		return name;
	}

	auto it = name_map_.find(name);
	return (it != name_map_.end()) ? it->second : name;
}

bool
SpefParser::parse(const std::string &spef_file)
{
	pin_coordinates_.clear();
	name_map_.clear();

	std::ifstream file(spef_file);
	if (!file.is_open()) {
		std::cerr << "Error: Cannot open SPEF file: " << spef_file
		          << std::endl;
		return false;
	}

	enum class State { IDLE, IN_NAME_MAP, IN_D_NET, IN_CONN };
	State state = State::IDLE;

	std::string line;
	while (std::getline(file, line)) {
		line.erase(0, line.find_first_not_of(" \t\r\n"));
		line.erase(line.find_last_not_of(" \t\r\n") + 1);

		if (line.empty())
			continue;

		if (line.rfind("*NAME_MAP", 0) == 0) {
			state = State::IN_NAME_MAP;
			continue;
		}

		if (state == State::IN_NAME_MAP) {
			if (line[0] == '*') {
				auto tokens = tokenize(line);
				if (tokens.size() >= 2) {
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
			}
		}

		if (line.rfind("*D_NET", 0) == 0) {
			state = State::IN_D_NET;
			continue;
		}
		if (line.rfind("*END", 0) == 0) {
			state = State::IDLE;
			continue;
		}

		if (state == State::IN_D_NET) {
			if (line.rfind("*CONN", 0) == 0) {
				state = State::IN_CONN;
				continue;
			}
		}

		if (state == State::IN_CONN) {
			if (line.rfind("*P", 0) == 0 || line.rfind("*I", 0) == 0) {
				auto tokens = tokenize(line);
				if (tokens.size() < 2)
					continue;

				std::string pin_name = tokens[1];
				auto it = std::find(tokens.begin(), tokens.end(), "*C");

				if (it != tokens.end()
				    && std::distance(it, tokens.end()) > 2) {
					try {
						float x = std::stof(*(it + 1));
						float y = std::stof(*(it + 2));
						std::string resolved = resolveName(pin_name);
						pin_coordinates_[resolved] = {x, y};
					}
					catch (const std::exception &e) {
						std::cerr << "Warning: bad coordinate: "
						          << line << std::endl;
					}
				}
			}
			else if (line[0] == '*') {
				state = State::IN_D_NET;
			}
		}
	}

	std::cerr << "SPEF: " << name_map_.size() << " name mappings, "
	          << pin_coordinates_.size() << " coordinates" << std::endl;
	return true;
}

const std::map<std::string, PinCoordinates> &
SpefParser::getPinCoordinates() const
{
	return pin_coordinates_;
}

} // namespace sta