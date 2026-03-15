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

#ifndef FEASTA_SPEF_PARSER_HH
#define FEASTA_SPEF_PARSER_HH

#include <string>
#include <map>

namespace sta {

struct PinCoordinates {
	float x, y;
};

class SpefParser {
public:
	SpefParser();
	bool parse(const std::string &spef_file);
	const std::map<std::string, PinCoordinates> &getPinCoordinates() const;

private:
	std::map<std::string, PinCoordinates> pin_coordinates_;
	std::map<std::string, std::string> name_map_;

	std::string resolveName(const std::string &name) const;
};

} // namespace sta

#endif // FEASTA_SPEF_PARSER_HH