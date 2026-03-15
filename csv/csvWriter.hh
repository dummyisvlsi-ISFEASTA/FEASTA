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

#ifndef FEASTA_CSV_WRITER_HH
#define FEASTA_CSV_WRITER_HH

#include "Network.hh"
#include "Sta.hh"
#include <string>

namespace sta {

void writeVerilogCsv(const Network *network, const std::string &filename);
void writeNetworkNodes(const Sta *sta, const std::string &filename,
                       bool useInternalNodes);
void writeNetworkArcs(const Sta *sta, const std::string &filename,
                      bool useInternalNodes);
void writePinPropertiesCsv(const Sta *sta, const std::string &filename,
                           const std::string &spef_file = "");
void writeCellPropertiesCsv(const Sta *sta, const std::string &filename);
void writeInstancePropertiesBenchmark(const Sta *sta,
                                      const std::string &filename);

} // namespace sta

#endif // FEASTA_CSV_WRITER_HH