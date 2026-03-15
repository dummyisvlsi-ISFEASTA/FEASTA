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

%module csvWriter

%{
#include "csv/csvWriter.hh"
#include "Sta.hh"

using namespace sta;
%}

%inline %{

void
write_network_nodes_cmd(const char *filename, bool use_internal_nodes)
{
  Sta *sta = Sta::sta();
  writeNetworkNodes(sta, filename, use_internal_nodes);
}

void
write_network_arcs_cmd(const char *filename, bool use_internal_nodes)
{
  Sta *sta = Sta::sta();
  writeNetworkArcs(sta, filename, use_internal_nodes);
}

void
write_pin_properties_cmd(const char *filename, const char *spef_file)
{
  Sta *sta = Sta::sta();
  std::string spef = spef_file ? spef_file : "";
  writePinPropertiesCsv(sta, filename, spef);
}

void
write_cell_properties_cmd(const char *filename)
{
  Sta *sta = Sta::sta();
  writeCellPropertiesCsv(sta, filename);
}

void
write_instance_properties_benchmark_cmd(const char *filename)
{
  Sta *sta = Sta::sta();
  writeInstancePropertiesBenchmark(sta, filename);
}

%} // inline
