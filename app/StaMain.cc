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

#include "StaMain.hh"

#include <tcl.h>
#include <cstdlib>
#include <sys/stat.h>

#include "Machine.hh"
#include "StringUtil.hh"
#include "Vector.hh"
#include "Sta.hh"
#include "csv/csvWriter.hh"
#include "csv/SpefParser.hh"
#include <fstream>
#include <cstring>

namespace sta {

int
parseThreadsArg(int &argc,
		char *argv[])
{
  char *thread_arg = findCmdLineKey(argc, argv, "-threads");
  if (thread_arg) {
    if (stringEqual(thread_arg, "max"))
      return processorCount();
    else if (isDigits(thread_arg))
      return atoi(thread_arg);
    else
      fprintf(stderr,"Warning: -threads must be max or a positive integer.\n");
  }
  return 1;
}

bool
findCmdLineFlag(int &argc,
		char *argv[],
		const char *flag)
{
  for (int i = 1; i < argc; i++) {
    char *arg = argv[i];
    if (stringEq(arg, flag)) {
      // Remove flag from argv.
      for (int j = i + 1; j < argc; j++, i++)
	argv[i] = argv[j];
      argc--;
      argv[argc] = nullptr;
      return true;
    }
  }
  return false;
}

char *
findCmdLineKey(int &argc,
	       char *argv[],
	       const char *key)
{
  for (int i = 1; i < argc; i++) {
    char *arg = argv[i];
    if (stringEq(arg, key) && i + 1 < argc) {
      char *value = argv[i + 1];
      // Remove key and value from argv.
      for (int j = i + 2; j < argc; j++, i++)
	argv[i] = argv[j];
      argc -= 2;
      argv[argc] = nullptr;
      return value;
    }
  }
  return nullptr;
}

// Use overridden version of source to echo cmds and results.
int
sourceTclFile(const char *filename,
	      bool echo,
	      bool verbose,
	      Tcl_Interp *interp)
{
  std::string cmd;
  stringPrint(cmd, "sta::include_file %s %s %s",
	      filename,
	      echo ? "1" : "0",
	      verbose ? "1" : "0");
  int code = Tcl_Eval(interp, cmd.c_str());
  const char *result = Tcl_GetStringResult(interp);
  if (result[0] != '\0')
    printf("%s\n", result);
  return code;
}

void
evalTclInit(Tcl_Interp *interp,
	    const char *inits[])
{
  char *unencoded = unencode(inits);
  if (Tcl_Eval(interp, unencoded) != TCL_OK) {
    // Get a backtrace for the error.
    Tcl_Eval(interp, "$errorInfo");
    const char *tcl_err = Tcl_GetStringResult(interp);
    fprintf(stderr, "Error: TCL init script: %s.\n", tcl_err);
    fprintf(stderr, "       Try deleting TclInitVar.cc and rebuilding.\n");
    exit(0);
  }
  delete [] unencoded;
}

char *
unencode(const char *inits[])
{
  size_t length = 0;
  for (const char **e = inits; *e; e++) {
    const char *init = *e;
    length += strlen(init);
  }
  char *unencoded = new char[length / 3 + 1];
  char *u = unencoded;
  for (const char **e = inits; *e; e++) {
    const char *init = *e;
    size_t init_length = strlen(init);
    for (const char *s = init; s < &init[init_length]; s += 3) {
      char code[4] = {s[0], s[1], s[2], '\0'};
      char ch = atoi(code);
      *u++ = ch;
    }
  }
  *u = '\0';
  return unencoded;
}

// Hack until c++17 filesystem is better supported.
bool
is_regular_file(const char *filename)
{
  struct stat sb;
  return stat(filename, &sb) == 0 && S_ISREG(sb.st_mode);
}

// ===== Custom CSV Export Commands =====

static int
dumpCsvDataCmd(ClientData clientData, Tcl_Interp* interp, int /*argc*/, const char* /*argv*/[])
{
    Sta* sta = reinterpret_cast<Sta*>(clientData);
    Network* network = sta->network();
    if (!network) {
        Tcl_SetResult(interp, const_cast<char*>("Error: Network is not initialized."), TCL_STATIC);
        return TCL_ERROR;
    }
    const char* filename = "design_data.csv";
    writeVerilogCsv(network, filename);
    return TCL_OK;
}

void registerDumpCsvDataCmd(Tcl_Interp* interp, Sta* sta)
{
    Tcl_CreateCommand(interp, "dump_csv_data", dumpCsvDataCmd, sta, nullptr);
}

static int
dumpNetworkNodesCmd(ClientData clientData, Tcl_Interp* interp, int argc, const char* argv[])
{
    Sta* sta = reinterpret_cast<Sta*>(clientData);
    if (!sta) {
        Tcl_SetResult(interp, const_cast<char*>("Error: STA is not initialized."), TCL_STATIC);
        return TCL_ERROR;
    }
    const char* filename = "network_nodes.csv";
    bool useInternalNodes = (argc > 1 && std::strcmp(argv[1], "-use_internal_nodes") == 0);
    writeNetworkNodes(sta, filename, useInternalNodes);
    Tcl_SetResult(interp, const_cast<char*>("Network nodes written to network_nodes.csv"), TCL_STATIC);
    return TCL_OK;
}

static int
dumpNetworkArcsCmd(ClientData clientData, Tcl_Interp* interp, int argc, const char* argv[])
{
    Sta* sta = reinterpret_cast<Sta*>(clientData);
    if (!sta) {
        Tcl_SetResult(interp, const_cast<char*>("Error: STA is not initialized."), TCL_STATIC);
        return TCL_ERROR;
    }
    const char* filename = "network_arcs.csv";
    bool useInternalNodes = (argc > 1 && std::strcmp(argv[1], "-use_internal_nodes") == 0);
    writeNetworkArcs(sta, filename, useInternalNodes);
    Tcl_SetResult(interp, const_cast<char*>("Network arcs written to network_arcs.csv"), TCL_STATIC);
    return TCL_OK;
}

static int
writePinPropertiesCmd(ClientData clientData, Tcl_Interp* interp, int argc, const char* argv[])
{
    Sta* sta = reinterpret_cast<Sta*>(clientData);
    if (!sta) {
        Tcl_SetResult(interp, const_cast<char*>("Error: STA is not initialized."), TCL_STATIC);
        return TCL_ERROR;
    }
    
    // Usage: dump_pin_properties [output_csv] [spef_file]
    const char* filename = "pin_properties.csv";
    std::string spef_file = "";
    
    if (argc > 3) {
        Tcl_SetResult(interp, const_cast<char*>("Usage: dump_pin_properties [<output_csv_file>] [<spef_file>]"), TCL_STATIC);
        return TCL_ERROR;
    }
    if (argc >= 2) {
        filename = argv[1];
    }
    if (argc >= 3) {
        spef_file = argv[2];
    }
    
    writePinPropertiesCsv(sta, filename, spef_file);
    std::string result = std::string("Pin properties written to ") + filename;
    if (!spef_file.empty()) {
        result += " (with coordinates from " + spef_file + ")";
    }
    Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
    return TCL_OK;
}

static int
dumpPinCoordsCmd(ClientData /*clientData*/, Tcl_Interp* interp, int argc, const char* argv[])
{
    if (argc != 3) {
        Tcl_SetResult(interp, const_cast<char*>("Usage: dump_pin_coords <spef_file> <output_csv_file>"), TCL_STATIC);
        return TCL_ERROR;
    }
    SpefParser parser;
    if (!parser.parse(argv[1])) {
        std::string msg = std::string("Error: Failed to parse SPEF file: ") + argv[1];
        Tcl_SetResult(interp, const_cast<char*>(msg.c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
    std::ofstream csv(argv[2]);
    if (!csv.is_open()) {
        std::string msg = std::string("Error: Could not open CSV file: ") + argv[2];
        Tcl_SetResult(interp, const_cast<char*>(msg.c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
    csv << "pin_name,x,y\n";
    for (const auto& p : parser.getPinCoordinates()) {
        csv << p.first << "," << p.second.x << "," << p.second.y << "\n";
    }
    std::string result = std::string("Pin coordinates written to ") + argv[2];
    Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
    return TCL_OK;
}

static int
writeCellPropertiesCmd(ClientData clientData, Tcl_Interp* interp, int argc, const char* argv[])
{
    Sta* sta = reinterpret_cast<Sta*>(clientData);
    if (!sta) {
        Tcl_SetResult(interp, const_cast<char*>("Error: STA is not initialized."), TCL_STATIC);
        return TCL_ERROR;
    }
    const char* filename = (argc > 1) ? argv[1] : "cell_properties.csv";
    try {
        writeCellPropertiesCsv(sta, filename);
        std::string result = std::string("Cell properties written to ") + filename;
        Tcl_SetResult(interp, const_cast<char*>(result.c_str()), TCL_VOLATILE);
        return TCL_OK;
    } catch (...) {
        Tcl_SetResult(interp, const_cast<char*>("Error writing cell properties."), TCL_STATIC);
        return TCL_ERROR;
    }
}

// ===== Cached Pin Coordinates for get_pin_coords command =====
static std::string cached_spef_file_;
static std::map<std::string, PinCoordinates> cached_coords_;

static int
getPinCoordsCmd(ClientData /*clientData*/, Tcl_Interp* interp, int argc, const char* argv[])
{
    if (argc < 3 || argc > 5) {
        Tcl_SetResult(interp, const_cast<char*>("Usage: get_pin_coords <pin_name> <spef_file> [-csv <csv_file>]"), TCL_STATIC);
        return TCL_ERROR;
    }
    
    const char* pin_name = argv[1];
    const char* spef_file = argv[2];
    const char* csv_file = nullptr;
    
    // Check for optional -csv argument
    for (int i = 3; i < argc; i++) {
        if (std::strcmp(argv[i], "-csv") == 0 && i + 1 < argc) {
            csv_file = argv[i + 1];
            break;
        }
    }
    
    // Check if we need to parse the SPEF file (different file or first time)
    if (cached_spef_file_ != spef_file) {
        SpefParser parser;
        if (!parser.parse(spef_file)) {
            std::string msg = std::string("Error: Failed to parse SPEF file: ") + spef_file;
            Tcl_SetResult(interp, const_cast<char*>(msg.c_str()), TCL_VOLATILE);
            return TCL_ERROR;
        }
        // Update cache
        cached_spef_file_ = spef_file;
        cached_coords_ = parser.getPinCoordinates();
        
        std::cerr << "SPEF coordinates cached: " << cached_coords_.size() 
                  << " pins from " << spef_file << std::endl;
    }
    
    // Look up the pin in the cache
    auto it = cached_coords_.find(pin_name);
    if (it != cached_coords_.end()) {
        // Format result
        std::ostringstream oss;
        oss << pin_name << ": x=" << it->second.x << " y=" << it->second.y;
        
        // Export to CSV if requested
        if (csv_file) {
            std::ofstream csv(csv_file, std::ios::app);
            if (csv.is_open()) {
                // Check if file is empty to write header
                csv.seekp(0, std::ios::end);
                if (csv.tellp() == 0) {
                    csv << "pin_name,x,y\n";
                }
                csv << pin_name << "," << it->second.x << "," << it->second.y << "\n";
                csv.close();
                oss << " (appended to " << csv_file << ")";
            } else {
                oss << " (warning: could not write to " << csv_file << ")";
            }
        }
        
        Tcl_SetResult(interp, const_cast<char*>(oss.str().c_str()), TCL_VOLATILE);
        return TCL_OK;
    } else {
        std::string msg = std::string("Pin not found in SPEF: ") + pin_name;
        Tcl_SetResult(interp, const_cast<char*>(msg.c_str()), TCL_VOLATILE);
        return TCL_ERROR;
    }
}

void registerNetworkGraphCmds(Tcl_Interp* interp, Sta* sta)
{
    Tcl_CreateCommand(interp, "dump_network_nodes", dumpNetworkNodesCmd, sta, nullptr);
    Tcl_CreateCommand(interp, "dump_network_arcs", dumpNetworkArcsCmd, sta, nullptr);
    Tcl_CreateCommand(interp, "dump_pin_properties", writePinPropertiesCmd, sta, nullptr);
    Tcl_CreateCommand(interp, "dump_cell_properties", writeCellPropertiesCmd, sta, nullptr);
    Tcl_CreateCommand(interp, "dump_pin_coords", dumpPinCoordsCmd, sta, nullptr);
    Tcl_CreateCommand(interp, "get_pin_coords", getPinCoordsCmd, sta, nullptr);
}

} // namespace
