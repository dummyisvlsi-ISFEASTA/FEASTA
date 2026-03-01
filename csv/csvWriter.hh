#ifndef __CSV_WRITER_HH__
#define __CSV_WRITER_HH__

#include "Network.hh"
#include "Sta.hh"
#include <string>

namespace sta {

void writeVerilogCsv(const Network* network, const std::string& filename);
void writeNetworkNodes(const Sta* sta, const std::string& filename, bool useInternalNodes);
void writeNetworkArcs(const Sta* sta, const std::string& filename, bool useInternalNodes);
void writePinPropertiesCsv(const Sta* sta, const std::string& filename, const std::string& spef_file = "");
void writeCellPropertiesCsv(const Sta* sta, const std::string& filename);

// Benchmark function: extracts only Tcl get_property accessible fields
void writeInstancePropertiesBenchmark(const Sta* sta, const std::string& filename);

} 

#endif // __CSV_WRITER_HH__