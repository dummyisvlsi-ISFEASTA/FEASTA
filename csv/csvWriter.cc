// FEASTA: Native feature extraction for OpenSTA
// Exports circuit and timing properties to CSV for ML workflows.

#include "sta/Network.hh"
#include "sta/TimingRole.hh"
#include "sta/ConcreteNetwork.hh"
#include "sta/Liberty.hh"
#include "sta/TimingArc.hh"
#include "sta/TimingModel.hh"
#include "sta/Graph.hh"
#include "sta/Sta.hh"
#include "sta/Corner.hh"
#include "sta/Path.hh"
#include "sta/PathEnd.hh"
#include "sta/PortDirection.hh"
#include "sta/PathAnalysisPt.hh"
#include "sta/Clock.hh"
#include "sta/Search.hh"
#include "sta/Sdc.hh"
#include "sta/SearchPred.hh"
#include "csv/csvWriter.hh"
#include "csv/SpefParser.hh"

#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <set>
#include <functional>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <cstring>

namespace sta {

namespace {

// ---- Formatting utilities ----

std::string csvEscape(const std::string &field) {
  if (field.find(',') == std::string::npos &&
      field.find('"') == std::string::npos &&
      field.find('\n') == std::string::npos)
    return field;
  std::string esc = field;
  size_t pos = 0;
  while ((pos = esc.find('"', pos)) != std::string::npos) {
    esc.insert(pos, "\"");
    pos += 2;
  }
  return "\"" + esc + "\"";
}

std::string safeStr(const char *s) {
  return s ? std::string(s) : "N/A";
}

std::string fmtTimingNs(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF)
    return "N/A";
  float ns = val * 1e9f;
  if (std::abs(ns) < 1e-6f) return "N/A";
  std::ostringstream oss;
  int prec = (std::abs(ns) < 1.0f) ? 6 : 3;
  oss << std::fixed << std::setprecision(prec) << ns;
  return oss.str();
}

// Fixed 3-decimal variant for cell-level delay aggregates
std::string fmtTimingNs3(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF)
    return "N/A";
  float ns = val * 1e9f;
  if (std::abs(ns) < 1e-6f) return "N/A";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << ns;
  return oss.str();
}

// Arc delay variant: empty string for unavailable, "0.000" for near-zero
std::string fmtDelayNs(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF)
    return "";
  float ns = val * 1e9f;
  if (std::abs(ns) < 1e-6f) return "0.000";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << ns;
  return oss.str();
}

std::string fmtCapPf(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF || val <= 0.0f)
    return "N/A";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << (val * 1e12f);
  return oss.str();
}

std::string fmtResOhm(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF || val <= 0.0f)
    return "N/A";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << val;
  return oss.str();
}

std::string fmtPowerPw(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF || val < 0.0f)
    return "N/A";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << (val * 1e12f);
  return oss.str();
}

std::string fmtAreaUm2(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF || val <= 0.0f)
    return "N/A";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << val;
  return oss.str();
}

std::string fmtActivity(float val) {
  if (std::isnan(val) || std::isinf(val) || val == INF || val == -INF || val < 0.0f)
    return "N/A";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << val;
  return oss.str();
}

std::string fmtToggleRateMHz(float activity, float clk_period) {
  if (std::isnan(activity) || activity < 0.0f) return "N/A";
  if (clk_period <= 0.0f) return fmtActivity(activity);
  float mhz = (activity / clk_period) / 1e6f;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << mhz;
  return oss.str();
}

std::string joinSet(const std::set<std::string> &s, int limit = 15) {
  if (s.empty()) return "N/A";
  std::string r;
  int n = 0;
  for (auto it = s.begin(); it != s.end() && n < limit; ++it, ++n) {
    if (n > 0) r += ";";
    r += *it;
  }
  return r;
}

// ---- Hierarchy traversal helpers (used by writeVerilogCsv) ----

void writeInstPins(const Network *network, std::ofstream &csv,
                   std::ofstream &log, Instance *inst,
                   const std::string &hier) {
  const char *nc = network->name(inst);
  std::string iname = nc ? nc : "Unnamed";
  std::string full = hier.empty() ? iname : hier + "/" + iname;
  const Cell *cell = network->cell(inst);

  InstancePinIterator *pi = network->pinIterator(inst);
  if (!pi) { log << "No pin iterator for " << full << "\n"; return; }
  while (Pin *pin = pi->next()) {
    const PortDirection *dir = network->direction(pin);
    const Net *net = network->net(pin);
    csv << safeStr(cell ? network->name(cell) : nullptr) << ","
        << full << ","
        << safeStr(cell ? network->name(cell) : nullptr) << ","
        << safeStr(network->name(pin)) << ","
        << safeStr(dir ? dir->name() : nullptr) << ","
        << safeStr(net ? network->name(net) : nullptr) << "\n";
  }
  delete pi;
}

void traverseForCsv(const Network *network, std::ofstream &csv,
                    std::ofstream &log, Instance *inst,
                    const std::string &hier) {
  writeInstPins(network, csv, log, inst, hier);
  InstanceChildIterator *ci = network->childIterator(inst);
  if (!ci) return;
  while (ci->hasNext()) {
    Instance *child = ci->next();
    const char *nc = network->name(inst);
    std::string h = hier.empty() ? safeStr(nc) : hier + "/" + safeStr(nc);
    traverseForCsv(network, csv, log, child, h);
  }
  delete ci;
}

// ---- Clock domain mapping via graph BFS ----

void buildClockDomainMap(const Sta *sta,
                         std::map<Pin*, std::set<std::string>> &domains,
                         std::map<Pin*, bool> &is_clk_src,
                         std::ofstream &/*log*/) {
  Sta *sm = const_cast<Sta*>(sta);
  Network *net = sta->network();
  Graph *graph = sta->graph();
  Sdc *sdc = sta->sdc();
  if (!net || !sdc) return;

  ClockSeq *clocks = sdc->clocks();
  if (!clocks) return;

  for (Clock *clk : *clocks) {
    if (!clk) continue;
    std::string cn = clk->name() ? std::string(clk->name()) : "unnamed_clock";

    for (const Pin *cp : clk->pins()) {
      if (!cp) continue;
      Pin *mp = const_cast<Pin*>(cp);
      is_clk_src[mp] = true;
      domains[mp].insert(cn);
    }

    if (!graph) continue;
    try {
      SearchPred *pred = sm->search()->searchAdj();
      if (!pred) continue;

      std::set<Vertex*> visited;
      std::queue<Vertex*> q;
      for (const Pin *cp : clk->pins()) {
        if (!cp) continue;
        Vertex *v = graph->pinDrvrVertex(cp);
        if (!v) v = graph->pinLoadVertex(cp);
        if (v && visited.insert(v).second) q.push(v);
      }

      int traced = 0;
      while (!q.empty() && traced < 500000) {
        Vertex *v = q.front(); q.pop();
        traced++;
        if (!v) continue;

        Pin *vp = v->pin();
        if (vp) {
          domains[vp].insert(cn);
          try {
            LibertyPort *lp = net->libertyPort(vp);
            if (lp && lp->isRegClk()) domains[vp].insert(cn);
          } catch (...) {}
        }

        try {
          VertexOutEdgeIterator ei(v, graph);
          while (ei.hasNext()) {
            Edge *e = ei.next();
            if (e && pred->searchThru(e)) {
              Vertex *to = e->to(graph);
              if (to && visited.insert(to).second) q.push(to);
            }
          }
        } catch (...) {}
      }
    } catch (...) {}
  }

  // Propagate clock domain to register clock pins via incoming edges
  try {
    LeafInstanceIterator *li = net->leafInstanceIterator();
    if (!li) return;
    while (li->hasNext()) {
      Instance *inst = li->next();
      if (!inst) continue;
      InstancePinIterator *pi = net->pinIterator(inst);
      if (!pi) continue;
      while (Pin *pin = pi->next()) {
        if (!pin) continue;
        try {
          LibertyPort *lp = net->libertyPort(pin);
          if (!lp || !lp->isRegClk() || !domains[pin].empty()) continue;
          if (graph) {
            Vertex *lv = graph->pinLoadVertex(pin);
            if (lv) {
              VertexInEdgeIterator iei(lv, graph);
              while (iei.hasNext()) {
                Edge *e = iei.next();
                if (!e) continue;
                Vertex *fv = e->from(graph);
                if (!fv) continue;
                Pin *fp = fv->pin();
                if (fp && !domains[fp].empty()) {
                  for (const auto &c : domains[fp]) domains[pin].insert(c);
                  break;
                }
              }
            }
          }
          // Default to sole clock if only one exists
          if (domains[pin].empty() && clocks->size() == 1) {
            Clock *dc = (*clocks)[0];
            if (dc && dc->name()) domains[pin].insert(dc->name());
          }
        } catch (...) {}
      }
      delete pi;
    }
    delete li;
  } catch (...) {}
}

} // anonymous namespace


// ---- writeVerilogCsv ----

void writeVerilogCsv(const Network *network, const std::string &filename) {
  std::ofstream csv(filename);
  std::ofstream log("log.txt", std::ios::app);
  if (!csv.is_open()) { log << "Cannot open " << filename << "\n"; return; }

  csv << "Module,Instance,CellType,PinName,Direction,Net\n";
  Instance *top = network->topInstance();
  if (!top) { log << "Top instance is null\n"; return; }
  traverseForCsv(network, csv, log, top, "");
}


// ---- writeNetworkNodes ----

void writeNetworkNodes(const Sta *sta, const std::string &filename,
                       bool useInternalNodes) {
  std::ofstream csv(filename);
  std::ofstream log("/tmp/nodes_log.txt", std::ios::app);
  if (!csv.is_open()) { log << "Cannot open " << filename << "\n"; return; }

  csv << "Name,InstanceName,PinName,Direction,IsPort,Type,"
         "IsClockNetwork\n";

  Network *network = sta->network();
  Instance *top = network->topInstance();
  if (!top) { log << "Top instance is null\n"; return; }

  // Initialize timing engine
  try {
    Sta *sm = const_cast<Sta*>(sta);
    sm->ensureGraph();
    sm->ensureLevelized();
    sm->searchPreamble();
    sm->search()->findAllArrivals();
    sm->ensureClkNetwork();
  } catch (...) {
    log << "Warning: could not initialize clock network\n";
  }

  // BFS to find clock network pins using hierarchical string names
  std::unordered_set<std::string> clockNodes;
  try {
    const Sdc *sdc = sta->sdc();
    if (sdc) {
      std::queue<std::pair<const Pin*, std::string>> bfs;
      for (Clock *clk : const_cast<Sdc*>(sdc)->clks()) {
        for (const Pin *src : clk->pins()) {
          Instance *pi = network->instance(src);
          std::string ip;
          if (pi && pi != top) {
            const char *pc = network->pathName(pi);
            if (pc) ip = pc;
          }
          std::string pn = safeStr(network->name(src));
          std::string nn = ip.empty() ? pn : ip + ":" + pn;
          if (clockNodes.insert(nn).second)
            bfs.push({src, ip});
        }
      }

      int limit = 500000;
      while (!bfs.empty() && limit-- > 0) {
        auto [pin, instPath] = bfs.front(); bfs.pop();

        PinConnectedPinIterator *ci = network->connectedPinIterator(pin);
        if (!ci) continue;
        while (ci->hasNext()) {
          const Pin *cp = ci->next();
          Instance *cInst = network->instance(cp);
          std::string cip;
          if (cInst && cInst != top) {
            const char *pc = network->pathName(cInst);
            if (pc) cip = pc;
          }
          std::string cpn = safeStr(network->name(cp));
          std::string cnn = cip.empty() ? cpn : cip + ":" + cpn;
          if (!clockNodes.insert(cnn).second) continue;

          const PortDirection *cd = network->direction(cp);
          if (cd && cd->isInput()) {
            LibertyPort *lp = network->libertyPort(cp);
            bool isClkIn = lp && (lp->isRegClk() || lp->isClockGateClock()
                                  || lp->isClock());
            if (isClkIn && cInst) {
              // Follow clock through buffers, inverters, ICGs
              InstancePinIterator *ipi = network->pinIterator(cInst);
              if (ipi) {
                while (ipi->hasNext()) {
                  Pin *op = ipi->next();
                  const PortDirection *od = network->direction(op);
                  if (!od || !od->isOutput()) continue;
                  LibertyPort *olp = network->libertyPort(op);
                  bool isClkOut = olp && olp->isClockGateOut();
                  if (!isClkOut && lp) {
                    LibertyCell *lc = lp->libertyCell();
                    if (lc) isClkOut = lc->isBuffer() || lc->isInverter();
                  }
                  if (isClkOut) {
                    std::string opn = safeStr(network->name(op));
                    std::string onn = cip.empty() ? opn : cip + ":" + opn;
                    if (clockNodes.insert(onn).second)
                      bfs.push({op, cip});
                  }
                }
                delete ipi;
              }
            }
          }
          if (cInst && network->isHierarchical(cInst))
            bfs.push({cp, cip});
        }
        delete ci;
      }
    }
    log << "Clock network pins found: " << clockNodes.size() << "\n";
  } catch (...) {
    log << "Warning: could not build clock network set\n";
  }

  // Pin-to-node-name map (also written to file for arc creation)
  std::unordered_map<Pin*, std::string> pinNodeMap;

  auto processPin = [&](Pin *pin, const std::string &instName, bool isPort) {
    std::string pn = safeStr(network->name(pin));
    const PortDirection *dir = network->direction(pin);
    std::string direction = dir ? (dir->name() ? dir->name() : "unknown")
                                : "unknown";
    if (!useInternalNodes && direction == "internal") return;

    std::string type = "unknown";
    if (isPort) {
      type = "port";
    } else {
      Instance *pi = network->instance(pin);
      if (pi) {
        Cell *c = network->cell(pi);
        type = network->isLeaf(c) ? "leaf" : "hierarchical";
      }
    }

    std::string node = (instName == "top" || instName.empty())
                       ? pn : instName + ":" + pn;
    bool isClock = clockNodes.count(node) || sta->isClock(pin);
    pinNodeMap[pin] = node;

    csv << csvEscape(node) << "," << csvEscape(instName) << ","
        << csvEscape(pn) << "," << csvEscape(direction) << ","
        << (isPort ? "true" : "false") << "," << csvEscape(type) << ","
        << (isClock ? "true" : "false") << "\n";
  };

  // Top-level ports
  InstancePinIterator *tpi = network->pinIterator(top);
  if (tpi) {
    while (Pin *pin = tpi->next()) processPin(pin, "top", true);
    delete tpi;
  }

  // Recursive instance traversal
  std::function<void(Instance*, const std::string&)> traverse;
  traverse = [&](Instance *inst, const std::string &hier) {
    std::string full;
    if (inst != top) {
      std::string in = safeStr(network->name(inst));
      full = hier.empty() ? in : hier + "/" + in;
      InstancePinIterator *pi = network->pinIterator(inst);
      if (pi) {
        while (Pin *pin = pi->next()) processPin(pin, full, false);
        delete pi;
      }
    }
    InstanceChildIterator *ci = network->childIterator(inst);
    if (ci) {
      while (ci->hasNext()) traverse(ci->next(), full);
      delete ci;
    }
  };
  traverse(top, "");

  // Persist pin-to-node mapping for external tools
  std::ofstream mapFile("pin_node_map.dat");
  if (mapFile.is_open()) {
    for (const auto &p : pinNodeMap)
      mapFile << reinterpret_cast<uintptr_t>(p.first) << ","
              << p.second << "\n";
  }
  log << "Node extraction completed.\n";
}


// ---- writeNetworkArcs ----

void writeNetworkArcs(const Sta *sta, const std::string &filename,
                      bool useInternalNodes) {
  std::ofstream csv(filename);
  std::ofstream log("arcs_log.txt", std::ios::app);
  std::unordered_set<std::string> seen;

  if (!csv.is_open()) { log << "Cannot open " << filename << "\n"; return; }

  csv << "Source,Sink,NetName,Connection,"
        "Delay_Min_RR,Delay_Min_RF,Delay_Min_FR,Delay_Min_FF,"
        "Delay_Max_RR,Delay_Max_RF,Delay_Max_FR,Delay_Max_FF,ArcType\n";

  Network *network = sta->network();
  if (!network) return;
  Instance *top = network->topInstance();
  if (!top) return;

  // Build in-memory pin-to-node-name map
  std::unordered_map<uintptr_t, std::string> pinMap;
  {
    auto reg = [&](Pin *pin, const std::string &inst) {
      const PortDirection *dir = network->direction(pin);
      std::string d = dir ? (dir->name() ? dir->name() : "unknown") : "unknown";
      if (!useInternalNodes && d == "internal") return;
      pinMap[reinterpret_cast<uintptr_t>(pin)] =
          (inst == "top" || inst.empty()) ? safeStr(network->name(pin)) : inst + ":" + safeStr(network->name(pin));
    };

    InstancePinIterator *tpi = network->pinIterator(top);
    if (tpi) {
      while (Pin *p = tpi->next()) reg(p, "top");
      delete tpi;
    }

    std::function<void(Instance*, const std::string&)> walk;
    walk = [&](Instance *inst, const std::string &hier) {
      std::string full;
      if (inst != top) {
        std::string in = safeStr(network->name(inst));
        full = hier.empty() ? in : hier + "/" + in;
        InstancePinIterator *pi = network->pinIterator(inst);
        if (pi) {
          while (Pin *p = pi->next()) reg(p, full);
          delete pi;
        }
      }
      InstanceChildIterator *ci = network->childIterator(inst);
      if (ci) {
        while (ci->hasNext()) walk(ci->next(), full);
        delete ci;
      }
    };
    walk(top, "");
    log << "Pin map: " << pinMap.size() << " entries\n";
  }

  auto nodeName = [&](const Pin *pin) -> std::string {
    auto it = pinMap.find(reinterpret_cast<uintptr_t>(pin));
    return it != pinMap.end() ? it->second : "";
  };

  // Helpers for min/max delay extraction across rise/fall transition pairs.
  auto trIdx = [](const RiseFall *from, const RiseFall *to) -> int {
    if (!from || !to) return -1;
    int fi = (from == RiseFall::rise()) ? 0 : 1;
    int ti = (to == RiseFall::rise()) ? 0 : 1;
    return fi * 2 + ti; // 0:RR 1:RF 2:FR 3:FF
  };

  auto getAllEdgeDelays = [&](const std::vector<Edge*> &edges) -> std::string {
    float mn[4] = {1e30f, 1e30f, 1e30f, 1e30f};
    float mx[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    bool fmn[4] = {}, fmx[4] = {};
    Sta *sm = const_cast<Sta*>(sta);

    for (Edge *edge : edges) {
      if (!edge) continue;
      TimingArcSet *as = edge->timingArcSet();
      for (const Corner *corner : *sta->corners()) {
        for (const MinMax *mm : {MinMax::min(), MinMax::max()}) {
          DcalcAnalysisPt *ap = corner->findDcalcAnalysisPt(mm);
          if (!ap) continue;
          if (as && !as->arcs().empty()) {
            for (TimingArc *arc : as->arcs()) {
              float d = sm->arcDelay(edge, arc, ap);
              if (std::isnan(d)) continue;
              int ti = (arc->fromEdge() && arc->toEdge())
                       ? trIdx(arc->fromEdge()->asRiseFall(),
                               arc->toEdge()->asRiseFall())
                       : -1;
              if (ti < 0) continue;
              if (mm == MinMax::min()) {
                if (!fmn[ti] || d < mn[ti]) { mn[ti] = d; fmn[ti] = true; }
              } else {
                if (!fmx[ti] || d > mx[ti]) { mx[ti] = d; fmx[ti] = true; }
              }
            }
          } else {
            float d = sm->arcDelay(edge, nullptr, ap);
            if (std::isnan(d)) continue;
            for (int i = 0; i < 4; i++) {
              if (mm == MinMax::min()) {
                if (!fmn[i] || d < mn[i]) { mn[i] = d; fmn[i] = true; }
              } else {
                if (!fmx[i] || d > mx[i]) { mx[i] = d; fmx[i] = true; }
              }
            }
          }
        }
      }
    }

    std::string r;
    for (int i = 0; i < 4; i++)
      r += (fmn[i] ? fmtDelayNs(mn[i]) : std::string("N/A")) + ",";
    for (int i = 0; i < 4; i++)
      r += (fmx[i] ? fmtDelayNs(mx[i]) : std::string("N/A"))
           + (i < 3 ? "," : "");
    return r;
  };

  // Single-edge wrapper.
  auto getEdgeDelays = [&](Edge *edge) -> std::string {
    if (!edge) return "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A";
    return getAllEdgeDelays({edge});
  };

  const int MAX_CONN = 1000000;
  int connCount = 0;

  auto writeArc = [&](const std::string &src, const std::string &sink,
                      const std::string &netName, const std::string &connType,
                      const std::string &delays, const std::string &arcType) {
    std::string id = src + "-" + sink;
    if (!seen.insert(id).second) return;
    csv << csvEscape(src) << "," << csvEscape(sink) << ","
        << csvEscape(netName) << "," << connType << ","
        << delays << "," << arcType << "\n";
    connCount++;
  };

  // Ensure the timing graph is built before any pass uses it.
  Graph *graph = sta->graph();
  if (!graph) {
    const_cast<Sta*>(sta)->ensureGraph();
    graph = sta->graph();
  }

  // Pass 1: Net arcs (driver-to-load connections).
  log << "Processing net arcs...\n";
  int netArcCount = 0;
  NetIterator *ni = network->netIterator(top);
  if (ni) {
    while (Net *net = ni->next()) {
      if (connCount >= MAX_CONN) break;
      const char *nn = network->name(net);
      std::string netName = nn ? nn : "UnnamedNet";

      PinSet *drivers = network->drivers(net);
      if (!drivers || drivers->empty()) continue;

      for (const Pin *drv : *drivers) {
        std::string sn = nodeName(drv);
        if (sn.empty()) continue;
        const PortDirection *dd = network->direction(const_cast<Pin*>(drv));
        if (!useInternalNodes && dd && dd->name() &&
            std::string(dd->name()) == "internal")
          continue;

        PinConnectedPinIterator *ci = network->connectedPinIterator(net);
        if (!ci) continue;
        int pinCnt = 0;
        while (const Pin *ld = ci->next()) {
          if (++pinCnt > 5000 || connCount >= MAX_CONN) break;
          if (ld == drv) continue;
          std::string tn = nodeName(ld);
          if (tn.empty()) continue;
          const PortDirection *ld_d = network->direction(const_cast<Pin*>(ld));
          if (!useInternalNodes && ld_d && ld_d->name() &&
              std::string(ld_d->name()) == "internal")
            continue;

          // Find graph edge for delay.
          Edge *target = nullptr;
          if (graph) {
            Vertex *fv = graph->pinDrvrVertex(const_cast<Pin*>(drv));
            if (!fv) fv = graph->pinLoadVertex(const_cast<Pin*>(drv));
            Vertex *tv = graph->pinLoadVertex(const_cast<Pin*>(ld));
            if (!tv) tv = graph->pinDrvrVertex(const_cast<Pin*>(ld));
            if (fv && tv) {
              VertexOutEdgeIterator ei(fv, graph);
              while (ei.hasNext()) {
                Edge *e = ei.next();
                if (e->to(graph) == tv) { target = e; break; }
              }
            }
          }
          writeArc(sn, tn, netName, "net",
                   target ? getEdgeDelays(target) : "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A",
                   "net_arc");
          netArcCount++;
        }
        delete ci;
      }
    }
    delete ni;
  }
  log << "Net arcs: " << netArcCount << "\n";

  // Pass 2: Intra-cell timing arcs from liberty models.
  log << "Processing cell timing arcs...\n";
  int cellArcCount = 0;

  std::function<void(Instance*)> processCellArcs = [&](Instance *inst) {
    if (inst == top) return;
    Cell *cell = network->cell(inst);
    if (!cell) return;
    LibertyCell *lc = sta->networkReader()->libertyCell(cell);
    if (!lc) return;

    // One output row per (from-pin, to-pin) pair per instance.
    std::unordered_set<std::string> cellSeen;

    for (TimingArcSet *as : lc->timingArcSets()) {
      LibertyPort *fp = as->from(), *tp = as->to();
      if (!fp || !tp) continue;

      // Only combinational and clock-to-Q arcs.
      bool valid = false;
      for (TimingArc *arc : as->arcs()) {
        const TimingRole *role = arc->role();
        if (role == TimingRole::combinational() ||
            role == TimingRole::regClkToQ()) {
          valid = true;
          break;
        }
      }
      if (!valid) continue;

      Pin *fpin = network->findPin(inst, fp->name());
      Pin *tpin = network->findPin(inst, tp->name());
      if (!fpin || !tpin) continue;

      std::string sn = nodeName(fpin), tn = nodeName(tpin);
      if (sn.empty() || tn.empty()) continue;

      // Skip duplicate (from, to) pairs for this instance.
      std::string pairKey = sn + "-" + tn;
      if (!cellSeen.insert(pairKey).second) continue;

      // Collect all graph edges between this pin pair.
      std::vector<Edge*> targets;
      if (graph) {
        Vertex *fv = graph->pinDrvrVertex(fpin);
        if (!fv) fv = graph->pinLoadVertex(fpin);
        Vertex *tv = graph->pinLoadVertex(tpin);
        if (!tv) tv = graph->pinDrvrVertex(tpin);
        if (fv && tv) {
          VertexOutEdgeIterator ei(fv, graph);
          while (ei.hasNext()) {
            Edge *e = ei.next();
            if (e->to(graph) == tv) targets.push_back(e);
          }
        }
      }
      writeArc(sn, tn, "cell_internal", "internal",
               targets.empty() ? "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
                               : getAllEdgeDelays(targets),
               "timing_arc");
      cellArcCount++;
      if (connCount >= MAX_CONN) return;
    }
  };

  std::function<void(Instance*)> walkInst = [&](Instance *inst) {
    processCellArcs(inst);
    InstanceChildIterator *ci = network->childIterator(inst);
    if (ci) {
      while (ci->hasNext()) walkInst(ci->next());
      delete ci;
    }
  };
  walkInst(top);
  log << "Cell timing arcs: " << cellArcCount << "\n";

  // Pass 3: Internal wire arcs from the timing graph (only with useInternalNodes).
  if (useInternalNodes && graph) {
    log << "Processing internal wire arcs...\n";
    int wireArcCount = 0;
    VertexIterator vi(graph);
    while (vi.hasNext() && connCount < MAX_CONN) {
      Vertex *fv = vi.next();
      const Pin *fp = fv->pin();
      std::string sn = nodeName(fp);
      if (sn.empty()) continue;

      VertexOutEdgeIterator ei(fv, graph);
      while (ei.hasNext() && connCount < MAX_CONN) {
        Edge *e = ei.next();
        if (e->role() != TimingRole::wire()) continue;
        const Pin *tp = e->to(graph)->pin();
        std::string tn = nodeName(tp);
        if (tn.empty()) continue;
        if (network->isTopLevelPort(const_cast<Pin*>(fp)) ||
            network->isTopLevelPort(const_cast<Pin*>(tp)))
          continue;
        writeArc(sn, tn, "internal_net", "internal",
                 getEdgeDelays(e), "internal_arc");
        wireArcCount++;
      }
    }
    log << "Internal wire arcs: " << wireArcCount << "\n";
  }

  log << "Total arcs written: " << connCount << "\n";
}


// ---- writePinPropertiesCsv ----

void writePinPropertiesCsv(const Sta *sta, const std::string &filename,
                           const std::string &spef_file) {
  std::ofstream csv(filename);
  std::ofstream log("pin_properties_debug.txt");
  if (!csv.is_open()) { log << "Cannot open " << filename << "\n"; return; }

  csv << "FullName,Direction,IsPort,IsHierarchical,IsRegisterClock,"
         "LibPinName,SlewRise_ns,SlewFall_ns,SlewMinRise_ns,SlewMinFall_ns,"
         "SlackRise_ns,SlackFall_ns,SlackWorst_ns,"
         "SlackMinRise_ns,SlackMinFall_ns,SlackMinWorst_ns,"
         "IsClock,ClockNames,"
         "Capacitance_pf,DriveResistance_ohm,"
         "Activity,StaticProbability,ToggleRate_MHz,ActivityOrigin,"
         "CoordX_um,CoordY_um\n";

  // Parse SPEF for pin coordinates if provided
  std::map<std::string, PinCoordinates> spefCoords;
  if (!spef_file.empty()) {
    SpefParser parser;
    if (parser.parse(spef_file)) {
      spefCoords = parser.getPinCoordinates();
      log << "Loaded " << spefCoords.size() << " coordinates from SPEF\n";
    }
  }

  if (!sta) return;
  Sta *sm = const_cast<Sta*>(sta);
  Network *network = sta->network();
  Graph *graph = sta->graph();
  Sdc *sdc = sta->sdc();
  if (!network) return;

  Instance *top = network->topInstance();
  if (!top) return;

  if (!graph) {
    try { sm->ensureGraph(); graph = sta->graph(); } catch (...) {}
  }
  try { sm->search(); } catch (...) {}

  // Build clock domain mapping
  std::map<Pin*, std::set<std::string>> pinClkDomains;
  std::map<Pin*, bool> pinIsClkSrc;
  buildClockDomainMap(sta, pinClkDomains, pinIsClkSrc, log);

  // Default clock period for toggle rate computation
  float clkPeriod = 10e-9f;
  try {
    if (sdc) {
      ClockSeq *clks = sdc->clocks();
      if (clks && !clks->empty() && (*clks)[0])
        clkPeriod = (*clks)[0]->period();
    }
  } catch (...) {}

  // Pin processing lambda
  auto processPin = [&](Pin *pin, const std::string &fullName, bool isPort) {
    if (!pin) return;
    try {
      std::string pname = safeStr(network->name(pin));

      // Direction
      std::string dir = "unknown";
      const PortDirection *pd = network->direction(pin);
      if (pd) {
        if (pd->isInput()) dir = "input";
        else if (pd->isOutput()) dir = "output";
        else if (pd->isBidirect()) dir = "bidirect";
        else if (pd->isTristate()) dir = "tristate";
        else if (pd->isPower()) dir = "power";
        else if (pd->isGround()) dir = "ground";
        else if (pd->isPowerGround()) dir = "pg_pin";
        else if (pd->isInternal()) dir = "internal";
        else {
          LibertyPort *lp = network->libertyPort(pin);
          dir = (lp && lp->isClock()) ? "clock" : "pg_pin";
        }
      }

      bool isHier = false;
      try { isHier = network->isHierarchical(pin); } catch (...) {}
      bool isRegClk = false;
      std::string libPin = "N/A", libPort = "N/A";
      std::string cap = "N/A", drvRes = "N/A";

      try {
        LibertyPort *lp = network->libertyPort(pin);
        if (lp) {
          libPin = safeStr(lp->name());
          libPort = libPin;
          isRegClk = lp->isRegClk();
          try { cap = fmtCapPf(lp->capacitance()); } catch (...) {}
          try { drvRes = fmtResOhm(lp->driveResistance()); } catch (...) {}
        }
      } catch (...) {}

      // Timing: slew and slack
      std::string sr = "N/A", sf = "N/A", smr = "N/A", smf = "N/A";
      std::string skr = "N/A", skf = "N/A", skw = "N/A";
      std::string skmr = "N/A", skmf = "N/A", skmw = "N/A";

      if (graph && sm) {
        try {
          Vertex *dv = graph->pinDrvrVertex(pin);
          Vertex *lv = graph->pinLoadVertex(pin);
          Vertex *v = dv ? dv : lv;
          if (v) {
            try {
              sr = fmtTimingNs(sm->vertexSlew(v, RiseFall::rise(), MinMax::max()));
              sf = fmtTimingNs(sm->vertexSlew(v, RiseFall::fall(), MinMax::max()));
            } catch (...) {}
            try {
              smr = fmtTimingNs(sm->vertexSlew(v, RiseFall::rise(), MinMax::min()));
              smf = fmtTimingNs(sm->vertexSlew(v, RiseFall::fall(), MinMax::min()));
            } catch (...) {}
            try {
              skr = fmtTimingNs(sm->vertexSlack(v, RiseFall::rise(), MinMax::max()));
              skf = fmtTimingNs(sm->vertexSlack(v, RiseFall::fall(), MinMax::max()));
              skw = fmtTimingNs(sm->vertexSlack(v, MinMax::max()));
            } catch (...) {}
            try {
              skmr = fmtTimingNs(sm->vertexSlack(v, RiseFall::rise(), MinMax::min()));
              skmf = fmtTimingNs(sm->vertexSlack(v, RiseFall::fall(), MinMax::min()));
              skmw = fmtTimingNs(sm->vertexSlack(v, MinMax::min()));
            } catch (...) {}
          }
        } catch (...) {}
      }

      // Clock info
      bool isClock = false;
      std::string clockNames = "N/A", clockDomains = "N/A";
      if (pinIsClkSrc.count(pin)) isClock = pinIsClkSrc[pin];
      if (pinClkDomains.count(pin) && !pinClkDomains[pin].empty()) {
        clockNames = joinSet(pinClkDomains[pin]);
        clockDomains = clockNames;
      }

      // Activity estimation based on pin role
      std::string act = "N/A", sprob = "N/A", trate = "N/A";
      std::string aorigin = "unknown";
      float actVal = 0, spVal = 0.5f;
      bool actFound = false;

      if (isClock) {
        actVal = 1.0f; spVal = 0.5f; actFound = true;
        aorigin = "clock_source";
      } else if (isRegClk) {
        actVal = 1.0f; spVal = 0.5f; actFound = true;
        aorigin = "register_clock";
      } else if (pinClkDomains.count(pin) && !pinClkDomains[pin].empty()) {
        actVal = 1.0f; spVal = 0.5f; actFound = true;
        aorigin = "clock_network";
      } else if (dir == "input" && isPort) {
        actVal = 0.2f; spVal = 0.5f; actFound = true;
        aorigin = "input_port";
      } else if (dir == "output" && isPort) {
        actVal = 0.15f; spVal = 0.4f; actFound = true;
        aorigin = "output_port";
      } else if (!isPort) {
        if (dir == "output") { actVal = 0.12f; spVal = 0.35f; aorigin = "internal_output"; }
        else if (dir == "input") { actVal = 0.15f; spVal = 0.45f; aorigin = "internal_input"; }
        else { actVal = 0.1f; spVal = 0.4f; aorigin = "internal_default"; }
        actFound = true;
      }
      if (actFound) {
        act = fmtActivity(actVal);
        sprob = fmtActivity(spVal);
        trate = fmtToggleRateMHz(actVal, clkPeriod);
      }

      // SPEF coordinate lookup
      std::string cx = "N/A", cy = "N/A";
      std::string key3 = fullName;
      std::replace(key3.begin(), key3.end(), '/', ':');
      if (spefCoords.count(pname)) {
        cx = std::to_string(spefCoords[pname].x);
        cy = std::to_string(spefCoords[pname].y);
      } else if (spefCoords.count(key3)) {
        cx = std::to_string(spefCoords[key3].x);
        cy = std::to_string(spefCoords[key3].y);
      }

      csv << fullName << "," << dir << ","
          << (isPort ? "true" : "false") << ","
          << (isHier ? "true" : "false") << ","
          << (isRegClk ? "true" : "false") << ","
          << libPin << ","
          << sr << "," << sf << "," << smr << "," << smf << ","
          << skr << "," << skf << "," << skw << ","
          << skmr << "," << skmf << "," << skmw << ","
          << (isClock ? "true" : "false") << ","
          << clockNames << ","
          << cap << "," << drvRes << ","
          << act << "," << sprob << "," << trate << "," << aorigin << ","
          << cx << "," << cy << "\n";
    } catch (...) {
      log << "Exception processing pin " << fullName << "\n";
    }
  };

  // Process top-level ports
  try {
    InstancePinIterator *tpi = network->pinIterator(top);
    if (tpi) {
      while (Pin *pin = tpi->next())
        if (pin) processPin(pin, "top/" + safeStr(network->name(pin)), true);
      delete tpi;
    }
  } catch (...) {}

  // Process leaf instances
  try {
    LeafInstanceIterator *li = network->leafInstanceIterator();
    if (li) {
      while (li->hasNext()) {
        Instance *inst = li->next();
        if (!inst || inst == top) continue;
        std::string iname = safeStr(network->name(inst));
        try {
          InstancePinIterator *pi = network->pinIterator(inst);
          if (pi) {
            while (Pin *pin = pi->next())
              if (pin) processPin(pin, iname + "/" + safeStr(network->name(pin)), false);
            delete pi;
          }
        } catch (...) {}
      }
      delete li;
    }
  } catch (...) {}

  // Process hierarchical instance pins
  std::function<void(Instance*, const std::string&)> procHier;
  procHier = [&](Instance *parent, const std::string &path) {
    try {
      InstanceChildIterator *ci = network->childIterator(parent);
      if (!ci) return;
      while (ci->hasNext()) {
        Instance *child = ci->next();
        if (!child || !network->isHierarchical(child)) continue;
        std::string cp = path.empty() ? safeStr(network->name(child))
                                      : path + "/" + safeStr(network->name(child));
        try {
          InstancePinIterator *pi = network->pinIterator(child);
          if (pi) {
            while (Pin *pin = pi->next())
              if (pin) processPin(pin, cp + "/" + safeStr(network->name(pin)), false);
            delete pi;
          }
        } catch (...) {}
        procHier(child, cp);
      }
      delete ci;
    } catch (...) {}
  };
  procHier(top, "");
  csv.close();
}


// ---- writeCellPropertiesCsv ----

void writeCellPropertiesCsv(const Sta *sta, const std::string &filename) {
  std::ofstream csv(filename);
  std::ofstream log("cell_properties_debug.txt");
  if (!csv.is_open()) { log << "Cannot open " << filename << "\n"; return; }

  csv << "FullInstanceName,LibertyCell,Library,"
         "CellType,IsBuffer,IsInverter,IsMemory,IsMacro,IsHierarchical,"
         "Area_um2,LeakagePower_pW,PinCount,InputPinCount,OutputPinCount,"
         "BiDirectPinCount,ClockPinCount,DataPinCount,AsyncPinCount,"
         "FanoutLoad,FaninLoad,IsCombinational,IsSequential,IsClockGating,"
         "SetupTime_ns,HoldTime_ns,"
         "TimingArcCount,HasClockInput,ClockDomains,"
         "SwitchingPower_pW,InternalPower_pW,TotalPower_pW,"
         "Process,Voltage_V,Temperature_C\n";

  if (!sta) return;
  Network *network = sta->network();
  if (!network) return;
  Instance *top = network->topInstance();
  if (!top) return;

  Sta *sm = const_cast<Sta*>(sta);
  Graph *graph = sta->graph();


  if (!graph) {
    try { sm->ensureGraph(); graph = sta->graph(); } catch (...) {}
  }
  try { sm->search(); } catch (...) {}

  // Clock domain mapping
  std::map<Pin*, std::set<std::string>> pinClkDomains;
  std::map<Pin*, bool> pinIsClkSrc;
  buildClockDomainMap(sta, pinClkDomains, pinIsClkSrc, log);



  // Clock info per instance
  auto getClockInfo = [&](Instance *inst) -> std::tuple<bool, std::string> {
    bool hasClk = false;
    std::set<std::string> clkNames;
    if (!inst) return {false, "N/A"};
    try {
      InstancePinIterator *pi = network->pinIterator(inst);
      if (!pi) return {false, "N/A"};
      int pc = 0;
      while (pi->hasNext() && pc < 50) {
        pc++;
        try {
          Pin *pin = pi->next();
          if (!pin) continue;
          LibertyPort *lp = network->libertyPort(pin);
          if (!lp || !lp->isRegClk()) continue;
          hasClk = true;
          if (pinClkDomains.count(pin))
            for (const auto &c : pinClkDomains[pin]) clkNames.insert(c);
        } catch (...) {}
      }
      delete pi;
    } catch (...) {}
    return {hasClk, joinSet(clkNames)};
  };

  // Main cell processing lambda
  auto processCell = [&](Instance *inst, const std::string &fullName) {
    if (!inst || inst == top) return;
    try {
      const char *inc = network->name(inst);
      if (!inc) return;
      std::string iname(inc);

      Cell *cell = network->cell(inst);
      LibertyCell *lc = nullptr;
      std::string cellName = "N/A", libName = "N/A";
      if (cell) {
        cellName = safeStr(network->name(cell));
        lc = network->libertyCell(cell);
        if (lc) {
          ConcreteLibrary *lib = lc->library();
          if (lib && lib->name()) libName = lib->name();
        }
      }

      std::string cellType = "unknown";
      bool isBuf = false, isInv = false, isMem = false;
      bool isMacro = false, isHier = false;
      std::string area = "N/A", leakPwr = "N/A";

      if (lc) {
        isBuf = lc->isBuffer(); isInv = lc->isInverter();
        isMem = lc->isMemory(); isMacro = lc->isMacro();
        if (isBuf) cellType = "buffer";
        else if (isInv) cellType = "inverter";
        else if (isMem) cellType = "memory";
        else if (isMacro) cellType = "macro";
        else cellType = "combinational";

        area = fmtAreaUm2(lc->area());
        float lv = 0; bool le = false;
        lc->leakagePower(lv, le);
        if (le) leakPwr = fmtPowerPw(lv);
      }
      try { isHier = network->isHierarchical(inst); } catch (...) {}

      // Pin counting
      int pinCnt = 0, inCnt = 0, outCnt = 0, biCnt = 0;
      int clkCnt = 0, dataCnt = 0, asyncCnt = 0;
      float inCap = 0, outCap = 0;
      try {
        InstancePinIterator *pi = network->pinIterator(inst);
        if (pi) {
          int lim = 0;
          while (pi->hasNext() && lim < 100) {
            lim++;
            try {
              Pin *pin = pi->next();
              if (!pin) continue;
              const PortDirection *d = network->direction(pin);
              if (d && d->isPowerGround()) continue;
              pinCnt++;
              if (d && d->isInput()) {
                inCnt++;
                bool isClkP = false, isAsync = false;
                try {
                  LibertyPort *lp = network->libertyPort(pin);
                  if (lp) {
                    isClkP = lp->isRegClk() || lp->isClockGateClock() || lp->isClock();
                    if (!isClkP && lp->isCheckClk()) isAsync = true;
                    float c = lp->capacitance();
                    if (!std::isnan(c) && c > 0 && c < 1e-6) inCap += c;
                  }
                } catch (...) {}
                if (isClkP) clkCnt++; else if (isAsync) asyncCnt++;
              } else if (d && d->isOutput()) {
                outCnt++;
                try {
                  LibertyPort *lp = network->libertyPort(pin);
                  if (lp) {
                    float c = lp->capacitance();
                    if (!std::isnan(c) && c > 0 && c < 1e-6) outCap += c;
                  }
                } catch (...) {}
              } else if (d && d->isBidirect()) biCnt++;
            } catch (...) {}
          }
          delete pi;
        }
      } catch (...) {}

      pinCnt = inCnt + outCnt + biCnt;
      dataCnt = pinCnt - clkCnt - asyncCnt;

      // Fanout/fanin load counting
      int foLoad = 0, fiLoad = 0;
      try {
        InstancePinIterator *fi = network->pinIterator(inst);
        if (fi) {
          while (fi->hasNext()) {
            Pin *pin = fi->next();
            if (!pin) continue;
            Net *pn = network->net(pin);
            if (!pn) continue;
            const PortDirection *pd = network->direction(pin);
            if (!pd || pd->isPowerGround()) continue;
            if (pd->isOutput() || pd->isBidirect()) {
              NetConnectedPinIterator *cp = network->connectedPinIterator(pn);
              if (cp) {
                while (cp->hasNext()) {
                  const Pin *p = cp->next();
                  if (p != pin && network->isLoad(p)) foLoad++;
                }
                delete cp;
              }
            } else if (pd->isInput()) {
              NetConnectedPinIterator *cp = network->connectedPinIterator(pn);
              if (cp) {
                while (cp->hasNext()) {
                  const Pin *p = cp->next();
                  if (p != pin && network->isDriver(p)) fiLoad++;
                }
                delete cp;
              }
            }
          }
          delete fi;
        }
      } catch (...) {}



      bool isComb = true, isSeq = false, isClkGate = false;
      if (lc) try { isClkGate = lc->isClockGate(); } catch (...) {}
      if (clkCnt > 0 || isClkGate) { isSeq = true; isComb = false; }
      if (cellType == "combinational" || cellType == "unknown") {
        if (isClkGate) cellType = "clock_gating";
        else if (isSeq) cellType = "sequential";
      }

      // Timing: setup, hold
      std::string setupT = "N/A", holdT = "N/A";
      int arcCount = 0;
      if (lc) try { arcCount = lc->timingArcSets().size(); } catch (...) {}

      try {
        float mSetup = -1e10f, mHold = -1e10f;
        InstancePinIterator *pi = network->pinIterator(inst);
        if (pi) {
          const Corner *corner = sta->cmdCorner();
          DcalcAnalysisPt *apMin = corner ? corner->findDcalcAnalysisPt(MinMax::min()) : nullptr;
          DcalcAnalysisPt *apMax = corner ? corner->findDcalcAnalysisPt(MinMax::max()) : nullptr;
          while (pi->hasNext()) {
            Pin *pin = pi->next();
            if (!pin) continue;
            Vertex *v = graph->pinLoadVertex(pin);
            if (!v) continue;
            VertexOutEdgeIterator ei(v, graph);
            while (ei.hasNext()) {
              Edge *e = ei.next();
              if (!e) continue;
              TimingArcSet *as = e->timingArcSet();
              if (!as || !as->role()) continue;
              const TimingRole *role = as->role();
              Vertex *tv = e->to(graph);
              if (!tv || !tv->pin() || network->instance(tv->pin()) != inst) continue;
              DcalcAnalysisPt *ap = (role == TimingRole::hold() ||
                                     role == TimingRole::nonSeqHold()) ? apMin : apMax;
              if (!ap) continue;
              float md = -INF;
              if (!as->arcs().empty()) {
                for (TimingArc *a : as->arcs()) {
                  float d = sm->arcDelay(e, a, ap);
                  if (!std::isnan(d) && d != INF && d != -INF) md = std::max(md, d);
                }
              } else {
                float d = sm->arcDelay(e, nullptr, ap);
                if (!std::isnan(d) && d != INF && d != -INF) md = std::max(md, d);
              }
              if (md == -INF) continue;
              if (role == TimingRole::setup() || role == TimingRole::nonSeqSetup()) mSetup = std::max(mSetup, md);
              else if (role == TimingRole::hold() || role == TimingRole::nonSeqHold()) mHold = std::max(mHold, md);
            }
          }
          delete pi;
          if (mSetup > -1e9f) setupT = fmtTimingNs3(mSetup);
          if (mHold > -1e9f) holdT = fmtTimingNs3(mHold);
        }
      } catch (...) {}

      auto [hasClkIn, clkDomains] = getClockInfo(inst);

      // Power analysis
      std::string swPwr = "N/A", intPwr = "N/A", totPwr = "N/A";
      try {
        const Corner *corner = sta->cmdCorner();
        if (corner) {
          PowerResult pwr = sm->power(inst, corner);
          if (!std::isnan(pwr.switching())) swPwr = fmtPowerPw(pwr.switching());
          if (!std::isnan(pwr.internal())) intPwr = fmtPowerPw(pwr.internal());
          if (!std::isnan(pwr.total())) totPwr = fmtPowerPw(pwr.total());
        }
      } catch (...) {}

      // PVT extraction: instance -> global operating conditions -> library default
      std::string pvtProc = "N/A", pvtVolt = "N/A", pvtTemp = "N/A";
      std::string threshGrp = "N/A", opCond = "typical", procCorner = "tt";
      try {
        float pr = 0, vo = 0, te = 0;
        bool found = false;
        const Pvt *ipvt = sta->sdc()->pvt(inst, MinMax::max());
        if (ipvt) { pr = ipvt->process(); vo = ipvt->voltage(); te = ipvt->temperature(); found = true; }
        if (!found) {
          OperatingConditions *gop = sta->sdc()->operatingConditions(MinMax::max());
          if (gop) { pr = gop->process(); vo = gop->voltage(); te = gop->temperature(); found = true; }
        }
        if (!found && lc) {
          LibertyLibrary *lib = lc->libertyLibrary();
          if (lib) {
            OperatingConditions *dop = lib->defaultOperatingConditions();
            if (dop) { pr = dop->process(); vo = dop->voltage(); te = dop->temperature(); found = true; }
          }
        }
        if (found) {
          std::ostringstream o;
          o << std::fixed << std::setprecision(4) << pr; pvtProc = o.str();
          o.str(""); o << std::fixed << std::setprecision(4) << vo; pvtVolt = o.str();
          o.str(""); o << std::fixed << std::setprecision(2) << te; pvtTemp = o.str();
        }
      } catch (...) {}

      csv << fullName << "," << cellName << "," << libName << ","
          << cellType << ","
          << (isBuf?"true":"false") << "," << (isInv?"true":"false") << ","
          << (isMem?"true":"false") << "," << (isMacro?"true":"false") << ","
          << (isHier?"true":"false") << "," << area << "," << leakPwr << ","
          << pinCnt << "," << inCnt << "," << outCnt << ","
          << biCnt << "," << clkCnt << "," << dataCnt << "," << asyncCnt << ","
          << foLoad << "," << fiLoad << ","
          << (isComb?"true":"false") << "," << (isSeq?"true":"false") << ","
          << (isClkGate?"true":"false") << ","
          << setupT << "," << holdT << ","
          << arcCount << ","
          << (hasClkIn?"true":"false") << "," << clkDomains << ","
          << swPwr << "," << intPwr << "," << totPwr << ","
          << pvtProc << "," << pvtVolt << "," << pvtTemp << "\n";
    } catch (...) {
      log << "Exception processing " << fullName << "\n";
    }
  };

  // Pass 1: Leaf instances
  log << "Processing leaf instances...\n";
  try {
    LeafInstanceIterator *li = network->leafInstanceIterator();
    if (li) {
      int cnt = 0;
      while (li->hasNext()) {
        try {
          Instance *inst = li->next();
          if (!inst || inst == top) continue;
          cnt++;
          if (cnt % 1000 == 0) { log << "Processed " << cnt << " instances\n"; csv.flush(); }
          processCell(inst, safeStr(network->pathName(inst)));
        } catch (...) {}
      }
      delete li;
      log << "Leaf instances: " << cnt << "\n";
    }
  } catch (...) {}

  // Pass 2: Hierarchical instances
  log << "Processing hierarchical instances...\n";
  try {
    int hc = 0;
    std::function<void(Instance*)> walkHier = [&](Instance *parent) {
      if (!parent) return;
      InstanceChildIterator *ci = network->childIterator(parent);
      if (!ci) return;
      while (ci->hasNext()) {
        try {
          Instance *child = ci->next();
          if (!child || child == top || !network->isHierarchical(child)) continue;
          hc++;
          processCell(child, safeStr(network->pathName(child)));
          walkHier(child);
        } catch (...) {}
      }
      delete ci;
    };
    walkHier(top);
    log << "Hierarchical instances: " << hc << "\n";
  } catch (...) {}

  csv.close();
}


// ---- writeInstancePropertiesBenchmark ----
// Extracts only Tcl get_property accessible fields for speedup comparison.

void writeInstancePropertiesBenchmark(const Sta *sta, const std::string &filename) {
  const Network *network = sta->network();
  std::ofstream csv(filename);
  if (!csv.is_open()) return;

  csv << "full_name,name,ref_name,liberty_cell,is_buffer,is_inverter,"
         "is_macro,is_memory,is_clock_gate,is_hierarchical\n";

  LeafInstanceIterator *li = network->leafInstanceIterator();
  if (!li) { csv.close(); return; }

  int count = 0;
  while (li->hasNext()) {
    Instance *inst = li->next();
    if (!inst) continue;
    try {
      std::string fullName = safeStr(network->pathName(inst));
      std::string name = safeStr(network->name(inst));
      const Cell *cell = network->cell(inst);
      std::string refName = safeStr(cell ? network->name(cell) : nullptr);

      const LibertyCell *lc = network->libertyCell(inst);
      std::string lcName = "";
      bool isBuf = false, isInv = false, isMacro = false;
      bool isMem = false, isClkGate = false;
      if (lc) {
        lcName = safeStr(lc->name());
        isBuf = lc->isBuffer(); isInv = lc->isInverter();
        isMacro = lc->isMacro(); isMem = lc->isMemory();
        isClkGate = lc->isClockGate();
      }
      bool isHier = !network->isLeaf(inst);

      csv << csvEscape(fullName) << "," << csvEscape(name) << ","
          << csvEscape(refName) << "," << csvEscape(lcName) << ","
          << (isBuf?"true":"false") << "," << (isInv?"true":"false") << ","
          << (isMacro?"true":"false") << "," << (isMem?"true":"false") << ","
          << (isClkGate?"true":"false") << "," << (isHier?"true":"false") << "\n";
      count++;
    } catch (...) {}
  }
  delete li;
  csv.close();
  std::cout << "Wrote " << count << " instances to " << filename << std::endl;
}

} // namespace sta
