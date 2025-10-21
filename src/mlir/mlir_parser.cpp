/**
 * MLIR Parser with Context-Based Implementation
 *
 * Implements the universal MLIR parser pipeline as documented in:
 * devdocs/parser/universal-mlir-parser-design.md
 *
 * Pipeline:
 * 1. Context setup with extended dialect registration
 * 2. Parse MLIR text to ModuleOp
 * 3. Conditional normalization (VHLO→StableHLO for HLO dialects) - TODO
 * 4. CreateUniqueOpNamesPass() for stable IDs - TODO
 * 5. Graph building with recursive region traversal
 *
 * Features Implemented:
 * - Extended dialect support (Linalg, Tosa, Math, MemRef)
 * - Location-based naming (NameLoc, FusedLoc, CallSiteLoc)
 * - Per-function SSA value scoping
 * - SubgraphIds population for function calls
 * - Recursive region traversal with nested namespaces
 * - Helper input nodes for region block arguments
 *
 * Requires:
 * - MLIR libraries (from LLVM project)
 * - CMake build system
 * - C++17 or later
 */

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Location.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// Dialect includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// JSON output
#include <nlohmann/json.hpp>

#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cctype>

#include "llvm/ADT/DenseMap.h"

using json = nlohmann::json;
using namespace mlir;

namespace {

/**
 * Extract deterministic name from MLIR location metadata
 * Supports NameLoc and FusedLoc for stable node naming
 */
std::string extractLocationName(Location loc) {
    if (auto nameLoc = mlir::dyn_cast<NameLoc>(loc)) {
        return nameLoc.getName().str();
    }

    if (auto fusedLoc = mlir::dyn_cast<FusedLoc>(loc)) {
        // For fused locations, concatenate all names
        std::string result;
        for (auto subLoc : fusedLoc.getLocations()) {
            if (auto subNameLoc = mlir::dyn_cast<NameLoc>(subLoc)) {
                if (!result.empty()) result += "_";
                result += subNameLoc.getName().str();
            }
        }
        if (!result.empty()) return result;
    }

    // Callsite locations - extract from callee
    if (auto callSiteLoc = mlir::dyn_cast<CallSiteLoc>(loc)) {
        return extractLocationName(callSiteLoc.getCallee());
    }

    return "";  // No meaningful name available
}

/**
 * MLIR Context Manager
 * Handles dialect registration and context configuration
 */
class MLIRContextManager {
public:
    MLIRContextManager() {
        // Enable unregistered dialects FIRST for tolerant parsing
        // This allows custom/unknown dialects to be parsed without explicit registration
        context_.allowUnregisteredDialects(true);

        // Register common dialects for better pretty-printing and structural op support
        context_.loadDialect<func::FuncDialect>();
        context_.loadDialect<arith::ArithDialect>();
        context_.loadDialect<shape::ShapeDialect>();
        context_.loadDialect<scf::SCFDialect>();
        context_.loadDialect<tensor::TensorDialect>();
        context_.loadDialect<linalg::LinalgDialect>();
        context_.loadDialect<tosa::TosaDialect>();
        context_.loadDialect<math::MathDialect>();
        context_.loadDialect<memref::MemRefDialect>();
        context_.loadDialect<gpu::GPUDialect>();
        context_.loadDialect<vector::VectorDialect>();
        context_.loadDialect<cf::ControlFlowDialect>();

        // NOTE: Other dialects (TF, TFL, StableHLO, custom) work automatically
        // via allowUnregisteredDialects(true) without explicit registration

        std::cerr << "✓ MLIR context initialized with CF dialect support" << std::endl;
    }

    MLIRContext& getContext() { return context_; }

private:
    MLIRContext context_;
};

/**
 * Graph Builder
 * Converts MLIR ModuleOp to Model Explorer graph format
 */
class GraphBuilder {
public:
    GraphBuilder() : nodeIdCounter_(0) {}

    /**
     * Build graphs from MLIR module
     * Returns multi-graph format with one graph per function
     */
    json buildGraphs(ModuleOp module) {
        // First, collect all function names for subgraph ID resolution
        std::vector<std::string> functionNames;
        module.walk([&](func::FuncOp funcOp) {
            functionNames.push_back(funcOp.getSymName().str());
        });

        json result;
        result["graphs"] = json::array();

        // Walk all func.func operations and build graphs
        module.walk([&](func::FuncOp funcOp) {
            currentFunctionName_ = funcOp.getSymName().str();
            auto graph = buildFunctionGraph(funcOp, functionNames);
            result["graphs"].push_back(graph);
        });

        return result;
    }

private:
    int nodeIdCounter_;
    llvm::DenseMap<Value, std::string> valueToNodeId_;
    std::string currentFunctionName_;

    using NodeList = std::vector<json>;
    struct SectionInfo {
        std::string parentNamespace;
        std::string label;
        size_t nodeCount;
    };

    json buildFunctionGraph(func::FuncOp funcOp, const std::vector<std::string> &functionNames) {
        valueToNodeId_.clear();

        NodeList nodes;
        std::string funcName = funcOp.getSymName().str();

        auto &entryBlock = funcOp.getBody().front();
        for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
            auto arg = entryBlock.getArgument(i);
            auto inputNode = createInputNode(arg, i, funcName);
            nodes.push_back(inputNode);
            valueToNodeId_[arg] = inputNode["id"];
        }

        processRegion(funcOp.getBody(), funcName, functionNames, nodes);

        funcOp.walk([&](func::ReturnOp returnOp) {
            for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
                auto outputNode = createOutputNode(returnOp.getOperand(i), i, funcName);
                nodes.push_back(outputNode);
            }
        });

        size_t threshold = getLayerThreshold();
        auto sections = applyArtificialPartitioning(nodes, threshold);
        auto nodesWithLayers = addLayerGroupNodes(nodes, funcName, sections);

        json graph;
        graph["id"] = funcName;
        graph["nodes"] = json::array();
        for (const auto &node : nodesWithLayers) {
            graph["nodes"].push_back(node);
        }

        json tasksData = generateEdgeOverlaysForGraph(nodesWithLayers, funcName);
        if (!tasksData.is_null()) {
            graph["tasksData"] = tasksData;
        }

        return graph;
    }

    void processRegion(Region &region, const std::string &namespace_,
                       const std::vector<std::string> &functionNames, NodeList &nodes) {
        for (auto &block : region) {
            for (auto &op : block) {
                auto opNode = createOperationNode(&op, namespace_, functionNames);
                nodes.push_back(opNode);

                for (unsigned i = 0; i < op.getNumResults(); ++i) {
                    valueToNodeId_[op.getResult(i)] = opNode["id"];
                }

                if (op.getNumRegions() > 0) {
                    std::string opLabel = opNode["label"];
                    std::string opId = opNode["id"];

                    for (unsigned regionIdx = 0; regionIdx < op.getNumRegions(); ++regionIdx) {
                        auto &nestedRegion = op.getRegion(regionIdx);

                        std::string regionNamespace = "(" + opLabel + "_" + opId + ")/(region_" +
                                                     std::to_string(regionIdx) + ")";

                        if (!nestedRegion.empty()) {
                            auto &regionBlock = nestedRegion.front();
                            for (unsigned argIdx = 0; argIdx < regionBlock.getNumArguments(); ++argIdx) {
                                auto arg = regionBlock.getArgument(argIdx);
                                auto helperNode = createRegionInputNode(arg, argIdx, regionNamespace);
                                nodes.push_back(helperNode);
                                valueToNodeId_[arg] = helperNode["id"];
                            }
                        }

                        processRegion(nestedRegion, regionNamespace, functionNames, nodes);
                    }
                }
            }
        }
    }

    json createInputNode(BlockArgument arg, unsigned index, const std::string &funcName) {
        json node;
        node["id"] = funcName + "_input_" + std::to_string(index);
        node["label"] = "Input";
        node["namespace"] = funcName + "/Inputs";
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();

        json inputMetadata;
        inputMetadata["id"] = "0";
        inputMetadata["attrs"] = json::array();

        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType())) {
            json shapeAttr;
            shapeAttr["key"] = "tensor_shape";
            std::string typeStr;
            llvm::raw_string_ostream ss(typeStr);
            tensorType.print(ss);
            shapeAttr["value"] = ss.str();
            inputMetadata["attrs"].push_back(shapeAttr);
        }

        node["outputsMetadata"] = json::array();
        node["outputsMetadata"].push_back(inputMetadata);

        return node;
    }

    json createRegionInputNode(BlockArgument arg, unsigned index, const std::string &regionNamespace) {
        json node;
        node["id"] = regionNamespace + "_input_" + std::to_string(index);
        node["label"] = "input_" + std::to_string(index);
        node["namespace"] = regionNamespace;
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();

        json inputMetadata;
        inputMetadata["id"] = "0";
        inputMetadata["attrs"] = json::array();

        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType())) {
            json shapeAttr;
            shapeAttr["key"] = "tensor_shape";
            std::string typeStr;
            llvm::raw_string_ostream ss(typeStr);
            tensorType.print(ss);
            shapeAttr["value"] = ss.str();
            inputMetadata["attrs"].push_back(shapeAttr);
        }

        node["outputsMetadata"] = json::array();
        node["outputsMetadata"].push_back(inputMetadata);

        return node;
    }

    json createOutputNode(Value value, unsigned index, const std::string &funcName) {
        json node;
        node["id"] = funcName + "_output_" + std::to_string(index);
        node["label"] = "Output";
        node["namespace"] = funcName + "/Outputs";
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();

        if (valueToNodeId_.count(value)) {
            json edge;
            edge["sourceNodeId"] = valueToNodeId_[value];
            edge["sourceNodeOutputId"] = "0";
            edge["targetNodeInputId"] = "0";
            node["incomingEdges"].push_back(edge);
        }

        return node;
    }

    json createOperationNode(Operation *op, const std::string &currentNamespace,
                             const std::vector<std::string> &functionNames) {
        json node;
        node["id"] = currentFunctionName_ + "_op_" + std::to_string(nodeIdCounter_++);

        std::string locationName = extractLocationName(op->getLoc());
        if (!locationName.empty()) {
            node["label"] = locationName;
        } else {
            node["label"] = op->getName().getStringRef().str();
        }

        node["namespace"] = currentNamespace;
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();
        node["subgraphIds"] = json::array();

        if (auto callOp = dyn_cast<func::CallOp>(op)) {
            std::string callee = callOp.getCallee().str();
            if (std::find(functionNames.begin(), functionNames.end(), callee) != functionNames.end()) {
                node["subgraphIds"].push_back(callee);
            }
        }

        for (auto namedAttr : op->getAttrs()) {
            json attr;
            attr["key"] = namedAttr.getName().str();
            std::string attrValue;
            llvm::raw_string_ostream os(attrValue);
            namedAttr.getValue().print(os);
            attr["value"] = os.str();
            node["attrs"].push_back(attr);
        }

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            Value operand = op->getOperand(i);
            if (valueToNodeId_.count(operand)) {
                json edge;
                edge["sourceNodeId"] = valueToNodeId_[operand];
                edge["sourceNodeOutputId"] = "0";
                edge["targetNodeInputId"] = std::to_string(i);
                node["incomingEdges"].push_back(edge);
            }
        }

        node["outputsMetadata"] = json::array();
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
            json outputMeta;
            outputMeta["id"] = std::to_string(i);
            outputMeta["attrs"] = json::array();

            auto result = op->getResult(i);
            if (auto tensorType = mlir::dyn_cast<RankedTensorType>(result.getType())) {
                json shapeAttr;
                shapeAttr["key"] = "tensor_shape";
                std::string typeStr;
                llvm::raw_string_ostream ss(typeStr);
                tensorType.print(ss);
                shapeAttr["value"] = ss.str();
                outputMeta["attrs"].push_back(shapeAttr);
            }

            node["outputsMetadata"].push_back(outputMeta);
        }

        return node;
    }

    size_t getLayerThreshold() const {
        const char *env = std::getenv("MLIR_LAYER_THRESHOLD");
        if (env) {
            try {
                size_t value = static_cast<size_t>(std::stoull(env));
                if (value > 0) return value;
            } catch (...) {
            }
        }
        return 1000;
    }

    static std::string parentNamespaceOf(const std::string &ns) {
        if (ns.empty()) return "";
        auto pos = ns.find_last_of('/');
        if (pos == std::string::npos) return "";
        return ns.substr(0, pos);
    }

    static std::string lastSegmentOf(const std::string &ns) {
        if (ns.empty()) return "";
        auto pos = ns.find_last_of('/');
        if (pos == std::string::npos) return ns;
        return ns.substr(pos + 1);
    }

    static std::string sanitizeNamespaceForId(const std::string &ns) {
        std::string result;
        result.reserve(ns.size());
        for (char c : ns) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                result.push_back(c);
            } else {
                result.push_back('_');
            }
        }
        if (result.empty()) return "root";
        return result;
    }

    static std::string humanizeSegment(const std::string &segment, const std::string &fallback) {
        if (segment.empty()) return fallback;
        if (segment.rfind("__section_", 0) == 0) {
            std::string suffix = segment.substr(std::strlen("__section_"));
            return "Section " + suffix;
        }
        std::string cleaned = segment;
        if (!cleaned.empty() && cleaned.front() == '(' && cleaned.back() == ')') {
            cleaned = cleaned.substr(1, cleaned.size() - 2);
        }
        if (cleaned.empty()) return fallback;
        std::replace(cleaned.begin(), cleaned.end(), '_', ' ');
        return cleaned;
    }

    std::map<std::string, SectionInfo> applyArtificialPartitioning(NodeList &nodes, size_t threshold) {
        std::unordered_map<std::string, std::vector<size_t>> namespaceToIndices;
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::string ns = nodes[i].value("namespace", "");
            namespaceToIndices[ns].push_back(i);
        }

        std::map<std::string, SectionInfo> sections;

        for (auto &entry : namespaceToIndices) {
            const std::string &ns = entry.first;
            auto &indices = entry.second;

            std::vector<size_t> candidates;
            for (size_t idx : indices) {
                std::string label = nodes[idx].value("label", "");
                if (label == "Input" || label == "Output") continue;
                candidates.push_back(idx);
            }

            if (threshold == 0 || candidates.size() <= threshold) continue;

            size_t partitions = (candidates.size() + threshold - 1) / threshold;
            for (size_t part = 0; part < partitions; ++part) {
                size_t start = part * threshold;
                size_t end = std::min(start + threshold, candidates.size());
                std::string sectionNamespace = ns.empty()
                    ? "__section_" + std::to_string(part + 1) + "__"
                    : ns + "/__section_" + std::to_string(part + 1) + "__";

                for (size_t i = start; i < end; ++i) {
                    nodes[candidates[i]]["namespace"] = sectionNamespace;
                }

                sections.emplace(sectionNamespace, SectionInfo{
                    ns,
                    "Section " + std::to_string(part + 1),
                    end - start
                });
            }
        }

        return sections;
    }

    NodeList addLayerGroupNodes(const NodeList &inputNodes,
                                const std::string &rootNamespace,
                                const std::map<std::string, SectionInfo> &sections) {
        NodeList result = inputNodes;
        std::set<std::string> namespaceSet;

        for (const auto &node : inputNodes) {
            std::string ns = node.value("namespace", "");
            if (!ns.empty()) namespaceSet.insert(ns);
        }
        if (!rootNamespace.empty()) namespaceSet.insert(rootNamespace);

        std::set<std::string> expandedNamespaces;
        for (const auto &ns : namespaceSet) {
            std::string current = ns;
            while (!current.empty()) {
                expandedNamespaces.insert(current);
                current = parentNamespaceOf(current);
            }
        }
        if (!rootNamespace.empty()) expandedNamespaces.insert(rootNamespace);

        for (const auto &section : sections) {
            expandedNamespaces.insert(section.first);
        }

        std::vector<std::string> sortedNamespaces(expandedNamespaces.begin(), expandedNamespaces.end());
        std::sort(sortedNamespaces.begin(), sortedNamespaces.end(), [](const std::string &a, const std::string &b) {
            auto depth = [](const std::string &ns) {
                return std::count(ns.begin(), ns.end(), '/');
            };
            int depthA = depth(a);
            int depthB = depth(b);
            if (depthA == depthB) return a < b;
            return depthA < depthB;
        });

        std::unordered_set<std::string> existingIds;
        for (const auto &node : result) {
            existingIds.insert(node.value("id", ""));
        }

        for (const auto &ns : sortedNamespaces) {
            std::string parent = parentNamespaceOf(ns);
            bool isRoot = ns == rootNamespace;
            auto sectionIt = sections.find(ns);
            std::string segment = lastSegmentOf(ns);
            std::string label = sectionIt != sections.end()
                ? sectionIt->second.label + " (" + std::to_string(sectionIt->second.nodeCount) + ")"
                : humanizeSegment(segment, isRoot ? rootNamespace : (segment.empty() ? "Layer" : segment));

            std::string groupId = sanitizeNamespaceForId(ns) + "___group___";
            if (existingIds.count(groupId)) continue;
            existingIds.insert(groupId);

            json attrs = json::array();
            json layerAttr;
            layerAttr["key"] = "__layer__";
            layerAttr["value"] = "true";
            attrs.push_back(layerAttr);

            json style;
            style["backgroundColor"] = sectionIt != sections.end() ? "#FFFDE7" : "#E3F2FD";
            style["borderColor"] = sectionIt != sections.end() ? "#F9A825" : "#64B5F6";
            style["borderWidth"] = sectionIt != sections.end() ? 2.0 : 1.5;

            if (sectionIt != sections.end()) {
                json attr;
                attr["key"] = "__artificial_layer__";
                attr["value"] = "true";
                attrs.push_back(attr);

                json countAttr;
                countAttr["key"] = "__node_count__";
                countAttr["value"] = std::to_string(sectionIt->second.nodeCount);
                attrs.push_back(countAttr);
            }

            if (isRoot) {
                json rootAttr;
                rootAttr["key"] = "__root_layer__";
                rootAttr["value"] = "true";
                attrs.push_back(rootAttr);
            }

            json groupNode;
            groupNode["id"] = groupId;
            groupNode["label"] = label;
            groupNode["namespace"] = parent;
            groupNode["attrs"] = attrs;
            groupNode["incomingEdges"] = json::array();
            groupNode["style"] = style;

            result.push_back(groupNode);
        }

        return result;
    }

    json generateEdgeOverlaysForGraph(const NodeList &nodes, const std::string &functionName) {
        std::unordered_map<std::string, const json*> nodeById;
        for (const auto &node : nodes) {
            nodeById[node.value("id", "")] = &node;
        }

        std::vector<json> allEdges;
        std::vector<json> tensorEdges;
        std::vector<json> scalarEdges;

        for (const auto &node : nodes) {
            std::string targetId = node.value("id", "");
            if (!node.contains("incomingEdges")) continue;

            for (const auto &edge : node["incomingEdges"]) {
                std::string sourceId = edge.value("sourceNodeId", "");
                std::string outputId = edge.value("sourceNodeOutputId", "");
                if (sourceId.empty()) continue;

                auto it = nodeById.find(sourceId);
                if (it == nodeById.end()) continue;

                const auto *sourceNode = it->second;
                if (!sourceNode->contains("outputsMetadata")) continue;

                std::string label;
                for (const auto &meta : (*sourceNode)["outputsMetadata"]) {
};

} // anonymous namespace

/**
 * Main parsing function
 * Implements the documented MLIR parsing pipeline
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    try {
        // Read MLIR content from stdin
        std::ostringstream buffer;
        buffer << std::cin.rdbuf();
        std::string mlirContent = buffer.str();

        if (mlirContent.empty()) {
            json error;
            error["error"] = "Empty input";
            error["message"] = "No MLIR content provided";
            std::cout << error.dump(2) << std::endl;
            return 1;
        }

        // Step 1: Create MLIR context with dialect registration
        MLIRContextManager contextManager;
        auto &context = contextManager.getContext();

        // Step 2: Parse MLIR text to ModuleOp
        auto module = parseSourceString<ModuleOp>(mlirContent, &context);
        if (!module) {
            json error;
            error["error"] = "MLIR parsing failed";
            error["message"] = "Failed to parse MLIR content. Check syntax and dialect usage.";
            std::cout << error.dump(2) << std::endl;
            return 1;
        }

        std::cerr << "✓ MLIR parsed successfully" << std::endl;

        // Step 3: Verify module
        if (failed(verify(*module))) {
            json error;
            error["error"] = "MLIR verification failed";
            error["message"] = "Module verification failed. Check IR validity.";
            std::cout << error.dump(2) << std::endl;
            return 1;
        }

        std::cerr << "✓ MLIR module verified" << std::endl;

        // Step 4: Apply passes for normalization and uniquing
        // TODO: Implement conditional normalization (VHLO→StableHLO)
        // TODO: Implement CreateUniqueOpNamesPass
        PassManager pm(&context);
        // pm.addPass(createSymbolDCEPass());  // Example pass

        if (failed(pm.run(*module))) {
            std::cerr << "⚠ Pass execution failed, continuing with unoptimized module" << std::endl;
        }

        // Step 5: Build graphs from module
        GraphBuilder builder;
        json graphs = builder.buildGraphs(*module);

        // Add metadata
        graphs["_metadata"] = {
            {"parser", "mlir-context-cpp"},
            {"functions_parsed", graphs["graphs"].size()}
        };

        // Output JSON
        std::cout << graphs.dump(2) << std::endl;

        std::cerr << "✓ Graph generation complete" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        json error;
        error["error"] = "Exception during parsing";
        error["message"] = e.what();
        std::cout << error.dump(2) << std::endl;
        return 1;
    }
}
