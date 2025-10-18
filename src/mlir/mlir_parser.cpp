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
            auto graph = buildFunctionGraph(funcOp, functionNames);
            result["graphs"].push_back(graph);
        });

        return result;
    }

private:
    int nodeIdCounter_;
    llvm::DenseMap<Value, std::string> valueToNodeId_;

    /**
     * Build graph for a single function
     */
    json buildFunctionGraph(func::FuncOp funcOp, const std::vector<std::string> &functionNames) {
        // IMPORTANT: Reset SSA map for each function to ensure proper scoping
        valueToNodeId_.clear();

        json graph;
        graph["id"] = funcOp.getSymName().str();
        graph["nodes"] = json::array();

        std::string funcName = funcOp.getSymName().str();

        // Create input nodes from function arguments
        auto &entryBlock = funcOp.getBody().front();
        for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
            auto arg = entryBlock.getArgument(i);
            auto inputNode = createInputNode(arg, i, funcName);
            graph["nodes"].push_back(inputNode);

            // Map argument value to node ID for edge creation
            valueToNodeId_[arg] = inputNode["id"];
        }

        // Process the function body region recursively
        processRegion(funcOp.getBody(), funcName, functionNames, graph["nodes"]);

        // Create output nodes from return operation
        funcOp.walk([&](func::ReturnOp returnOp) {
            for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
                auto outputNode = createOutputNode(returnOp.getOperand(i), i, funcName);
                graph["nodes"].push_back(outputNode);
            }
        });

        return graph;
    }

    /**
     * Process a region recursively, creating nodes with proper namespaces
     */
    void processRegion(Region &region, const std::string &namespace_,
                       const std::vector<std::string> &functionNames, json &nodes) {
        // Process each block in the region
        for (auto &block : region) {
            // Process each operation in the block
            for (auto &op : block) {
                // Create node for this operation
                auto opNode = createOperationNode(&op, namespace_, functionNames);
                nodes.push_back(opNode);

                // Map operation results to node IDs for edge creation
                for (unsigned i = 0; i < op.getNumResults(); ++i) {
                    valueToNodeId_[op.getResult(i)] = opNode["id"];
                }

                // Process nested regions if this operation has any
                if (op.getNumRegions() > 0) {
                    std::string opLabel = opNode["label"];
                    std::string opId = opNode["id"];

                    // Process each nested region
                    for (unsigned regionIdx = 0; regionIdx < op.getNumRegions(); ++regionIdx) {
                        auto &nestedRegion = op.getRegion(regionIdx);

                        // Create namespace for nested region: (opLabel_id)/(region_i)
                        std::string regionNamespace = "(" + opLabel + "_" + opId + ")/(region_" +
                                                     std::to_string(regionIdx) + ")";

                        // Create helper input nodes for region block arguments
                        if (!nestedRegion.empty()) {
                            auto &regionBlock = nestedRegion.front();
                            for (unsigned argIdx = 0; argIdx < regionBlock.getNumArguments(); ++argIdx) {
                                auto arg = regionBlock.getArgument(argIdx);
                                auto helperNode = createRegionInputNode(arg, argIdx, regionNamespace);
                                nodes.push_back(helperNode);

                                // Map region argument to helper node ID
                                valueToNodeId_[arg] = helperNode["id"];
                            }
                        }

                        // Recursively process the nested region
                        processRegion(nestedRegion, regionNamespace, functionNames, nodes);
                    }
                }
            }
        }
    }

    /**
     * Create input node from function argument
     */
    json createInputNode(BlockArgument arg, unsigned index, const std::string &funcName) {
        json node;
        node["id"] = funcName + "_input_" + std::to_string(index);
        node["label"] = "Input";
        node["namespace"] = funcName + "/Inputs";
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();

        // Add input metadata
        json inputMetadata;
        inputMetadata["id"] = "0";
        inputMetadata["attrs"] = json::array();

        // Add tensor shape information
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

    /**
     * Create helper input node for region block arguments
     */
    json createRegionInputNode(BlockArgument arg, unsigned index, const std::string &regionNamespace) {
        json node;
        node["id"] = regionNamespace + "_input_" + std::to_string(index);
        node["label"] = "input_" + std::to_string(index);
        node["namespace"] = regionNamespace;
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();

        // Add input metadata with tensor shape
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

    /**
     * Create output node from return operand
     */
    json createOutputNode(Value value, unsigned index, const std::string &funcName) {
        json node;
        node["id"] = funcName + "_output_" + std::to_string(index);
        node["label"] = "Output";
        node["namespace"] = funcName + "/Outputs";
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();

        // Add incoming edge from producing operation
        if (valueToNodeId_.count(value)) {
            json edge;
            edge["sourceNodeId"] = valueToNodeId_[value];
            edge["sourceNodeOutputId"] = "0";  // Simplified
            edge["targetNodeInputId"] = "0";
            node["incomingEdges"].push_back(edge);
        }

        return node;
    }

    /**
     * Create operation node with location-based naming and subgraph resolution
     */
    json createOperationNode(Operation *op, const std::string &funcName, const std::vector<std::string> &functionNames) {
        json node;
        node["id"] = funcName + "_op_" + std::to_string(nodeIdCounter_++);

        // Try to extract deterministic name from location metadata
        std::string locationName = extractLocationName(op->getLoc());
        if (!locationName.empty()) {
            node["label"] = locationName;  // Use location-based name for determinism
        } else {
            node["label"] = op->getName().getStringRef().str();  // Fall back to op name
        }

        node["namespace"] = funcName;
        node["attrs"] = json::array();
        node["incomingEdges"] = json::array();
        node["subgraphIds"] = json::array();  // Initialize for function calls

        // Populate subgraphIds for function call operations
        if (auto callOp = dyn_cast<func::CallOp>(op)) {
            std::string callee = callOp.getCallee().str();
            // Check if callee is a known function in this module
            if (std::find(functionNames.begin(), functionNames.end(), callee) != functionNames.end()) {
                node["subgraphIds"].push_back(callee);
            }
        }

        // Add attributes
        for (auto namedAttr : op->getAttrs()) {
            json attr;
            attr["key"] = namedAttr.getName().str();
            std::string attrValue;
            llvm::raw_string_ostream os(attrValue);
            namedAttr.getValue().print(os);
            attr["value"] = os.str();
            node["attrs"].push_back(attr);
        }

        // Add incoming edges from operands
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            Value operand = op->getOperand(i);
            if (valueToNodeId_.count(operand)) {
                json edge;
                edge["sourceNodeId"] = valueToNodeId_[operand];
                edge["sourceNodeOutputId"] = "0";  // Simplified
                edge["targetNodeInputId"] = std::to_string(i);
                node["incomingEdges"].push_back(edge);
            }
        }

        // Add output metadata
        node["outputsMetadata"] = json::array();
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
            json outputMeta;
            outputMeta["id"] = std::to_string(i);
            outputMeta["attrs"] = json::array();

            // Add tensor shape if available
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
